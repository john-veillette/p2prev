import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import minmax_scale
import pandas as pd
import mne
mne.set_log_level(False)
import json
from time import time
import argparse
import arviz as az

import sys
import os
import tempfile
cwd = os.getcwd()
pardir = os.path.abspath(os.path.join(cwd, os.pardir))
sys.path.append(pardir)
from p2prev import PCurveMixture
from p2prev._benchmarking import BinomialOutcomesModel

PREV_HIGH = .9
PREV_LOW = .3
EFFSIZE_HIGH = 2.5 # not on scale with effect size param of model
EFFSIZE_LOW = 1.5
ALPHA = .05
HDI_PROB = .95
N_SUBJECTS = 30 # n_trials is hard coded in make_epochs() as 100

def load_eeg():
    '''
    Loads some raw EEG to use as background noise and a realistic
    ERP topography + timecourse as in Sassenhagen and Draschkow (2019),
    see their orginal simulation code at https://osf.io/xf53t/.

    Returns
    ---------
    raw : mne.Raw
    erp : np.array
    '''
    ## retrieve example EEGLab dataset contained in MNE installation
    sample_data_dir = mne.datasets.testing.data_path()
    eeg_fpath = os.path.join(sample_data_dir, 'EEGLAB', 'test_raw_onefile.set')
    raw = mne.io.read_raw_eeglab(eeg_fpath, preload = True)
    ## and add channel names and locations
    locs_fpath = os.path.join(sample_data_dir, 'EEGLAB', 'test_chans.locs')
    dig = mne.channels.read_custom_montage(locs_fpath)
    mapping = {raw.ch_names[i]: dig.ch_names[i] for i in range(len(raw.ch_names))}
    raw = raw.rename_channels(mapping)
    raw = raw.set_montage(dig)
    ## preprocess the raw time series
    # default at time of Sassenhagen and Draschkow (2019) was max_iter = 200
    # but MNE changed the default to max_iter = 1000 for fastica
    # so we hard code S&D's settings for reproducibility
    ica = mne.preprocessing.ICA(n_components=20, random_state=0, max_iter=200)
    ica.fit(raw.copy().filter(20, 50))
    ica.exclude = [3, 13, 16]
    raw = ica.apply(raw, exclude = ica.exclude).filter(.1, 30)
    raw.resample(100)
    ## now generate a fake ERP timecourse with one of the IC topographies
    topo = ica.get_components()[:, 1]
    pre_stim = np.zeros(15)
    post_stim = np.zeros(15)
    erp = minmax_scale(norm.pdf(np.linspace(-1.5, 1.5, 21)))
    erp = np.hstack((pre_stim, erp, post_stim)) * 1e-5 * 1.5
    erp = np.array([erp] * 32) * -topo[:, np.newaxis]
    return raw, erp

def make_epochs(raw, erp, effsize, seed = None):
    '''
    Arguments
    -----------
    raw : mne.Raw
        From which to cut out chunks to use as realistic EEG background noise.
    erp : np.array
        An ERP topography/timecourse
    effsize : float
        to be multiplied by `erp` before adding it to background noise

    Returns
    ----------
    epochs : mne.Epochs
    '''
    # make arbitrary events
    rng = np.random.default_rng(seed)
    raw_onset = rng.uniform()
    raw_ = raw.copy().crop(raw_onset)
    events = mne.make_fixed_length_events(raw_, duration = .5)
    # subsample to 100 events
    events = events[sorted(np.random.choice(len(events), size = 100, replace = False))]
    # randomly assign condition labels
    conds = np.array(events.shape[0]//2 * [1] + events.shape[0]//2 * [2])
    rng.shuffle(conds)
    events[:, 2] = conds
    # now slice into epochs
    epochs = mne.Epochs(raw_, events, preload = True).apply_baseline().crop(0, .5)
    events = epochs.events
    # add effect to only conditions 2
    effect = np.array([
        erp if events[i, 2] == 2 else np.zeros(erp.shape)
        for i in range(events.shape[0])
    ])
    epochs._data += (effect * effsize) # in place
    return epochs.drop_channels(["T8"])

def simulate_group(raw, erp, prev, effsize, n_subs = N_SUBJECTS, seed = None):
    '''
    simulates a group of subjects with specified H1 prevalence and effect size

    Returns
    ---------
    erp_data : np.array of shape (n_subs, n_times, n_channels)
        The ERP difference waves for subjects in group
    ps_sub : np.array of shape (n_subs,)
        The within-subject p-value for each subject in group
    adjacency
        Adjacency matrix for channels
    H1_frac_rej : int
        Fraction of subjects for whom H1 where null was rejected
    '''
    ps_sub = []
    cond1 = []
    cond2 = []
    n_rej = 0

    # decide which subjects will express H1
    rng = np.random.default_rng(seed)
    H1_true = rng.binomial(1, prev, size = n_subs)

    for i in range(n_subs):
        # generate data for one subject
        epochs = make_epochs(raw, erp, effsize * H1_true[i], seed = rng)
        if i == 0:
            adj, _ = mne.channels.find_ch_adjacency(epochs.info, 'eeg')
        # perform indep. sample permutation test within subject
        epo1 = epochs['1'].get_data(copy = True)
        epo2 = epochs['2'].get_data(copy = True)
        data = [epo1.swapaxes(1, 2), epo2.swapaxes(1, 2)]
        _, _, ps, _ = mne.stats.permutation_cluster_test(
            data, adjacency = adj,
            tail = 1, # 1-tailed F-test same as 2-tailed t-test
            n_jobs = -1, seed = rng,
            n_permutations = 1024
        )
        if ps.size > 0:
            p = ps.min()
        else: # no clusters found!
            p = 1/(1024 + 1) # so assign highest p-val
        if H1_true[i] == 1:
            n_rej += int(p <= ALPHA)
        ps_sub.append(p)
        # save noisy ERPs for later group-level perm test
        cond1.append(epochs['1'].average())
        cond2.append(epochs['2'].average())

    # now do a group-level test
    evo1 = np.stack([evo.get_data() for evo in cond1])
    evo2 = np.stack([evo.get_data() for evo in cond2])
    data = (evo2 - evo1).swapaxes(1, 2)
    return data, np.array(ps_sub), adj, n_rej / H1_true.sum()

def fit_models(pvals):
    model = PCurveMixture(pvals, progressbar = False)
    model.fit()
    pcurve = dict(
        prev = model.prevalence,
        pow = model.posterior_predictive_power(ALPHA).power.to_numpy()
    )
    k = (pvals <= ALPHA).sum()
    n = len(pvals)
    model = BinomialOutcomesModel(k, n, ALPHA, progressbar = False)
    model.fit()
    binom = dict(
        prev = model.prevalence,
        pow = model.power
    )
    return pcurve, binom

def compare_models(mod0, mod1):
    '''
    If population prevalence and power per group are considered as
    fixed effects, then there's no pooling between them and we can just
    subtract samples.
    '''
    prev_diff = mod1['prev'] - mod0['prev']
    pow_diff = mod1['pow'] - mod0['pow']
    res = dict(
        prevdiff_exp = prev_diff.mean(),
        prevdiff_hdi = az.hdi(prev_diff, hdi_prob = HDI_PROB).tolist(),
        prevdiff_prob = (prev_diff > 0).mean(),
        powdiff_exp = pow_diff.mean(),
        powdiff_hdi = az.hdi(pow_diff, hdi_prob = HDI_PROB).tolist(),
        powdiff_prob = (pow_diff > 0).mean(),
    )
    return res

def test_difference(erp0, erp1, adj, seed = None):
    rng = np.random.default_rng(seed)
    data = [erp0, erp1]
    _, _, ps, _ = mne.stats.permutation_cluster_test(
        data, adjacency = adj,
        tail = 1, # 1-tailed F-test same as 2-tailed t-test
        n_jobs = -1, seed = rng,
        n_permutations = 1024
    )
    if ps.size > 0:
        return ps.min()
    else:
        return 1.

def main(seed):

    # setup
    rng = np.random.default_rng(seed)
    raw, erp = load_eeg()
    res = dict(
        increase_prev = dict(),
        increase_pow = dict(),
        baseline = dict()
    )

    # simulate baseline group
    erp0, ps0, adj, rej = simulate_group(raw, erp, PREV_LOW, EFFSIZE_LOW, seed = rng)
    pcurve0, binom0 = fit_models(ps0)
    res['baseline']['frac_H1_rej'] = rej

    # simulate another group with increase in prevalence
    erp1, ps1, _, rej = simulate_group(raw, erp, PREV_HIGH, EFFSIZE_LOW, seed = rng)
    pcurve1, binom1 = fit_models(ps1)
    res['increase_prev']['pcurve'] = compare_models(pcurve0, pcurve1)
    res['increase_prev']['binom'] = compare_models(binom0, binom1)
    res['increase_prev']['pval_group'] = test_difference(erp0, erp1, adj, rng)
    res['increase_prev']['frac_H1_rej'] = rej

    # and one with increase in power
    erp2, ps2, _, rej = simulate_group(raw, erp, PREV_LOW, EFFSIZE_HIGH, seed = rng)
    pcurve2, binom2 = fit_models(ps2)
    res['increase_pow']['pcurve'] = compare_models(pcurve0, pcurve2)
    res['increase_pow']['binom'] = compare_models(binom0, binom2)
    res['increase_pow']['pval_group'] = test_difference(erp0, erp2, adj, rng)
    res['increase_pow']['frac_H1_rej'] = rej

    # now save results
    out_dir = 'EEG'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    fpath = os.path.join(out_dir, 'seed-%d.json'%seed)
    with open(fpath, 'w', encoding = 'utf-8') as f:
        json.dump(res, f, ensure_ascii = False, indent = 4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('seed', type = int)
    args = parser.parse_args()
    print('Starting simulation.')
    t0 = time()
    main(args.seed)
    t1 = time()
    print('Completed simulation in %.01f seconds.'%(t1 - t0))
