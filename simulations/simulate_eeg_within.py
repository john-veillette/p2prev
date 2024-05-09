from time import time
import argparse
import numpy as np
import json
import mne

import sys
import os
import tempfile
cwd = os.getcwd()
pardir = os.path.abspath(os.path.join(cwd, os.pardir))
sys.path.append(pardir)
from p2prev import PCurveWithinGroupDifference

from simulate_eeg_between import (
    load_eeg, # functions
    simulate_group,
    PREV_HIGH, # hard-coded constants
    PREV_LOW,
    EFFSIZE_HIGH,
    EFFSIZE_LOW,
    ALPHA,
    HDI_PROB,
    N_SUBJECTS
)

def fit_model(pvals1, pvals2):
    model = PCurveWithinGroupDifference(pvals1, pvals2, progressbar = False)
    model.fit()
    res = dict(
        prevdiff_exp = model.prevalence_diff.mean(),
        prevdiff_hdi = model.prevalence_diff_hdi(HDI_PROB).tolist(),
        prevdiff_prob = model.prob_H2_prev_greater,
        powdiff_exp = model.power_diff(ALPHA).mean(),
        powdiff_hdi = model.power_diff_hdi(ALPHA, HDI_PROB).tolist(),
        powdiff_prob = model.prob_H2_effect_size_greater
    )
    return res


def test_difference(erp0, erp1, adj, seed = None):
    rng = np.random.default_rng(seed)
    _, _, ps, _ = mne.stats.permutation_cluster_1samp_test(
        erp1 - erp0, adjacency = adj,
        tail = 0, # two-tailed
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

    # decide which subjects will express effects
    H_true_high = rng.binomial(1, PREV_HIGH, size = N_SUBJECTS)
    # use conditional binomial rule to generate subset with PREV_LOW
    _H_true = rng.binomial(1, PREV_LOW / PREV_HIGH , size = N_SUBJECTS)
    H_true_low = np.logical_and(H_true_high, _H_true).astype(int)

    # simulate baseline group
    erp0, ps0, adj, rej = simulate_group(
        raw, erp, PREV_LOW, EFFSIZE_LOW,
        seed = rng, H1_true = H_true_low
    )
    res['baseline']['frac_H1_rej'] = rej

    # simulate another group with increase in prevalence
    erp1, ps1, _, rej = simulate_group(
        raw, erp, PREV_HIGH, EFFSIZE_LOW,
        seed = rng, H1_true = H_true_high
        )
    res['increase_prev']['pcurve'] = fit_model(ps0, ps1)
    res['increase_prev']['pval_group'] = test_difference(erp0, erp1, adj, rng)
    res['increase_prev']['frac_H1_rej'] = rej

    # and one with increase in power
    erp2, ps2, _, rej = simulate_group(
        raw, erp, PREV_LOW, EFFSIZE_HIGH,
        seed = rng, H1_true = H_true_low
    )
    res['increase_pow']['pcurve'] = fit_model(ps0, ps2)
    res['increase_pow']['pval_group'] = test_difference(erp0, erp2, adj, rng)
    res['increase_pow']['frac_H1_rej'] = rej

    # now save results
    out_dir = 'within'
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
