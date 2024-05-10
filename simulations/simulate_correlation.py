from scipy.stats import binom, pearsonr
import numpy as np
import pandas as pd
import argparse
import json
import sys
import os

cwd = os.getcwd()
pardir = os.path.abspath(os.path.join(cwd, os.pardir))
sys.path.append(pardir)
from p2prev import PCurveMixture
from p2prev._benchmarking import BinomialOutcomesModel

N_SIMULATIONS = 1000
N_TRIALS = 50
ALPHA = .05
HDI_PROB = .95
# average classifications accuracies for the two simulation types
EFFSIZE_LOW = .2
EFFSIZE_HIGH = .4


def sim_subs(n_subs, n_trials, effsize = 0, seed = None):
    '''
    generate data with specified correlation
    '''
    rng = np.random.default_rng(seed)
    rs = effsize * np.ones(n_subs)
    xy = [
        rng.multivariate_normal([0,0], [[1,r],[r,1]], size = n_trials)
        for r in rs
    ]
    ps = [
        pearsonr(dat[:,0], dat[:,1]).pvalue # two sided p-value
        for dat in xy
    ]
    return np.array(ps)

def select_pvals(pvals_H0, pvals_H1, prev_H1, n_subs, seed = None):
    rng = np.random.default_rng(seed)
    idxs = rng.choice(np.arange(pvals_H0.size), size = n_subs)
    H1_true = binom.rvs(1, prev_H1, size = n_subs)
    pvals = np.stack([pvals_H0, pvals_H1])[H1_true, idxs]
    return pvals

def simulation(pvals_H0, pvals_H1, prev_H1, n_subs, seed = None):
    pvals = select_pvals(pvals_H0, pvals_H1, prev_H1, n_subs, seed)
    model = PCurveMixture(pvals, progressbar = False, nuts_sampler = 'numpyro')
    model.fit()
    hdi = model.prevalence_hdi(HDI_PROB)
    res = dict(
        prevalence = prev_H1,
        pcurve_expectation = model.prevalence.mean(),
        pcurve_hdi_low = hdi[0],
        pcurve_hdi_high = hdi[1],
        pcurve_map = model.map
    )
    k = (pvals <= ALPHA).sum()
    n = len(pvals)
    model = BinomialOutcomesModel(
        k, n, ALPHA,
        progressbar = False,
        nuts_sampler = 'numpyro'
        )
    model.fit()
    res['binom_expectation'] = model.prevalence.mean()
    hdi = model.prevalence_hdi(HDI_PROB)
    res['binom_hdi_low'] = hdi[0]
    res['binom_hdi_high'] = hdi[1]
    res['binom_map'] = model.map
    return res

def main(n_subjects, power = 'high'):

    # generate a bunch of p-values from a permutation test for accuracy
    n_subs = 100000 # to choose from during simulations
    pvals_H0 = sim_subs(n_subs, N_TRIALS, 0., seed = 0)
    pvals1 = sim_subs(n_subs, N_TRIALS, EFFSIZE_LOW, seed = 1)
    pvals2 = sim_subs(n_subs, N_TRIALS, EFFSIZE_HIGH, seed = 2)

    # make prevalence for one simulation type
    # the same as power for the other type
    if power == 'low':
        pvals_H1 = pvals1
        prev_H1 = (pvals2 <= ALPHA).mean()
    elif power == 'high':
        pvals_H1 = pvals2
        prev_H1 = (pvals1 <= ALPHA).mean()

    results = []
    for sim in range(N_SIMULATIONS):
        if sim % 100 == 0:
            print('Beginning simulation %d...'%(sim + 1))
        res = simulation(pvals_H0, pvals_H1, prev_H1, n_subjects, seed = sim)
        results.append(res)
    print('Simulations complete!')

    df = pd.DataFrame(results)
    out_dir = 'correlation'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    fpath = os.path.join(out_dir, 'subjects-%d_power-%s.csv'%(n_subjects,power))
    df.to_csv(fpath, index = False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('n_subs', type = int)
    parser.add_argument('power', type = str)
    args = parser.parse_args()
    main(args.n_subs, args.power)
