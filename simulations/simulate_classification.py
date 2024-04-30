from scipy.stats import binom, beta, permutation_test
from pymc.distributions.continuous import Beta
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

N_SIMULATIONS = 5
N_TRIALS = 50
ALPHA = .05
HDI_PROB = .95
# average classifications accuracies for the two simulation types
EFFSIZE_LOW = .6
EFFSIZE_HIGH = .7


def half_beta_rv(p_mean, size = 1, seed = None):
    '''
    generates a random value between 0.5 and 1.0 with specified mean
    '''
    trunc_mean = (p_mean - .5) * 2
    a, b = Beta.get_alpha_beta(mu = trunc_mean, sigma = .1)
    trunc_rv = beta.rvs(a, b, size = size, random_state = seed)
    return np.squeeze(trunc_rv/2 + .5)

def simulate_subject(prob_correct, seed = None):
    '''
    Simulates actual (y) and predicted (y_hat) values with,
    on average, the given accuracy.
    '''
    if prob_correct == .5:
        accuracy = .5
    else:
        accuracy = half_beta_rv(prob_correct, seed = seed)
    y = binom.rvs(1, .5, size = N_TRIALS)
    y_not = np.logical_not(y).astype(int)
    correct = binom.rvs(1, accuracy, size = y.size, random_state = seed)
    y_hat = np.stack([y_not, y])[correct, np.arange(y.size)]
    return np.stack([y, y_hat], axis = 1)

def select_pvals(pvals_H0, pvals_H1, prev_H1, n_subs, seed = None):
    rng = np.random.default_rng(seed)
    idxs = rng.choice(np.arange(pvals_H0.size), size = n_subs)
    H1_true = binom.rvs(1, prev_H1, size = n_subs)
    pvals = np.stack([pvals_H0, pvals_H1])[H1_true, idxs]
    return pvals

def simulation(pvals_H0, pvals_H1, prev_H1, n_subs, seed = None):
    pvals = select_pvals(pvals_H0, pvals_H1, prev_H1, n_subs, seed)
    model = PCurveMixture(pvals, progressbar = False)
    model.fit()
    hdi = model.prevalence_hdi(HDI_PROB)
    res = dict(
        prevalence = prev_H1,
        pcurve_expectation = model.prevalence.mean(),
        pcurve_hdi_low = hdi[0],
        pcurve_hdi_high = hdi[1]
    )
    k = (pvals <= ALPHA).sum()
    n = len(pvals)
    model = BinomialOutcomesModel(k, n, ALPHA, progressbar = False)
    model.fit()
    res['binom_expectation'] = model.prevalence.mean()
    hdi = model.prevalence_hdi(HDI_PROB)
    res['binom_hdi_low'] = hdi[0]
    res['binom_hdi_high'] = hdi[1]
    return res

perm_test = lambda data: permutation_test(
    [data[...,0], data[...,1]],
    statistic = lambda y, y_hat, axis: (y == y_hat).mean(axis),
    vectorized = True,
    axis = 1,
    alternative = 'greater',
    permutation_type = 'pairings',
    n_resamples = 1000
).pvalue

def main(n_subjects, power = 'high'):

    # generate a bunch of p-values from a permutation test for accuracy
    n_subs = 10000 # to choose from during simulations
    data = np.stack([simulate_subject(.5, i) for i in range(n_subs)])
    pvals_H0 = perm_test(data)
    data = np.stack([simulate_subject(EFFSIZE_LOW, i) for i in range(n_subs)])
    pvals1 = perm_test(data)
    data = np.stack([simulate_subject(EFFSIZE_HIGH, i) for i in range(n_subs)])
    pvals2 = perm_test(data)

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
    out_dir = 'classification'
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
