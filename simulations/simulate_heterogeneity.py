from simulate_classification import (
    simulate_subject,
    simulation,
    perm_test
    )
import numpy as np
import pandas as pd
from time import time
import argparse
import os

AVG_EFFSIZE = .65
PREV = .4
N_SUB = 30

def run_simulation(sigma, seed):
    # generate p-values to draw from in simulation
    n_subs = N_SUB*10 # some extra cause `simulate` does subselection
    data = np.stack([simulate_subject(.5, seed=seed*i) for i in range(n_subs)])
    pvals_H0 = perm_test(data)
    data = np.stack([simulate_subject(
        AVG_EFFSIZE,
        seed = seed*i,
        sigma = sigma
        ) for i in range(n_subs)
        ])
    pvals_H1 = perm_test(data)
    res = simulation(pvals_H0, pvals_H1, PREV, N_SUB, seed = seed)
    res['sigma'] = sigma
    res['seed'] = seed
    return res


def main(seed):
    results = []
    for sigma in np.arange(.01, .11, .01):
        print('Starting sigma = %f'%sigma)
        results.append(run_simulation(sigma, seed))
    df = pd.DataFrame(results)
    out_dir = 'heterogeneity'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    fpath = os.path.join(out_dir, 'seed-%d.csv'%seed)
    df.to_csv(fpath, index = False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('seed', type = int)
    args = parser.parse_args()
    t0 = time()
    main(args.seed)
    t1 = time()
    print('Simulations took %.02f minutes'%((t1-t0)/60))
