from scipy.stats import norm, expon, uniform
from matplotlib import pyplot as plt
import numpy as np

def p_curve_lik(p, delta):
    '''
    probability mass function for a p-values of a
    one-tailed Z-test
    '''
    Z = norm.isf(p, loc = 0)
    return norm.pdf(Z, loc = delta) / norm.pdf(Z, loc = 0)

def p_cdf(p, delta):
    '''
    cumulative distribution function for p-values
    '''
    Z = norm.isf(p, loc = 0)
    return norm.sf(Z, loc = delta)

def _half_p_lik(p, d, nu = 3):
    n = (nu + 2)/2 # "sample size"
    mu = d * np.sqrt(n/2) # non-centrality parameter
    T = t.isf(p, df = nu, loc = 0)
    return t.pdf(T, df = nu, loc = mu) / t.pdf(T, df = nu, loc = 0)

def p_curve_lik_overdisp(p, d, nu = 5):
    '''
    probability mass function for a p-values of a
    two-tailed t-test with effect size `d` and degrees
    of freedom `nu`.
    '''
    return (_half_p_lik(p/2, d, nu) + _half_p_lik(1 - p/2, d, nu))/2
