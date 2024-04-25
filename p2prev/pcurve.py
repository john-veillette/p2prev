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
