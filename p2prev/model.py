import pymc as pm
import pandas as pd
import numpy as np
import arviz as az
from arviz.stats.stats import _calculate_ics

from .pcurve import p_cdf

def p_curve_loglik(p, delta):
    Z = -pm.Normal.icdf(p, 0, 1)
    logp = pm.Normal.logp(Z, delta, 1) - pm.Normal.logp(Z, 0, 1)
    return logp

class PCurveMixture:

    def __init__(self, pvals, effect_size_prior = 1.5, **sampler_kwargs):
        '''
        Arguments
        ---------
        pvals : np.array of size (n_observations,)
            The observed p-values
        effect_size_prior : float
            Mean of the exponential distribution used as an effect size prior.
            You can use `PCurveMixture.prior_predictive_power(alpha)` to see
            how this parameter translates to a prior over Type II error for a
            given false positive rate `alpha`.
        '''
        self._mix = None
        self._H1 = None
        self._ps = pvals
        assert(effect_size_prior > 0)
        self.prior = effect_size_prior
        if 'idata_kwargs' not in sampler_kwargs:
            sampler_kwargs['idata_kwargs'] = dict(log_likelihood = True)
        else:
            sampler_kwargs['idata_kwargs']['log_likelihood'] = True
        if 'random_seed' not in sampler_kwargs:
            sampler_kwargs['random_seed'] = 1
        if 'draws' not in sampler_kwargs:
            sampler_kwargs['draws'] = 1000
        if 'chains' not in sampler_kwargs:
            sampler_kwargs['chains'] = 5
        if 'cores' not in sampler_kwargs:
            sampler_kwargs['cores'] = 5
        self.sampler_kwargs = sampler_kwargs

    def fit(self):
        with pm.Model() as mixture_model:
            # define model
            delta = pm.Exponential('effect_size', lam = 1/self.prior)
            pcurve_H1 = pm.CustomDist.dist(delta, logp = p_curve_loglik)
            pcurve_H0 = pm.Uniform.dist(0, 1)
            prev = pm.Uniform('prevalence', 0, 1)
            pm.Mixture(
                'likelihood',
                w = [1 - prev, prev],
                comp_dists = [pcurve_H0, pcurve_H1],
                observed = self._ps
            )
            # and sample from it
            idata = pm.sample(**self.sampler_kwargs)

        self._mix = idata
        return idata

    def fit_alternative(self):
        with pm.Model() as alltrue_model:
            # define model
            delta = pm.Exponential('effect_size', lam = 1/self.prior)
            pcurve_H1 = pm.CustomDist(
                'p', delta, logp = p_curve_loglik,
                observed = self._ps
            )
            # and sample from it
            idata = pm.sample(**self.sampler_kwargs)

        self._H1 = idata
        return idata

    @property
    def mixture(self):
        if self._mix is None:
            raise Exception('Must call `PCurveMixture.fit()` first!')
        else:
            return self._mix

    def summary(self, **summary_kwargs):
        return az.summary(self.mixture, **summary_kwargs)

    def plot_trace(self, **kwargs):
        return az.plot_trace(self.mixture, **kwargs)

    @property
    def H1(self):
        if self._H1 is None:
            raise Exception('Must call `PCurveMixture.fit_alternative()` first!')
        else:
            return self._H1

    def compare(self):
        # compute loo-likelihood for sampled H1 & H1/H0 models
        ic = 'loo'
        scale = 'log'
        comp_dict = {"mixture": self.mixture, r"all $H_1$": self.H1}
        ics_dict, scale, ic = _calculate_ics(comp_dict, scale, ic)
        # create mock ELPDData object for unsampled H0/uniform model
        mix_elpd = ics_dict['mixture']
        unif_elpd = mix_elpd.copy(deep = True)
        assert(unif_elpd['scale'] == 'log')
        unif_elpd['elpd_loo'] = np.log(1.)
        unif_elpd['se'] = 0.
        unif_elpd['p_loo'] = 0.
        unif_elpd['loo_i'].values = np.full(mix_elpd.n_data_points, np.log(1.))
        unif_elpd['warning'] = False
        ics_dict[r'all $H_0$'] = unif_elpd # and add to ics_dict
        return az.compare(ics_dict, ic, method = 'stacking', scale = scale)

    def plot_compare(self, **plot_kwargs):
        comp = self.compare()
        return az.plot_compare(comp, **plot_kwargs)

    @property
    def prevalence(self):
        return self.mixture.posterior.prevalence.values.flatten()

    def prevalence_hdi(hdi_prob = .95):
        return az.hdi(self.prevalence, hdi_prob = hdi_prob)

    @property
    def effect_size(self):
        return self.mixture.posterior.effect_size.values.flatten()

    def effect_size_hdi(hdi_prob = .95):
        return az.hdi(self.effect_size, hdi_prob = hdi_prob)

    def posterior_predictive_power(self, alpha):
        pows = p_cdf(alpha, self.effect_size)
        return pd.DataFrame({'prevalence': self.prevalence, 'power': pows})

    def prior_predictive_power(self, alpha, random_seed = 0):
        rng = np.random.default_rng(random_seed)
        try:
            n = self.effect_size.size
        except:
            n = 10000
        prev = rng.uniform(size = n)
        deltas = rng.exponential(scale = self.prior, size = n)
        pows = p_cdf(alpha, deltas)
        return pd.DataFrame({'prevalence': prev, 'power': pows})

    def posterior_predictive_power_hdi(self, alpha, hdi_prob = .95):
        pow = self.posterior_predictive_power(alpha)
        return az.hdi(pow.power.to_numpy(), hdi_prob = hdi_prob)

    def prior_predictive_power_hdi(self, alpha, hdi_prob = .95):
        pow = self.prior_predictive_power(alpha)
        return az.hdi(pow.power.to_numpy(), hdi_prob = hdi_prob)
