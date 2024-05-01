import pymc as pm
import numpy as np
import arviz as az

class BinomialOutcomesModel:
    '''
    This is the population prevalence model used by
    Ince et al. (2021) @ https://doi.org/10.7554/eLife.62461
    but we make power/sensitivity a fit parameter instead of
    a fixed parameter so we can fairly compare with our own model.
    '''

    def __init__(self, k, n, alpha, **sampler_kwargs):
        self._k = k
        self._n = n
        self._alpha = alpha # Type I error
        self._trace = None
        if 'random_seed' not in sampler_kwargs:
            sampler_kwargs['random_seed'] = 1
        if 'draws' not in sampler_kwargs:
            sampler_kwargs['draws'] = 1000
        if 'chains' not in sampler_kwargs:
            sampler_kwargs['chains'] = 5
        if 'cores' not in sampler_kwargs:
            sampler_kwargs['cores'] = 5
        self.sampler_kwargs = sampler_kwargs
        self._model = None

    def fit(self):
        with pm.Model() as binom_model:
            # uniform priors
            prev = pm.Uniform('prevalence', 0, 1)
            power = pm.Uniform('power', 0, 1)
            # the probability of rejecting null hypothesis per subject
            theta = (1 - prev)*self._alpha + prev*power
            # yields the following likelihood
            pm.Binomial('rejections', self._n, theta, observed = self._k)
            # and now we sample...
            idata = pm.sample(**self.sampler_kwargs)
        self._trace = idata
        self._model = binom_model
        return idata

    @property
    def map(self):
        try:
            return 1. * pm.find_MAP(
                model = self._model, progressbar = False
                )['prevalence']
        except:
            self.trace # trigger not-yet-fit exception

    @property
    def trace(self):
        if self._trace is None:
            raise Exception('Must call `BinomialOutcomesModel.fit()` first')
        else:
            return self._trace

    def summary(self):
        return az.summary(self.trace)

    @property
    def power(self): # posterior samples
        return self.trace.posterior.power.values.flatten()

    @property
    def prevalence(self): # posterior samples
        return self.trace.posterior.prevalence.values.flatten()

    def power_hdi(self, hdi_prob = .95):
        return az.hdi(self.power, hdi_prob = hdi_prob)

    def prevalence_hdi(self, hdi_prob = .95):
        return az.hdi(self.prevalence, hdi_prob = hdi_prob)

    @property
    def expected_power(self):
        return self.power.mean()

    @property
    def expected_prevalence(self):
        return self.prevalence.mean()
