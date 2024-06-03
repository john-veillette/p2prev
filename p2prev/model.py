import pymc as pm
import pandas as pd
import numpy as np
import arviz as az
from arviz.stats.stats import _calculate_ics

from .pcurve import p_cdf

def p_curve_loglik(p, delta):
    '''
    log-likelihood of p-values from a one-tailed
    Z-test with known unit variance and one observation
    '''
    Z = -pm.Normal.icdf(p, 0, 1)
    logp = pm.Normal.logp(Z, delta, 1) - pm.Normal.logp(Z, 0, 1)
    return logp

def _half_p_loglik(p, d, nu):
    '''
    log-likelihood of  p-values of a
    one-tailed t-test with effect size `d` and degrees
    of freedom `nu`.
    '''
    n = (nu + 2)/2 # "sample size"
    nct = d * np.sqrt(n/2) # non-centrality parameter
    T = -1.*pm.StudentT.icdf(p, nu, mu = 0., sigma = 1.)
    numer =  pm.StudentT.logp(T, nu, mu = nct, sigma = 1.)
    denom = pm.StudentT.logp(T, nu, mu = 0., sigma = 1.)
    return numer - denom

def p_loglik_overdisp(p, d, nu):
    '''
    log-likelihood for p-values of a
    two-tailed t-test with effect size `d` and degrees
    of freedom `nu`.
    '''
    a = _half_p_loglik(p/2, d, nu)
    b = _half_p_loglik(1 - p/2, d, nu)
    return pm.logaddexp(a, b) - np.log(2.)

class PCurveMixture:
    '''
    A user-friendly wrapper for fitting a p-curve mixture model.

    Arguments
    ---------
    pvals : np.array of size (n_observations,)
        The observed p-values
    effect_size_prior : float
        Mean of the exponential distribution used as an effect size prior.
        You can use `PCurveMixture.prior_predictive_power(alpha)` to see
        how this parameter translates to a prior over Type II error for a
        given false positive rate `alpha`.
    **sampler_kwargs
        You can input any valid argument to `pymc.sample` if you wish the change
        the Monte-Carlo sampling settings. By default, 5 chains of 1000 samples
        will be drawn from the posterior, and this will be distributed across
        five CPUs if available. The same random seed will be used each time.
    '''

    def __init__(self, pvals, effect_size_prior = 1.5, **sampler_kwargs):

        self._mix = None
        self._H1 = None
        self._ps = pvals
        self._model = None
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
        '''
        Fits model. Must be called before doing anything else (except prior
        predictive simulation).
        '''
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
            # and sample from posterior
            idata = pm.sample(**self.sampler_kwargs)

        self._mix = idata
        self._model = mixture_model
        return idata

    @property
    def map(self):
        '''
        maximum a-posteriori estimate of prevalence
        '''
        try:
            return 1. * pm.find_MAP(
                model = self._model, progressbar = False
                )['prevalence']
        except:
            self.mixture # trigger not-yet-fit exception

    def fit_alternative(self):
        '''
        Fits alternative models. Must be called before model comparison
        can be performed.
        '''
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
        '''
        The trace of the fit mixture model.
        '''
        if self._mix is None:
            raise Exception('Must call `PCurveMixture.fit()` first!')
        else:
            return self._mix

    def summary(self, **summary_kwargs):
        '''
        Gives summary of posteriors for model parameters.
        '''
        return az.summary(self.mixture, **summary_kwargs)

    def plot_trace(self, **kwargs):
        '''
        Plots posterior traces.
        '''
        return az.plot_trace(self.mixture, **kwargs)

    @property
    def H1(self):
        if self._H1 is None:
            raise Exception('Must call `PCurveMixture.fit_alternative()` first!')
        else:
            return self._H1

    def compare(self):
        '''
        Performs model comaprison of mixture model against all-null and
        all-alternative models. This method can't be called until you've called
        the .fit_alternative() method.
        '''
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
        '''
        Plots model comparison
        '''
        comp = self.compare()
        return az.plot_compare(comp, **plot_kwargs)

    @property
    def prevalence(self):
        '''
        posterior samples for prevalence parameters
        '''
        return self.mixture.posterior.prevalence.values.flatten()

    def prevalence_hdi(self, hdi_prob = .95):
        '''
        HDI (default 95%) for prevalence parameters

        Arguments
        ----------
        hdi_prob : float
            Width of highest-density interval to return.
        '''
        return az.hdi(self.prevalence, hdi_prob = hdi_prob)

    @property
    def effect_size(self):
        '''
        posterior samples for abstract effect size
        '''
        return self.mixture.posterior.effect_size.values.flatten()

    def effect_size_hdi(hdi_prob = .95):
        '''
        HDI (default 95%) for abstract effect size

        Arguments
        ----------
        hdi_prob : float
            Width of highest-density interval to return.
        '''
        return az.hdi(self.effect_size, hdi_prob = hdi_prob)

    def posterior_predictive_power(self, alpha):
        '''
        Posterior samples for within-subject power given alternative hypothesis
        is true. Returned as pandas DataFrame with prevalence samples, so can
        be plotted as a joint distribution.
        '''
        pows = p_cdf(alpha, self.effect_size)
        return pd.DataFrame({'prevalence': self.prevalence, 'power': pows})

    def prior_predictive_power(self, alpha, random_seed = 0):
        '''
        Prior samples for within-subject power given alternative hypothesis is
        true. Returned as pandas DataFrame with prevalence samples, so can be
        plotted as a joint distribution.
        '''
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
        '''
        Posterior HDI for within-subject effect size at a given
        significance level.

        Arguments
        ----------
        alpha : float
            Significance level for which to get posterior power .
        hdi_prob : float
            Width of highest-density interval to return.
        '''
        pow = self.posterior_predictive_power(alpha)
        return az.hdi(pow.power.to_numpy(), hdi_prob = hdi_prob)

    def prior_predictive_power_hdi(self, alpha, hdi_prob = .95):
        '''
        HDI under prior for within-subject effect size at a given
        significance level.

        Arguments
        ----------
        alpha : float
            Significance level for which to get posterior power .
        hdi_prob : float
            Width of highest-density interval to return.
        '''
        pow = self.prior_predictive_power(alpha)
        return az.hdi(pow.power.to_numpy(), hdi_prob = hdi_prob)

class PCurveWithinGroupDifference:
    '''
    Fits p-curve mixture model for two within-subject hypothesis tests
    applied to the SAME group of subjects. Estimates the difference in
    prevalence of the two effects tested, again in the SAME subjects.
    (If instead you want to compare prevalence of the same effect in two
    different groups of subjects, i.e. a "between group" difference, you
    can just fit a `PCurveMixture` to each group individually and
    subtract posterior samples from the two models to get samples from the
    posterior of the difference.)

    We assume that the effect size for each test is fixed, i.e. no
    effect size information is explicitly pooled between tests.
    However, we account for the possibility that expressing H1 makes a
    subject more likely to express H2 or the reverse. (The degree of such
    covariation is something the model learns from the data.)

    Arguments
    ---------
    pvals1 : np.array of size (n_subjects,)
        The observed p-values for within-subject hypothesis test of H0 vs. H1.
    pvals2 : np.array of size (n_subjects,)
        The observed p-values for within-subject hypothesis test of H0 vs. H2.
        Subject order should be the same as in `pvals`.
    effect_size_prior : float
        Mean of the exponential distribution used as an effect size prior.
    **sampler_kwargs
        You can input any valid argument to `pymc.sample` if you wish the change
        the Monte-Carlo sampling settings. By default, 5 chains of 1000 samples
        will be drawn from the posterior, and this will be distributed across
        five CPUs if available. The same random seed will be used each time.

    Notes
    ---------
    I had to do some PyMC trickery to get PyMC to account for covariation
    between H1 and H2 prevalence the way we need it to, at the cost of
    being able to use built-in model comparison techniques as in the
    main `PCurveMixture` class. If you want to compare to the H0 only model
    or H1/H2 only models, then you should do that for H1 and H2 individually
    using `PCurveMixture`.
    '''

    def __init__(self, pvals1, pvals2, effect_size_prior = 1.5, **sampler_kwargs):

        try:
            assert(len(pvals1) == len(pvals2))
        except:
            raise Exception(
                'First two arguments must be paired series of p-values, ' + \
                'but args were different shapes!'
            )
        self._mix = None
        self._ps1 = pvals1
        self._ps2 = pvals2
        self._model = None
        assert(effect_size_prior > 0)
        self.prior = effect_size_prior
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
        '''
        fits model
        '''
        with pm.Model() as mixture_model:

            # define likelihoods of p-values under H0, H1, and H2
            delta = pm.Exponential('effect_size', lam = 1/self.prior, shape = 2)
            pm.Deterministic('effect_size_diff', delta[1] - delta[0])
            pcurve_H1 = pm.CustomDist.dist(delta[0], logp = p_curve_loglik)
            pcurve_H2 = pm.CustomDist.dist(delta[1], logp = p_curve_loglik)
            pcurve_H0 = pm.Uniform.dist(0, 1)

            # probabilities of combinations of H0, H1, and H2 being true
            k = pm.Dirichlet('prevalence', np.ones(4)) # prior
            k00, k10, k01, k11 = k[0], k[1], k[2], k[3] # notation from Ince
            prev1 = pm.Deterministic('prevalence_H1', k10 + k11)
            prev2 = pm.Deterministic('prevalence_H2', k01 + k11)
            pm.Deterministic('prevalence_diff', prev2 - prev1)

            # define likelihoods for combinations of H0, H1, and H2 being true
            H1_logl_1 = pm.logp(pcurve_H1, self._ps1)
            H0_logl_1 = pm.logp(pcurve_H0, self._ps1)
            H2_logl_2 = pm.logp(pcurve_H2, self._ps2)
            H0_logl_2 = pm.logp(pcurve_H0, self._ps2)
            logl_00 = H0_logl_1 + H0_logl_2
            logl_10 = H1_logl_1 + H0_logl_2
            logl_01 = H0_logl_1 + H2_logl_2
            logl_11 = H1_logl_1 + H2_logl_2

            # and marginalize these likelihoods over class probabilities
            logp_marg = pm.logaddexp(
                pm.math.log(k00) + logl_00,
                pm.math.log(k10) + logl_10,
                pm.math.log(k01) + logl_01,
                pm.math.log(k11) + logl_11
            )
            # logp_marg isn't technically a valid PyMC likelihood
            # (though it's a real log-likelihood), but it will be treated
            # as one during sampling if we specify it as a "potential"
            pm.Potential('likelihood', logp_marg)

            # so now we can sample from posterior
            idata = pm.sample(**self.sampler_kwargs)

        # keep model and posterior samples in memory
        self._mix = idata
        self._model = mixture_model

    @property
    def mixture(self):
        '''
        the trace of the fit model
        '''
        if self._mix is None:
            raise Exception('Must call `PCurveMixture.fit()` first!')
        else:
            return self._mix

    def summary(self, **summary_kwargs):
        '''
        returns a summary of the posterior
        '''
        if len(summary_kwargs) == 0:
            summary_kwargs['var_names'] = [
                'effect_size', 'effect_size_diff',
                'prevalence_H1', 'prevalence_H2',
                'prevalence_diff'
            ]
        return az.summary(self.mixture, **summary_kwargs)

    def plot_trace(self, **kwargs):
        '''
        plots the traces for parameters
        '''
        if len(kwargs) == 0:
            kwargs['var_names'] = [
                'effect_size', 'effect_size_diff',
                'prevalence_H1', 'prevalence_H2',
                'prevalence_diff'
            ]
        return az.plot_trace(self.mixture, **kwargs)

    @property
    def prevalence_H1(self):
        '''
        posterior samples for H1 prevalence
        '''
        return self.mixture.posterior.prevalence_H1.values.flatten()

    @property
    def prevalence_H2(self):
        '''
        posterior samples for H2 prevalence
        '''
        return self.mixture.posterior.prevalence_H2.values.flatten()

    @property
    def prevalence_diff(self):
        '''
        posterior samples for H2 prevalence minus H1 prevalence
        '''
        return self.mixture.posterior.prevalence_diff.values.flatten()

    @property
    def prob_H2_prev_greater(self):
        '''
        posterior probability H2 prevalence minus H1 prevalence is positive
        '''
        return (self.prevalence_diff > 0).mean()

    def prevalence_diff_hdi(self, hdi_prob = .95):
        '''
        HDI for H2 prevalence minus H1 prevalence

        Arguments
        ----------
        hdi_prob : float
            Width of highest-density interval to return.
        '''
        return az.hdi(self.prevalence_diff, hdi_prob = hdi_prob)

    @property
    def effect_size_H1(self):
        '''
        posterior samples for relative effect size of H1
        '''
        return self.mixture.posterior.effect_size.values[..., 0].flatten()

    @property
    def effect_size_H2(self):
        '''
        posterior samples for relative effect size of H2
        '''
        return self.mixture.posterior.effect_size.values[..., 1].flatten()

    @property
    def effect_size_diff(self):
        '''
        posterior samples for H2 - H1 effect sizes
        '''
        return self.mixture.posterior.effect_size_diff.values.flatten()

    @property
    def prob_H2_effect_size_greater(self):
        '''
        posterior probability H2 effect minus H1 effect is positive
        '''
        return (self.effect_size_diff > 0).mean()

    def effect_size_diff_hdi(self, hdi_prob = .95):
        '''
        HDI for H2 - H1 effect sizes

        Arguments
        ----------
        hdi_prob : float
            Width of highest-density interval to return.
        '''
        return az.hdi(self.effect_size_diff, hdi_prob = hdi_prob)

    def power_diff(self, alpha):
        '''
        posterior samples for power given H2 minus power given H1
        at a given significance level `alpha`.

        Arguments
        ----------
        alpha : float
            Significance level for which to get posterior power .
        '''
        pow1 = p_cdf(alpha, self.effect_size_H1)
        pow2 = p_cdf(alpha, self.effect_size_H2)
        pow_diff = pow2 - pow1
        return pow_diff

    def power_diff_hdi(self, alpha, hdi_prob = .95):
        '''
        HDI for difference in within-subject power

        Arguments
        ----------
        alpha : float
            Significance level for which to get posterior power .
        hdi_prob : float
            Width of highest-density interval to return.
        '''
        pow = self.power_diff(alpha)
        return az.hdi(pow, hdi_prob = hdi_prob)
