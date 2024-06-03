## Examples

The notebooks in this folder are meant to provide tutorial-style documentation for using the `p2prev` package, which provides a simplified interface for fitting $p$-curve mixture models, and to provide an introduction to $p$-curve mixture models in general. 

### For a conceptual introduction

`interoception.ipynb` is a good place to start for a conceptual introduction, as it includes a visualization of a of $p$-curve mixture model, which is compares to a more bespoke Bayesian mixture model (from a published paper) for the same dataset. It also defines the $p$-curve mixture model directly in [PyMC](https://www.pymc.io), rather than invoking `p2prev`'s simplified interface; this may be "a lot" if you're not familiar with Bayesian modelling, but it makes it easier to see what's going on for those who are. It also illustrates how to compute a per-subject posterior probability of group membership, which can't be done with the simplified `p2prev` interface.

### For an introduction to the package 

`prevalence-estimation.ipynb` is a good introduction to using `p2prev` for $p$-curve mixture models. It includes code for visualizing prior and posterior distributions (the rest of the tutorial notebooks just use the `p2prev`'s default priors), and it demonstrates the built-in model comparison functionality for checking whether a mixture model is a good fit to your data (compared to models in which all subjects' data come from the same distribution). This notebook also compares the $p$-curve mixture method for prevalence estimation to the a previous prevalence estimation method, to illustrate the advantage of using $p$-curves.

`absolute-pitch.ipynb` provides another example of fitting $p$-curve models to estimate prevalence in a real dataset, and it illustrates a concrete use case: estimating differences in effect prevalence and in within-subject effect size between two groups of subjects. This example is a good motivating example, since jointly estimating prevalence and within-subject effect size leads to a qualitatively different conclusion than would methods which assume the latter is fixed. 

### For advanced users 

`bayes-factor.ipynb` provides an example of comparing the $p$-curve mixture model to null-only and alternative-only models using Bayes Factors, instead of using `p2prev`'s built-in model comparison functionality. I think this approach to model comparison is conceptually superior to the built-in functionality in many cases (for reasons discussed in the notebook), but estimatation of Bayes Factors is not sufficiently numerically stable to incorporate into the main `p2prev` package. This means that, for now, users wishing to use Bayes Factors will have to interface with [PyMC](https://www.pymc.io) directly. 
