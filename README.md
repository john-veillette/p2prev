[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11459065.svg)](https://doi.org/10.5281/zenodo.11459065)
# p2prev: a convenient interface for fitting p-curve mixture models 

$p$-curve mixture models are a method for estimating the population prevalence of an effect from the results of multiple within-subject hypothesis tests (i.e. $p$-values) of that effect. Unlike other prevalence estimation methods for this setting, $p$-curve mixture models are applicable even in the presence of uncertainty about the sensitivity (i.e. statistical power) of the within-subject test.

$p$-curve mixtures are a type of Bayesian mixture model, but instead of modeling the data on their orginal scale, which may require unavailable domain knowledge or unrealistic parametric assumptions, they model the individual $p$-values obtained from each subject as a mixture between two [p-curves](https://www.p-curve.com/), a probability distribution over repeated $p$-values that has been used primarily in the meta-analysis literature. This distribution-agnostic approach allows $p$-curve mixtures to be used for any data from which one can compute a valid $p$-value for each subject (or other unit of analysis for which you want to estimate prevalence), which affords the ability to use Bayesian mixture modeling on data for which defining a parametric model would be insurmountably difficult -- as well as lowering the barrier for entry for non-experts to apply Bayesian mixture modeling. (Though, of course, we always recommend formally modeling your data when possible.)

The `p2prev` package provides a simplified interface to fit $p$-curve mixture models. Basic usage is basically to plug in your $p$-values and go:
```
from p2prev import PCurveMixture # import model class from p2prev
import numpy as np

pvals = [0.00060, 0.02999, 0.04939, 0.94601]
pvals = np.array(pvals) # p-vals should be in an array

model = PCurveMixture(pvals) # feed model the data
model.fit() # fit the model parameters to the data 
model.summary() # prints results
```

`PCurveMixture` results can be compared between two independent groups to estimate between-group differences in effect prevalence or in within-subject effect size (given that the subject shows the effect), for which we provide an [example]([PCurveMixture](https://github.com/john-veillette/p2prev/blob/main/examples/absolute-pitch.ipynb)). We also provide a `p2prev.PCurveWithinGroupDifference` class for estimating the prevalence difference between two tests performed on the same group of subjects (or e.g. the same group of subjects is given the same test in two experimental conditions).

Check out our [tutorial examples](https://github.com/john-veillette/p2prev/tree/main/examples) and [Documentation](http://p2prev.readthedocs.io/) for details.

### Graphical user interface

For users unfamiliar with Python, some of the basic functionality can be accessed online through a [Shiny app interface](https://jveillette.shinyapps.io/p2prev/), where _p_-values can simply be uploaded in a .csv file. This will generally be much slower, so be prepared to wait around for a while while models take over a minute (versus less than ten seconds locally) to fit. For more advanced functionality and full control of parameters, you will want to use Python instead.

### Installation

We recommend installing `pymc` before trying to install `p2prev`. (I recommend version 5.0 or greater, though I think `p2prev` will run with Version 4.) Installing `p2prev` as below will attempt to install `pymc` using Pip, but I find that `pymc` is much more likely to install correctly using Anaconda as suggested in the [PyMC documentation](https://www.pymc.io/projects/docs/en/latest/installation.html).

Once you've installed `pymc`, you can simply run 
```
pip install p2prev
```
to install the latest stable version.


You could also install the development version with
```
pip install git+https://github.com/john-veillette/p2prev.git
```
but do so at your own peril! I may sometimes break things as I work on the development version.

### Citing this package

If you use $p$-curve mixtures for prevalence estimation, with or without the `p2prev` package, please be so kind as to cite the [paper](https://doi.org/10.1038/s44271-025-00190-0) where we describe and validates the method. 

We have also minted a DOI so that you can cite the `p2prev` package itself (or other contents of this repository) and be confident that the link in your citation will always lead to the same place. The DOI for the current version should always be near the top of this page on GitHub, or you can use the permanent DOI ([10.5281/zenodo.11459064](https://zenodo.org/doi/10.5281/zenodo.11459064)), which always links to the latest release.

Importantly, `p2prev` is only a wrapper around a model implemented with the excellent [PyMC](https://www.pymc.io/) package, which performs the hardest parts of model fitting for us. So if you use `p2prev`, please also [cite PyMC](https://doi.org/10.7717/peerj-cs.1516) so its developers can get credit for their work. 
