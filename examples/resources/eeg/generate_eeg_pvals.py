from bids import BIDSLayout
import numpy as np
import pandas as pd
import mne
import os
from scipy.stats import lognorm
from scipy.special import expit
from joblib import Parallel, delayed

from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, accuracy_score

def get_sensitivity_p(events):
    import statsmodels.formula.api as smf
    import statsmodels.api as sm
    mod = smf.glm(
        'agency ~ latency',
        family = sm.families.Binomial(),
        data = events
    )
    res = mod.fit()
    p = res.pvalues['latency']
    return p

def load_behavioral_data(layout, sub):
    '''
    loads the behavioral data from subject's stimuluation block, with
    the original exclusion criteria from Veillette et al. JNeuro (2023).
    Exclusion criteria, which is just meant to remove trials where electrical
    stimulation was not effective, has already been applied to EEG data in the
    'preprocessing' derivatives folder on OpenNeuro.
    '''
    events = layout.get(subject = sub, suffix = 'events')[0].get_df()
    rm_idx = (events.pressed_first == True)&(events.trial_type == 'stimulation')
    rm_idx = (events.rt > .6) | rm_idx
    events = events[~rm_idx]
    movement_lag = events.rt - events.latency
    params = lognorm.fit(movement_lag[events.trial_type == 'stimulation'])
    lower = lognorm.ppf(.025, params[0], params[1], params[2])
    upper = lognorm.ppf(.975, params[0], params[1], params[2])
    events['outlier'] = (movement_lag > upper) | (movement_lag < lower)
    events = events[~events.outlier]
    events = events[events.trial_type == 'stimulation']
    return events

def shuffle_within_folds(v, fold, seed = None):
    rng = np.random.default_rng(seed)
    v = v.copy()
    for r in np.unique(fold):
        v_fold = v[fold == r]
        rng.shuffle(v_fold)
        v[fold == r] = v_fold
    return v

def generate_perm_dist(y, yhat, fold, score_func, n_perms = 5000):
    if score_func == accuracy_score:
        yhat = yhat >= .5
    H0 = [score_func(y, yhat)]
    for i in range(n_perms):
        yshuff = shuffle_within_folds(y, fold, i)
        H0.append(score_func(yshuff, yhat))
    return np.array(H0)

def evaluate_at_time(X, y, t):
    # subset data to time
    x = X[..., t]
    # build classification pipeline
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegressionCV(solver = 'liblinear')
    )
    # generate out-of-sample predictions
    cv_params = dict(n_splits = 10, shuffle = True, random_state = 0)
    yhat = cross_val_predict(
        pipe, x, y,
        method = 'predict_proba',
        cv = StratifiedKFold(**cv_params),
    )[:, 1]
    # reconstruct which cross-validation split each prediction comes from
    fold = np.empty_like(y)
    cv = StratifiedKFold(**cv_params)
    for i, (train, test) in enumerate(cv.split(x, y)):
        fold[test] = i
    # and shuffle only labels within same fold in permutation test
    H0_auroc = generate_perm_dist(y, yhat, fold, roc_auc_score)
    H0_accuracy = generate_perm_dist(y, yhat, fold, accuracy_score)
    return np.stack([H0_auroc, H0_accuracy])

def main(layout):
    '''
    Generates p-values for behavioral sensitivity to stimuluation latency,
    and then permutation distributions for classification AUROC and accuracy
    over time. Saves results in 'eeg' directory.

    Saved outputs are one (n_subjects,) array of p-values from behavioral data
    and two (n_permutations, n_times, n_subjects) arrays of AUROC/accuracy.
    The first "permutation" in the AUROC/accuracy arrays are observed values.
    Also saves the timestamps indicating when each time sample in the decoding
    arrays occured relative to the onset of electrical stimulation.

    Note
    ------------
    I originally ran this script with sklearn v1.4.2.
    '''

    aurocs = []
    accuracies = []
    ps_behav = []

    for sub in layout.get_subjects():

        print('\nStarting sub-%s...'%sub)

        # get p-value for behavioral sensitivity
        events = load_behavioral_data(layout, sub)
        p_behav = get_sensitivity_p(events)
        ps_behav.append(p_behav)

        # load preprocessed EEG data (same exclusion criteria already applied)
        f = layout.get(
                scope = 'preprocessing',
                subject = sub,
                suffix = 'epo',
                desc = 'clean30',
                extension = 'fif.gz')[0]
        epochs = mne.read_epochs(f.path, preload = True, verbose = False)
        X = epochs.get_data()
        y = epochs.events[:, 2]

        # generate null distributions for classification AUROC and accuracy
        res = Parallel(n_jobs = -1, verbose = 1)(
            delayed(evaluate_at_time)(X, y, t)
            for t in range(X.shape[-1])
        )
        auroc = np.stack([r[0] for r in res], axis = 1)
        accuracy = np.stack([r[1] for r in res], axis = 1)
        aurocs.append(auroc)
        accuracies.append(accuracy)

        print('Done with sub-%s.\n'%sub)

    p_behav = np.array(ps_behav)
    auroc = np.stack(aurocs, axis = -1)
    accuracy = np.stack(accuracies, axis = -1)

    # save results
    if not os.path.exists('eeg'):
        os.mkdir('eeg')
    f = os.path.join('eeg', 'pvals_behav.npy')
    np.save(f, p_behav, allow_pickle = False)
    f = os.path.join('eeg', 'H0_auroc.npy')
    np.save(f, auroc, allow_pickle = False)
    f = os.path.join('eeg', 'H0_accuracy.npy')
    np.save(f, accuracy, allow_pickle = False)
    f = os.path.join('eeg', 'timestamps.npy')
    np.save(f, epochs.times, allow_pickle = False)


if __name__ == '__main__':
    # 'bids_dataset' can be downloaded from OpenNeuro:
    # https://doi.org/10.18112/openneuro.ds004561.v1.0.0
    layout = BIDSLayout('bids_dataset', derivatives = True)
    main(layout)
