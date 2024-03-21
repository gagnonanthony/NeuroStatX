# -*- coding: utf-8 -*-

from math import sqrt
import warnings

from enum import Enum
import numpy as np
from sklearn.base import clone, is_classifier
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import (cross_val_predict, KFold, check_cv)
from sklearn.metrics import mean_squared_error, r2_score, check_scoring
from sklearn.utils import indexable, check_random_state
from sklearn.utils.metaestimators import _safe_split
from sklearn.utils.parallel import Parallel, delayed
from sklearn.model_selection._validation import _shuffle
from strenum import StrEnum
from tqdm import tqdm


class ScoringMethod(StrEnum, Enum):
    accuracy = "accuracy"
    balanced_accuracy = "balanced_accuracy"
    top_k_accuracy = "top_k_accuracy"
    average_precision = "average_precision"
    neg_brier_score = "neg_brier_score"
    f1 = "f1"
    f1_micro = "f1_micro"
    f1_macro = "f1_macro"
    f1_weighted = "f1_weighted"
    f1_samples = "f1_samples"
    neg_log_loss = "neg_log_loss"
    precision = "precision"
    recall = "recall"
    jaccard = "jaccard"
    roc_auc = "roc_auc"
    roc_auc_ovr = "roc_auc_ovr"
    roc_auc_ovo = "roc_auc_ovo"
    roc_auc_ovr_weighted = "roc_auc_ovr_weighted"
    roc_auc_ovo_weighted = "roc_auc_ovo_weighted"
    adjusted_rand_score = "adjusted_rand_score"
    adjusted_mutual_info_score = "adjusted_mutual_info_score"
    completeness_score = "completeness_score"
    fowlkes_mallows_score = "fowlkes_mallows_score"
    homogeneity_score = "homogeneity_score"
    mutual_info_score = "mutual_info_score"
    normalized_mutual_info_score = "normalized_mutual_info_score"
    v_measure_score = "v_measure_score"
    rand_score = "rand_score"
    explained_variance = "explained_variance"
    max_error = "max_error"
    neg_mean_absolute_error = "neg_mean_absolute_error"
    neg_mean_squared_error = "neg_mean_squared_error"
    neg_mean_squared_log_error = "neg_mean_squared_log_error"
    neg_median_absolute_error = "neg_median_absolute_error"
    neg_root_mean_squared_error = "neg_root_mean_squared_error"
    r2 = "r2"
    neg_mean_poisson_deviance = "neg_mean_poisson_deviance"
    neg_mean_gamma_deviance = "neg_mean_gamma_deviance"
    neg_mean_absolute_percentage_error = "neg_mean_absolute_percentage_error"
    d2_absolute_error_score = "d2_absolute_error_score"
    d2_pinball_score = "d2_pinball_score"
    d2_tweedie_score = "d2_tweedie_score"


def plsr_cv(X,
            Y,
            nb_comp,
            splits=10,
            processes=4,
            verbose=False):
    """Function to perform a PLSR model with cross-validation between a set of
    predictor and dependent variables.

    Args:
        X (pd.DataFrame):               Dataframe containing the predictor
                                        variables.
        Y (pd.DataFrame):               Dataframe containing the dependent
                                        variables.
        nb_comp (int):                  Number of components to use.
        splits (int, optional):         Number of fold to use in
                                        cross-validation. Defaults to 10.
        processes (int, optional):      Number of cpus to use during
                                        processing. Defaults to 4.
        verbose (bool, optional):       Verbose mode. Defaults to False.

    Returns:
        plsr:                           PLSR model.
        mse:                            List of mean squared errors.
        score_c:                        R2 score for the model.
        score_cv:                       R2 score for the cross-validation.
        rscore:                         Square root of the R2 score.
        mse_c:                          Mean squared error for the model.
        mse_cv:                         Mean squared error for the
                                        cross-validation.
    """
    v = False if verbose else False

    mse = []
    component = np.arange(1, nb_comp + 1)
    kf_10 = KFold(n_splits=splits, shuffle=True, random_state=1)

    # Initialize a PLSR object.
    plsr = PLSRegression()

    for i in tqdm(component, disable=v):
        plsr.n_components = i

        y_cv = cross_val_predict(plsr, X, Y, cv=kf_10, n_jobs=processes)

        mse.append(mean_squared_error(Y, y_cv))

    msemin = np.argmin(mse)

    # Fit the optimal model.
    plsr.n_components = msemin + 1
    plsr.fit(X, Y)

    y_c = plsr.predict(X)

    # Cross-validation.
    y_cv = cross_val_predict(plsr, X, Y, cv=kf_10)

    score_c = r2_score(Y, y_c)
    score_cv = r2_score(Y, y_cv)
    rscore = sqrt(score_c)
    mse_c = mean_squared_error(Y, y_c)
    mse_cv = mean_squared_error(Y, y_cv)

    return plsr, mse, score_c, score_cv, rscore, mse_c, mse_cv


def permutation_testing(X,
                        Y,
                        nb_comp,
                        nb_permutations=1000,
                        scoring='r2',
                        splits=10,
                        processes=4,
                        verbose=False):
    """Function to perform permutation testing on a model.

    Args:
        X (pd.DataFrame):                   Dataframe containing the predictor
                                            variables.
        Y (pd.DataFrame):                   Dataframe containing the dependent
                                            variables.
        nb_comp (int):                      Number of components to use.
        nb_permutations (int, optional):    Number of iterations to perform.
                                            Defaults to 1000.
        scoring (str, optional):            Scoring method to use. Defaults
                                            to 'r2'.
        splits (int, optional):             Number of fold to use in
                                            cross-validation. Defaults to 10.
        processes (int, optional):          Number of cpus to use during
                                            processing. Defaults to 4.
        verbose (bool, optional):           Verbose mode. Defaults to False.

    Returns:
        score:                              R2 score for the model.
        perm_score:                         R2 scores for the permutation
                                            testing.
        score_pvalue:                       P-value for the model.
        perm_coef:                          Coefficients for the permutation
                                            testing.
        coef_pvalue:                        P-value for the coefficients.
    """
    v = 1 if verbose else 0

    plsr = PLSRegression(n_components=nb_comp)
    kf_10 = KFold(n_splits=splits, shuffle=True, random_state=1)

    # Lauching permutation testing.
    score, perm_score, score_pvalue, perm_coef, coef_pvalue = permutation_test(
        plsr,
        X,
        Y,
        scoring=scoring,
        cv=kf_10,
        n_permutations=nb_permutations,
        n_jobs=processes,
        verbose=v)

    return score, perm_score, score_pvalue, perm_coef, coef_pvalue


def _permutation_scorer(estimator,
                        X,
                        Y,
                        groups,
                        cv,
                        scorer,
                        fit_params=None):
    """Core worker for permutation testing.

    Args:
        estimator (Model):              Model to use.
        X (pd.DataFrame):               Dataframe containing the predictor
                                        variables.
        Y (pd.DataFrame):               Dataframe containing the dependent
                                        variables.
        groups (pd.DataFrame):          Dataframe containing the groups.
        cv (int):                       Number of fold to use in
                                        cross-validation.
        scorer (str):                   Scoring method to use.
        fit_params (dict, optional):    Parameters to use during fitting.
                                        Defaults to None.

    Returns:
        avg_score:                      Average score.
        coefficients:                   Coefficients.
    """

    fit_params = fit_params if fit_params is not None else {}
    avg_score = []
    coefficients = []

    for train, test in cv.split(X, Y, groups):
        X_train, y_train = _safe_split(estimator, X, Y, train)
        X_test, y_test = _safe_split(estimator, X, Y, test, train)
        estimator.fit(X_train, y_train, **fit_params)
        avg_score.append(scorer(estimator, X_test, y_test))
        coefficients.append(estimator.coef_.T)

    return np.mean(avg_score), np.mean(np.array(coefficients), axis=0)


def permutation_test(estimator,
                     X,
                     Y,
                     *,
                     groups=None,
                     cv=None,
                     n_permutations=100,
                     n_jobs=None,
                     random_state=0,
                     verbose=0,
                     scoring=None,
                     fit_params=None):
    """Core function to perform permutation testing.

    Args:
        estimator (Model):                  Model to use.
        X (pd.DataFrame):                   Dataframe containing the predictor
                                            variables.
        Y (pd.DataFrame):                   Dataframe containing the dependent
                                            variables.
        groups (str, optional):             Group variable. Defaults to None.
        cv (int, optional):                 Number of fold to use in
                                            cross-validation. Defaults to None.
        n_permutations (int, optional):     Number of permutations to perform.
                                            Defaults to 100.
        n_jobs (int, optional):             Number of jobs to run in parallel.
                                            Defaults to None.
        random_state (int, optional):       Random state seed. Defaults to 0.
        verbose (int, optional):            Verbose mode. Defaults to 0.
        scoring (str, optional):            Scoring method to use. Defaults
                                            to None.
        fit_params (dict, optional):        Parameters to use during fitting.
                                            Defaults to None.

    Returns:
        score:                              R2 score for the model.
        perm_score:                         R2 scores for the permutation
                                            testing.
        score_pvalue:                       P-value for the model.
        perm_coef:                          Coefficients for the permutation
                                            testing.
        coef_pvalue:                        P-value for the coefficients.
    """

    warnings.filterwarnings("ignore")

    X, y, groups = indexable(X, Y, groups)

    cv = check_cv(cv, y, classifier=is_classifier(estimator))
    scorer = check_scoring(estimator, scoring=scoring)
    random_state = check_random_state(random_state)

    score = _permutation_scorer(clone(estimator),
                                X, y, groups, cv, scorer,
                                fit_params=fit_params)

    results = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(_permutation_scorer)(
            clone(estimator),
            X,
            _shuffle(y, groups, random_state),
            groups,
            cv,
            scorer,
            fit_params=fit_params,
        )
        for _ in range(n_permutations)
    )

    perm_score, perm_coef = zip(*results)

    perm_score = np.array(perm_score)
    perm_coef = np.array(perm_coef)
    score_pvalue = (np.sum(perm_score >= score[0]) + 1) / (n_permutations + 1)
    coef_pvalue = (np.sum(abs(perm_coef) >= abs(score[1]),
                          axis=0) + 1) / (n_permutations + 1)

    return score, perm_score, score_pvalue, perm_coef, coef_pvalue
