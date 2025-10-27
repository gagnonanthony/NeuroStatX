# -*- coding: utf-8 -*-

from math import sqrt
import warnings

from enum import Enum
import numpy as np
from sklearn.base import clone, is_classifier
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict, KFold, check_cv
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


class Penalty(StrEnum, Enum):
    l1 = "l1"
    l2 = "l2"
    elasticnet = "elasticnet"


class Solver(StrEnum, Enum):
    newton_cg = "newton-cg"
    newton_cholesky = "newton-cholesky"
    lbfgs = "lbfgs"
    liblinear = "liblinear"
    sag = "sag"
    saga = "saga"


def plsr_cv(X,
            Y,
            nb_comp,
            max_iter=1000,
            splits=10,
            processes=1,
            verbose=False):
    """
    Function to perform a PLSR model with cross-validation between a set of
    predictor and dependent variables.

    Parameters
    ----------
    X : pd.DataFrame
        Dataframe containing the predictor variables.
    Y : pd.DataFrame
        Dataframe containing the dependent variables.
    nb_comp : int
        Number of components to use.
    max_iter : int, optional
        Maximum number of iterations. Defaults to 1000.
    splits : int, optional
        Number of fold to use in cross-validation. Defaults to 10.
    processes : int, optional
        Number of cpus to use during processing. Defaults to 1.
    verbose : bool, optional
        Verbose mode. Defaults to False.

    Returns
    -------
    plsr : PLSR model
        PLSR model.
    mse : list
        List of mean squared errors.
    score_c : float
        R2 score for the model.
    score_cv : float
        R2 score for the cross-validation.
    rscore : float
        Square root of the R2 score.
    mse_c : float
        Mean squared error for the model.
    mse_cv : float
        Mean squared error for the cross-validation.
    """

    v = False if verbose else True

    mse = []
    component = np.arange(1, nb_comp + 1)
    kf_10 = KFold(n_splits=splits, shuffle=True, random_state=1)

    # Initialize a PLSR object.
    plsr = PLSRegression(max_iter=max_iter, scale=True, tol=1e-06, copy=True)

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


def permutation_testing(
    estimator,
    X,
    Y,
    binary=False,
    nb_permutations=1000,
    scoring="r2",
    splits=10,
    processes=1,
    verbose=False,
):
    """
    Function to perform permutation testing on a model.

    Parameters
    ----------
    estimator : Model
        Model to use.
    X : pd.DataFrame
        Dataframe containing the predictor variables.
    Y : pd.DataFrame
        Dataframe containing the dependent variables.
    binary : bool, optional
        If the dependent variable is binary. Defaults to False.
    nb_permutations : int, optional
        Number of iterations to perform. Defaults to 1000.
    scoring : str, optional
        Scoring method to use. Defaults to 'r2'.
    splits : int, optional
        Number of fold to use in cross-validation. Defaults to 10.
    processes : int, optional
        Number of cpus to use during processing. Defaults to 1.
    verbose : bool, optional
        Verbose mode. Defaults to False.

    Returns
    -------
    mod : Model
        Model.
    score : float
        Score for the model.
    coef : list
        Coefficients for the model.
    perm_score : list
        Scores for the permutation testing.
    score_pvalue : float
        P-value for the model.
    perm_coef : list
        Coefficients for the permutation testing.
    coef_pvalue : list
        P-value for the coefficients.
    """

    v = 1 if verbose else 0

    # Lauching permutation testing.
    mod, score, coef, perm_score, score_pvalue, perm_coef, coef_pvalue = (
        permutation_test(
            estimator,
            X,
            Y,
            scoring=scoring,
            cv=splits,
            n_permutations=nb_permutations,
            n_jobs=processes,
            verbose=v,
        )
    )

    return mod, score, coef, perm_score, score_pvalue, perm_coef, coef_pvalue


def _permutation_scorer(estimator, X, Y, groups, cv, scorer, fit_params):
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
        estimator:                      Model.
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

    return estimator, np.mean(avg_score), np.mean(np.array(coefficients),
                                                  axis=0)


def permutation_test(
    estimator,
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
    fit_params=None
):
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
        mod:                                Model.
        score:                              Score for the model.
        coef:                               Coefficients for the model.
        perm_score:                         Scores for the permutation
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

    mod, score, coef = _permutation_scorer(
        clone(estimator), X, y, groups, cv, scorer, fit_params=fit_params
    )

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

    models, perm_score, perm_coef = zip(*results)

    perm_score = np.array(perm_score)
    perm_coef = np.array(perm_coef)
    score_pvalue = (np.sum(perm_score >= score) + 1) / (n_permutations + 1)
    coef_pvalue = (np.sum(abs(perm_coef) >= abs(coef), axis=0) + 1) / (
        n_permutations + 1
    )

    return mod, score, coef, perm_score, score_pvalue, perm_coef, coef_pvalue


class PHQ9Labeler:
    def __init__(self):
        pass

    def fit(self, X, y=None):
        """
        Method kept for consistency with the scikit-learn API. But in this
        case, will simply call the `transform` method since no actual model
        gets fitted here.

        Needs to contain only 9 columns representing the ordered PHQ-9 items.
        Should be in the form of a DataFrame with shape (n_samples, 9).

        Parameters
        ----------
        X : pd.DataFrame
            Input features (n_samples, 9) with only the 9 PHQ-9 items as
            columns and subject as rows.
        y : pd.Series, optional
            Target variable. Not used in this model. Keeps the API consistent.

        Returns
        -------
        pd.DataFrame
            Transformed features.
        """
        if X.shape[1] != 9:
            raise ValueError(
                "Input DataFrame must contain exactly 9 columns"
                " representing the ordered PHQ-9 items."
            )

        return self.transform(X)

    def transform(self, X):
        """
        Transform the input features to assign the label based on the fuzzy
        weighting of the PHQ-9 items. Needs to contain only 9 columns
        representing the ordered PHQ-9 items. Should be in the form of a
        DataFrame with shape (n_samples, 9).

        Final labels will be either:

        - Not depressed
        - Mild
        - Moderate
        - Mod-Severe
        - Severe

        Parameters
        ----------
        X : pd.DataFrame
            Input features.

        Returns
        -------
        pd.DataFrame
            Transformed features.
        """
        if X.shape[1] != 9:
            raise ValueError(
                "Input DataFrame must contain exactly 9 columns"
                " representing the ordered PHQ-9 items."
            )

        def _transform_row(row):
            # Apply the fuzzy matching logic to each row.
            q1, q2, q3, q4, q5, q6, q7, q8, q9 = row.values

            if not (q1 > 1 or q2 > 1):
                return "Not Depressed"

            # Compute uD (sum of q1, q2, q6, q9) divided by 12
            uD = np.sum([q1, q2, q6, q9]) / 12

            if uD < 0.33:
                return "Not Depressed"

            # Sort the array for easier comparisons.
            s_array = np.sort([q1, q2, q6, q9])

            # --- uD = 0.33 ---
            if np.isclose(uD, 0.33, atol=0.01) and (q1 >= 2 and q2 >= 2):
                return "Moderate"

            # --- uD = 0.417 ---
            if np.isclose(uD, 0.417, atol=0.01):
                if np.array_equal(s_array, np.array([1, 1, 1, 2])):
                    return "Mild"
                elif q1 > 2 or q2 > 2 or q6 > 2 or q9 > 2:
                    return "Moderate"

            # --- uD = 0.5 ---
            if np.isclose(uD, 0.5, atol=0.01):
                if (
                    s_array[3] == 2
                    and s_array[2] == 2
                    and s_array[1] == 2
                    and s_array[0] < 2
                ):
                    return "Mod-Severe"
                elif (
                    s_array[3] == 2
                    and s_array[2] == 2
                    and s_array[1] < 2
                    and s_array[0] < 2
                ):
                    return "Moderate"
                elif np.array_equal(s_array, np.array([1, 1, 1, 3])):
                    return "Moderate"
                elif s_array[3] == 3 and s_array[2] == 3:
                    return "Mod-Severe"

            # --- uD = 0.58 ---
            if np.isclose(uD, 0.58, atol=0.01):
                return "Mod-Severe"

            # --- uD = 0.66 ---
            if np.isclose(uD, 0.66, atol=0.01):
                if np.array_equal(s_array, np.array([1, 1, 3, 3])):
                    return "Mod-Severe"
                elif np.array_equal(s_array, np.array([2, 2, 2, 2])):
                    return "Severe"
                elif np.array_equal(s_array, np.array([1, 2, 2, 3])):
                    return "Mod-Severe"
                elif s_array[3] == 3 and s_array[2] == 3 and s_array[1] == 2:
                    return "Severe"

            # --- uD = 0.75 ---
            if np.isclose(uD, 0.75, atol=0.01):
                if np.array_equal(s_array, np.array([2, 2, 2, 3])):
                    return "Severe"
                elif s_array[3] == 3 and s_array[2] == 3 and s_array[1] == 2:
                    return "Mod-Severe"
                elif s_array[3] == 3 and s_array[2] == 3 and s_array[1] == 3:
                    return "Severe"

            # --- uD = 0.83, 0.91, or 1 ---
            if (
                np.isclose(uD, 0.83, atol=0.01)
                or np.isclose(uD, 0.91, atol=0.01)
                or np.isclose(uD, 1, atol=0.01)
            ):
                return "Severe"

        return X.apply(_transform_row, axis=1)


class GAD7Labeler:
    def __init__(self):
        pass

    def fit(self, X, y=None):
        """
        Method kept for consistency with the scikit-learn API. But in this
        case, will simply call the `transform` method since no actual model
        gets fitted here.

        Needs to contain only 7 columns representing the GAD-7 items.
        Should be in the form of a DataFrame with shape (n_samples, 7).

        Parameters
        ----------
        X : pd.DataFrame
            Input features (n_samples, 7) with only the 7 GAD-7 items as
            columns and subject as rows.
        y : pd.Series, optional
            Target variable. Not used in this model. Keeps the API consistent.

        Returns
        -------
        pd.DataFrame
            Transformed features.
        """
        if X.shape[1] != 7:
            raise ValueError(
                "Input DataFrame must contain exactly 7 columns"
                " representing the GAD-7 items."
            )

        return self.transform(X)

    def transform(self, X):
        """
        Transform the input features to assign the label based on the GAD-7
        scoring.

        Needs to contain only 7 columns representing the GAD-7 items.
        Should be in the form of a DataFrame with shape (n_samples, 7).

        Final labels will be either:

        - Not anxious
        - Mild
        - Moderate
        - Severe

        Parameters
        ----------
        X : pd.DataFrame
            Input features.

        Returns
        -------
        pd.DataFrame
            Transformed features.
        """
        if X.shape[1] != 7:
            raise ValueError(
                "Input DataFrame must contain exactly 7 columns"
                " representing the GAD-7 items."
            )

        def _transform_row(row):
            # Apply the GAD-7 scoring fuzzy logic to each row.
            q1, q2, q3, q4, q5, q6, q7 = row.values

            # Compute uA (sum of q1 to q7) divided by 21
            uA = np.sum([q1, q2, q3, q4, q5, q6, q7]) / 21

            if uA < 0.285:
                return "Not Anxious"

            # Sort the array for easier comparisons.
            s_array = np.sort([q1, q2, q3, q4, q5, q6, q7])

            # --- 0.285 < uA < 0.428 ---
            if 0.285 < uA < 0.428:
                return "Mild"

            # --- uA = 0.428 ---
            if np.isclose(uA, 0.428, atol=0.01):
                if (
                    s_array[6] == 2
                    and s_array[5] == 2
                    and s_array[4] == 1
                    and s_array[3] == 1
                    and s_array[2] == 1
                    and s_array[1] == 1
                    and s_array[0] == 1
                ):
                    return "Mild"
                else:
                    return "Moderate"

            # --- uA = 0.476 ---
            if np.isclose(uA, 0.476, atol=0.01):
                if (
                    s_array[6] == 3
                    and s_array[5] == 2
                    and s_array[4] == 1
                    and s_array[3] == 1
                    and s_array[2] == 1
                    and s_array[1] == 1
                    and s_array[0] == 1
                ):
                    return "Mild"
                elif (
                    s_array[6] == 2
                    and s_array[5] == 2
                    and s_array[4] == 2
                    and s_array[3] == 1
                    and s_array[2] == 1
                    and s_array[1] == 1
                    and s_array[0] == 1
                ):
                    return "Mild"
                else:
                    return "Moderate"

            # --- uA = 0.523 ---
            if np.isclose(uA, 0.523, atol=0.01):
                if (
                    s_array[6] == 3
                    and s_array[5] == 2
                    and s_array[4] == 2
                    and s_array[3] == 1
                    and s_array[2] == 1
                    and s_array[1] == 1
                    and s_array[0] == 1
                ) or (
                    s_array[6] == 2
                    and s_array[5] == 2
                    and s_array[4] == 2
                    and s_array[3] == 2
                    and s_array[2] == 1
                    and s_array[1] == 1
                    and s_array[0] == 1
                ):
                    return "Mild"
                else:
                    return "Moderate"

            # --- uA = 0.571 ---
            if np.isclose(uA, 0.571, atol=0.01):
                return "Moderate"

            # --- uA = 0.619 ---
            if np.isclose(uA, 0.619, atol=0.01):
                if (
                    s_array[6] == 3
                    and s_array[5] == 3
                    and s_array[4] == 3
                    and s_array[3] == 2
                    and s_array[2] == 2
                ) or (
                    s_array[6] == 3
                    and s_array[5] == 3
                    and s_array[4] == 2
                    and s_array[3] == 2
                    and s_array[2] == 2
                    and s_array[1] == 1
                ):
                    return "Severe"
                else:
                    return "Moderate"

            # --- uA = 0.666 ---
            if np.isclose(uA, 0.666, atol=0.01):
                if (
                    (
                        s_array[6] == 3
                        and s_array[5] == 3
                        and s_array[4] == 3
                        and s_array[3] == 3
                        and s_array[2] == 1
                        and s_array[1] == 1
                        and s_array[0] == 0
                    )
                    or (
                        s_array[6] == 3
                        and s_array[5] == 3
                        and s_array[4] == 3
                        and s_array[3] == 2
                        and s_array[2] == 1
                        and s_array[1] == 1
                        and s_array[0] == 1
                    )
                    or (np.array_equal(s_array,
                                       np.array([2, 2, 2, 2, 2, 2, 2])))
                ):
                    return "Moderate"
                else:
                    return "Severe"

            # --- uA = 0.714 ---
            if np.isclose(uA, 0.714, atol=0.01):
                if (
                    (
                        s_array[6] == 3
                        and s_array[5] == 3
                        and s_array[4] == 3
                        and s_array[3] == 3
                        and s_array[2] == 3
                    )
                    or (
                        s_array[6] == 3
                        and s_array[5] == 3
                        and s_array[4] == 3
                        and s_array[3] == 3
                    )
                    or (
                        s_array[6] == 3
                        and s_array[5] == 3
                        and s_array[4] == 3
                        and s_array[3] == 2
                        and s_array[2] == 2
                        and s_array[1] == 2
                    )
                ):
                    return "Severe"
                else:
                    return "Moderate"

            # --- uA = 0.761, 0.809, 0.857, 0.904, or 1 ---
            if uA > 0.760:
                return "Severe"

        return X.apply(_transform_row, axis=1)
