# -*- coding: utf-8 -*-

import warnings

import numpy as np
import skfuzzy as fuzz
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_random_state
from sklearn.utils.parallel import Parallel, delayed
from sklearn.utils.validation import check_array, check_is_fitted

from neurostatx.clustering.metrics import (
    compute_evaluation_metrics,
    compute_gap_stats,
    compute_sse,
)
from neurostatx.clustering.viz import plot_fuzzy_cmeans_solutions


class FuzzyCMeans(ClusterMixin, BaseEstimator):
    """Fuzzy C-Means clustering.

    Soft clustering estimator with a sklearn-compatible API. Fits a single
    number of clusters; use [search_fuzzy_cmeans][neurostatx.clustering.fuzzy.search_fuzzy_cmeans] to evaluate a range
    of ``k``. Based on scikit-fuzzy ``cmeans`` /
    ``cmeans_predict``.

    Parameters
    ----------
    n_clusters : int, optional
        Number of clusters to form. Defaults to 2.
    m : float, optional
        Fuzziness exponent applied to the membership function. ``m=1``
        yields crisp clusters; larger values increase fuzziness.
        Defaults to 2.
    max_iter : int, optional
        Maximum number of iterations. Defaults to 1000.
    tol : float, optional
        Convergence tolerance (stopping criterion). Defaults to 1e-6.
    metric : str, optional
        Distance metric passed to scikit-fuzzy. Defaults to ``"euclidean"``.
    init : array-like of shape (n_clusters, n_samples), optional
        Initial fuzzy c-partitioned matrix. Defaults to None (random init).
    random_state : int, numpy.RandomState instance or None, optional
        Seed used for centroid initialization. Defaults to None.

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centroids.
    u_ : ndarray of shape (n_clusters, n_samples)
        Fuzzy membership matrix (scikit-fuzzy layout).
    labels_ : ndarray of shape (n_samples,)
        Hard cluster labels (argmax of membership).
    fpc_ : float
        Fuzzy partition coefficient.
    n_iter_ : int
        Number of iterations run until convergence.
    inertia_ : float
        Within-cluster sum of squared errors (WSS).
    n_features_in_ : int
        Number of features seen during [fit][neurostatx.clustering.fuzzy.FuzzyCMeans.fit].

    Examples
    --------
    >>> import numpy as np
    >>> from neurostatx.clustering.fuzzy import FuzzyCMeans
    >>> X = np.array([[0.0, 0.0], [0.1, 0.0], [1.0, 1.0], [0.9, 1.0]])
    >>> fcm = FuzzyCMeans(n_clusters=2, random_state=0).fit(X)
    >>> fcm.labels_.shape
    (4,)
    """

    def __init__(
        self,
        n_clusters=2,
        m=2.0,
        max_iter=1000,
        tol=1e-6,
        metric="euclidean",
        init=None,
        random_state=None,
    ):
        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.tol = tol
        self.metric = metric
        self.init = init
        self.random_state = random_state

    def _get_seed(self):
        """Return an integer seed for scikit-fuzzy, or None."""
        if self.random_state is None:
            return None
        if isinstance(self.random_state, (int, np.integer)):
            return int(self.random_state)
        rng = check_random_state(self.random_state)
        return int(rng.randint(0, np.iinfo(np.int32).max))

    def _validate_predict_input(self, X):
        check_is_fitted(self, "cluster_centers_")
        X = check_array(X, dtype=np.float64, ensure_2d=True)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but {type(self).__name__} "
                f"is expecting {self.n_features_in_} features as input."
            )
        return X

    def _cmeans_predict_u(self, X):
        """Return membership in scikit-fuzzy layout (n_clusters, n_samples)."""
        u, _, _, _, _, _ = fuzz.cmeans_predict(
            X.T,
            self.cluster_centers_,
            m=self.m,
            error=self.tol,
            maxiter=self.max_iter,
            metric=self.metric,
            init=None,
            seed=self._get_seed(),
        )
        return u

    def fit(self, X, y=None):
        """Fit the fuzzy C-Means model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : Ignored
            Not used, present for sklearn API compatibility.

        Returns
        -------
        self : FuzzyCMeans
            Fitted estimator.

        Examples
        --------
        >>> import numpy as np
        >>> from neurostatx.clustering.fuzzy import FuzzyCMeans
        >>> X = np.array([[0.0, 0.0], [0.1, 0.0], [1.0, 1.0], [0.9, 1.0]])
        >>> fcm = FuzzyCMeans(n_clusters=2, random_state=0).fit(X)
        >>> fcm.n_features_in_
        2
        """
        X = check_array(X, dtype=np.float64, ensure_2d=True)
        cntr, u, _, _, _, n_iter, fpc = fuzz.cmeans(
            X.T,
            self.n_clusters,
            m=self.m,
            error=self.tol,
            maxiter=self.max_iter,
            metric=self.metric,
            init=self.init,
            seed=self._get_seed(),
        )
        self.cluster_centers_ = cntr
        self.u_ = u
        self.labels_ = np.argmax(u, axis=0)
        self.fpc_ = fpc
        self.n_iter_ = n_iter
        self.inertia_ = compute_sse(X, cntr, self.labels_)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        """Predict hard cluster labels for ``X``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data to predict.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Hard cluster labels.

        Examples
        --------
        >>> import numpy as np
        >>> from neurostatx.clustering.fuzzy import FuzzyCMeans
        >>> X = np.array([[0.0, 0.0], [0.1, 0.0], [1.0, 1.0], [0.9, 1.0]])
        >>> fcm = FuzzyCMeans(n_clusters=2, random_state=0).fit(X)
        >>> fcm.predict(X).shape
        (4,)
        """
        X = self._validate_predict_input(X)
        u = self._cmeans_predict_u(X)
        return np.argmax(u, axis=0)

    def predict_proba(self, X):
        """Predict fuzzy membership for ``X``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data to predict.

        Returns
        -------
        membership : ndarray of shape (n_samples, n_clusters)
            Membership degree of each sample to each cluster. Rows sum to 1.

        Examples
        --------
        >>> import numpy as np
        >>> from neurostatx.clustering.fuzzy import FuzzyCMeans
        >>> X = np.array([[0.0, 0.0], [0.1, 0.0], [1.0, 1.0], [0.9, 1.0]])
        >>> fcm = FuzzyCMeans(n_clusters=2, random_state=0).fit(X)
        >>> proba = fcm.predict_proba(X)
        >>> proba.shape
        (4, 2)
        """
        X = self._validate_predict_input(X)
        u = self._cmeans_predict_u(X)
        return u.T

    def _more_tags(self):
        return {"X_types": ["2darray"], "allow_nan": False}

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = False
        return tags


def process_cluster(
    X,
    n_cluster,
    min_clusters,
    m,
    tol,
    max_iter,
    init,
    metric,
    random_state,
    compute_gap,
):
    """
    Core worker of [search_fuzzy_cmeans][neurostatx.clustering.fuzzy.search_fuzzy_cmeans]. Fit one ``k``, compute
    metrics, and subsample points for visualization.

    Must remain a module-level function so it can be pickled by
    multiprocessing.
    """
    if init is not None:
        init_mat = init[n_cluster - min_clusters]
    else:
        init_mat = None

    fcm = FuzzyCMeans(
        n_clusters=n_cluster,
        m=m,
        max_iter=max_iter,
        tol=tol,
        metric=metric,
        init=init_mat,
        random_state=random_state,
    )
    fcm.fit(X)

    ss, chi, dbi = compute_evaluation_metrics(
        X, fcm.labels_, metric=metric
    )
    wss = fcm.inertia_

    if compute_gap:
        gap, sk = compute_gap_stats(
            X,
            wss,
            nrefs=100,
            n_cluster=n_cluster,
            m=m,
            error=tol,
            metric=metric,
            maxiter=max_iter,
            init=None,
        )
    else:
        gap, sk = None, None

    xpts = X[:, 0]
    ypts = X[:, 1]

    if X.shape[0] > 500:
        indices = np.random.choice(X.shape[0], size=500, replace=False)
        xpts_for_viz = xpts[indices]
        ypts_for_viz = ypts[indices]
        cluster_membership_for_viz = fcm.labels_[indices]
    else:
        xpts_for_viz = xpts
        ypts_for_viz = ypts
        cluster_membership_for_viz = fcm.labels_

    return (
        n_cluster,
        fcm.n_iter_,
        fcm.fpc_,
        fcm.cluster_centers_,
        fcm.u_,
        wss,
        ss,
        chi,
        dbi,
        gap,
        sk,
        xpts_for_viz,
        ypts_for_viz,
        cluster_membership_for_viz,
    )


def search_fuzzy_cmeans(
    X,
    min_clusters=2,
    max_clusters=10,
    n_jobs=1,
    compute_gap=True,
    output=None,
    verbose=False,
    m=2.0,
    max_iter=1000,
    tol=1e-6,
    metric="euclidean",
    init=None,
    random_state=None,
):
    """Fit Fuzzy C-Means for each k in ``[min_clusters, max_clusters]``.

    Parallelizes over ``k`` and optionally writes a multi-panel
    visualization of all solutions.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data to cluster.
    min_clusters : int, optional
        Smallest number of clusters to fit. Defaults to 2.
    max_clusters : int, optional
        Largest number of clusters to fit. Defaults to 10.
    n_jobs : int, optional
        Number of parallel jobs. Defaults to 1.
    compute_gap : bool, optional
        If True, compute GAP statistics (expensive). Defaults to True.
    output : str, optional
        Folder in which to save ``viz_multiple_cluster_nb.png``. If None,
        no figure is written. Defaults to None.
    verbose : bool, optional
        Verbosity flag forwarded to sklearn ``Parallel``. Defaults to False.
    m : float, optional
        Fuzziness exponent. Defaults to 2.
    max_iter : int, optional
        Maximum iterations per model. Defaults to 1000.
    tol : float, optional
        Convergence tolerance. Defaults to 1e-6.
    metric : str, optional
        Distance metric. Defaults to ``"euclidean"``.
    init : list of array-like, optional
        Initial membership matrices, one per ``k`` from ``min_clusters``
        to ``max_clusters``. Defaults to None.
    random_state : int, numpy.RandomState instance or None, optional
        Seed forwarded to each [FuzzyCMeans][neurostatx.clustering.fuzzy.FuzzyCMeans] fit. Defaults to None.

    Returns
    -------
    cntr : tuple of ndarray
        Cluster centroids for each ``k``.
    u : tuple of ndarray
        Membership matrices for each ``k``.
    wss : tuple of float
        Within-cluster sum of squared errors.
    fpc : tuple of float
        Fuzzy partition coefficients.
    ss : tuple of float
        Silhouette scores.
    chi : tuple of float
        Calinski-Harabasz indices.
    dbi : tuple of float
        Davies-Bouldin indices.
    gap : tuple of float
        GAP statistics (or None if ``compute_gap`` is False).
    sk : tuple of float
        GAP standard errors (or None if ``compute_gap`` is False).

    Examples
    --------
    >>> import numpy as np
    >>> from neurostatx.clustering.fuzzy import search_fuzzy_cmeans
    >>> X = np.random.RandomState(0).rand(30, 4)
    >>> cntr, u, wss, fpc, ss, chi, dbi, gap, sk = search_fuzzy_cmeans(
    ...     X, min_clusters=2, max_clusters=3, compute_gap=False, n_jobs=1
    ... )
    >>> len(cntr)
    2
    """
    X = check_array(X, dtype=np.float64, ensure_2d=True)
    if max_clusters < min_clusters:
        raise ValueError(
            f"max_clusters ({max_clusters}) must be >= min_clusters "
            f"({min_clusters})."
        )

    results = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(process_cluster)(
            X,
            k,
            min_clusters=min_clusters,
            m=m,
            tol=tol,
            max_iter=max_iter,
            init=init,
            metric=metric,
            random_state=random_state,
            compute_gap=compute_gap,
        )
        for k in range(min_clusters, max_clusters + 1)
    )

    _, _, fpc, cntr, u, wss, ss, chi, dbi, gap, sk, _, _, _ = zip(*results)

    if output is not None:
        plot_fuzzy_cmeans_solutions(
            results,
            output=output,
            min_clusters=min_clusters,
            max_clusters=max_clusters,
        )

    return cntr, u, wss, fpc, ss, chi, dbi, gap, sk


def fuzzyCmeans(
    X,
    max_cluster=10,
    m=2,
    error=1e-6,
    maxiter=1000,
    init=None,
    metric="euclidean",
    output="./",
    processes=1,
    verbose=False,
):
    """Deprecated wrapper around [search_fuzzy_cmeans][neurostatx.clustering.fuzzy.search_fuzzy_cmeans].

    Notes
    -----
    Deprecated. Use [FuzzyCMeans][neurostatx.clustering.fuzzy.FuzzyCMeans]
    for a single number of clusters, or
    [search_fuzzy_cmeans][neurostatx.clustering.fuzzy.search_fuzzy_cmeans]
    to evaluate a range of ``k``.

    Parameters
    ----------
    X : np.array
        Array of data to cluster (subjects x features).
    max_cluster : int, optional
        Maximum number of clusters to fit a model for. Defaults to 10.
    m : float, optional
        Exponentiation value to apply on the membership function.
        Defaults to 2.
    error : float, optional
        Stopping criterion. Defaults to 1e-6.
    maxiter : int, optional
        Maximum number of iterations. Defaults to 1000.
    init : 2d array, optional
        Initial fuzzy c-partitioned matrix. Defaults to None.
    metric : str, optional
        Distance metric used for intra/inter subject and cluster distance.
        Defaults to ``"euclidean"``.
    output : str, optional
        Output folder for the visualization. Defaults to ``"./"``.
    processes : int, optional
        Number of processes to use. Defaults to 1.
    verbose : bool, optional
        If True, produce verbose output. Defaults to False.

    Returns
    -------
    cntr : list
        Cluster centroids for each ``k``.
    u : list
        Membership matrices for each ``k``.
    wss : list
        Within-cluster sum of squared errors.
    fpc : list
        Fuzzy partition coefficients.
    ss : list
        Silhouette scores.
    chi : list
        Calinski-Harabasz indices.
    dbi : list
        Davies-Bouldin indices.
    gap : list
        GAP statistics.
    sk : list
        GAP standard errors.

    Examples
    --------
    >>> import numpy as np
    >>> from neurostatx.clustering.fuzzy import fuzzyCmeans
    >>> X = np.random.RandomState(0).rand(20, 3)
    >>> cntr, u, wss, fpc, ss, chi, dbi, gap, sk = fuzzyCmeans(
    ...     X, max_cluster=3, processes=1, output=None
    ... )
    >>> len(cntr)
    2
    """
    warnings.warn(
        "fuzzyCmeans() is deprecated and will be removed in a future "
        "release. Use FuzzyCMeans for a single k or search_fuzzy_cmeans "
        "to evaluate a range of cluster counts.",
        DeprecationWarning,
        stacklevel=2,
    )
    return search_fuzzy_cmeans(
        X,
        min_clusters=2,
        max_clusters=max_cluster,
        n_jobs=processes,
        compute_gap=True,
        output=output,
        verbose=verbose,
        m=m,
        max_iter=maxiter,
        tol=error,
        metric=metric,
        init=init,
        random_state=0,
    )
