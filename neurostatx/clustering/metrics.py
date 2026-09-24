# -*- coding: utf-8 -*-

from kneed import KneeLocator
import numpy as np
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score
)
import skfuzzy as fuzz


def compute_evaluation_metrics(X, labels, metric="euclidean"):
    """Compute silhouette, Calinski-Harabasz, and Davies-Bouldin scores.

    Parameters
    ----------
    X : array-like
        Data used to evaluate the clustering solution.
    labels : list
        Cluster labels for each sample.
    metric : str, optional
        Distance metric passed to the silhouette score. Defaults to
        ``"euclidean"``. Accepts options from
        ``sklearn.metrics.pairwise.pairwise_distances``.

    Returns
    -------
    ss : float
        Silhouette score.
    chi : float
        Calinski-Harabasz index.
    dbi : float
        Davies-Bouldin index.

    Examples
    --------
    >>> import numpy as np
    >>> from neurostatx.clustering.metrics import compute_evaluation_metrics
    >>> X = np.array([[0.0, 0.0], [0.1, 0.0], [1.0, 1.0], [0.9, 1.1]])
    >>> ss, chi, dbi = compute_evaluation_metrics(X, [0, 0, 1, 1])
    >>> ss > 0
    True
    """

    # Storing Silhouette score.
    ss = silhouette_score(X, labels, metric=metric)

    # Storing Calinski-Harabasz Indices (CHI).
    chi = calinski_harabasz_score(X, labels)

    # Storing the Davies-Bouldin Index (DBI).
    dbi = davies_bouldin_score(X, labels)

    return ss, chi, dbi


def compute_knee_location(lst, direction="decreasing"):
    """Return the elbow location of a clustering metric curve.

    Parameters
    ----------
    lst : list
        Metric values used to locate the elbow, typically one per ``k``.
    direction : str, optional
        Curve direction passed to Kneed. Defaults to ``"decreasing"``.

    Returns
    -------
    elbow : int
        Estimated elbow location (cluster count).

    Examples
    --------
    >>> from neurostatx.clustering.metrics import compute_knee_location
    >>> compute_knee_location([10.0, 4.0, 3.5, 3.2])
    3
    """

    knee = KneeLocator(
        range(2, len(lst) + 2), lst, S=1, curve="convex", direction=direction
    )
    elbow = knee.elbow

    return elbow


def compute_sse(X, cntr, labels):
    """Compute the within-cluster sum of squared errors (WSS).

    Parameters
    ----------
    X : array
        Original data of shape (n_samples, n_features).
    cntr : array
        Cluster centroids of shape (n_clusters, n_features).
    labels : array
        Hard cluster labels of shape (n_samples,).

    Returns
    -------
    WSS : float
        Within-cluster sum of squared errors.

    Examples
    --------
    >>> import numpy as np
    >>> from neurostatx.clustering.metrics import compute_sse
    >>> X = np.array([[0.0, 0.0], [1.0, 1.0]])
    >>> cntr = np.array([[0.0, 0.0], [1.0, 1.0]])
    >>> compute_sse(X, cntr, np.array([0, 1]))
    0.0
    """

    WSS = 0
    for k in np.unique(labels):
        data_k = X[labels == k]
        WSS += np.sum([np.abs(data_k - cntr[k]) ** 2])

    return WSS


def compute_gap_stats(
    X,
    wss,
    nrefs,
    n_cluster,
    m=2,
    error=1e-6,
    maxiter=1000,
    metric="euclidean",
    init=None,
):
    """Compute the GAP statistic against uniformly sampled reference data.

    Parameters
    ----------
    X : np.array
        Data array used to generate reference samples.
    wss : float
        Within-cluster sum of squared errors for the fitted model.
    nrefs : int
        Number of random reference datasets to average.
    n_cluster : int
        Number of clusters in the fitted model.
    m : float, optional
        Fuzziness exponent passed to Fuzzy C-Means. Defaults to 2.
    error : float, optional
        Convergence tolerance. Defaults to 1e-6.
    maxiter : int, optional
        Maximum iterations for each reference fit. Defaults to 1000.
    metric : str, optional
        Distance metric. Defaults to ``"euclidean"``.
    init : array, optional
        Initial fuzzy c-partitioned matrix. Defaults to None.

    Returns
    -------
    gap : float
        GAP statistic.
    sk : float
        Standard error of the GAP statistic.

    Examples
    --------
    >>> import numpy as np
    >>> from neurostatx.clustering.metrics import compute_gap_stats,
    ...                                           compute_sse
    >>> X = np.array([[0.0, 0.0], [0.1, 0.0], [1.0, 1.0], [0.9, 1.0]])
    >>> cntr = np.array([[0.05, 0.0], [0.95, 1.0]])
    >>> labels = np.array([0, 0, 1, 1])
    >>> gap, sk = compute_gap_stats(X, compute_sse(X, cntr, labels),
    ...                             nrefs=2, n_cluster=2)
    >>> gap > 0
    True
    """

    refDisps = np.zeros(nrefs)
    a, b = X.min(axis=0, keepdims=True), X.max(axis=0, keepdims=True)

    for i in range(nrefs):
        randomRef = np.random.random_sample(size=X.shape) * (b - a) + a

        cntr, u, u0, d, jm, p, fpc = fuzz.cmeans(
            randomRef.T,
            n_cluster,
            m=m,
            error=error,
            maxiter=maxiter,
            metric=metric,
            init=init,
        )

        labels = np.argmax(u, axis=0)
        refDisps[i] = compute_sse(randomRef, cntr, labels)

    gap = np.mean(np.log(refDisps)) - np.log(wss)

    # Compute standard deviation.
    sdk = np.sqrt(np.mean((np.log(refDisps) - np.mean(np.log(refDisps))
                           ) ** 2.0))
    sk = np.sqrt(1.0 + 1.0 / nrefs) * sdk

    return gap, sk


def find_optimal_gap(gap, sk):
    """Return the first k satisfying Tibshirani's GAP rule.

    Selects the first index where ``GAP[k] >= GAP[k+1] - SD[k+1]``.

    Parameters
    ----------
    gap : np.array
        GAP statistic values for a range of cluster counts.
    sk : np.array
        Standard errors aligned with ``gap``.

    Returns
    -------
    optimal : int
        Index of the selected solution.

    Examples
    --------
    >>> import numpy as np
    >>> from neurostatx.clustering.metrics import find_optimal_gap
    >>> find_optimal_gap(np.array([0.2, 0.8, 0.7]), np.array([0.1, 0.1, 0.1]))
    1
    """

    for i in range(len(gap)):
        if i + 1 in range(len(gap)):
            if gap[i] >= (gap[i + 1] - sk[i + 1]):
                optimal = i
                break
        else:
            optimal = i
            break

    return optimal


def compute_rand_index(dict):
    """Compute pairwise adjusted Rand indices from membership dataframes.

    Hard labels are obtained with ``argmax`` before scoring, because the
    adjusted Rand index applies only to crisp partitions.

    Parameters
    ----------
    dict : dict
        Mapping of labels to membership dataframes.

    Returns
    -------
    ari : np.array
        Symmetric pairwise adjusted Rand index matrix.

    Examples
    --------
    >>> import pandas as pd
    >>> from neurostatx.clustering.metrics import compute_rand_index
    >>> a = pd.DataFrame([[0.9, 0.1], [0.2, 0.8]])
    >>> b = pd.DataFrame([[0.8, 0.2], [0.1, 0.9]])
    >>> compute_rand_index({"a": a, "b": b}).shape
    (2, 2)
    """

    ari = []
    keys = list(dict.keys())
    for k in keys:
        for k2 in keys:
            val1 = dict[k].values.argmax(axis=1)
            val2 = dict[k2].values.argmax(axis=1)
            ari.append(adjusted_rand_score(val1, val2))

    return np.array(ari).reshape(len(keys), len(keys))
