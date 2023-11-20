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

#
# from CCPM.clustering.cluster import cmeans


def compute_evaluation_metrics(X, labels, metric="euclidean"):
    """
    Function to compute a variety of metrics to evaluate the goodness of fit
    of a clustering model.

    Args:
        X (Array-like):             Data from clustering algorithm to
                                    derive metrics from.
        labels (List):              List of labels.
        metric (str, optional):     Distance metric to use. Defaults
                                    to 'euclidean'. Accept options from
                                    sklearn.metrics.pairwise.pairwise_distances.

    Returns:
        ss:                         Silhouette Score (SS).
        chi:                        Calinski Harabasz Score (CHI).
        dbi:                        Davies Bouldin Score (DBI).
    """

    # Storing Silhouette score.
    ss = silhouette_score(X, labels, metric=metric)

    # Storing Calinski-Harabasz Indices (CHI).
    chi = calinski_harabasz_score(X, labels)

    # Storing the Davies-Bouldin Index (DBI).
    dbi = davies_bouldin_score(X, labels)

    return ss, chi, dbi


def compute_knee_location(lst, direction="decreasing"):
    """
    Funtion to compute the Elbow location using the Kneed package.

    Args:
        lst (List):     List of values representing the indicators to identify
                        the elbow location.

    Returns:
        elbow:          Elbow location.
    """

    knee = KneeLocator(
        range(2, len(lst) + 2), lst, S=1, curve="convex", direction=direction
    )
    elbow = knee.elbow

    return elbow


def compute_sse(X, cntr, umat):
    """
    Function to compute within cluster sum of square error (WSS).
    Adapted from :
    https://towardsdatascience.com/how-to-determine-the-right-number-of-clusters-with-code-d58de36368b1

    Args:
        X (Array):          Original data (S, N)
        cntr (Array):       Centroid points (N, F)
        umat (Array):       Membership array (S, N)

    Returns:
        WSS:                Within Sum-of-Squares Error (WSS)
    """

    labels = np.argmax(umat, axis=0)

    WSS = 0
    for k in np.unique(labels):
        data_k = X[labels == k]
        WSS += np.sum([np.abs(data_k - cntr[k]) ** 2])

    return WSS


def compute_sse_from_dmat(dmat, umat):
    """
    ** Not to be used for now **
    Function to compute within cluster sum of square error (WSS). It uses the
    precomputed distance matrix from cmeans().

    Args:
        dmat (Array):       Distance matrix (S, N)
        umat (Array):       Membership matrix (S, N)

    Returns:
        WSS:             Within Sum-of-Squares Error (WSS)
    """

    min_dist = np.min(dmat, axis=0)
    labels = np.argmax(umat, axis=0)

    WSS = 0
    for k in np.unique(labels):
        data_k = min_dist[labels == k]
        WSS += (data_k**2).sum()

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
    """
    Function to compute the GAP Statistics to determine the optimal number of
    clusters.
    Adapted from :
    https://towardsdatascience.com/cheat-sheet-to-implementing-7-methods-for-selecting-optimal-number-of-clusters-in-python-898241e1d6ad
    and https://github.com/milesgranger/gap_statistic

    Args:
        X (array):                      Data array on which clustering will be
                                        computed.
        wss (float):                    Within Cluster Sum of Squared Error
                                        (WSS) for this clustering model.
        nrefs (int):                    Number of random reference data to
                                        generate and average.
        n_cluster (int):                Number of cluster in for this model.
        m (int, optional):              Exponentiation value as used in the
                                        main script. Defaults to 2.
        error (float, optional):        Convergence error threshold. Defaults
                                        to 1E-6.
        maxiter (int, optional):        Maximum iterations to perform.
                                        Defaults to 1000.
        metric (str, optional):         Distance metric to use. Defaults to
                                        'euclidean'.
        init (array, optional):         Initial fuzzy c-partitioned matrix.
                                        Defaults to None.

    Returns:
        gap:                            GAP Statistics.
        sk:                             Standard deviation of the GAP
                                        statistic.
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

        refDisps[i] = compute_sse(randomRef, cntr, u)

    gap = np.mean(np.log(refDisps)) - np.log(wss)

    # Compute standard deviation.
    sdk = np.sqrt(np.mean((np.log(refDisps) - np.mean(np.log(refDisps))
                           ) ** 2.0))
    sk = np.sqrt(1.0 + 1.0 / nrefs) * sdk

    return gap, sk


def find_optimal_gap(gap, sk):
    """
    Function to find the optimal k number based on the GAP statistics using
    the method from Tibshirani R. et al., 2001
    (https://hastie.su.domains/Papers/gap.pdf). Highlights the first
    k value where GAP[k] >= GAP[k+1] - SD[k+1].

    Args:
        gap (ndarray):              Ndarray of GAP statistics values for a
                                    range of k clusters.
        sk (ndarray):               Ndarray of standard deviation for each GAP
                                    values.

    Returns:
        optimal:                    Optimal k value.
    """

    for k in range(len(gap)):
        if gap[k] >= (gap[k + 1] - sk[k + 1]):
            optimal = k
            break

    return optimal


def compute_rand_index(dict):
    """
    Compute the adjusted Rand Index from a list of fuzzy membership matrices
    using sklearn.metrics.adjusted_rand_score. A defuzzification step is
    required since this method applies only to crisp clusters.

    Args:
        dict (dictionnary):             Dictonnary containing all dataframes.

    Return:
        Ndarray                         Symmetric ndarray.
    """

    ari = []
    keys = list(dict.keys())
    for k in keys:
        for k2 in keys:
            val1 = dict[k].values.argmax(axis=1)
            val2 = dict[k2].values.argmax(axis=1)
            ari.append(adjusted_rand_score(val1, val2))

    return np.array(ari).reshape(len(keys), len(keys))
