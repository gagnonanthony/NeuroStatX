from kneed import KneeLocator
import numpy as np
from sklearn.metrics import (silhouette_score,
                             calinski_harabasz_score,
                             davies_bouldin_score)
import skfuzzy as fuzz


def compute_evaluation_metrics(X, labels, distance='euclidean'):
    """
    Function to compute a variety of metrics to evaluate the goodness of fit of a clustering model. 

    Args:
        X (Array-like):                     Data from clustering algorithm to derive metrics from.
        labels (List):                      List of labels. 
        distance (str, optional):           Distance method to use. Defaults to 'euclidean'.

    Returns:
        ss:                             Silhouette Score (SS).
        chi:                            Calinski Harabasz Score (CHI).
        dbi:                            Davies Bouldin Score (DBI).
    """
    
    # Storing Silhouette score.
    ss = silhouette_score(X, labels, metric=distance)
            
    # Storing Calinski-Harabasz Indices (CHI). 
    chi = calinski_harabasz_score(X, labels)
            
    # Storing the Davies-Bouldin Index (DBI).
    dbi = davies_bouldin_score(X, labels)
    
    return ss, chi, dbi


def compute_knee_location(lst):
    """
    Funtion to compute the Elbow location using the Kneed package. 

    Args:
        lst (List):     List of values representing the indicators to identify the elbow location,

    Returns:
        elbow:          Elbow location. 
    """
    
    knee = KneeLocator(
        range(2, len(lst)+2), lst, curve='convex', direction='decreasing'
    )
    elbow = knee.elbow
    
    return elbow


def compute_sse(dmat, umat):
    """
    Function to compute within cluster sum of square error (WSS).
    Adapted from : https://towardsdatascience.com/how-to-determine-the-right-number-of-clusters-with-code-d58de36368b1

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


def compute_gap_stats(X, wss, nrefs, n_cluster, m=2, error=1E-6, maxiter=1000, init=None):
    """
    Function to compute the GAP Statistics to determine the optimal number of clusters. 
    Adapted from : https://towardsdatascience.com/cheat-sheet-to-implementing-7-methods-for-selecting-optimal-number-of-clusters-in-python-898241e1d6ad

    Args:
        X (array):                      Data array on which clustering will be computed.
        wss (float):                    Within Cluster Sum of Squared Error (WSS) for this clustering model.
        nrefs (int):                    Number of random reference data to generate and average. 
        n_cluster (int):                Number of cluster in for this model. 
        m (int, optional):              Exponentiation value as used in the main script. Defaults to 2.
        error (float, optional):        Convergence error threshold. Defaults to 1E-6.
        maxiter (int, optional):        Maximum iterations to perform. Defaults to 1000.
        init (array, optional):         Initial fuzzy c-partitioned matrix. Defaults to None.

    Returns:
        gap:                            GAP Statistics. 
    """
    
    refDisps = np.zeros(nrefs)
    
    for i in range(nrefs):
        randomRef = np.random.random_sample(size=X.shape)
        
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
                randomRef, n_cluster, m=m, error=error, maxiter=maxiter, init=init
            )

        refDisps[i] = compute_sse(d, u)
    
    gap = np.log(np.mean(refDisps)) - np.log(wss)
    
    return gap