# -*- coding: utf-8 -*-

from enum import Enum
from scipy.spatial.distance import cdist


class DistanceMetrics(str, Enum):
    braycurtis = 'braycurtis', 
    canberra = 'canberra', 
    chebyshev = 'chebyshev', 
    cityblock = 'cityblock', 
    correlation = 'correlation', 
    cosine = 'cosine', 
    dice = 'dice', 
    euclidean = 'euclidean', 
    hamming = 'hamming', 
    jaccard = 'jaccard', 
    jensenshannon = 'jensenshannon', 
    kulczynski1 = 'kulczynski1', 
    mahalanobis = 'mahalanobis', 
    matching = 'matching', 
    minkowski = 'minkowski', 
    rogerstanimoto = 'rogerstanimoto', 
    russellrao = 'russellrao', 
    seuclidean = 'seuclidean', 
    sokalmichener = 'sokalmichener', 
    sokalsneath = 'sokalsneath', 
    sqeuclidean = 'sqeuclidean', 
    yule = 'yule'
    

def compute_distance(data, centers, metric='euclidean'):
    """
    Function computing the distance between two rectangle arrays.
    Available distance metric are the ones from scipy.spatial.distance.cdist().
    List can be found here : 
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html#scipy.spatial.distance.cdist
    
    Args:
        data (Array):                   Array containing original data points.
        centers (Array):                Array containing clusters' centroids. 
        method (str, optional):         Distance metric to use. Defaults to 'euclidean'.

    Returns:
        d:                              Distance matrix. 
    """
    
    return cdist(data, centers, metric=metric).T
