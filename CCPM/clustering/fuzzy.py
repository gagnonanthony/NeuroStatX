import math

import numpy as np
from sklearn.decomposition import PCA
import skfuzzy as fuzz
from matplotlib import pyplot as plt
from matplotlib.pyplot import get_cmap
from matplotlib.colors import rgb2hex
import pandas as pd

from CCPM.clustering.metrics import compute_evaluation_metrics, compute_sse, compute_gap_stats


def fuzzyCmeans(X, max_cluster=10, m=2, error=1E-6, maxiter=1000, init=None, distance='euclidean', output='./'):
    """ Fuzzy C-Means clustering function. Iteratively test and report statistics on multiple number
        of clusters. Based on documentation found here : 
        https://pythonhosted.org/scikit-fuzzy/auto_examples/plot_cmeans.html
        
    Args:
        X (Numpy array):                Numpy array with data to cluster (Subject x Features).
        max_cluster (int, optional):    Maximum number of clusters to fit a model for. Defaults to 10.
        m (float, optional):            Exponentiation value to apply on the membership function. Defaults to 2.
        error (float, optional):        Stopping criterion. Defaults to 1E-6.
        maxiter (int, optional):        Maximum iteration value. Defaults to 1000.
        init (2d array, optional):      Initial fuzzy c-partitioned matrix. Defaults to None.
        distance (str, optional):       Distance method to use to compute intra/inter subjects/clusters distance. Defaults to
                                        euclidean.
        output (String, optional):      Output folder. Defaults to './'.

    Returns:
        _type_: _description_
    """
    
    num_clusters = max_cluster
    wss = list()
    fpcs = list()
    ss = list()
    chi = list()
    dbi = list()
    gap = list()
    
    # Setting color palette.
    cmap = get_cmap('PiYG', max_cluster)
    colors = [rgb2hex(cmap(i)) for i in range(cmap.N)]
    
    # Reducing the features to 2 for better visualization.
    #viz = PCA(n_components=2).fit_transform(X)
    xpts = X[:, 0]
    ypts = X[:, 1]
    
    # Fixing plot grid dimensions.
    grid = math.ceil(math.sqrt(num_clusters))
    
    fig1, axes1 = plt.subplots(grid, grid, figsize=(8,8))
    for n_cluster, ax in enumerate(axes1.reshape(-1), 2):
        if n_cluster <= num_clusters:
            cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
                X.T, n_cluster, m=m, error=error, maxiter=maxiter, init=init
            )
            
            # Storing fuzzy partition coefficient (FPC) in a list for plotting.
            fpcs.append(fpc)
            
            # Plotting clusters with hard assignment of membership.
            cluster_membership = np.argmax(u, axis=0)
            
            # Computing evaluation metrics. 
            ss_u, chi_u, dbi_u = compute_evaluation_metrics(X, cluster_membership, distance=distance)
            ss.append(ss_u)
            chi.append(chi_u)
            dbi.append(dbi_u)
            wss_ = compute_sse(d, u)
            wss.append(wss_)
            
            # Compute gap statistics. 
            gap_ = compute_gap_stats(X.T, wss_, nrefs=3, n_cluster=n_cluster, m=m, error=error, maxiter=maxiter, init=init)
            gap.append(gap_)
            
            # Selecting a random sample for plotting to avoid useless memory consumption.
            if X.shape[0] > 500:
                indices = np.random.choice(X.shape[0], size=500, replace=False)
                xpts_for_viz = xpts[indices]
                ypts_for_viz = ypts[indices]
                cluster_membership_for_viz = cluster_membership[indices]
            
            # Plotting results.
            for j in range(n_cluster):
                ax.plot(xpts_for_viz[cluster_membership_for_viz == j], 
                        ypts_for_viz[cluster_membership_for_viz == j], '.', color=colors[j])
            
            for pt in cntr:
                ax.plot(pt[0], pt[1], 'rs')
            
            ax.set_title('Clusters = {0}; FPC = {1:.2f}'.format(n_cluster, fpc), fontdict={'fontsize': 8})
            ax.axis('off')
        else:
            ax.remove()
    
    fig1.tight_layout()
    fig1.savefig(f'{output}/viz_multiple_cluster_nb.png')
    plt.close()
    
    return cntr, u, wss, fpcs, ss, chi, dbi, gap
