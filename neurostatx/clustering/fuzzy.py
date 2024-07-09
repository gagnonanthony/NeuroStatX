# -*- coding: utf-8 -*-

import math

from matplotlib import pyplot as plt
from matplotlib.colors import rgb2hex
import numpy as np
import skfuzzy as fuzz
from sklearn.utils.parallel import Parallel, delayed

from neurostatx.clustering.metrics import (
    compute_evaluation_metrics,
    compute_sse,
    compute_gap_stats,
)


# Define process_cluster outside fuzzyCmeans
def process_cluster(X, n_cluster, max_clusters, m, error, maxiter, init,
                    metric):
    """
    Core worker function of fuzzyCmeans(). Compute the clustering, metrics,
    and visualization dataset.

    Args:
        X (Numpy array):                Numpy array with data to cluster
                                        (Subject x Features).
        max_cluster (int, optional):    Maximum number of clusters to fit a
                                        model for. Defaults to 10.
        m (float, optional):            Exponentiation value to apply on the
                                        membership function. Defaults to 2.
        error (float, optional):        Stopping criterion. Defaults to 1E-6.
        maxiter (int, optional):        Maximum iteration value.
                                        Defaults to 1000.
        init (2d array, optional):      Initial fuzzy c-partitioned matrix.
                                        Defaults to None.
        metric (str, optional):         Distance metric to use to compute
                                        intra/inter subjects/clusters distance.
                                        Defaults to euclidean.

    Returns:
        n_cluster:                      Number of cluster.
        p:                              Number of iterations run.
        fpc:                            Fuzzy partition coefficient.
        cntr:                           Cluster centroids array.
        u:                              Membership array.
        wss:                            Within-cluster Sum of Square Error.
        ss:                             Silhouette Coefficient Score.
        chi:                            Calinski-Harabasz Index.
        dbi:                            Davies-Bouldin Index.
        gap:                            GAP statistic.
        sk:                             GAP standard error.
        xpts_for_viz:                   X data points for visualization.
        ypts_for_viz:                   Y data points for visualization.
        cluster_membership_for_viz:     Hard membership value for visualization
                                        data points.
    """

    if n_cluster <= max_clusters:
        if init is not None:
            init_mat = init[n_cluster - 2]
        else:
            init_mat = None

        cntr, u, _, _, _, p, fpc = fuzz.cmeans(
            X.T,
            n_cluster,
            m=m,
            error=error,
            maxiter=maxiter,
            metric=metric,
            init=init_mat,
            seed=0)

        cluster_membership = np.argmax(u, axis=0)

        ss, chi, dbi = compute_evaluation_metrics(X, cluster_membership,
                                                  metric=metric)
        labels = np.argmax(u, axis=0)
        wss = compute_sse(X, cntr, labels)

        gap, sk = compute_gap_stats(
            X,
            wss,
            nrefs=100,
            n_cluster=n_cluster,
            m=m,
            error=error,
            metric=metric,
            maxiter=maxiter,
            init=None,
        )

        xpts = X[:, 0]
        ypts = X[:, 1]

        if X.shape[0] > 500:
            indices = np.random.choice(X.shape[0], size=500, replace=False)
            xpts_for_viz = xpts[indices]
            ypts_for_viz = ypts[indices]
            cluster_membership_for_viz = cluster_membership[indices]
        else:
            xpts_for_viz = xpts
            ypts_for_viz = ypts
            cluster_membership_for_viz = cluster_membership

        return (
            n_cluster,
            p,
            fpc,
            cntr,
            u,
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

    else:
        return None


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
    """
    Fuzzy C-Means clustering function. Iteratively test and report statistics
    on multiple number of clusters. Based on documentation found here :
    https://pythonhosted.org/scikit-fuzzy/auto_examples/plot_cmeans.html

    Args:
        X (Numpy array):                Numpy array with data to cluster
                                        (Subject x Features).
        max_cluster (int, optional):    Maximum number of clusters to fit a
                                        model for. Defaults to 10.
        m (float, optional):            Exponentiation value to apply on the
                                        membership function. Defaults to 2.
        error (float, optional):        Stopping criterion. Defaults to 1E-6.
        maxiter (int, optional):        Maximum iteration value. Defaults to
                                        1000.
        init (folder, optional):        Folder containing the c-partitioned
                                        matrices for each cluster number.
                                        Defaults to None.
        metric (str, optional):         Distance metric to use to compute
                                        intra/inter subjects/clusters distance.
                                        Defaults to euclidean.
        output (String, optional):      Output folder. Defaults to './'.
        processes (int, optional):      Number of processes to use for
                                        multiprocessing. Defaults to 1.
        verbose (bool, optional):       Set verbose output. Defaults to False.

    Returns:
        cntr:                           List of cluster centroids arrays for
                                        each model.
        u:                              List of membership arrays for each
                                        model.
        d:                              List of distance arrays for each model.
        wss:                            List of Within-cluster Sum of Square
                                        Error value for each model.
        fpc:                            List of fuzzy partition coefficient
                                        value for each model.
        ss:                             List of Silhouette Coefficient Score
                                        for each model.
        chi:                            List of Calinski-Harabasz Index for
                                        each model.
        dbi:                            List of Davies-Bouldin Index for each
                                        model.
        gap:                            List of GAP statistic for each model.
    """

    num_clusters = max_cluster
    grid = math.ceil(math.sqrt(num_clusters))

    # Spawing multiprocessing using sklearn parallel and delayed function.
    results = Parallel(n_jobs=processes, verbose=verbose)(
        delayed(process_cluster)(
            X,
            k,
            max_clusters=num_clusters,
            m=m,
            error=error,
            maxiter=maxiter,
            init=init,
            metric=metric
        )
        for k in range(2, num_clusters + 2)
    )

    # Process results.
    valid_results = [r for r in results if r is not None]

    # Prepare outputs.
    _, _, fpc, cntr, u, wss, ss, chi, dbi, gap, sk, _, _, _ = zip(
        *valid_results)

    # Save viz.
    fig, axes = plt.subplots(grid, grid, figsize=(8, 8))

    for result in valid_results:
        (
            n_cluster,
            p,
            fpc_,
            cntr_,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            xpts_for_viz,
            ypts_for_viz,
            cluster_membership_for_viz,
        ) = result

        ax = axes[(n_cluster - 2) // grid, (n_cluster - 2) % grid]

        cmap = plt.get_cmap("plasma", n_cluster)
        colors = [rgb2hex(cmap(i)) for i in range(cmap.N)]

        for j in range(n_cluster):
            ax.plot(
                xpts_for_viz[cluster_membership_for_viz == j],
                ypts_for_viz[cluster_membership_for_viz == j],
                ".",
                color=colors[j],
            )

        for pt in cntr_:
            ax.plot(pt[0], pt[1], "rs")

        ax.set_title(
            "Clusters = {0}; FPC = {1:.2f}\nIterations = {iteration}"
            .format(
                n_cluster, fpc_, iteration=p
            ),
            fontdict={"fontsize": 8},
        )
        ax.axis("off")

    # Removing unused axis.
    for ax in axes.flat[(num_clusters - 1):]:
        ax.remove()

    plt.tight_layout()
    plt.savefig(f"{output}/viz_multiple_cluster_nb.png")
    plt.close()

    return cntr, u, wss, fpc, ss, chi, dbi, gap, sk
