#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import coloredlogs
import logging
import os
import sys
import dill as pickle

from cyclopts import App, Parameter
import pandas as pd
import numpy as np
from typing_extensions import Annotated

from neurostatx.io.utils import assert_input, assert_output_dir_exist
from neurostatx.io.loader import DatasetLoader
from neurostatx.io.viz import flexible_barplot
from neurostatx.clustering.fuzzy import search_fuzzy_cmeans
from neurostatx.utils.preprocessing import compute_pca
from neurostatx.clustering.viz import (
    plot_clustering_results,
    plot_dendrogram,
    plot_parallel_plot,
    radar_plot
)
from neurostatx.clustering.metrics import (compute_knee_location,
                                           find_optimal_gap)
from neurostatx.clustering.distance import DistanceMetrics


# Initializing the app.
app = App(default_parameter=Parameter(negative=()))
"""Cyclopts application for the FuzzyClustering command-line tool."""


@app.default()
def FuzzyClustering(
    in_dataset: Annotated[
        str,
        Parameter(
            show_default=False,
            group="Essential Files Options",
        ),
    ],
    id_column: Annotated[
        str,
        Parameter(
            show_default=False,
            group="Essential Files Options",
        ),
    ],
    desc_columns: Annotated[
        int,
        Parameter(
            show_default=False,
            group="Essential Files Options",
        ),
    ],
    k: Annotated[
        int,
        Parameter(
            show_default=True,
            group="Clustering Options",
        ),
    ] = 10,
    m: Annotated[
        float,
        Parameter(
            show_default=True,
            group="Clustering Options",
        ),
    ] = 2,
    error: Annotated[
        float,
        Parameter(
            show_default=True,
            group="Clustering Options",
        ),
    ] = 1e-6,
    maxiter: Annotated[
        int,
        Parameter(
            show_default=True,
            group="Clustering Options",
        ),
    ] = 1000,
    init: Annotated[
        str,
        Parameter(
            show_default=True,
            group="Clustering Options",
        ),
    ] = None,
    metric: Annotated[
        DistanceMetrics,
        Parameter(
            show_default=True,
            group="Clustering Options",
        ),
    ] = DistanceMetrics.euclidean,
    pca: Annotated[
        bool,
        Parameter(
            "--pca",
            show_default=True,
            group="Clustering Options",
        ),
    ] = False,
    out_folder: Annotated[
        str,
        Parameter(
            group="Essential Files Options",
        ),
    ] = "./fuzzy_results/",
    processes: Annotated[
        int,
        Parameter(
            show_default=True,
            group="Computational Options",
        ),
    ] = 1,
    parallelplot: Annotated[
        bool,
        Parameter(
            show_default=True,
            group="Visualization Options",
        ),
    ] = False,
    radarplot: Annotated[
        bool,
        Parameter(
            show_default=True,
            group="Visualization Options",
        ),
    ] = True,
    cmap: Annotated[
        str,
        Parameter(
            show_default=True,
            group="Visualization Options",
        ),
    ] = "magma",
    verbose: Annotated[
        bool,
        Parameter(
            "-v",
            "--verbose",
            group="Optional parameters",
        ),
    ] = False,
    save_parameters: Annotated[
        bool,
        Parameter(
            "-s",
            "--save_parameters",
            group="Optional parameters",
        ),
    ] = False,
    overwrite: Annotated[
        bool,
        Parameter(
            "-f",
            "--overwrite",
            group="Optional parameters",
        ),
    ] = False,
):
    """Wrap Fuzzy C-Means clustering over k=2 to k.

    FuzzyClustering runs the analysis for each cluster count up to --k and
    writes evaluation metrics plus summary barplots and parallel plots.

    Notes
    -----

    **Evaluation metrics**

    The fuzzy partition coefficient (FPC) is defined between 0 and 1, with 1
    the better score. It represents how well the data are described by the
    clustering model, so a higher FPC is a better fit. On real-world data,
    local maxima can also be interpreted as an optimal solution [1], [5].

    The Silhouette Coefficient evaluates cluster definition. The score is
    bounded from -1 to 1, with 1 the perfect score. A higher Silhouette
    Coefficient relates to better-defined clusters. It tends to be higher for
    clusters generated from density-based methods [2].

    The Calinski-Harabasz Index (Variance Ratio Criterion) can be used when no
    known labels are available. It represents the density and separation of
    clusters, and also tends to be higher for density-based methods. A higher
    index relates to better-defined clusters.

    The Davies-Bouldin Index is reported for all cluster models. A lower DBI
    relates to better cluster separation. It measures similarity between
    clusters from quantities and features of the dataset, tends to be higher
    for convex clusters, and uses centroid distance, so the distance metric
    is limited to Euclidean space.

    Within-cluster sum of squared error (WSS) is the average distance from
    each point to its cluster centroid. Combined with the elbow method, WSS
    helps determine the optimal k [3], [4].

    GAP statistics are based on WSS. They compute the difference in cluster
    compactness between the actual data and simulated data with a null
    distribution. The optimal k is identified by a maximized GAP statistic
    (local maxima can also suggest possible solutions).

    **Fuzziness and initialization**

    The --m parameter defines the degree of fuzziness of the resulting
    membership matrix. Using --m 1 returns crisp clusters, whereas --m > 1
    returns increasingly fuzzy clusters. The c-partitioned matrix can be
    pre-initialized from previous membership matrices. Specify a folder
    containing a membership matrix for each k (if clustering up to k=10, a
    matrix is needed for each of them) using this name convention:

    ```text
                [init_folder]
                    |-- cluster_membership_1.npy
                    |-- cluster_membership_2.npy
                    |-- [...]
                    └-- cluster_membership_{k}.npy
    ```

    **Output folder structure**

    The script creates a default output structure in the destination specified
    by --out-folder:

    ```text
                [out_folder]
                    |-- CENTROIDS
                    |       |-- clusters_centroids_2.xlsx
                    |       |-- [...]
                    |       └-- clusters_centroids_{k}.xlsx
                    |-- MEMBERSHIP_DF
                    |       |-- clusters_membership_2.xlsx
                    |       |-- [...]
                    |       └-- clusters_membership_{k}.xlsx
                    |-- MEMBERSHIP_MAT (in .npy format)
                    |-- METRICS
                    |       |-- chi.png
                    |       |-- [...]
                    |       └-- wss.png
                    |-- PARALLEL_PLOTS (optional)
                    |       |-- parallel_plot_2clusters.png
                    |       |-- [...]
                    |       |-- parallel_plot_{k}clusters.png
                    |-- PCA (optional)
                    |       |-- transformed_data.xlsx
                    |       |-- variance_explained.xlsx
                    |       └-- pca_model.joblib
                    |-- RADAR_PLOTS (optional)
                    |       |-- radar_plot_2clusters.png
                    |       |-- [...]
                    |       |-- radar_plot_{k}clusters.png
                    |-- validation_indices.xlsx
                    └-- viz_multiple_cluster_nb.png
    ```

    References
    ----------
    [1] [scikit-fuzzy c-means example](https://pythonhosted.org/scikit-fuzzy/auto_examples/plot_cmeans.html)

    [2] [scikit-learn clustering performance evaluation](https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation)

    [3] [Selecting the optimal number of clusters](https://towardsdatascience.com/cheat-sheet-to-implementing-7-methods-for-selecting-optimal-number-of-clusters-in-python-898241e1d6ad)

    [4] [How to determine the right number of clusters](https://towardsdatascience.com/how-to-determine-the-right-number-of-clusters-with-code-d58de36368b1)

    [5] [scikit-fuzzy GitHub repository](https://github.com/scikit-fuzzy/scikit-fuzzy)

    Parameters
    ----------
    in_dataset : str
        Input dataset.
    id_column : str
        Name of the column containing the subject's ID tag. Required for
        proper handling of IDs and merging multiple datasets.
    desc_columns : int
        Number of descriptive columns at the beginning of the dataset to
        exclude in statistics and descriptive tables.
    k : int, optional
        Maximum k number of clusters to fit a model for. The script iterates
        until k is met. Defaults to 10.
    m : float, optional
        Exponentiation value to apply on the membership function. Determines
        the degree of fuzziness of the membership matrix. Defaults to 2.
    error : float, optional
        Error threshold for convergence stopping criterion. Defaults to 1e-6.
    maxiter : int, optional
        Maximum number of iterations to perform. Defaults to 1000.
    init : str, optional
        Initial fuzzy c-partitioned matrix. Defaults to None.
    metric : DistanceMetrics, optional
        Metric to use to compute distance between original points and cluster
        centroids. Defaults to euclidean.
    pca : bool, optional
        If set, will perform PCA decomposition to 2 components before
        clustering. Defaults to False.
    out_folder : str, optional
        Path of the folder in which the results will be written. If not
        specified, current folder and default name will be used. Defaults to
        ``./fuzzy_results/``.
    processes : int, optional
        Number of processes to launch in parallel. Defaults to 1.
    parallelplot : bool, optional
        If true, will output a parallel plot for each cluster solution.
        Defaults to False.
    radarplot : bool, optional
        If true, will output a radar plot for each cluster solution. Defaults
        to True.
    cmap : str, optional
        Colormap to use for plotting. Defaults to ``magma``. See
        https://matplotlib.org/stable/tutorials/colors/colormaps.html.
    verbose : bool, optional
        If true, produce verbose output. Defaults to False.
    save_parameters : bool, optional
        If true, will save input parameters to .txt file. Defaults to False.
    overwrite : bool, optional
        If true, force overwriting of existing output files. Defaults to False.

    Examples
    --------
    ```bash
    FuzzyClustering --in-dataset dataset.csv --id-column ID --desc-columns
    1 --k 10 --m 2 --error 1e-6 --maxiter 1000 --init init_folder --metric
    euclidean --pca --out-folder ./fuzzy_results/ --processes 4 --verbose
    ```
    """

    if verbose:
        logging.getLogger().setLevel(logging.INFO)
        coloredlogs.install(level=logging.INFO)

    assert_input(in_dataset)
    assert_output_dir_exist(overwrite, out_folder, create_dir=True)

    if save_parameters:
        parameters = list(locals().items())
        with open(f"{out_folder}/parameters.txt", "w+") as f:
            for param in parameters:
                f.writelines(str(param))

    # Creating substructures for output folder.
    os.mkdir(f"{out_folder}/METRICS/")

    # Loading dataframe.
    logging.info("Loading dataset(s)...")
    df = DatasetLoader().load_data(in_dataset)
    descriptive_columns = [n for n in range(0, desc_columns)]

    # Creating the array.
    desc_data = df.get_descriptive_columns(descriptive_columns)
    df.drop_columns(descriptive_columns).set_type("float")

    # Decomposing into 2 components if asked.
    if pca:
        logging.info("Applying PCA dimensionality reduction.")
        X, model, variance, components, chi, kmo = df.custom_function(
            compute_pca,
            n_components=3)
        logging.info(
            "Bartlett's test of sphericity returned a p-value of {} and "
            "Keiser-Meyer-Olkin (KMO)"
            " test returned a value of {}.".format(chi, kmo)
        )

        # Exporting variance explained data.
        os.mkdir(f"{out_folder}/PCA/")
        DatasetLoader().import_data(
            variance, columns=["Variance Explained"]).save_data(
            f"{out_folder}/PCA/variance_explained.csv",
            index=True,
            header=True
        )

        # Exporting PCA components and transformed data.
        components = DatasetLoader().import_data(
            components, columns=df.get_data().columns)
        components.save_data(
            f"{out_folder}/PCA/components.csv",
            index=True,
            header=True)
        df = DatasetLoader().import_data(
            X,
            columns=["Component #1", "Component #2", "Component #3"]
        ).join(
            desc_data, left=True
        )
        df.save_data(
            f"{out_folder}/PCA/transformed_data.csv",
            index=True,
            header=True
        )
        df.drop_columns(descriptive_columns).set_type("float")

        # Transpose the components matrix to get the loadings values.
        components.data = components.get_data().T
        components.custom_function(
            flexible_barplot,
            nb_axes=3,
            output=f"{out_folder}/PCA/barplot_loadings.png",
            cmap=cmap,
            title="Loadings values for the three components.",
            ylabel="Loading values"
        )

        # Exporting model in .joblib format.
        with open(f"{out_folder}/PCA/pca_model.pkl", "wb") as f:
            pickle.dump(model, f)

    # Plotting the dendrogram.
    logging.info("Generating dendrogram.")
    sys.setrecursionlimit(50000)
    df.custom_function(
        plot_dendrogram,
        output=f"{out_folder}/METRICS/dendrogram.png",
    )

    # Load initialisation matrix if any.
    if init is not None:
        init_mat = [
            np.load(f"{init}/clusters_membership_{i}.npy")
            for i in range(2, k + 1)
        ]
    else:
        init_mat = None

    # Computing a range of C-means clustering method.
    logging.info("Computing FCM from k=2 to k={}".format(k))
    cntr, u, wss, fpcs, ss, chi, dbi, gap, sk = search_fuzzy_cmeans(
        df.get_data().values,  # This will change in the future.
        min_clusters=2,
        max_clusters=k,
        m=m,
        tol=error,
        max_iter=maxiter,
        init=init_mat,
        metric=metric,
        output=out_folder,
        n_jobs=processes,
        verbose=verbose,
        random_state=1234,
    )

    # Compute knee location on Silhouette Score.
    logging.info("Plotting validation indicators and outputting final "
                 "matrices.")
    elbow_wss = compute_knee_location(wss)

    # Creating a dataframe to export statistics.
    DatasetLoader().import_data(
        {"FPC": fpcs, "WSS": wss, "Silhouette Score": ss,
         "CHI": chi, "DBI": dbi, "GAP": gap},
        index=[f"{i}-Cluster Model" for i in range(2, len(ss) + 2)]
    ).save_data(
        f"{out_folder}/validation_indices.csv",
        header=True,
        index=True
    )

    # Plotting results for each indicators.
    plot_clustering_results(
        wss,
        title="Within Cluster Sum of Square Error (WSS)",
        metric="WSS",
        output=f"{out_folder}/METRICS/wss.png",
        annotation=f"Elbow threshold (Optimal cluster nb): {elbow_wss}",
    )
    fpcs_index = fpcs.index(max(fpcs))
    plot_clustering_results(
        fpcs,
        title="Fuzzy Partition Coefficient (FPC)",
        metric="FPC",
        output=f"{out_folder}/METRICS/fpc.png",
        annotation=f"Optimal Number of Cluster: {fpcs_index+2}",
    )
    ss_index = ss.index(max(ss))
    plot_clustering_results(
        ss,
        title="Silhouette Score Coefficient (SS)",
        metric="SS",
        output=f"{out_folder}/METRICS/ss.png",
        annotation=f"Optimal Number of Clusters: {ss_index+2}",
    )
    chi_index = chi.index(max(chi))
    plot_clustering_results(
        chi,
        title="Calinski-Harabasz Index (CHI)",
        metric="CHI",
        output=f"{out_folder}/METRICS/chi.png",
        annotation=f"Optimal Number of Clusters: {chi_index+2}",
    )
    dbi_index = dbi.index(min(dbi))
    plot_clustering_results(
        dbi,
        title="Davies-Bouldin Index (DBI)",
        metric="DBI",
        output=f"{out_folder}/METRICS/dbi.png",
        annotation=f"Optimal Number of Clusters: {dbi_index+2}",
    )
    gap_index = find_optimal_gap(gap, sk)
    plot_clustering_results(
        gap,
        title="GAP Statistics.",
        metric="GAP",
        output=f"{out_folder}/METRICS/gap.png",
        errorbar=sk,
        annotation=f"Optimal Number of Clusters: {gap_index+2}",
    )

    # Exporting plots and graphs for each cluster solution.
    os.mkdir(f"{out_folder}/MEMBERSHIP_MAT/")
    os.mkdir(f"{out_folder}/MEMBERSHIP_DF/")
    os.mkdir(f"{out_folder}/PARALLEL_PLOTS/")
    os.mkdir(f"{out_folder}/RADAR_PLOTS/")
    os.mkdir(f"{out_folder}/CENTROIDS/")

    # Iterating and saving every elements.
    for i in range(len(u)):
        membership = np.argmax(u[i], axis=0)
        viz_df = DatasetLoader().load_data(
            in_dataset
        ).drop_columns(descriptive_columns).set_type("float")
        if parallelplot:
            viz_df.custom_function(
                plot_parallel_plot,
                labels=membership,
                mean_values=True,
                output=f"{out_folder}/PARALLEL_PLOTS/parallel_plot_{i+2}"
                       "clusters.png",
                cmap=cmap,
                title=f"Parallel Coordinates plot for {i+2} clusters solution."
            )
        if radarplot:
            viz_df.custom_function(
                radar_plot,
                labels=membership,
                title=f"Radar plot for {i+2} clusters solution.",
                frame='circle',
                cmap=cmap,
                output=f"{out_folder}/RADAR_PLOTS/radar_plot_{i+2}clusters.png"
            )

        if i == 0:
            shape = df.get_data().shape[1]

        DatasetLoader().import_data(
            cntr[i],
            columns=[f"v{i}" for i in range(shape)],
            index=[f"Cluster #{n+1}" for n in range(u[i].shape[0])],
        ).save_data(
            f"{out_folder}/CENTROIDS/clusters_centroids_{i+2}.xlsx",
            header=True,
            index=True,
        )

        # Converting membership arrays to df.
        member = DatasetLoader().import_data(
            u[i].T,
            columns=[f"Cluster #{n+1}" for n in range(u[i].shape[0])],
            index=None,
        ).get_data()

        # Appending subject ids and descriptive columns.
        DatasetLoader().import_data(
            pd.concat([
                desc_data,
                df.get_data(),
                member
            ], axis=1)
        ).save_data(
            f"{out_folder}/MEMBERSHIP_DF/clusters_membership_{i+2}.xlsx",
            header=True,
            index=False,
        )

        # Saving original matrix.
        np.save(f"{out_folder}/MEMBERSHIP_MAT/clusters_membership_{i+2}.npy",
                u[i])


if __name__ == "__main__":
    app()
