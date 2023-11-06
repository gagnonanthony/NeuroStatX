#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import coloredlogs
import logging
import os
import sys

import numpy as np
import pandas as pd
import typer
from typing import List
from typing_extensions import Annotated

from CCPM.io.utils import (load_df_in_any_format, assert_input,
                           assert_output_dir_exist)
from CCPM.clustering.fuzzy import fuzzyCmeans
from CCPM.utils.preprocessing import merge_dataframes, compute_pca
from CCPM.clustering.viz import (
    plot_clustering_results,
    plot_dendrogram,
    plot_parallel_plot,
    plot_grouped_barplot,
)
from CCPM.clustering.metrics import compute_knee_location, find_optimal_gap
from CCPM.clustering.distance import DistanceMetrics


# Initializing the app.
app = typer.Typer(add_completion=False)


@app.command()
def main(
    in_dataset: Annotated[
        List[str],
        typer.Option(
            help="Input dataset(s) to filter. If multiple files are"
            "provided as input, will be merged according to "
            "the subject id columns.",
            show_default=False,
            rich_help_panel="Essential Files Options",
        ),
    ],
    id_column: Annotated[
        str,
        typer.Option(
            help="Name of the column containing the subject's ID tag. "
            "Required for proper handling of IDs and "
            "merging multiple datasets.",
            show_default=False,
            rich_help_panel="Essential Files Options",
        ),
    ],
    desc_columns: Annotated[
        int,
        typer.Option(
            help="Number of descriptive columns at the beginning of the "
            "dataset to exclude in statistics and descriptive tables.",
            show_default=False,
            rich_help_panel="Essential Files Options",
        ),
    ],
    k: Annotated[
        int,
        typer.Option(
            help="Maximum k number of cluster to fit a model for. (Script will"
            " iterate until k is met.)",
            show_default=True,
            rich_help_panel="Clustering Options",
        ),
    ] = 10,
    m: Annotated[
        float,
        typer.Option(
            help="Exponentiation value to apply on the membership function, "
            "will determined the degree of fuzziness of the membership matrix",
            show_default=True,
            rich_help_panel="Clustering Options",
        ),
    ] = 2,
    error: Annotated[
        float,
        typer.Option(
            help="Error threshold for convergence stopping criterion.",
            show_default=True,
            rich_help_panel="Clustering Options",
        ),
    ] = 1e-6,
    maxiter: Annotated[
        int,
        typer.Option(
            help="Maximum number of iterations to perform.",
            show_default=True,
            rich_help_panel="Clustering Options",
        ),
    ] = 1000,
    init: Annotated[
        str,
        typer.Option(
            help="Initial fuzzy c-partitioned matrix",
            show_default=True,
            rich_help_panel="Clustering Options",
            case_sensitive=False,
        ),
    ] = None,
    cluster_solution: Annotated[
        int,
        typer.Option(
            help="k value to export and plot solution",
            show_default=False,
            rich_help_panel="Clustering Options",
        ),
    ] = None,
    metric: Annotated[
        DistanceMetrics,
        typer.Option(
            help="Metric to use to compute distance between original points"
            " and clusters centroids.",
            show_default=True,
            rich_help_panel="Clustering Options",
        ),
    ] = DistanceMetrics.euclidean,
    pca: Annotated[
        bool,
        typer.Option(
            "--pca",
            help="If set, will perform PCA decomposition to 2 components "
            "before clustering",
            show_default=True,
            rich_help_panel="Clustering Options",
        ),
    ] = False,
    out_folder: Annotated[
        str,
        typer.Option(
            help="Path of the folder in which the results will be written. "
            "If not specified, current folder and default "
            "name will be used (e.g. = ./output/).",
            rich_help_panel="Essential Files Options",
        ),
    ] = "./fuzzy_results/",
    verbose: Annotated[
        bool,
        typer.Option(
            "-v",
            "--verbose",
            help="If true, produce verbose output.",
            rich_help_panel="Optional parameters",
        ),
    ] = False,
    save_parameters: Annotated[
        bool,
        typer.Option(
            "-s",
            "--save_parameters",
            help="If true, will save input parameters to .txt file.",
            rich_help_panel="Optional parameters",
        ),
    ] = False,
    overwrite: Annotated[
        bool,
        typer.Option(
            "-f",
            "--overwrite",
            help="If true, force overwriting of existing " "output files.",
            rich_help_panel="Optional parameters",
        ),
    ] = False,
):
    """
    \b
    =============================================================================
                ________    ________   ________     ____     ____
               /    ____|  /    ____| |   ___  \   |    \___/    |
              /   /       /   /       |  |__|   |  |             |
             |   |       |   |        |   _____/   |   |\___/|   |
              \   \_____  \   \_____  |  |         |   |     |   |
               \________|  \________| |__|         |___|     |___|
                  Children Cognitive Profile Mapping Toolbox©
    =============================================================================
    \b
    FUZZY CLUSTERING
    ----------------
    CCPM_fuzzy_clustering.py is a wrapper script for a Fuzzy C-Means
    clustering analysis. By design, the script will compute the analysis for
    k specified cluster (chosen by --k) and returns various
    evaluation metrics and summary barplot/parallel plot.
    \b
    EVALUATION METRICS
    ------------------
    The fuzzy partition coefficient (FPC) is a metric defined between 0 and 1
    with 1 representing the better score. It represents how well the data is
    described by the clustering model. Therefore, a higher FPC represents a
    better fitted model. On real-world data, local maxima can also be
    interpreted as one of the optimal solution.
    \b
    The Silhouette Coefficient represents an evaluation of cluster's
    definition. The score is bounded (-1 to 1) with 1 as the perfect score and
    -1 as not a good clustering result. A higher Silhouette Coefficient relates
    to a model with better defined clusters (therefore a better model). It
    tends to have higher score with cluster generated from density-
    based methods.
    \b
    The Calinski-Harabasz Index (or the Variance Ratio Criterion) can be used
    when no known labels are available. It represents the density and
    separation of clusters. Although it tends to be higher for cluster
    generated from density-based methods. A higher Calinski-Harabasz Index
    relates to better defined clusters.
    \b
    Davies-Bouldin Index is reported for all cluster-models. A lower DBI
    relates to a model with better cluster separation. It represents a measure
    of similarity between clusters and is solely based on quantities and
    features of the dataset. It also tends to be generally higher for
    convex clusters and it uses the centroid distance between clusters
    therefore limiting the distance metric to euclidean space.
    \b
    Within cluster Sum of Squared error (WSS) represents the average distance
    from each point to their cluster centroid. WSS is combined with the elbow
    method to determine the optimal k number of clusters.
    \b
    The GAP statistics is based on the WSS. It relies on computing the
    difference in cluster compactness between the actual data and simulated
    data with a null distribution. The optimal k-number of clusters is
    identified by a maximized GAP statistic (local maxima can also suggest
    possible solutions.).
    \b
    PARAMETERS
    ----------
    Details regarding the parameters can be seen below. Regarding the
    --m parameter, it defines the degree of fuzziness of the resulting
    membership matrix. Using --m 1 will returns crisp clusters, whereas
    --m >1 will returned more and more fuzzy clusters. It is also possible
    to pre-initialize the c-partitioned matrix from previous membership matrix.
    If you want to do that, you need to specify a folder containing all
    membership matrices for each k number (meaning that if you want to perform
    clustering up to k=10, you need a membership matrices for each of them.).
    It also must respect this name convention:
                    [init_folder]
                        |-- cluster_membership_1.npy
                        |-- cluster_membership_2.npy
                        |-- [...]
                        └-- cluster_membership_{k}.npy
    \b
    OUTPUT FOLDER STRUCTURE
    -----------------------
    The script creates a default output structure in a destination specified
    by using --out-folder. Output structure is as follows:
                    [out_folder]
                        |-- BARPLOTS
                        |       |-- barplot_2clusters.png
                        |       |-- [...]
                        |       └-- barplot_{k}clusters.png
                        |-- CENTROIDS
                        |       |-- clusters_centroids_2.xlsx
                        |       |-- [...]
                        |       └-- clusters_centroids_{k}.xlsx
                        |-- MEMBERSHIP_DF
                        |       |-- clusters_membership_2.xlsx
                        |       |-- [...]
                        |       └-- clusters_membership_{k}.xlsx
                        |-- MEMBERSHIP_MAT (identical to MEMBERSHIP_DF but in
                                            .npy format)
                        |-- METRICS
                        |       |-- chi.png
                        |       |-- [...]
                        |       └-- wss.png
                        |-- PARALLEL_PLOTS
                        |       |-- parallel_plot_2clusters.png
                        |       |-- [...]
                        |       |-- parallel_plot_{k}clusters.png
                        |-- PCA (optional)
                        |       |-- transformed_data.xlsx
                        |       └-- variance_explained.xlsx
                        |-- validation_indices.xlsx
                        └-- viz_multiple_cluster_nb.png
    \b
    REFERENCES
    ----------
    [1]
    https://pythonhosted.org/scikit-fuzzy/auto_examples/plot_cmeans.html
    [2]
    https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation
    [3]
    https://towardsdatascience.com/cheat-sheet-to-implementing-7-methods-for-selecting-optimal-number-of-clusters-in-python-898241e1d6ad
    [4]
    https://towardsdatascience.com/how-to-determine-the-right-number-of-clusters-with-code-d58de36368b1
    [5]     https://github.com/scikit-fuzzy/scikit-fuzzy

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
    if len(in_dataset) > 1:
        if id_column is None:
            sys.exit(
                "Column name for index matching is required when inputting "
                "multiple dataframe."
            )
        dict_df = {i: load_df_in_any_format(i) for i in in_dataset}
        raw_df = merge_dataframes(dict_df, id_column)
    else:
        raw_df = load_df_in_any_format(in_dataset[0])
    descriptive_columns = [n for n in range(0, desc_columns)]

    # Creating the array.
    desc_data = raw_df[raw_df.columns[descriptive_columns]]
    df_for_clust = raw_df.drop(
        raw_df.columns[descriptive_columns], axis=1, inplace=False
    ).astype("float")
    X = df_for_clust.values

    # Decomposing into 2 components if asked.
    if pca:
        logging.info("Applying PCA dimensionality reduction.")
        X, variance, chi, kmo = compute_pca(X, 2)
        logging.info(
            "Bartlett's test of sphericity returned a p-value of {} and "
            "Keiser-Meyer-Olkin (KMO)"
            " test returned a value of {}.".format(chi, kmo)
        )
        # Exporting variance explained data.
        os.mkdir(f"{out_folder}/PCA/")
        var_exp = pd.DataFrame(variance, columns=["Variance Explained"])
        var_exp.to_excel(
            f"{out_folder}/PCA/variance_explained.xlsx", index=True,
            header=True
        )
        out = pd.DataFrame(X, columns=["Component #1", "Component #2"])
        out.to_excel(f"{out_folder}/PCA/transformed_data.xlsx", index=True,
                     header=True)

    # Plotting the dendrogram.
    logging.info("Generating dendrogram.")
    sys.setrecursionlimit(50000)
    plot_dendrogram(X, f"{out_folder}/METRICS/dendrogram.png")

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
    cntr, u, d, wss, fpcs, ss, chi, dbi, gap, sk = fuzzyCmeans(
        X,
        max_cluster=k,
        m=m,
        error=error,
        maxiter=maxiter,
        init=init_mat,
        metric=metric,
        output=out_folder,
        verbose=verbose,
    )

    # Compute knee location on Silhouette Score.
    logging.info("Plotting validation indicators and outputting final "
                 "matrices.")
    elbow_wss = compute_knee_location(wss)

    # Creating a dataframe to export statistics.
    stats = pd.DataFrame(
        data={
            "FPC": fpcs,
            "Silhouette Score": ss,
            "CHI": chi,
            "DBI": dbi,
            "WSS": wss,
            "GAP": gap,
        },
        index=[f"{i}-Cluster Model" for i in range(2, len(ss) + 2)],
    )
    stats.to_excel(f"{out_folder}/validation_indices.xlsx", header=True,
                   index=True)

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
    os.mkdir(f"{out_folder}/CENTROIDS/")
    os.mkdir(f"{out_folder}/BARPLOTS")

    # Iterating and saving every elements.
    for i in range(len(u)):
        membership = np.argmax(u[i], axis=0)
        plot_parallel_plot(
            df_for_clust,
            membership,
            mean_values=True,
            output=f"{out_folder}/PARALLEL_PLOTS/parallel_plot_{i+2}"
                   "clusters.png",
            title=f"Parallel Coordinates plot for {i+2} clusters solution.",
        )
        plot_grouped_barplot(
            df_for_clust,
            membership,
            title=f"Barplot of {i+2} clusters solution.",
            output=f"{out_folder}/BARPLOTS/barplot_{i+2}clusters.png",
        )

        # Converting membership arrays to df.
        member = pd.DataFrame(
            u[i].T,
            index=None,
            columns=[f"Cluster #{n+1}" for n in range(u[i].shape[0])],
        )
        centroids = pd.DataFrame(
            cntr[i],
            index=[f"Cluster #{n+1}" for n in range(u[i].shape[0])],
            columns=["x", "y"],
        )

        # Appending subject ids and descriptive columns.
        member_out = pd.concat([desc_data, member], axis=1)
        member_out.to_excel(
            f"{out_folder}/MEMBERSHIP_DF/clusters_membership_{i+2}.xlsx",
            header=True,
            index=False,
        )
        centroids.to_excel(
            f"{out_folder}/CENTROIDS/clusters_centroids_{i+2}.xlsx",
            header=True,
            index=True,
        )

        # Saving original matrix.
        np.save(f"{out_folder}/MEMBERSHIP_MAT/clusters_membership_{i+2}.npy",
                u[i])

if __name__ == "__main__":
    app()
