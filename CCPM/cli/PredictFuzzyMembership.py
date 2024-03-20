#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import coloredlogs
import logging
import os
import sys
from joblib import load

from cyclopts import App, Parameter
import numpy as np
import pandas as pd
from skfuzzy import cmeans_predict
from typing import List
from typing_extensions import Annotated

from CCPM.io.utils import (load_df_in_any_format, assert_input,
                           assert_output_dir_exist)
from CCPM.clustering.viz import (
    plot_parallel_plot,
    plot_grouped_barplot,
    radar_plot)
from CCPM.io.viz import flexible_barplot
from CCPM.utils.preprocessing import merge_dataframes, compute_pca
from CCPM.clustering.distance import DistanceMetrics


# Initializing the app.
app = App(default_parameter=Parameter(negative=()))


@app.default()
def PredictFuzzyMembership(
    in_dataset: Annotated[
        List[str],
        Parameter(
            show_default=False,
            group="Essential Files Options",
        ),
    ],
    in_cntr: Annotated[
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
    out_folder: Annotated[
        str,
        Parameter(
            show_default=False,
            group="Essential Files Options",
        ),
    ] = "./output/",
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
    pca_model: Annotated[
        str,
        Parameter(
            show_default=True,
            group="Clustering Options",
        ),
    ] = None,
    parallelplot: Annotated[
        bool,
        Parameter(
            show_default=True,
            group="Visualization Options",
        ),
    ] = False,
    barplot: Annotated[
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
    """FUZZY MEMBERSHIP PREDICTION
    ---------------------------
    This script will predict the membership matrix of a dataset using a
    trained Cmeans model (only the centroids are necessary for the prediction).

    PARAMETERS
    ----------
    Details regarding the parameters can be seen below. Regarding the
    --m parameter, it defines the degree of fuzziness of the resulting
    membership matrix. Using --m 1 will returns crisp clusters, whereas
    --m >1 will returned more and more fuzzy clusters. It is recommended
    to use the same m value as used during training.

    EXAMPLE USAGE
    -------------
    ::

        PredictFuzzyMembership --in_dataset dataset.xlsx --in_cntr
        centroids.xlsx --id_column ID --desc_columns 1 --out_folder
        predicted_membership_matrix/ --m 2 --error 1e-6 --maxiter 1000
        --metric euclidean --verbose --save_parameters --overwrite

    Parameters
    ----------
    in_dataset : List[str]
        Input dataset(s) to filter. If multiple files are provided as input,
        will be merged according to the subject id columns.
    in_cntr : str
        Centroid file to use for prediction. Should come from a trained Cmeans
        model (such as ``CCPM_fuzzy_clustering.py``).
    id_column : str
        Name of the column containing the subject's ID tag. Required for proper
        handling of IDs and merging multiple datasets.
    desc_columns : int
        Number of descriptive columns at the beginning of the dataset to
        exclude in statistics and descriptive tables.
    out_folder : str, optional
        Output folder for the predicted membership matrix.
    m : float, optional
        Exponentiation value to apply on the membership function, will
        determined the degree of fuzziness of the membership matrix.
    error : float, optional
        Error threshold for convergence stopping criterion.
    maxiter : int, optional
        Maximum number of iterations to perform.
    metric : DistanceMetrics, optional
        Metric to use to compute distance between original points and clusters
        centroids.
    pca : bool, optional
        If set, will perform PCA decomposition to 3 components before
        clustering.
    parallelplot : bool, optional
        If true, will output parallel plot for each cluster solution. Default
        is False.
    barplot : bool, optional
        If true, will output barplot for each cluster solution. Default is
        False.
    radarplot : bool, optional
        If true, will output radar plot for each cluster solution. Default is
        True.
    cmap : str, optional
        Colormap to use for plotting. Default is "magma". See
        [Matplotlib](https://matplotlib.org/stable/tutorials/colors/colormaps.html).
    verbose : bool, optional
        If true, produce verbose output.
    save_parameters : bool, optional
        If true, will save input parameters to .txt file.
    overwrite : bool, optional
        If true, force overwriting of existing output files.
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

    cntr = load_df_in_any_format(in_cntr)
    cntr.drop(cntr.columns[0], axis=1, inplace=True)

    # Creating the array.
    desc_data = raw_df[raw_df.columns[descriptive_columns]]
    df_for_clust = raw_df.drop(
        raw_df.columns[descriptive_columns], axis=1, inplace=False
    ).astype("float")
    X = df_for_clust.values

    # Decomposing into 2 components if asked.
    if pca:
        if pca_model is not None:
            logging.info("Loading PCA model...")
            pca_model = load(pca_model)
            X = pca_model.transform(X)
        else:
            logging.info("Applying PCA dimensionality reduction.")
            X, variance, components, chi, kmo = compute_pca(X, 3)
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
            components_df = pd.DataFrame(components,
                                         columns=df_for_clust.columns)
            components_df.to_excel(f"{out_folder}/PCA/components.xlsx",
                                   index=True,
                                   header=True)
            out = pd.DataFrame(X, columns=["Component #1", "Component #2",
                                           "Component #3"])
            out.to_excel(f"{out_folder}/PCA/transformed_data.xlsx", index=True,
                         header=True)

            flexible_barplot(
                components,
                df_for_clust.columns,
                3,
                title="Loadings values for the two components.",
                filename=f"{out_folder}/PCA/barplot_loadings.png",
                ylabel="Loading value")

    logging.info("Predicting membership matrix...")
    u, u0, d, jm, p, fpc = cmeans_predict(
        X.T,
        cntr.values,
        m=m,
        error=error,
        maxiter=maxiter,
        metric=metric,
        init=None,
        seed=42,
    )

    # Saving results.
    logging.info("Saving results...")
    member = pd.DataFrame(
        u.T,
        index=None,
        columns=[f"Cluster #{n+1}" for n in range(u.shape[0])],
    )
    out = pd.concat([desc_data, member], axis=1)
    out.to_excel(
        f"{out_folder}/predicted_membership_matrix.xlsx",
        header=True,
        index=False,
    )

    os.mkdir(f"{out_folder}/PARALLEL_PLOTS/")
    os.mkdir(f"{out_folder}/BARPLOTS/")
    os.mkdir(f"{out_folder}/RADAR_PLOTS/")
    membership = np.argmax(u, axis=0)
    if parallelplot:
        plot_parallel_plot(
            df_for_clust,
            membership,
            mean_values=True,
            output=f"{out_folder}/PARALLEL_PLOTS/parallel_plot_{u.shape[0]}"
                   "clusters.png",
            title=f"Parallel Coordinates plot for {u.shape[0]} clusters "
            "solution.",
            cmap=cmap
        )
    if barplot:
        plot_grouped_barplot(
            df_for_clust,
            membership,
            title=f"Barplot of {u.shape[0]} clusters solution.",
            cmap=cmap,
            output=f"{out_folder}/BARPLOTS/barplot_{u.shape[0]}clusters.png",
        )
    if radarplot:
        radar_plot(
            df_for_clust,
            membership,
            title=f"Radar plot of {u.shape[0]} clusters solution.",
            frame='circle',
            cmap=cmap,
            output=(
                f"{out_folder}/RADAR_PLOTS/radar_plot_{u.shape[0]}clusters.png"
            ),
        )


if __name__ == "__main__":
    app()
