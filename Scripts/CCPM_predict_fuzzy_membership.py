#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import coloredlogs
import logging
import os
import sys

import numpy as np
import pandas as pd
from skfuzzy import cmeans_predict
import typer
from typing import List
from typing_extensions import Annotated

from CCPM.io.utils import (load_df_in_any_format, assert_input,
                           assert_output_dir_exist)
from CCPM.clustering.viz import (
    plot_parallel_plot,
    plot_grouped_barplot)
from CCPM.io.viz import flexible_barplot
from CCPM.utils.preprocessing import merge_dataframes, compute_pca
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
    in_cntr: Annotated[
        str,
        typer.Option(
            help="Centroid file to use for prediction. Should come from a "
            "trained Cmeans model (such as ``CCPM_fuzzy_clustering.py``).",
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
    out_folder: Annotated[
        str,
        typer.Option(
            help="Output folder for the predicted membership matrix.",
            show_default=False,
            rich_help_panel="Essential Files Options",
        ),
    ],
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
    FUZZY MEMBERSHIP PREDICTION
    ---------------------------
    This script will predict the membership matrix of a dataset using a
    trained Cmeans model (only the centroids are necessary for the prediction).
    \b
    PARAMETERS
    ----------
    Details regarding the parameters can be seen below. Regarding the
    --m parameter, it defines the degree of fuzziness of the resulting
    membership matrix. Using --m 1 will returns crisp clusters, whereas
    --m >1 will returned more and more fuzzy clusters. It is recommended
    to use the same m value as used during training.
    \b
    EXAMPLE USAGE
    -------------
    CCPM_predict_fuzzy_membership.py \
        --in_dataset dataset.xlsx \
        --in_cntr centroids.xlsx \
        --id_column ID \
        --desc_columns 1 \
        --out_folder predicted_membership_matrix/ \
        --m 2 \
        --error 1e-6 \
        --maxiter 1000 \
        --metric euclidean \
        --verbose \
        --save_parameters \
        --overwrite
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
        logging.info("Applying PCA dimensionality reduction.")
        X, variance, components, chi, kmo = compute_pca(X, 2)
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
        components_df = pd.DataFrame(components, columns=df_for_clust.columns)
        components_df.to_excel(f"{out_folder}/PCA/components.xlsx", index=True,
                               header=True)
        out = pd.DataFrame(X, columns=["Component #1", "Component #2"])
        out.to_excel(f"{out_folder}/PCA/transformed_data.xlsx", index=True,
                     header=True)

        flexible_barplot(
            components,
            df_for_clust.columns,
            2,
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
    membership = np.argmax(u, axis=0)
    plot_parallel_plot(
        df_for_clust,
        membership,
        mean_values=True,
        output=f"{out_folder}/PARALLEL_PLOTS/parallel_plot_{u.shape[0]}"
               "clusters.png",
        title=f"Parallel Coordinates plot for {u.shape[0]} clusters "
        "solution.",
    )
    plot_grouped_barplot(
        df_for_clust,
        membership,
        title=f"Barplot of {u.shape[0]} clusters solution.",
        output=f"{out_folder}/BARPLOTS/barplot_{u.shape[0]}clusters.png",
    )


if __name__ == "__main__":
    app()
