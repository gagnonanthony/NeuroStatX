#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import coloredlogs
import logging
import os
import warnings

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
import typer
from typing import List
from typing_extensions import Annotated

from CCPM.io.utils import (assert_input, assert_output_dir_exist)
from CCPM.network.utils import fetch_attributes_df, fetch_edge_data
from CCPM.statistics.plsr import (plsr_cv, permutation_testing,
                                  ScoringMethod)
from CCPM.io.viz import generate_coef_plot

# Initializing the app.
app = typer.Typer(add_completion=False)


@app.command()
def main(
    in_graph: Annotated[
        str,
        typer.Option(
            help="Graph file to add attributes to.",
            show_default=False,
            rich_help_panel="Essential Files Options",
        ),
    ],
    out_folder: Annotated[
        str,
        typer.Option(
            help="Output folder.",
            show_default=False,
            rich_help_panel="Essential Files Options",
        ),
    ],
    attributes: Annotated[
        List[str],
        typer.Option(
            help="Attributes names to include in the PLSR model. If None are "
            "provided, all attributes will be included.",
            rich_help_panel="Model Parameters",
            show_default=True,
        ),
    ] = None,
    weight: Annotated[
        str,
        typer.Option(
            help="Edge weight to use for the PLSR model.",
            rich_help_panel="Model Parameters",
            show_default=True,
        ),
    ] = "membership",
    splits: Annotated[
        int,
        typer.Option(
            help="Number of splits to use for the cross-validation.",
            rich_help_panel="Model Parameters",
            show_default=True,
        ),
    ] = 10,
    permutations: Annotated[
        int,
        typer.Option(
            help="Number of permutations to use for the permutation testing.",
            rich_help_panel="Model Parameters",
            show_default=True,
        ),
    ] = 1000,
    scoring: Annotated[
        ScoringMethod,
        typer.Option(
            help="Scoring method to use for the permutation testing.",
            rich_help_panel="Model Parameters",
            show_default=True,
        ),
    ] = ScoringMethod.r2,
    processes: Annotated[
        int,
        typer.Option(
            help="Number of processes to use for the cross-validation.",
            rich_help_panel="Model Parameters",
            show_default=True,
        ),
    ] = 4,
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
    ] = False
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
    PARTIAL LEAST SQUARE REGRESSION (PLSR)
    --------------------------------------
    CCPM_plsr.py performs a Partial Least Square Regression (PLSR) on a graph
    using the nodes' attributes as predictors and the edges' weights as
    response variable. The script will perform a cross-validation to determine
    the optimal number of components to use for the PLSR model. It will then
    perform a permutation testing to determine if the model is statistically
    significant. Finally, it will output the PLSR coefficients and statistics
    as well as plots of the distributions of the attributes and edges' weights
    and the PLSR coefficients.
    \b
    PREPROCESSING
    -------------
    The script will scale the data to unit variance and zero mean and will
    perform a log transformation on the edges' weights (for now, it assumes
    that the weights represent a membership value resulting from a fuzzy
    clustering analysis).
    \b
    NODES' ATTRIBUTES
    -----------------
    The script takes only one graph file as input. The graph file must be in
    .gexf format. The script will then fetch the attributes from the graph
    file and will perform the PLSR analysis on the attributes and edges'
    weights. If no attributes are provided, the script will use all attributes
    found in the graph file. To set attributes to the nodes in the graph file,
    please see CCPM_set_nodes_attributes.py.
    \b
    SCORING OPTIONS
    ---------------
    The script will perform a permutation testing to determine if the model is
    statistically significant. The script will compute the p-value for the
    permutation testing using the R2 score by default. However, the user can
    also choose multiple scores to compute the p-value. The available scores
    can be seen in [1]. The equation used to compute the single-tailed p-value
    is:
        \b
        p-value = ∑(score_perm >= score) / (nb_permutations)
    \b
    COEFFICIENT SIGNIFICANCE
    ------------------------
    The script will also compute the p-value for the coefficients using the
    permutation testing. The p-value for the coefficients is computed by
    comparing the coefficients obtained from the PLSR model with the
    coefficients obtained from the permutation testing. The equation used to
    compute the two-tailed p-value is:
        \b
        p-value = ∑(abs(coef_perm) >= abs(coef)) / (nb_permutations)
    \b
    REFERENCES
    ----------
    [1]
    https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    \b
    EXAMPLE USAGE
    -------------
    CCPM_plsr.py --in-graph graph.gexf --out-folder output_folder \\
        -v -s
    """

    warnings.filterwarnings("ignore")

    if verbose:
        logging.getLogger().setLevel(logging.INFO)
        coloredlogs.install(level=logging.INFO)

    assert_input(in_graph)
    assert_output_dir_exist(overwrite, out_folder, create_dir=True)

    # Creating output structure.
    os.makedirs(f"{out_folder}/Distributions", exist_ok=True)
    os.makedirs(f"{out_folder}/Coefficients", exist_ok=True)

    # Saving parameters
    if save_parameters:
        parameters = list(locals().items())
        with open("nodes_attributes_parameters.txt", "w+") as f:
            for param in parameters:
                f.writelines(str(param))

    # Load graph file.
    logging.info("Loading graph and dataset.")
    G = nx.read_gexf(in_graph)

    # Fetching dataframe.
    logging.info("Fetching nodes' attributes dataframe.")
    attr_df = fetch_attributes_df(G, attributes)

    # Fetching edge data.
    logging.info("Fetching edge data.")
    edge_df = fetch_edge_data(G, weight=weight)

    # Scaling data to unit variance and zero mean.
    logging.info("Scaling data.")
    attr = scale(attr_df)
    edge = edge_df.apply(lambda x: np.log(x))
    edge = scale(edge)

    # Performing Cross-Validation.
    logging.info("Performing Cross-Validation.")
    (plsr, mse, score_c, score_cv,
     rscore, mse_c, mse_cv) = plsr_cv(
        attr,
        edge,
        len(attr_df.columns),
        splits=splits,
        processes=processes,
        verbose=verbose)

    # Permutation testing.
    logging.info("Performing permutation testing.")
    score, perm_score, score_pval, perm_coef, coef_pval = permutation_testing(
        attr,
        edge,
        nb_comp=np.argmin(mse) + 1,
        nb_permutations=permutations,
        scoring=scoring,
        splits=splits,
        processes=processes,
        verbose=verbose)

    # Exporting statistics.
    logging.info("Exporting statistics.")
    coef = {
        f'coef{i+1}': plsr.coef_[i, :] for i in range(0, plsr.coef_.shape[0])}
    coef['varname'] = attr_df.columns
    coef_df = pd.DataFrame(coef)
    coef_df.to_excel(f"{out_folder}/Coefficients/coef_df.xlsx")

    coef_pval_df = pd.DataFrame(coef_pval,
                                index=attr_df.columns,
                                columns=edge_df.columns)
    coef_pval_df.to_excel(f"{out_folder}/Coefficients/coef_pval_df.xlsx")

    stats = pd.DataFrame([mse_c, mse_cv, score_c, score_cv, rscore,
                          score, perm_score, score_pval],
                         columns=['Statistics'],
                         index=['MSE_c', 'MSE_cv', 'R2_c', 'R2_cv', 'R_c',
                                'R2_score', 'R2_perm', 'pval'])
    stats.to_excel(f"{out_folder}/statistics.xlsx", header=True, index=True)

    # Plotting results.
    logging.info("Generating plots.")

    # Plotting histograms.
    with plt.rc_context(
        {"font.size": 10, "font.weight": "bold", "axes.titleweight": "bold"}
    ):
        # Edge histogram.
        fig, ax = plt.subplots(figsize=(10, 10))
        pd.DataFrame(edge, columns=edge_df.columns).hist(ax=ax)
        plt.tight_layout()
        plt.savefig(f"{out_folder}/Distributions/edge_histogram.png")
        plt.close()

        # MSE plot.
        fig, ax = plt.subplots(figsize=(10, 10))
        plt.plot(
            np.arange(1, len(attr_df.columns)),
            np.array(mse),
            '-v',
            color='blue',
            mfc='blue')
        plt.plot(
            np.arange(1, len(attr_df.columns))[np.argmin(mse)],
            np.array(mse)[np.argmin(mse)],
            'P',
            ms=10,
            mfc='red')
        plt.xlabel('Number of PLS components')
        plt.ylabel('MSE')
        plt.title('MSE plot for multiple PLS components number.')
        plt.xlim(left=-1)
        plt.tight_layout()
        plt.savefig(f"{out_folder}/mse_plot.png")
        plt.close()

        # Coefficient plot.
        for i in range(0, len(edge_df.columns)):
            generate_coef_plot(
                coef_df,
                perm_coef[:, :, i],
                coef_pval[:, i],
                coefname=f'coef{i+1}',
                varname='varname',
                output=f"{out_folder}/Coefficients/coef_plot_cluster_{i+1}.png"
            )


if __name__ == "__main__":
    app()
