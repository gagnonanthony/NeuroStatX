#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import coloredlogs
import logging
import os
import warnings

from cyclopts import App, Parameter
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import scale
from typing import List
from typing_extensions import Annotated

from neurostatx.io.utils import (assert_input, assert_output_dir_exist)
from neurostatx.io.loader import DatasetLoader, GraphLoader
from neurostatx.statistics.models import (plsr_cv, permutation_testing,
                                          ScoringMethod)
from neurostatx.io.viz import generate_coef_plot, flexible_hist

# Initializing the app.
app = App(default_parameter=Parameter(negative=()))


@app.default()
def PartialLeastSquareRegression(
    in_graph: Annotated[
        str,
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
    ],
    attributes: Annotated[
        List[str],
        Parameter(
            group="Model Parameters",
            show_default=True,
        ),
    ] = None,
    weight: Annotated[
        str,
        Parameter(
            group="Model Parameters",
            show_default=True,
        ),
    ] = "membership",
    splits: Annotated[
        int,
        Parameter(
            group="Model Parameters",
            show_default=True,
        ),
    ] = 10,
    permutations: Annotated[
        int,
        Parameter(
            group="Model Parameters",
            show_default=True,
        ),
    ] = 1000,
    scoring: Annotated[
        ScoringMethod,
        Parameter(
            group="Model Parameters",
            show_default=True,
        ),
    ] = ScoringMethod.r2,
    processes: Annotated[
        int,
        Parameter(
            group="Model Parameters",
            show_default=True,
        ),
    ] = 1,
    plot_distributions: Annotated[
        bool,
        Parameter(
            show_default=True,
            group="Visualization Options",
        ),
    ] = False,
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
    ] = False
):
    """Partial Least Square Regression (PLSR)
    --------------------------------------
    Plsr performs a Partial Least Square Regression (PLSR) on a graph
    using the nodes' attributes as predictors and the edges' weights as
    response variable. The script will perform a cross-validation to determine
    the optimal number of components to use for the PLSR model. It will then
    perform a permutation testing to determine if the model is statistically
    significant. Finally, it will output the PLSR coefficients and statistics
    as well as plots of the distributions of the attributes and edges' weights
    and the PLSR coefficients.

    Preprocessing
    -------------
    The script will scale the data to unit variance and zero mean and will
    perform a log transformation on the edges' weights (for now, it assumes
    that the weights represent a membership value resulting from a fuzzy
    clustering analysis).

    Nodes' Attributes
    -----------------
    The script takes only one graph file as input. The graph file must be in
    .gexf format. The script will then fetch the attributes from the graph
    file and will perform the PLSR analysis on the attributes and edges'
    weights. If no attributes are provided, the script will use all attributes
    found in the graph file. To set attributes to the nodes in the graph file,
    please see AddNodesAttributes.

    Scoring Options
    ---------------
    The script will perform a permutation testing to determine if the model is
    statistically significant. The script will compute the p-value for the
    permutation testing using the R2 score by default. However, the user can
    also choose multiple scores to compute the p-value. The available scores
    can be seen in [1]. The equation used to compute the single-tailed p-value
    is:

        p-value = ∑(score_perm >= score) / (nb_permutations)

    Coefficient Significance
    ------------------------
    The script will also compute the p-value for the coefficients using the
    permutation testing. The p-value for the coefficients is computed by
    comparing the coefficients obtained from the PLSR model with the
    coefficients obtained from the permutation testing. The equation used to
    compute the two-tailed p-value is:

        p-value = ∑(abs(coef_perm) >= abs(coef)) / (nb_permutations)

    References
    ----------
    [1] Scikit-learn scoring methods
    (https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter)

    Example Usage
    -------------
    ::

        PartialLeastSquareRegression --in-graph graph.gexf
        --out-folder output_folder -v -s

    Parameters
    ----------
    in_graph : str
        Graph file containing the data for the PLSR model.
    out_folder : str
        Output folder.
    attributes : List[str], optional
        Attributes names to include in the PLSR model. If None are provided,
        all attributes will be included.
    weight : str, optional
        Edge weight to use for the PLSR model.
    splits : int, optional
        Number of splits to use for the cross-validation.
    permutations : int, optional
        Number of permutations to use for the permutation testing.
    scoring : ScoringMethod, optional
        Scoring method to use for the permutation testing.
    processes : int, optional
        Number of processes to use for the cross-validation.
    verbose : bool, optional
        If true, produce verbose output.
    plot_distributions : bool, optional
        If true, will plot the distributions of the edges' weights.
    save_parameters : bool, optional
        If true, will save input parameters to .txt file.
    overwrite : bool, optional
        If true, force overwriting of existing output files.
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
        with open(f"{out_folder}/nodes_attributes_parameters.txt", "w+") as f:
            for param in parameters:
                f.writelines(str(param))

    # Load graph file.
    logging.info("Loading graph and dataset.")
    G = GraphLoader().load_graph(in_graph)

    # Fetching dataframe.
    logging.info("Fetching nodes' attributes dataframe.")
    attr_df = G.fetch_attributes_df(attributes=attributes)

    # Fetching edge data.
    logging.info("Fetching edge data.")
    edge_df = G.fetch_edge_data(weight=weight)

    # Scaling data to unit variance and zero mean.
    logging.info("Scaling data.")
    attr_df.custom_function(scale)
    edge = edge_df.get_data().apply(lambda x: np.log(x))
    edge = DatasetLoader().import_data(scale(edge),
                                       columns=edge_df.get_data().columns,
                                       index=edge_df.get_data().index)

    # Performing Cross-Validation.
    logging.info("Performing Cross-Validation.")
    (plsr, mse, score_c, score_cv,
     rscore, mse_c, mse_cv) = attr_df.custom_function(
        plsr_cv,
        Y=edge.get_data(),
        nb_comp=len(attr_df.get_data().columns),
        splits=splits,
        processes=processes,
        verbose=verbose,
    )

    # Permutation testing.
    logging.info("Performing permutation testing.")
    mod, score, coef, perm_score, score_pval, perm_coef, coef_pval = \
        permutation_testing(
            plsr,
            attr_df.get_data(),
            edge.get_data(),
            nb_permutations=permutations,
            scoring=scoring,
            splits=splits,
            processes=processes,
            verbose=verbose)

    # Exporting statistics.
    logging.info("Exporting statistics.")
    coef = {
        f'coef{i+1}': coef[:, i] for i in range(0, coef.shape[0])}
    coef['varname'] = attr_df.get_data().columns
    coef_df = DatasetLoader().import_data(
        coef
    )
    coef_df.save_data(
        f"{out_folder}/Coefficients/coef_df.xlsx",
        header=True,
        index=False
    )

    DatasetLoader().import_data(
        coef_pval,
        index=attr_df.get_data().columns,
        columns=edge_df.get_data().columns
    ).save_data(
        f"{out_folder}/Coefficients/coef_pval_df.xlsx",
        header=True,
        index=True
    )

    DatasetLoader().import_data(
        [mse_c, mse_cv, score_c, score_cv, rscore, score, perm_score,
         score_pval],
        columns=['Statistics'],
        index=['MSE_c', 'MSE_cv', 'R2_c', 'R2_cv', 'R_c', 'R2_score',
               'R2_perm', 'pval']).save_data(
        f"{out_folder}/statistics.xlsx",
        header=True,
        index=True
    )

    # Plotting results.
    logging.info("Generating plots.")

    # Plotting histograms.
    with plt.rc_context(
        {"font.size": 10, "font.weight": "bold", "axes.titleweight": "bold"}
    ):
        if plot_distributions:
            # Edge's distributions.
            edge.custom_function(
                flexible_hist,
                output=f"{out_folder}/Distributions/edges_distribution.png",
                cmap="magma", title="Edges' distributions",
                xlabel="Edges' weights", ylabel="Density"
            )

        # MSE plot.
        fig, ax = plt.subplots(figsize=(10, 10))
        plt.plot(
            np.arange(1, len(attr_df.get_data().columns) + 1),
            np.array(mse),
            '-v',
            color='blue',
            mfc='blue')
        plt.plot(
            np.arange(1, len(attr_df.get_data().columns) + 1)[np.argmin(mse)],
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
        for i in range(0, len(edge_df.get_data().columns)):
            generate_coef_plot(
                coef_df.get_data(),
                coef_pval[:, i],
                coefname=f'coef{i+1}',
                varname='varname',
                output=f"{out_folder}/Coefficients/coef_plot_cluster_{i+1}.png"
            )


if __name__ == "__main__":
    app()
