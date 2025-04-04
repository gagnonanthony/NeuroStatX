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
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import RocCurveDisplay
from typing import List
from typing_extensions import Annotated

from neurostatx.io.utils import (assert_input, assert_output_dir_exist)
from neurostatx.io.loader import DatasetLoader, GraphLoader
from neurostatx.statistics.models import (permutation_testing, ScoringMethod,
                                          Penalty, Solver)
from neurostatx.io.viz import generate_coef_plot, flexible_hist

# Initializing the app.
app = App(default_parameter=Parameter(negative=()))


@app.default()
def LogisticRegression(
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
            show_default=False,
        ),
    ],
    covariates: Annotated[
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
    test_size: Annotated[
        float,
        Parameter(
            group="Model Parameters",
            show_default=True,
        ),
    ] = 0.2,
    cs: Annotated[
        int,
        Parameter(
            group="Model Parameters",
            show_default=True,
        ),
    ] = 10,
    max_iter: Annotated[
        int,
        Parameter(
            group="Model Parameters",
            show_default=True,
        ),
    ] = 1000,
    penalty: Annotated[
        Penalty,
        Parameter(
            group="Model Parameters",
            show_default=True,
            show_choices=True,
        ),
    ] = Penalty.l2,
    solver: Annotated[
        Solver,
        Parameter(
            group="Model Parameters",
            show_default=True,
            show_choices=True,
        ),
    ] = Solver.lbfgs,
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
    """Logistic Regression Analysis
    ----------------------------
    This script performs a Logistic Regression on a graph
    using the edges' weights as predictors and the nodes' attributes as
    response variable. The script will perform a cross-validation within a
    training dataset before testing the model on test data. It will then
    perform a permutation testing to determine if the model is statistically
    significant. Finally, it will output the coefficients and statistics
    as well as plots of the distributions of the attributes and edges' weights
    and the coefficients.

    Preprocessing
    -------------
    The script will scale the data to unit variance and zero mean and will
    perform a log transformation on the edges' weights (for now, it assumes
    that the weights represent a membership value resulting from a fuzzy
    clustering analysis).

    Nodes' Attributes
    -----------------
    The script takes only one graph file as input. The graph file must be in
    .gml format. The script will then fetch the attributes from the graph
    file and will perform the analysis on the attributes and edges'
    weights. At least one attribute needs to be provided in order to fit a
    model. To set attributes to the nodes in the graph file,
    please see AddNodesAttributes.

    Scoring Options
    ---------------
    The script will perform a permutation testing to determine if the model is
    statistically significant. The script will compute the p-value for the
    permutation testing using the area under the curve (AUC) by default.
    However, the user can also choose multiple scores to compute the p-value.
    The available scores can be seen in [1]. The equation used to compute
    the single-tailed p-value is:

        p-value = ∑(score_perm >= score) / (nb_permutations)

    Coefficient Significance
    ------------------------
    The script will also compute the p-value for the coefficients using the
    permutation testing. The p-value for the coefficients is computed by
    comparing the coefficients obtained from the model with the
    coefficients obtained from the permutation testing. The equation used to
    compute the two-tailed p-value is:

        p-value = ∑(abs(coef_perm) >= abs(coef)) / (nb_permutations)

    References
    ----------
    [1]
    https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter

    Example Usage
    -------------
    ::

        LogisticRegression --in-graph graph.gexf --out-folder output_folder -v
        -s

    Parameters
    ----------
    in_graph : str
        Graph file containing the data for the model.
    out_folder : str
        Output folder.
    attributes : List[str],
        Attributes names to include in the model. Must be present in the graph
        file. At least one attribute is required.
    covariates : List[str], optional
        Covariates to include in the model. Must be present in the graph file.
    weight : str, optional
        Edge weight to use for the model.
    splits : int, optional
        Number of splits to use for the cross-validation.
    test_size : float, optional
        Size of the testing set. Must be between 0 and 1.
    cs : int, optional
        Inverse of regularization strength. Smaller values specify stronger
        regularization.
    max_iter : int, optional
        Maximum number of iterations for the solver.
    penalty : Penalty, optional
        Regularization penalty to use for the LogisticRegression model.
    solver : Solver, optional
        Solver to use for the LogisticRegression model.
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

    # Fetching covariates, attributes and edge data.
    logging.info("Fetching covariates, attributes and edge data.")
    attributes_df = G.fetch_attributes_df(attributes=attributes)
    edge_df = G.fetch_edge_data(weight=weight)
    covariates_df = G.fetch_attributes_df(attributes=covariates)

    # Preprocessing data.
    logging.info("Scaling data.")
    edge = edge_df.get_data().apply(lambda x: np.log(x))
    edge = edge_df.import_data(scale(edge), columns=edge_df.get_data().columns,
                               index=edge_df.get_data().index)
    covariates_df.join(edge_df.get_data(), left=True)

    # Performing Cross-Validation.
    logging.info("Fitting model with permutation testing.")

    # Splitting into a training and testing set.
    X_train, X_test, y_train, y_test = train_test_split(
        covariates_df.get_data(),
        attributes_df.get_data(),
        test_size=test_size,
        random_state=1234
    )

    if penalty == Penalty.elasticnet:
        l1_ratios = [0.1, 0.5, 0.7, 0.9]
    else:
        l1_ratios = None

    # Stratified KFold.
    # kf_10 = StratifiedKFold(n_splits=splits, shuffle=True, random_state=1)

    classifier = LogisticRegressionCV(
        Cs=cs,
        cv=splits,
        max_iter=max_iter,
        penalty=penalty,
        scoring=scoring,
        solver=solver,
        l1_ratios=l1_ratios,
        n_jobs=processes
    )

    mod, score, coef, perm_score, score_pval, coef_perm, coef_pval = \
        permutation_testing(
            classifier,
            X_train,
            y_train.values.ravel(),
            splits=splits,
            nb_permutations=permutations,
            scoring=scoring,
            processes=processes,
            verbose=verbose
        )

    logging.info("Exporting statistics and plots.")
    for i in range(0, len(attributes_df.get_data().columns)):
        display = RocCurveDisplay.from_predictions(
            LabelBinarizer().fit(y_train).transform(y_test)[:, i],
            mod.predict_proba(X_test)[:, 1],
            name="Logistic Regression",
            color='black',
            plot_chance_level=True
        )
        # Set labels and title.
        display.ax_.set(
            xlabel="False Positive Rate (FPR)",
            ylabel="True Positive Rate (TPR)",
            title=f"Receiver Operating Characteristic (ROC) Curve for \
            {attributes_df.get_data().columns[i]}"
        )
        plt.tight_layout()
        plt.savefig(
            f"{out_folder}/roc_curve_{attributes_df.get_data().columns[i]}.png"
        )
        plt.close()

    # Generating distribution plots.
    if plot_distributions:
        edge.custom_function(
            flexible_hist,
            output=f'{out_folder}/Distributions/edges_distributions.png',
            cmap="magma", title="Edges' Distributions",
            xlabel="Edges' Weights", ylabel="Density"
        )

    # Generating coefficient plot.
    coef = {
        f'coef{i+1}': coef[:, i].T for i in range(0, coef.shape[1])
    }
    coef['varname'] = covariates_df.get_data().columns
    coef_df = DatasetLoader().import_data(
        coef
    )
    coef_df.save_data(
        f"{out_folder}/Coefficients/coefficients.csv",
        header=True,
        index=False
    )

    DatasetLoader().import_data(
        coef_pval,
        columns=attributes_df.get_data().columns,
        index=covariates_df.get_data().columns
    ).save_data(
        f"{out_folder}/Coefficients/coefficients_pval.csv",
        header=True
    )

    DatasetLoader().import_data(
        [score, np.median(perm_score), score_pval],
        columns=['Statistics'],
        index=['Score', 'Median Permuted Score', 'P-Value']
    ).save_data(
        f"{out_folder}/statistics.csv",
        header=True,
        index=True
    )

    for i in range(0, len(attributes_df.get_data().columns)):
        generate_coef_plot(
            coef_df.get_data(),
            coef_pval[:, i],
            coefname=f'coef{i+1}',
            varname='varname',
            output=f"{out_folder}/Coefficients/coef_plot_cluster_{i+1}.png"
        )


if __name__ == "__main__":
    app()
