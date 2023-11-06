#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import coloredlogs

import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import seaborn as sns
import typer
from typing_extensions import Annotated
from typing import List

from CCPM.io.utils import assert_input, assert_output_dir_exist
from CCPM.network.metrics import weightedpath, PathLengthsMethods

# Initializing the app.
app = typer.Typer(add_completion=False)


@app.command()
def main(
    in_graph: Annotated[
        str,
        typer.Option(
            help="Input graph file (.gexf).",
            show_default=False,
            rich_help_panel="Essential Files Options",
        ),
    ],
    id_column: Annotated[
        str,
        typer.Option(
            help="Name of the column in --data-for-label containing the "
            "subjects ids.",
            show_default=False,
            rich_help_panel="Essential Files Options",
        ),
    ],
    data_for_label: Annotated[
        str,
        typer.Option(
            help="Dataset containing binary columns used to select nodes.",
            show_default=False,
            rich_help_panel="Essential Files Options",
        ),
    ],
    label_name: Annotated[
        List[str],
        typer.Option(
            help="Label(s) name(s) to select group(s) of nodes. Can be "
            "multiple.",
            show_default=False,
            rich_help_panel="Essential Files Options",
        ),
    ],
    out_folder: Annotated[
        str,
        typer.Option(
            help="Output folder where files will be exported.",
            show_default=False,
            rich_help_panel="Essential Files Options",
        ),
    ] = None,
    iterations: Annotated[
        int,
        typer.Option(
            help="Number of iterations to perform to generate the null "
            "distribution.",
            show_default=True,
            rich_help_panel="Computational Options",
        ),
    ] = 5000,
    weight: Annotated[
        str,
        typer.Option(
            help="Name of the edge attributes to use during the weighted path "
            "computation.",
            show_default=True,
            rich_help_panel="Computational Options",
        ),
    ] = "membership",
    method: Annotated[
        PathLengthsMethods,
        typer.Option(
            help="Method to use for the estimation of the weighted path.",
            show_choices=True,
            show_default=True,
            rich_help_panel="Computational Options",
        ),
    ] = PathLengthsMethods.Dijkstra,
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
    GRAPH NETWORK METRICS
    ---------------------
    CCPM_graph_metrics.py is a wrapper script that allows the user to compute
    various network metrics on an existing graph network. Depending on your
    hardware and the metric you want to compute, the script could be running
    for ~10 mins for a large graph (~10 000 nodes).
    \b
    AVAILABLE METRICS
    -----------------
    Available metrics come from the Networkx implemented algorithms. As of
    now, only the algorithms that can handle undirected weighted graph are
    implemented (see operation argument). For more details regarding those
    algorithms, please see [1]. Details regarding specific operations: if
    NODE is required, you have to provide a single node label (such as 'c1').
    If NODES is required, you can provide multiple nodes within quotation
    marks (such as "c1 c2 c3").
    \b
    REFERENCE
    ---------
    [1]
    https://networkx.org/documentation/stable/reference/algorithms/index.html#
    \b
    EXAMPLE USAGE
    -------------
    CCPM_graph_metrics.py --out-folder output/ eigencentrality graph.gexf
        membership
    """

    if verbose:
        verbose = True
        logging.getLogger().setLevel(logging.INFO)
        coloredlogs.install(level=logging.INFO)

    assert_input(in_graph, data_for_label)
    assert_output_dir_exist(overwrite, out_folder, create_dir=True)

    if save_parameters:
        parameters = list(locals().items())
        with open(f"{out_folder}/parameters.txt", "w+") as f:
            for param in parameters:
                f.writelines(str(param))

    logging.info("Loading graph and dataset.")
    G = nx.read_gexf(in_graph)

    # Loading dataset and generating list of nodes to include.
    df = pd.read_excel(data_for_label)

    # Compute weigted path metric.
    for var in label_name:
        logging.info("Computing average weighted path for variable : {}"
                     .format(var))
        avg_weighted_path, null_dist = weightedpath(
            G,
            df,
            label_name=var,
            id_column=id_column,
            iterations=iterations,
            weight=weight,
            method=method,
            verbose=True,
        )

        # Plotting distribution.
        plt.rcParams["figure.figsize"] = [12, 7]
        plt.rcParams["figure.autolayout"] = True
        sns.set_style("white")
        sns.set_context("poster", 0.5)

        fig = plt.figure(figsize=(12, 7))
        ax = fig.add_subplot()

        # Plotting 1st highest membership value.
        sns.histplot(data=null_dist, stat="density", bins=50, kde=True, ax=ax)
        ax.set_xlabel("Average weighted path metric.")
        ax.axvline(x=avg_weighted_path, ymin=0, ymax=1)

        plt.tight_layout()
        plt.savefig(f"{out_folder}/results_{var}.png")
        plt.close()


if __name__ == "__main__":
    app()
