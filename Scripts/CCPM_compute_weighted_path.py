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
    ] = 1000,
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
    distribution: Annotated[
        str,
        typer.Option(
            help="Pre-computed null distribution, needs to be an .xlsx file"
            "containing label name as --label-name.",
            show_choices=False,
            show_default=True,
            rich_help_panel="Computational Options",
        ),
    ] = None,
    processes: Annotated[
        int,
        typer.Option(
            help="Number of processes to use to compute the null distribution",
            show_default=True,
            rich_help_panel="Computational Options",
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
    CCPM_compute_weighted_path.py is a script that computes the average
    weighted shortest path length for a group of nodes. The script will
    compare the result against a null distribution generated by randomly
    selecting the same number of nodes and computing the average weighted
    shortest path length. The script will also compute the p-value for the
    comparison.
    \b
    NULL DISTRIBUTION
    -----------------
    The null distribution is generated by randomly selecting the same number
    of nodes as the group of interest and computing the average weighted
    shortest path length. This process is repeated for a number of iterations
    (default: 5000) and the resulting distribution is used to compute the
    p-value. Please note that for large graphs, the null distribution could
    take a long time to compute. It is recommended to use the --distribution
    option to provide a pre-computed null distribution (see below). One could
    also reduce the number of iterations to speed up the process and have a
    rough estimate of the p-value.
    \b
    MULTIPROCESSING
    ---------------
    By default, CCPM_compute_weighted_path.py will use 4 CPUs to
    compute the null distribution. This can be changed by setting the
    --processes option.
    \b
    METHOD TO COMPUTE THE WEIGHTED PATH
    -----------------------------------
    CCPM_compute_weighted_path.py can use different methods to compute the
    weighted path. The default method is Dijkstra. The other methods are
    Bellman-Ford and Floyd-Warshall. Please note that the Floyd-Warshall
    method is not recommended for large graphs. For more details, see [1].
    \b
    REFERENCE
    ---------
    [1]
    https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.shortest_paths.generic.average_shortest_path_length.html#networkx.algorithms.shortest_paths.generic.average_shortest_path_length
    \b
    EXAMPLE USAGE
    -------------
    CCPM_compute_weighted_path.py \
        --in-graph ./graph.gexf \
        --id-column ID \
        --data-for-label ./data_for_label.xlsx \
        --label-name Group1 \
        --label-name Group2 \
        --out-folder ./results \
        --iterations 5000 \
        --weight membership \
        --method Dijkstra
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
    if distribution is not None:
        dist = pd.read_excel(distribution)
    else:
        dist = None

    # Compute weigted path metric.
    for var in label_name:
        logging.info("Computing average weighted path for variable : {}"
                     .format(var))
        avg_weighted_path, null_dist, pvalue = weightedpath(
            G,
            df,
            label_name=var,
            id_column=id_column,
            iterations=iterations,
            weight=weight,
            method=method,
            distribution=dist,
            processes=processes,
            verbose=True,
        )

        with plt.rc_context(
            {"font.size": 10, "font.weight": "bold",
             "axes.titleweight": "bold"}
        ):
            sns.set_style("white")
            sns.set_context("poster", 0.5)

            fig = plt.figure(figsize=(12, 7))
            ax = fig.add_subplot()

            # Plotting 1st highest membership value.
            sns.histplot(data=null_dist, stat="density", bins=50, kde=True,
                         ax=ax)
            ax.set_xlabel("Average weighted path metric.")
            ax.axvline(x=avg_weighted_path, ymin=0, ymax=1)

            # Annotating graph with p-value.
            ax.annotate(
                "p = {:.3f}".format(pvalue),
                xy=((0.75 * ax.get_xlim()[1]), (0.8 * ax.get_ylim()[1])),
                fontsize=20,
                ha="center",
                va="center",
            )

            plt.tight_layout()
            plt.savefig(f"{out_folder}/results_{var}.png")
            plt.close()

        out = pd.DataFrame(
            null_dist,
            columns=[var])
        out.to_excel(f"{out_folder}/null_distributions_{var}.xlsx",
                     header=True, index=False)

        # Export metric with pvalue.
        stats = pd.DataFrame([[avg_weighted_path], [pvalue]],
                             columns=['Statistics'],
                             index=['Average weighted path', 'p-value'])
        stats.to_excel(f"{out_folder}/statistics_{var}.xlsx", header=True,
                       index=True)


if __name__ == "__main__":
    app()
