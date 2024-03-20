#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import coloredlogs

from cyclopts import App, Parameter
import pandas as pd
import networkx as nx
from typing_extensions import Annotated
from typing import List

from CCPM.io.utils import assert_input, assert_output_dir_exist
from CCPM.network.metrics import weightedpath, PathLengthsMethods
from CCPM.network.utils import fetch_attributes_df

# Initializing the app.
app = App(default_parameter=Parameter(negative=()))


@app.default()
def AverageWeightedPath(
    in_graph: Annotated[
        str,
        Parameter(
            show_default=False,
            group="Essential Files Options",
        ),
    ],
    label_name: Annotated[
        List[str],
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
    ] = None,
    iterations: Annotated[
        int,
        Parameter(
            show_default=True,
            group="Computational Options",
        ),
    ] = 1000,
    weight: Annotated[
        str,
        Parameter(
            show_default=True,
            group="Computational Options",
        ),
    ] = "membership",
    method: Annotated[
        PathLengthsMethods,
        Parameter(
            show_choices=True,
            show_default=True,
            group="Computational Options",
        ),
    ] = PathLengthsMethods.Dijkstra,
    distribution: Annotated[
        str,
        Parameter(
            show_choices=False,
            show_default=True,
            group="Computational Options",
        ),
    ] = None,
    processes: Annotated[
        int,
        Parameter(
            show_default=True,
            group="Computational Options",
        ),
    ] = 4,
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
    """AVEARGE WEIGHTED PATH
    ---------------------
    AverageWeightedPath is a script that computes the average
    weighted shortest path length for a group of nodes. The script will
    compare the result against a null distribution generated by randomly
    selecting the same number of nodes and computing the average weighted
    shortest path length. The script will also compute the p-value for the
    comparison.

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

    MULTIPROCESSING
    ---------------
    By default, AverageWeightedPath will use 4 CPUs to
    compute the null distribution. This can be changed by setting the
    --processes option.

    METHOD TO COMPUTE THE WEIGHTED PATH
    -----------------------------------
    AverageWeightedPath can use different methods to compute the
    weighted path. The default method is Dijkstra. The other methods are
    Bellman-Ford and Floyd-Warshall. Please note that the Floyd-Warshall
    method is not recommended for large graphs. For more details, see [1].

    REFERENCE
    ---------
    [1]
    https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.shortest_paths.generic.average_shortest_path_length.html#networkx.algorithms.shortest_paths.generic.average_shortest_path_length

    EXAMPLE USAGE
    -------------
    ::

        AverageWeightedPath --in-graph ./graph.gexf --id-column ID
        --data-for-label ./data_for_label.xlsx --label-name Group1
        --label-name Group2 --out-folder ./results --iterations 5000
        --weight membership --method Dijkstra

    Parameters
    ----------
    in_graph : str
        Input graph file (.gml).
    label_name : List[str]
        Label(s) name(s) to select group(s) of nodes. Can be multiple.
    out_folder : str, optional
        Output folder where files will be exported.
    iterations : int, optional
        Number of iterations to perform to generate the null distribution.
    weight : str, optional
        Name of the edge attributes to use during the weighted path computation
    method : PathLengthsMethods, optional
        Method to use for the estimation of the weighted path.
    distribution : str, optional
        Pre-computed null distribution, needs to be an .xlsx file containing
        label name as --label-name.
    processes : int, optional
        Number of processes to use to compute the null distribution.
    verbose : bool, optional
        If true, produce verbose output.
    save_parameters : bool, optional
        If true, will save input parameters to .txt file.
    overwrite : bool, optional
        If true, force overwriting of existing output files.
    """

    if verbose:
        verbose = True
        logging.getLogger().setLevel(logging.INFO)
        coloredlogs.install(level=logging.INFO)

    assert_input(in_graph)
    assert_output_dir_exist(overwrite, out_folder, create_dir=True)

    if save_parameters:
        parameters = list(locals().items())
        with open(f"{out_folder}/parameters.txt", "w+") as f:
            for param in parameters:
                f.writelines(str(param))

    logging.info("Loading graph.")
    G = nx.read_gml(in_graph)

    # Fetching labels from nodes' attributes.
    df = fetch_attributes_df(G, attributes=label_name)

    # Loading dataset and generating list of nodes to include.
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
            iterations=iterations,
            weight=weight,
            method=method,
            distribution=dist,
            processes=processes,
            verbose=True,
        )

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
