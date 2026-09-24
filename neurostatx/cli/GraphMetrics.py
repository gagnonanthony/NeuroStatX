#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import sys
import coloredlogs

from cyclopts import App, Parameter, Group
from typing_extensions import Annotated
from typing import List

from neurostatx.network.metrics import get_metrics_ops
from neurostatx.io.loader import DatasetLoader, GraphLoader
from neurostatx.io.utils import assert_input, assert_output

OPERATIONS = get_metrics_ops()
"""Mapping of metric names to NetworkX functions GraphMetrics can run."""

# Initializing the app.
app = App(default_parameter=Parameter(negative=()))
"""Cyclopts application for the GraphMetrics command-line tool."""


@app.default()
def GraphMetrics(
    operation: Annotated[
        List[str],
        Parameter(
            show_choices=True,
            show_default=False,
            group=Group("Arguments"),
        ),
    ],
    out_file: Annotated[
        str,
        Parameter(
            show_default=False,
            group="Essential Files Options",
        ),
    ] = None,
    verbose: Annotated[
        bool,
        Parameter(
            "-v",
            "--verbose",
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
    """Compute NetworkX metrics on an existing undirected weighted graph.

    Runtime depends on hardware and the chosen metric; a large graph
    (~10 000 nodes) can take about 10 minutes.

    Notes
    -----

    **Available metrics**

    Available metrics come from NetworkX algorithms that can handle undirected
    weighted graphs (see the operation argument) [1]. If NODE is required,
    provide a single node label (such as ``c1``). If NODES is required, provide
    multiple nodes within quotation marks (such as ``"c1 c2 c3"``).

    * eigencentrality: GRAPH WEIGHT — eigenvector centrality for all nodes.
    * closenesscentrality: GRAPH WEIGHT — closeness centrality for all nodes.
    * betweennesscentrality: GRAPH WEIGHT — betweenness centrality for all
      nodes.
    * informationcentrality: GRAPH WEIGHT — information centrality for all
      nodes.
    * currentflowbc: GRAPH WEIGHT — current flow betweenness centrality for
      all nodes.
    * loadcentrality: GRAPH WEIGHT — load centrality for all nodes.
    * harmoniccentrality: GRAPH WEIGHT — harmonic centrality for all nodes.
    * eccentricity: GRAPH WEIGHT — eccentricity for all nodes.
    * clustering: GRAPH WEIGHT — clustering coefficient for all nodes.
    * constraint: GRAPH NODES WEIGHT — constraint for all specified nodes.
    * effectivesize: GRAPH NODES WEIGHT — effective size for all specified
      nodes.
    * closenessvitality: GRAPH NODE WEIGHT — closeness vitality for a single
      node.
    * degree: GRAPH WEIGHT — degree of the specified node.

    References
    ----------
    [1] [NetworkX algorithms](https://networkx.org/documentation/stable/reference/algorithms/index.html#)

    Parameters
    ----------
    operation : List[str]
        List of arguments to provide to the script. The first argument is the
        operation to perform. The second argument is the input graph file. The
        rest of the arguments are the arguments required by the operation (see
        above).
    out_file : str, optional
        Path and name of the file containing the metrics for each node.
        Defaults to ``./operation_name.xlsx``.
    verbose : bool, optional
        If true, produce verbose output. Defaults to False.
    overwrite : bool, optional
        If true, force overwriting of existing output files. Defaults to False.

    Examples
    --------
    ```bash
    GraphMetrics --out-folder output/ eigencentrality graph.gexf membership
    ```
    """

    if verbose:
        logging.getLogger().setLevel(logging.INFO)
        coloredlogs.install(level=logging.INFO)

    if out_file is None:
        out_file = f"{operation[0]}.xlsx"

    assert_input(operation[1])
    assert_output(overwrite, out_file, check_dir=True)

    # Validating correct input number.
    if len(operation) < 3:
        sys.exit("Incorrect number of input provided. Please see {}"
                 .format(__doc__))

    # Validating input operation exist in dict.
    if operation[0] not in OPERATIONS.keys():
        sys.exit("Operation {} not implemented.".format(operation[0]))

    # Loading graph network file.
    logging.info("Loading graph network file...")
    G = GraphLoader().load_graph(operation[1])

    # Performing operation.
    try:
        logging.info("Running {} on input network..."
                     .format(operation[0].capitalize()))
        # Get the name of the argument in the function to call, since
        # we need to pass named arguments to the function.
        # Maximum 3 arguments are accepted, the first one is the graph.
        args = OPERATIONS[operation[0]].__code__.co_varnames

        # Building the arguments to pass to the function.
        kwargs = {arg: operation[i + 2] for i, arg in enumerate(args[1:])}

        output = G.custom_function(
            OPERATIONS[operation[0]],
            **kwargs,
        )

        if isinstance(output, float):
            # If the output is a float, we need to convert it to a dict.
            output = {operation[0]: output}

    except ValueError as msg:
        logging.error("{} operation failed.".format(operation[0].capitalize()))
        logging.error(msg)
        return

    # Exporting results in an .xlsx file.
    logging.info("Exporting results here: {}".format(out_file))
    DatasetLoader().import_data(
        dict(output),
        columns=[operation[0]],
        orient="index",
    ).save_data(
        out_file,
        header=True,
        index=True,
    )


if __name__ == "__main__":
    app()
