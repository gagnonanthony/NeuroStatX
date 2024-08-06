#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import sys
import coloredlogs

from cyclopts import App, Parameter, Group
import pandas as pd
import networkx as nx
from typing_extensions import Annotated
from typing import List

from neurostatx.network.metrics import get_metrics_ops
from neurostatx.io.utils import assert_input, assert_output

OPERATIONS = get_metrics_ops()

# Initializing the app.
app = App(default_parameter=Parameter(negative=()))


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
    """Graph Network Metrics
    ---------------------
    GraphMetrics is a wrapper script that allows the user to compute
    various network metrics on an existing graph network. Depending on your
    hardware and the metric you want to compute, the script could be running
    for ~10 mins for a large graph (~10 000 nodes).

    Available Metrics
    -----------------
    Available metrics come from the Networkx implemented algorithms. As of now,
    only the algorithms that can handle undirected weighted graph are
    implemented (see operation argument). For more details regarding those
    algorithms, please see [1]. Details regarding specific operations: if NODE
    is required, you have to provide a single node label (such as 'c1'). If
    NODES is required, you can provide multiple nodes within quotation marks
    (such as "c1 c2 c3").

    - eigencentrality: GRAPH WEIGHT -
        Will return a dictionary of the eigenvector centrality for all nodes.
    - closenesscentrality: GRAPH WEIGHT -
        Will return a dictionary of the closeness centrality for all nodes.
    - betweennesscentrality: GRAPH WEIGHT -
        Will return a dictionary of the betweenness centrality for all nodes.
    - informationcentrality: GRAPH WEIGHT -
        Will return a dictionary of the information centrality for all nodes.
    - currentflowbc: GRAPH WEIGHT -
        Will return a dictionary of the current flow betweenness centrality
        for all nodes.
    - loadcentrality: GRAPH WEIGHT -
        Will return a dictionary of the load centrality for all nodes.
    - harmoniccentrality: GRAPH WEIGHT -
        Will return a dictionary of the harmonic centrality for all nodes.
    - eccentricity: GRAPH WEIGHT -
        Will return a dictionary of the eccentricity for all nodes.
    - clustering: GRAPH WEIGHT -
        Will return a dictionary of the clustering coefficient for all nodes.
    - constraint: GRAPH NODES WEIGHT -
        Will return a dictionary of the constraint for all specified nodes.
    - effectivesize: GRAPH NODES WEIGHT -
        Will return a dictionary of the effective size for all specified nodes.
    - closenessvitality: GRAPH NODE WEIGHT -
        Will return a dictionary of the closeness vitality for a single node.
    - degree: GRAPH WEIGHT -
        Will return the degree of the specified node.

    Reference
    ---------
    [1] Networkx Algorithms
    (https://networkx.org/documentation/stable/reference/algorithms/index.html#)

    Example Usage
    -------------
    ::

        GraphMetrics --out-folder output/ eigencentrality graph.gexf membership

    Parameters
    ----------
    operation : List[str]
        List of arguments to provide to the script. The first argument is the
        operation to perform. The second argument is the input graph file. The
        rest of the arguments are the arguments required by the operation (see
        above).
    out_file : str, optional
        Path and name of the file containing the metrics for each nodes.
        Default is : ./operation_name.xlsx
    verbose : bool, optional
        If true, produce verbose output.
    overwrite : bool, optional
        If true, force overwriting of existing output files.
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
    G = nx.read_gml(operation[1])

    # Performing operation.
    try:
        logging.info("Running {} on input network..."
                     .format(operation[0].capitalize()))
        output = OPERATIONS[operation[0]](G, *operation[2:])

    except ValueError as msg:
        logging.error("{} operation failed.".format(operation[0].capitalize()))
        logging.error(msg)
        return

    # Exporting results in an .xlsx file.
    logging.info("Exporting results here: {}".format(out_file))
    output = pd.DataFrame.from_dict([output]).T
    output.index.name = "nodes"
    output.columns = [operation[0]]
    output.to_excel(out_file, header=True, index=True)


if __name__ == "__main__":
    app()
