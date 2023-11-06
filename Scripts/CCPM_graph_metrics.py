#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import sys
import coloredlogs

import pandas as pd
import networkx as nx
import typer
from typing_extensions import Annotated
from typing import List

from CCPM.network.metrics import get_metrics_ops, get_metrics_docs
from CCPM.io.utils import assert_input, assert_output

OPERATIONS = get_metrics_ops()

__doc__ = get_metrics_docs(OPERATIONS)

# Initializing the app.
app = typer.Typer(add_completion=False)


@app.command()
def main(
    operation: Annotated[
        List[str],
        typer.Argument(
            help=__doc__,
            show_choices=True,
            show_default=False,
            rich_help_panel="Operations Options",
        ),
    ],
    out_file: Annotated[
        str,
        typer.Option(
            help="Path and name of the file containing the metrics for each "
            "nodes. Default is : ./{operation_name}.xlsx",
            show_default=False,
            rich_help_panel="Essential Files Options",
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option(
            "-v",
            "--verbose",
            help="If true, produce verbose output.",
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
    Available metrics come from the Networkx implemented algorithms. As of now,
    only the algorithms that can handle undirected weighted graph are
    implemented (see operation argument). For more details regarding those
    algorithms, please see [1]. Details regarding specific operations: if NODE
    is required, you have to provide a single node label (such as 'c1'). If
    NODES is required, you can provide multiple nodes within quotation marks
    (such as "c1 c2 c3").
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
    G = nx.read_gexf(operation[1])

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
