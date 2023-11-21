#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import coloredlogs
import logging
import os

import networkx as nx
import typer
from typing import List
from typing_extensions import Annotated

from CCPM.io.utils import (assert_input, assert_output,
                           load_df_in_any_format)
from CCPM.network.utils import construct_attributes_dict

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
    in_dataset: Annotated[
        str,
        typer.Option(
            help="Dataset containing the variables to add as nodes' attributes"
            " to the graph.",
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
    labels: Annotated[
        List[str],
        typer.Option(
            help="Label(s) name(s) to add as nodes' attributes to the graph. "
            "Can be supplied multiple times or as a .txt file containing all "
            "names in a line separated format.",
            show_default=False,
            rich_help_panel="Essential Files Options",
        ),
    ],
    out_file: Annotated[
        str,
        typer.Option(
            help="Output graph file name.",
            show_default=False,
            rich_help_panel="Essential Files Options",
        ),
    ],
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
    SETTING NODES ATTRIBUTES
    ------------------------
    CCPM_set_nodes_attributes.py is a script that sets the attributes of the
    nodes of a graph. The attributes are provided via a xlsx file. The script
    will automatically match the IDs of the nodes with the IDs of the subjects
    in the xlsx file. Name of the variables to add as attributes can be
    supplied either via a .txt file or with multiple --labels arguments.
    \b
    EXAMPLE USAGE
    -------------
    CCPM_set_nodes_attributes.py \\
        --in-graph graph.gexf \\
        --in-dataset dataset.xlsx \\
        --id-column ID \\
        --labels label1 --labels label2 --labels label3 \\
        --out-file graph_attributes.gexf \\
        --verbose
    """

    if verbose:
        logging.getLogger().setLevel(logging.INFO)
        coloredlogs.install(level=logging.INFO)

    assert_input(in_graph)
    assert_input(in_dataset)
    assert_output(overwrite, out_file)

    # Saving parameters
    if save_parameters:
        parameters = list(locals().items())
        with open("nodes_attributes_parameters.txt", "w+") as f:
            for param in parameters:
                f.writelines(str(param))

    logging.info("Loading graph and dataset.")
    G = nx.read_gexf(in_graph)
    df = load_df_in_any_format(in_dataset)

    # Sorting if labels is a .txt file or not.
    if len(labels) == 1:
        filename, ext = os.path.splitext(labels[0])
        if ext == '.txt':
            with open(labels[0], 'r') as f:
                labels = f.read().splitlines()

    # Getting dictonary of labels and values to add.
    logging.info("Constructing dictionary of attributes to add.")
    attributes = construct_attributes_dict(df, labels, id_column)

    # Add attributes to graph.
    logging.info("Adding attributes to graph.")
    nx.set_node_attributes(G, attributes)

    logging.info("Saving graph.")
    nx.write_gexf(G, out_file)


if __name__ == "__main__":
    app()
