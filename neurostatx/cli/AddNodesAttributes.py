#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import coloredlogs
import logging
import os

from cyclopts import App, Parameter
from typing import List
from typing_extensions import Annotated

from neurostatx.io.utils import (assert_input, assert_output)
from neurostatx.io.loader import DatasetLoader, GraphLoader
from neurostatx.network.utils import construct_attributes_dict

# Initializing the app.
app = App(default_parameter=Parameter(negative=()))


@app.default()
def AddNodesAttributes(
    in_graph: Annotated[
        str,
        Parameter(
            show_default=False,
            group="Essential Files Options",
        ),
    ],
    in_dataset: Annotated[
        str,
        Parameter(
            show_default=False,
            group="Essential Files Options",
        ),
    ],
    id_column: Annotated[
        str,
        Parameter(
            show_default=False,
            group="Essential Files Options",
        ),
    ],
    labels: Annotated[
        List[str],
        Parameter(
            show_default=False,
            group="Essential Files Options",
        ),
    ],
    out_file: Annotated[
        str,
        Parameter(
            show_default=False,
            group="Essential Files Options",
        ),
    ] = "graph_with_attributes.gml",
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
    ] = False
):
    """Setting Nodes Attributes
    ------------------------
    AddNodesAttributes is a script that sets the attributes of the
    nodes of a graph. The attributes are provided via a tabulated data format
    (.csv, .xlsx or .txt) file. The script will automatically match the IDs of
    the nodes with the IDs of the subjects in the file. Name of the variables
    to add as attributes can be supplied either via a .txt file or with
    multiple --labels arguments.

    Example Usage
    -------------
    ::

        AddNodesAttributes --in-graph graph.gml --in-dataset dataset.xlsx
        --id-column ID --labels label1 --labels label2 --labels label3
        --out-file graph_attributes.gexf --verbose

    Parameters
    ----------
    in_graph : str
        Graph file to add attributes to.
    in_dataset : str
        Dataset containing the variables to add as nodes' attributes to the
        graph.
    id_column : str
        Name of the column containing the subject's ID tag. Required for proper
        handling of IDs and merging multiple datasets.
    labels : List[str]
        Label(s) name(s) to add as nodes' attributes to the graph. Can be
        supplied multiple times or as a .txt file containing all names in a
        line separated format.
    out_file : str, optional
        Output graph file name (.gml).
    verbose : bool, optional
        If true, produce verbose output.
    overwrite : bool, optional
        If true, force overwriting of existing output files.
    """

    if verbose:
        logging.getLogger().setLevel(logging.INFO)
        coloredlogs.install(level=logging.INFO)

    assert_input(in_graph)
    assert_input(in_dataset)
    assert_output(overwrite, out_file)

    logging.info("Loading graph and dataset.")
    network = GraphLoader().load_graph(in_graph)
    df = DatasetLoader().load_data(in_dataset)

    # Sorting if labels is a .txt file or not.
    if len(labels) == 1:
        _filename, ext = os.path.splitext(labels[0])
        if ext == '.txt':
            with open(labels[0], 'r') as f:
                labels = f.read().splitlines()

    # Getting dictionary of labels and values to add.
    logging.info("Constructing dictionary of attributes to add.")
    attr = df.custom_function(construct_attributes_dict,
                              labels=labels,
                              id_column=id_column)
    # attributes = construct_attributes_dict(df, labels, id_column)

    # Add attributes to graph.
    logging.info("Adding attributes to graph.")
    network.add_node_attribute(attr)

    logging.info("Saving graph.")
    network.save_graph(out_file)


if __name__ == "__main__":
    app()
