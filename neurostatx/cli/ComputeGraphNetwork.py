#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import coloredlogs
import logging

from cyclopts import App, Parameter
import networkx as nx
import pandas as pd
from typing_extensions import Annotated

from neurostatx.io.utils import (assert_input, assert_output_dir_exist,
                                 load_df_in_any_format)
from neurostatx.network.utils import (get_nodes_and_edges,
                                      construct_attributes_dict)
from neurostatx.network.viz import (
    compute_layout,
    set_nodes_position,
    membership_distribution,
    NetworkLayout)


# Initializing the app.
app = App(default_parameter=Parameter(negative=()))


@app.default()
def ComputeGraphNetwork(
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
    desc_columns: Annotated[
        int,
        Parameter(
            show_default=False,
            group="Essential Files Options",
        ),
    ],
    out_folder: Annotated[
        str,
        Parameter(
            show_default=True,
            group="Essential Files Options",
        ),
    ] = "./graph_results/",
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
    save_parameters: Annotated[
        bool,
        Parameter(
            "-s",
            "--save_parameters",
            group="Optional parameters",
        ),
    ] = False,
    plot_distribution: Annotated[
        bool,
        Parameter(
            show_default=True,
            group="Plotting Options",
        ),
    ] = False,
    import_data: Annotated[
        bool,
        Parameter(
            show_default=True,
            group="Essential Files Options",
        ),
    ] = False,
    layout: Annotated[
        NetworkLayout,
        Parameter(
            show_default=True,
            show_choices=True,
            group="Layout Options",
        ),
    ] = NetworkLayout.Spring,
    weight: Annotated[
        str,
        Parameter(
            show_default=True,
            group="Layout Options",
        ),
    ] = "membership",
):
    """Graph Network Clustering Visualization
    --------------------------------------
    ComputeGraphNetwork is a script computing an undirected weighted
    graph network from fuzzy clustering c-partitioned membership matrix. It is
    designed to work seemlessly with FuzzyClustering. Mapping
    membership matrices to a graph network allows the future use of graph
    theory statistics such as shortest path, betweenness centrality, etc.
    The concept of this script was initially proposed in [1].

    Layout Algorithms
    -----------------
    In order to generate a graph network, the nodes positions need to be
    determined in relation with their connections to other nodes (and the
    weigth of those connections). Those connections are also called edges and
    contain a weight in the case of a weighted graph network. Possible
    algorithms to choose from are :

    Kamada Kawai Layout: Use the Kamada-Kawai path-length cost-function. Not
                        the optimal solution for large network as it is
                        computer intensive. For details, see [2].

    Spectral Layout: Position is determined using the eigenvectors of the
                    graph Laplacian. For details, see [2].

    Spring Layout: Use the Fruchterman-Reingold force-directed algorithm.
                    Suitable for large network with high number of nodes.
                    For details, see [2]. This is the default method.

    Importing Data Within The .gml File
    -----------------------------------
    If the --import-data flag is set to True, the descriptive data will be
    imported within the .gml file. The imported data will be stored as node's
    attributes. This is useful for future use of the graph network in
    visualization scripts or in statistical analysis (view AverageWeightedPath
    or Plsr). This ensure a robust handling of data and reduce the probability
    of data mismatch between subjects.

    References
    ----------
    [1] Ariza-Jiménez, L., Villa, L. F., & Quintero, O. L. (2019). Memberships
        Networks for High-Dimensional Fuzzy Clustering Visualization., Applied
        Computer Sciences in Engineering (Vol. 1052, pp. 263–273). Springer
        International Publishing.(https://doi.org/10.1007/978-3-030-31019-6_23)

    [2] NetworkX Documentation
    (https://networkx.org/documentation/stable/reference/drawing.html)

    Example Usage
    -------------
    ::

        ComputeGraphNetwork --in-dataset cluster_membership.xlsx
        --id-column subjectkey --desc-columns 1 --out-folder output/

    **For large graphs (~10 000 nodes), it might take ~5 mins to run using**
    **the spring layout and depending on your hardware.**

    Parameters
    ----------
    in_dataset : str
        Input dataset containing membership values for each clusters.
    id_column : str
        Name of the column containing the subject's ID tag. Required for
        proper handling of IDs.
    desc_columns : int
        Number of descriptive columns at the beginning of the dataset.
    out_folder : str, optional
        Path of the folder in which the results will be written. If not
        specified, current folder and default name will be used.
    verbose : bool, optional
        If true, produce verbose output.
    overwrite : bool, optional
        If true, force overwriting of existing output files.
    save_parameters : bool, optional
        If true, save the parameters used in a .txt file.
    plot_distribution : bool, optional
        If true, will plot the membership distribution and delta.
    import_data : bool, optional
        If true, will import the data from the input dataset within the graph
        network file.
    layout : NetworkLayout, optional
        Layout algorithm to determine the nodes position.
    weight : str, optional
        Name of the column containing the edge weight. Default is 'membership'.
    """

    if verbose:
        logging.getLogger().setLevel(logging.INFO)
        coloredlogs.install(level=logging.INFO)

    assert_input(in_dataset)
    assert_output_dir_exist(overwrite, out_folder, create_dir=True)

    # Saving parameters
    if save_parameters:
        parameters = list(locals().items())
        with open(f"{out_folder}/graph_network_parameters.txt", "w+") as f:
            for param in parameters:
                f.writelines(str(param))

    # Loading membership matrix.
    logging.info("Loading membership data.")
    raw_df = load_df_in_any_format(in_dataset)
    descriptive_columns = [n for n in range(0, desc_columns)]

    # Creating the array.
    desc_data = raw_df[raw_df.columns[descriptive_columns]]
    clean_df = raw_df.drop(
        raw_df.columns[descriptive_columns], axis=1, inplace=False
    ).astype("float")
    df_with_ids = pd.concat([desc_data[desc_data.columns[0]], clean_df],
                            axis=1)

    # Plotting membership distributions and delta.
    if plot_distribution:
        membership_distribution(
            clean_df.values, output=f"{out_folder}/membership_distribution.png"
        )

    # Fetching dataframe of nodes and edges.
    df, _, _ = get_nodes_and_edges(df_with_ids)

    # Creating network graph.
    G = nx.from_pandas_edgelist(df, "node1", "node2", edge_attr="membership")

    # Computing graph network layout.
    logging.info("Computing graph network layout.")
    pos = compute_layout(G, layout=getattr(nx, layout), weight=weight)

    logging.info("Setting nodes position.")
    set_nodes_position(G, pos)

    if import_data:
        logging.info("Importing data within the .gml file.")
        attributes = construct_attributes_dict(desc_data,
                                               desc_data.columns[1:],
                                               id_column)
        nx.set_node_attributes(G, attributes)

    # Saving graph as a .gexf object for easy reloading.
    nx.write_gml(G, f"{out_folder}/network_graph_file.gml")


if __name__ == "__main__":
    app()
