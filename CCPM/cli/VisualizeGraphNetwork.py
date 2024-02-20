#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import coloredlogs
import logging
import sys

from cyclopts import App, Parameter
import networkx as nx
import numpy as np
from typing import List
from typing_extensions import Annotated

from CCPM.io.utils import (assert_input, assert_output_dir_exist,
                           load_df_in_any_format)
from CCPM.network.viz import (visualize_network, create_cmap_from_list)
from CCPM.network.utils import filter_node_subjects


# Initializing the app.
app = App(default_parameter=Parameter(negative=()))


@app.default()
def VisualizeGraphNetwork(
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
            show_default=True,
            group="Essential Files Options",
        ),
    ] = "./visualize_network/",
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
    data_for_label: Annotated[
        str,
        Parameter(
            show_default=True,
            group="Label Options",
        ),
    ] = None,
    id_column: Annotated[
        str,
        Parameter(
            show_default=False,
            group="Label Options",
        ),
    ] = 'subjectkey',
    label_name: Annotated[
        List[str],
        Parameter(
            show_default=False,
            group="Label Options",
        ),
    ] = None,
    background_alpha: Annotated[
        bool,
        Parameter(
            show_default=True,
            group="Label Options",
        ),
    ] = True,
    weight: Annotated[
        str,
        Parameter(
            show_default=False,
            group="Network Visualization Options",
        ),
    ] = "membership",
    label_centroids: Annotated[
        bool,
        Parameter(
            show_default=True,
            group="Network Visualization Options",
        ),
    ] = True,
    label_subjects: Annotated[
        bool,
        Parameter(
            show_default=True,
            group="Network Visualization Options",
        ),
    ] = False,
    centroids_size: Annotated[
        int,
        Parameter(
            show_default=True,
            group="Network Visualization Options",
        ),
    ] = 500,
    centroid_alpha: Annotated[
        float,
        Parameter(
            show_default=True,
            group="Network Visualization Options",
        ),
    ] = 1,
    centroid_node_color: Annotated[
        str,
        Parameter(
            show_default=True,
            group="Network Visualization Options",
        ),
    ] = "white",
    centroid_edge_color: Annotated[
        str,
        Parameter(
            show_default=True,
            group="Network Visualization Options",
        ),
    ] = "black",
    subject_node_size: Annotated[
        int,
        Parameter(
            show_default=True,
            group="Network Visualization Options",
        ),
    ] = 5,
    subject_node_alpha: Annotated[
        float,
        Parameter(
            show_default=True,
            group="Network Visualization Options",
        ),
    ] = 0.3,
    subject_node_color: Annotated[
        str,
        Parameter(
            show_default=True,
            group="Network Visualization Options",
        ),
    ] = "darkgrey",
    subject_edge_color: Annotated[
        str,
        Parameter(
            show_default=True,
            group="Network Visualization Options",
        ),
    ] = None,
    colormap: Annotated[
        str,
        Parameter(
            show_default=True,
            group="Network Visualization Options",
        ),
    ] = "plasma",
    title: Annotated[
        str,
        Parameter(
            show_default=False,
            group="Network Visualization Options",
        ),
    ] = "Network Graph of the clustering membership values.",
    legend_title: Annotated[
        str,
        Parameter(
            show_default=False,
            group="Network Visualization Options",
        ),
    ] = "Membership values",
):
    """VISUALIZING GRAPH NETWORK
    -------------------------
    VisualizeGraphNetwork is a command line tool to visualize the graph network
    generated from the clustering results. The script uses the NetworkX library
    to visualize the graph network. The script also provides options to label
    specific nodes based on a condition (e.g. a diagnosis, etc.). It is also
    possible to customize the graph appearance using the Network Visualization
    Options. The script is made to work hand in hand with the script
    ComputeGraphNetwork since it requires the precomputation of the nodes'
    position within the graph file.

    LABELLING GRAPH NETWORK NODES
    -----------------------------
    It is possible to label specific nodes based on a condition (e.g. a
    diagnosis, etc.). To do so, use --data-for-label argument to provide a
    dataframe containing the column(s) to use for labelling. You also need to
    specify the --label-name in order to use the correct column. It is also
    possible to provide multiple label name by using --label-name x
    --label-name y. The script will output multiple graphs for each label name.
    **LABEL DATA NEEDS TO BE IN THE SAME ORDER AS THE DATASET PROVIDED DURING**
    **CLUSTERING, IF NOT, LABEL AND SUBJECT WILL NOT MATCH**

    GRAPH NETWORK CUSTOMIZATION
    ---------------------------
    To customize the graph appearance, please see the Network Visualization
    Options below. It should be noted that using subjects_labelling will crowd
    the network if it contains a high number of nodes. Also, centroids are
    labelled by default 'c1, c2, ...' and subjects 's1, s2, ...'. The script
    also exports a graph_network_file.gexf. This file can be used to further
    customize the network using other APIs such as GEPHI (see [1]).

    REFERENCES
    ----------
    [1] https://gephi.org/

    EXAMPLE USAGE
    -------------
    ::

        VisualizeGraphNetwork --in-graph graph_network.gml
        --out-folder output/ --data-for-label label_data.csv
        --id-column subjectkey --label-name diagnosis
        --weight membership -v -f -s

    Parameters
    ----------
    in_graph : str
        Input graph network to visualize (.gml format).
    out_folder : str
        Output folder for the graph network visualization.
    verbose : bool
        Verbose mode.
    overwrite : bool
        Overwrite existing files.
    save_parameters : bool
        Save parameters to a .txt file.
    data_for_label : str
        Dataframe containing the column(s) to use for labelling.
    id_column : str
        Column name containing the IDs.
    label_name : List[str]
        List of label names to subsequently use for labelling.
    background_alpha : bool
        Use background alpha for the graph.
    weight : str
        Weight to use for the graph network.
    label_centroids : bool
        Label centroids.
    label_subjects : bool
        Label subjects.
    centroids_size : int
        Size of the centroids.
    centroid_alpha : float
        Alpha of the centroids.
    centroid_node_color : str
        Color of the centroids.
    centroid_edge_color : str
        Edge color of the centroids.
    subject_node_size : int
        Size of the subjects.
    subject_node_alpha : float
        Alpha of the subjects.
    subject_node_color : str
        Color of the subjects.
    subject_edge_color : str
        Edge color of the subjects.
    colormap : str
        Colormap to use for the graph network.
    title : str
        Title of the graph network.
    legend_title : str
        Legend title of the graph network.
    """
    # Saving parameters
    if save_parameters:
        parameters = list(locals().items())
        with open("nodes_attributes_parameters.txt", "w+") as f:
            for param in parameters:
                f.writelines(str(param))

    if verbose:
        logging.getLogger().setLevel(logging.INFO)
        coloredlogs.install(level=logging.INFO)

    assert_input(in_graph)
    assert_output_dir_exist(overwrite, out_folder, create_dir=True)

    # Loading graph.
    logging.info("Loading graph data.")
    G = nx.read_gml(in_graph)

    # Visualizing and saving network.
    logging.info("Generating graph.")
    pos = visualize_network(
        G,
        output=f"{out_folder}/graph_network.png",
        weight=weight,
        centroids_labelling=label_centroids,
        subjects_labelling=label_subjects,
        centroid_node_shape=centroids_size,
        centroid_alpha=centroid_alpha,
        centroid_node_color=centroid_node_color,
        centroid_edge_color=centroid_edge_color,
        subject_node_shape=subject_node_size,
        subject_alpha=subject_node_alpha,
        subject_node_color=subject_node_color,
        subject_edge_color=subject_edge_color,
        colormap=colormap,
        title=title,
        legend_title=legend_title,
    )

    # Plotting network with custom label.
    if data_for_label is not None:
        if label_name is None:
            sys.exit(
                "If --data-for-label is provided, you need to specify which "
                "column to use with --label-name."
            )

        logging.info("Constructing graph(s) with custom labels.")

        # Loading df.
        df_for_label = load_df_in_any_format(data_for_label)

        # Fetching nodes ids.
        sub_node = nx.subgraph_view(G, filter_node_subjects)
        nodes_id = np.array(sub_node.nodes())

        # Validating subject nodes and subject in data for label are equal.
        if not np.array_equal(nodes_id, df_for_label[id_column].to_numpy()):
            sys.exit(
                "Subject nodes and subject in data for label are not equal. "
                "Please verify that the order of the subjects is the same in "
                "the dataset and the graph network."
            )

        # Fetching data for label as array.
        for label in label_name:
            labels = df_for_label[label]

            nodes_cmap = create_cmap_from_list(labels)

            if background_alpha:
                sub_alpha = []
                for i in nodes_cmap:
                    if isinstance(i, str):
                        sub_alpha.append(0.1)
                    else:
                        sub_alpha.append(1)
            else:
                sub_alpha = np.array([1] * len(nodes_cmap))

            _ = visualize_network(
                G,
                output=f"{out_folder}/graph_network_{label}.png",
                weight=weight,
                centroids_labelling=label_centroids,
                subjects_labelling=label_subjects,
                centroid_node_shape=centroids_size,
                centroid_alpha=centroid_alpha,
                centroid_node_color=centroid_node_color,
                centroid_edge_color=centroid_edge_color,
                subject_node_shape=subject_node_size,
                subject_alpha=sub_alpha,
                subject_node_color=nodes_cmap,
                subject_edge_color=subject_edge_color,
                colormap="gray",
                title=f"{title} with {label} subjects colored.",
                legend_title=legend_title,
            )


if __name__ == "__main__":
    app()
