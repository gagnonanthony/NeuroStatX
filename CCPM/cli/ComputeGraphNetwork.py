#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import coloredlogs
import logging
import sys

from cyclopts import App, Parameter
import networkx as nx
import numpy as np
import pandas as pd
from typing import List
from typing_extensions import Annotated

from CCPM.io.utils import (assert_input, assert_output_dir_exist,
                           load_df_in_any_format)
from CCPM.network.utils import get_nodes_and_edges
from CCPM.network.viz import (
    visualize_network,
    membership_distribution,
    NetworkLayout,
    create_cmap_from_list,
)


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
    data_for_label: Annotated[
        str,
        Parameter(
            show_default=True,
            group="Label Options",
        ),
    ] = None,
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
    layout: Annotated[
        NetworkLayout,
        Parameter(
            show_default=True,
            show_choices=True,
            group="Network Visualization Options",
        ),
    ] = NetworkLayout.Spring,
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
    """GRAPH NETWORK CLUSTERING VISUALIZATION
    --------------------------------------
    ComputeGraphNetwork is a script computing an undirected weighted
    graph network from fuzzy clustering c-partitioned membership matrix. It is
    designed to work seemlessly with CCPM_fuzzy_clustering.py. Mapping
    membership matrices to a graph network allows the future use of graph
    theory statistics such as shortest path, betweenness centrality, etc.
    The concept of this script was initially proposed in [1].

    LAYOUT ALGORITHMS
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

    **Layout is only computed once and is reused in all other network to**
    **reduce the computational burden.**

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
    customize the network using other APIs such as GEPHI (see [3]).

    REFERENCES
    ----------
    [1] Ariza-Jiménez, L., Villa, L. F., & Quintero, O. L. (2019). Memberships
        Networks for High-Dimensional Fuzzy Clustering Visualization., Applied
        Computer Sciences in Engineering (Vol. 1052, pp. 263–273). Springer
        International Publishing.
        https://doi.org/10.1007/978-3-030-31019-6_23

    [2] https://networkx.org/documentation/stable/reference/drawing.html

    [3] https://gephi.org/

    EXAMPLE USAGE
    -------------
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
    data_for_label : str, optional
        Variable within the dataframe to use for labelling specific subjects.
    label_name : List[str], optional
        Variable within the --data-for-label to use for subject nodes
        labelling.
    background_alpha : bool, optional
        If set, background nodes alpha will be set to 0.1 (more transparent).
    layout : NetworkLayout, optional
        Layout algorithm to determine the nodes position.
    label_centroids : bool, optional
        If true, centroids will be labelled.
    label_subjects : bool, optional
        If true, will label subjects nodes.
    centroids_size : int, optional
        Size of the centroids nodes.
    centroid_alpha : float, optional
        Alpha value representing the transparency of the centroids nodes.
    centroid_node_color : str, optional
        Centroids nodes color to use.
    centroid_edge_color : str, optional
        Assign a color to the edge of the centroids nodes.
    subject_node_size : int, optional
        Assign the size of the subjects nodes.
    subject_node_alpha : float, optional
        Assign the transparency alpha value to the subjects nodes.
    subject_node_color : str, optional
        Assign a color to the subjects nodes.
    subject_edge_color : str, optional
        Assign a color to the edge of the subjects nodes.
    colormap : str, optional
        Colormap to use when coloring the edges of the network based on the
        membership values to each clusters. Available colormap are those from
        plt.cm.
    title : str, optional
        Title of the network graph.
    legend_title : str, optional
        Legend title (colormap).
    """

    if verbose:
        logging.getLogger().setLevel(logging.INFO)
        coloredlogs.install(level=logging.INFO)

    assert_input(in_dataset)
    assert_output_dir_exist(overwrite, out_folder, create_dir=True)

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
    membership_distribution(
        clean_df.values, output=f"{out_folder}/membership_distribution.png"
    )

    # Fetching dataframe of nodes and edges.
    df, _, _ = get_nodes_and_edges(df_with_ids)

    # Creating network graph.
    G = nx.from_pandas_edgelist(df, "node1", "node2", edge_attr="membership")

    # Visualizing and saving network.
    logging.info("Constructing the layout and generating graph.")
    pos = visualize_network(
        G,
        output=f"{out_folder}/graph_network.png",
        layout=getattr(nx, layout),
        weight="membership",
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

    # Saving graph as a .gexf object for easy reloading.
    nx.write_gexf(G, f"{out_folder}/network_graph_file.gexf")

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

        assert (
            df_with_ids[id_column].all() == df_for_label[id_column].all()
        ), "Label and input data IDs does not match."

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
                layout=getattr(nx, layout),
                pos=pos,
                weight="membership",
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
