#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging

import networkx as nx
import numpy as np
import typer
from typing_extensions import Annotated

from CCPM.io.utils import assert_input, assert_output_dir_exist
from CCPM.network.utils import get_nodes_and_edges
from CCPM.network.viz import visualize_network, NetworkLayout


# Initializing the app.
app = typer.Typer(add_completion=False)


@app.command()
def main(
    in_matrix: Annotated[
        str,
        typer.Option(
            help="Numpy array containing the fuzzy clustering membership "
                 "values (.npy).",
            show_default=False,
            rich_help_panel="Essential Files Options",
        ),
    ],
    out_folder: Annotated[
        str,
        typer.Option(
            help="Path of the folder in which the results will be written. "
            "If not specified, current folder and default "
            "name will be used.",
            rich_help_panel="Essential Files Options",
        ),
    ] = "./graph_results/",
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
    layout: Annotated[
        NetworkLayout,
        typer.Option(
            help="Layout algorithm to determine the nodes position.",
            show_default=True,
            show_choices=True,
            rich_help_panel="Network Visualization Options",
        ),
    ] = NetworkLayout.Spring,
    label_centroids: Annotated[
        bool,
        typer.Option(
            help="If true, centroids will be labelled.",
            show_default=True,
            rich_help_panel="Network Visualization Options",
        ),
    ] = True,
    label_subjects: Annotated[
        bool,
        typer.Option(
            help="If true, will label subjects nodes.",
            show_default=True,
            rich_help_panel="Network Visualization Options",
        ),
    ] = False,
    centroids_size: Annotated[
        int,
        typer.Option(
            help="Size of the centroids nodes.",
            show_default=True,
            rich_help_panel="Network Visualization Options",
        ),
    ] = 500,
    centroid_alpha: Annotated[
        float,
        typer.Option(
            help="Alpha value representing the transparency of the centroids "
                 "nodes.",
            show_default=True,
            rich_help_panel="Network Visualization Options",
        ),
    ] = 1,
    centroid_node_color: Annotated[
        str,
        typer.Option(
            help="Centroids nodes color to use.",
            show_default=True,
            rich_help_panel="Network Visualization Options",
        ),
    ] = "white",
    centroid_edge_color: Annotated[
        str,
        typer.Option(
            help="Assign a color to the edge of the centroids nodes.",
            show_default=True,
            rich_help_panel="Network Visualization Options",
        ),
    ] = "black",
    subject_node_size: Annotated[
        int,
        typer.Option(
            help="Assign the size of the subjects nodes.",
            show_default=True,
            rich_help_panel="Network Visualization Options",
        ),
    ] = 5,
    subject_node_alpha: Annotated[
        float,
        typer.Option(
            help="Assign the transparency alpha value to the subjects "
                 "nodes.",
            show_default=True,
            rich_help_panel="Network Visualization Options",
        ),
    ] = 0.3,
    subject_node_color: Annotated[
        str,
        typer.Option(
            help="Assign a color to the subjects nodes.",
            show_default=True,
            rich_help_panel="Network Visualization Options",
        ),
    ] = "darkgrey",
    subject_edge_color: Annotated[
        str,
        typer.Option(
            help="Assign a color to the edge of the subjects nodes.",
            show_default=True,
            rich_help_panel="Network Visualization Options",
        ),
    ] = None,
    colormap: Annotated[
        str,
        typer.Option(
            help="Colormap to use when coloring the edges of the network "
                 "based on the membership values to each clusters. Available "
                 "colormap are those from plt.cm.",
            show_default=True,
            rich_help_panel="Network Visualization Options",
        ),
    ] = "plasma",
    title: Annotated[
        str,
        typer.Option(
            help="Title of the network graph.",
            show_default=False,
            rich_help_panel="Network Visualization Options",
        ),
    ] = "Network Graph of the clustering membership values.",
    legend_title: Annotated[
        str,
        typer.Option(
            help="Legend title (colormap).",
            show_default=False,
            rich_help_panel="Network Visualization Options",
        ),
    ] = "Membership values",
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
    GRAPH NETWORK CLUSTERING VISUALIZATION
    --------------------------------------
    CCPM_compute_graph_network.py is a script computing an undirected weighted
    graph network from fuzzy clustering c-partitioned membership matrix. It is
    designed to work seemlessly with CCPM_fuzzy_clustering.py. Mapping
    membership matrices to a graph network allows the future use of graph
    theory statistics such as shortest path, betweenness centrality, etc.
    The concept of this script was initially proposed in [1].
    \b
    LAYOUT ALGORITHMS
    -----------------
    In order to generate a graph network, the nodes positions need to be
    determined in relation with their connections to other nodes (and the
    weigth of those connections). Those connections are also called edges and
    contain a weight in the case of a weighted graph network. Possible
    algorithms to choose from are :
    \b
    Kamada Kawai Layout: Use the Kamada-Kawai path-length cost-function. Not
                         the optimal solution for large network as it is
                         computer intensive. For details, see [2].
    Spectral Layout: Position is determined using the eigenvectors of the
                     graph Laplacian. For details, see [2].
    Spring Layout: Use the Fruchterman-Reingold force-directed algorithm.
                   Suitable for large network with high number of nodes. For
                   details, see [2]. This is the default method.
    \b
    GRAPH NETWORK CUSTOMIZATION
    ---------------------------
    To customize the graph appearance, please see the Network Visualization
    Options below. It should be noted that using subjects_labelling will crowd
    the network if it contains a high number of nodes. Also, centroids are
    labelled by default 'c1, c2, ...' and subjects 's1, s2, ...'. The script
    also exports a graph_network_file.gexf. This file can be used to further
    customize the network using other APIs such as GEPHI (see [3]).
    \b
    REFERENCES
    ----------
    [1] Ariza-Jiménez, L., Villa, L. F., & Quintero, O. L. (2019). Memberships
        Networks for High-Dimensional Fuzzy Clustering Visualization., Applied
        Computer Sciences in Engineering (Vol. 1052, pp. 263–273). Springer
        International Publishing.
        https://doi.org/10.1007/978-3-030-31019-6_23
    [2] https://networkx.org/documentation/stable/reference/drawing.html
    [3] https://gephi.org/
    \b
    EXAMPLE USAGE
    -------------
    CCPM_compute_graph_network.py --in-matrix cluster_membership.npy
        --out-folder output/
    ** For large graphs (~10 000 nodes), it might take ~5 mins to run using
       the spring layout and depending on your hardware. **
    """

    if verbose:
        logging.getLogger().setLevel(logging.INFO)

    assert_input(in_matrix)
    assert_output_dir_exist(overwrite, out_folder, create_dir=True)

    # Loading membership matrix.
    logging.info("Loading membership matrix.")
    membership_mat = np.load(in_matrix)

    # Fetching dataframe of nodes and edges.
    df, _, _ = get_nodes_and_edges(membership_mat)

    # Creating network graph.
    G = nx.from_pandas_edgelist(df, "node1", "node2", edge_attr="membership")

    # Visualizing and saving network.
    logging.info("Constructing the layout and generating graph.")
    visualize_network(
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


if __name__ == "__main__":
    app()
