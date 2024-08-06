from enum import Enum
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import matplotlib
import seaborn as sns
from strenum import StrEnum

from neurostatx.network.utils import (filter_node_centroids,
                                      filter_node_subjects)


class NetworkLayout(StrEnum, Enum):
    KamadaKawai = ("kamada_kawai_layout",)
    Spectral = ("spectral_layout",)
    Spring = "spring_layout"


def compute_layout(G,
                   layout=nx.spring_layout,
                   weight="weight",
                   seed=1234):
    """Function to compute the optimal layout for a graph network.

    Parameters
    ----------
    G : nx.Graph
        Graph Network to draw.
    layout : nx.layout, optional
        Layout algorithm to use. Defaults to nx.spring_layout.
    weight : str, optional
        Edges weight to use while computing the layout. Defaults to "weight".
    seed : int, optional
        Random seed. Defaults to 1234.

    Returns
    -------
    pos
        Layout positions that can be reused for replotting of the same network.
    """

    return layout(G, weight=weight, seed=seed)


def set_nodes_position(G, pos):
    """
    Function to set the nodes' position for a graph network.

    Position array has to be converted into python float in order for
    Gephi file format export.

    Parameters
    ----------
    G : nx.Graph
        Graph Network to draw.
    pos : dict
        Dictionary with nodes' positions.

    Returns
    -------
    G : nx.Graph
        Network with nodes' positions set.
    """

    positions = {k: list(map(float, pos[k])) for k in pos}

    return nx.set_node_attributes(G, positions, "pos")


def visualize_network(
    G,
    output,
    weight="weight",
    centroids_labelling=True,
    subjects_labelling=False,
    centroid_node_shape=500,
    centroid_alpha=1,
    centroid_node_color="white",
    centroid_edge_color="black",
    subject_node_shape=5,
    subject_alpha=0.3,
    subject_node_color="black",
    subject_edge_color=None,
    colormap="plasma",
    title="Graph Network",
    legend_title="Membership values",
):
    """
    Function to visualize a weighted undirected graph network. Based on the
    concept from:
    Ariza-Jiménez, L., Villa, L. F., & Quintero, O. L. (2019). Memberships
        Networks for High-Dimensional Fuzzy Clustering Visualization.,
        Applied Computer Sciences in Engineering (Vol. 1052, pp. 263–273).
        Springer International Publishing.
        https://doi.org/10.1007/978-3-030-31019-6_23

    Parameters
    ----------
    G : nx.Graph
        Graph Network to draw.
    output : str
        Filename and path for the output png image.
    weight : str, optional
        Edge attribute to use as weight. Defaults to "weight".
    centroids_labelling : bool, optional
        Label centroid nodes. Defaults to True.
    subjects_labelling : bool, optional
        Label subject nodes. Defaults to False.
    centroid_node_shape : int, optional
        Centroid's nodes shape. Defaults to 500.
    centroid_alpha : int, optional
        Centroid's nodes alpha. Defaults to 1.
    centroid_node_color : str, optional
        Centroid's nodes color. Defaults to "white".
    centroid_edge_color : str, optional
        Centroid's nodes edge color. Defaults to "black".
    subject_node_shape : int, optional
        Subject's nodes shape. Defaults to 5.
    subject_alpha : float, optional
        Subject's nodes alpha value. Defaults to 0.3.
    subject_node_color : str, optional
        Subject's nodes color. Defaults to "black".
    subject_edge_color : str, optional
        Subject's nodes edge color. Defaults to None.
    colormap : str, optional
        Colormap to use to draw edges' weights. Defaults to "plasma".
    title : str, optional
        Graph title. Defaults to "Graph Network".
    legend_title : str, optional
        Legend title. Defaults to "Membership values".
    """

    # Fetching nodes position.
    pos = nx.get_node_attributes(G, "pos")

    # Fetching edges widths.
    widths = nx.get_edge_attributes(G, weight)

    # Sorting which nodes to label.
    labels = {}
    if centroids_labelling:
        for node in G.nodes():
            if "c" in node:
                labels[node] = node
    elif centroids_labelling and subjects_labelling:
        for node in G.nodes():
            labels[node] = node
    else:
        for node in G.nodes():
            labels[node] = ""

    # Setting z-order of nodes.
    cntr_node = nx.subgraph_view(G, filter_node=filter_node_centroids)
    sub_node = nx.subgraph_view(G, filter_node=filter_node_subjects)

    # Centroids customization lists.
    cntr_shape = np.array([centroid_node_shape] * len(cntr_node.nodes()))
    cntr_alpha = np.array([centroid_alpha] * len(cntr_node.nodes()))

    # Subjects customization lists.
    sub_shape = np.array([subject_node_shape] * len(sub_node.nodes()))
    # sub_alpha = np.array([subject_alpha] * len(sub_node.nodes()))

    # Plotting the graph.
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot()

    nodes1 = nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=sub_node.nodes(),
        node_size=sub_shape,
        node_color=subject_node_color,
        alpha=subject_alpha,
        ax=ax,
    )
    nodes2 = nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=cntr_node.nodes(),
        node_size=cntr_shape,
        node_color=centroid_node_color,
        alpha=cntr_alpha,
        ax=ax,
    )

    # Drawing edges.
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=widths.keys(),
        width=(list(widths.values()) * 10),
        edge_color=list(widths.values()),
        edge_cmap=getattr(plt.cm, colormap),
        alpha=list(widths.values()),
        ax=ax,
    )

    # Setting z-order.
    nodes1.set_zorder(2)
    if subject_edge_color is not None:
        nodes1.set_edgecolor(subject_edge_color)
    nodes2.set_zorder(3)
    nodes2.set_edgecolor(centroid_edge_color)

    # Plotting labels if set.
    nx.draw_networkx_labels(G, pos, labels=labels, font_color="black", ax=ax)

    # Adding colorbar, titles, etc.
    cmappable = ScalarMappable(matplotlib.colors.Normalize(0, 1),
                               getattr(plt.cm, colormap))
    cbar = plt.colorbar(cmappable, ax=ax, location="right", shrink=0.5)

    plt.box(False)
    ax.set_title(title)
    cbar.ax.set_title(legend_title)

    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def membership_distribution(mat, output="./membership_distribution.png"):
    """
    Function returning a plot of the distribution of the maximum membership
    values for each subject and the delta plot of the difference with the
    second highest membership from the fuzzy c-partitioned matrix.

    Parameters
    ----------
    mat : Array
        Membership fuzzy c-partitioned matrix.
    output : str, optional
        Output filename and path. Defaults to "./membership_distribution.png".
    """

    # Fetching 1st highest membership value.
    high1st = np.max(mat, axis=1)

    # Fetching 2nd highest membership value.
    high2nd = np.partition(mat, -2, axis=1)[:, -2]

    # Calculating delta.
    delta = high1st - high2nd

    # Plotting the distribution.
    with plt.rc_context(
        {"font.family": "Sans Serif",
         "font.size": 12, "font.weight": "normal", "axes.titleweight": "bold"}
    ):

        fig, ax = plt.subplots(2, 2, figsize=(16, 16))

        # Plotting 1st highest membership value.
        sns.histplot(data=high1st, stat="density", bins=50, kde=True,
                     ax=ax[0, 0], color="lightgray")
        ax[0, 0].set_xlabel(
            "1st highest membership value distribution amongst all subjects."
        )
        ax[0, 0].axvline(x=np.median(high1st), ymin=0, ymax=1, color="gray",
                         linestyle="--")

        # Plotting 2nd highest memberhsip value.
        sns.histplot(data=high2nd, stat="density", bins=50, kde=True,
                     ax=ax[0, 1], color="lightgray")
        ax[0, 1].set_xlabel(
            "2nd highest membership value distribution amongst all subjects."
        )
        ax[0, 1].axvline(x=np.median(high2nd), ymin=0, ymax=1, color="gray",
                         linestyle="--")

        # Plotting delta.
        sns.histplot(data=delta, stat="density", bins=50, kde=True,
                     ax=ax[1, 0], color="lightgray")
        ax[1, 0].set_xlabel(
            "Delta (1st highest value - 2nd highest value) distribution "
            "amongst all subjects."
        )
        ax[1, 0].axvline(x=np.median(delta), ymin=0, ymax=1, color="gray",
                         linestyle="--")

        ax[1, 1].remove()
        plt.tight_layout()
        plt.savefig(f"{output}")
        plt.close()


def creating_node_colormap(percentile_dict):
    """
    Function to create a colormap for a set of nodes based on a percentile
    dictionary.

    Parameters
    ----------
    percentile_dict : dict
        Percentile dictionary with keys identifying clusters.

    Returns
    -------
    cmap
        List containing all colors.
    """

    cmap = plt.cm.tab10(np.linspace(0, 1, 10))

    # Assuming it is impossible for node to have 2 main clusters.
    percentile_array = np.max(np.array(list(percentile_dict.values())), axis=0)

    nodes_cmap = []
    for i in percentile_array:
        if i == 0:
            nodes_cmap.append("black")
        else:
            nodes_cmap.append(cmap[i])

    return nodes_cmap


def create_cmap_from_list(array):
    """
    Function to create a cmap from a list of values (corresponding to a
    condition for each subject.) For example, subject with value = 0 will be
    mapped black, whereas subject = >1 will be mapped to different colors.

    Parameters
    ----------
    array : np.array
        Array of integers corresponding to condiditions (can be binary also.).

    Returns
    -------
    cmap
        List containing all colors.
    """

    if array.dtype == "float64":
        cmap = plt.cm.plasma(np.linspace(0, 1, len(array)))

        ranks = array.argsort().argsort()

        nodes_cmap = []
        for i in ranks:
            nodes_cmap.append(cmap[i])

    else:
        cmap = plt.cm.tab10(np.linspace(0, 1, 10))

        nodes_cmap = []
        for i in array:
            if i == 0:
                nodes_cmap.append("darkgrey")
            else:
                nodes_cmap.append(cmap[i])

    return nodes_cmap
