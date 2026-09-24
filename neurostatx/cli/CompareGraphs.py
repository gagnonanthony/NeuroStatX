#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import coloredlogs
import logging

from cyclopts import App, Parameter
import numpy as np
from typing_extensions import Annotated

from neurostatx.io.utils import assert_input, assert_output_dir_exist
from neurostatx.io.loader import GraphLoader
from neurostatx.network.utils import extract_subject_percentile
from neurostatx.network.viz import creating_node_colormap


# Initializing the app.
app = App(default_parameter=Parameter(negative=()))
"""Cyclopts application for the CompareGraphs command-line tool."""


@app.default()
def CompareGraphs(
    in_graph1: Annotated[
        str,
        Parameter(
            show_default=False,
            group="Essential Files Options",
        ),
    ],
    weight: Annotated[
        str,
        Parameter(
            show_default=False,
            group="Essential Files Options",
        ),
    ],
    percentile: Annotated[
        float,
        Parameter(
            show_default=False,
            group="Essential Files Options",
        ),
    ],
    in_graph2: Annotated[
        str,
        Parameter(
            show_default=False,
            group="Essential Files Options",
        ),
    ],
    out_folder: Annotated[
        str,
        Parameter(
            group="Essential Files Options",
        ),
    ] = "./comparison_results/",
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
    background_alpha: Annotated[
        bool,
        Parameter(
            show_default=True,
            group="Network Visualization Options",
        ),
    ] = True,
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
    ] = "gray",
    legend_title: Annotated[
        str,
        Parameter(
            show_default=False,
            group="Network Visualization Options",
        ),
    ] = "Membership values",
):
    """Extract high-percentile nodes from one graph and color them on another.

    CompareGraphs compares two undirected weighted graph networks by extracting
    the Xth-percentile nodes from --in-graph1 and labelling those nodes on
    --in-graph2. The percentile is set by --percentile.

    Notes
    -----

    **Graph appearance**

    Graph appearance can be customized with the visualization parameters
    below. Using subjects labelling will crowd the network if it contains a
    high number of nodes. Centroids are labelled by default ``c1, c2, ...``
    and subjects ``s1, s2, ...``.

    Parameters
    ----------
    in_graph1 : str
        First graph from which subjects above --percentile will be extracted
        and colored.
    weight : str
        Edge weight to use for the graph.
    percentile : float
        Percentile value used to extract subjects.
    in_graph2 : str
        Second graph to color extracted subjects on.
    out_folder : str, optional
        Path of the folder in which the results will be written. If not
        specified, current folder and default name will be used. Defaults to
        ``./comparison_results/``.
    verbose : bool, optional
        If true, produce verbose output. Defaults to False.
    overwrite : bool, optional
        If true, force overwriting of existing output files. Defaults to False.
    label_centroids : bool, optional
        If true, centroids will be labelled. Defaults to True.
    label_subjects : bool, optional
        If true, will label subjects nodes. Defaults to False.
    centroids_size : int, optional
        Size of the centroids nodes. Defaults to 500.
    centroid_alpha : float, optional
        Alpha value representing the transparency of the centroids nodes.
        Defaults to 1.
    centroid_node_color : str, optional
        Centroids nodes color to use. Defaults to ``white``.
    centroid_edge_color : str, optional
        Assign a color to the edge of the centroids nodes. Defaults to
        ``black``.
    subject_node_size : int, optional
        Assign the size of the subjects nodes. Defaults to 5.
    background_alpha : bool, optional
        If true, will plot the background nodes with alpha = 0.2. Defaults to
        True.
    subject_edge_color : str, optional
        Assign a color to the edge of the subjects nodes. Defaults to None.
    colormap : str, optional
        Colormap to use when coloring the edges of the network based on the
        membership values of each cluster. Available colormaps are those from
        Matplotlib
        (https://matplotlib.org/stable/tutorials/colors/colormaps.html).
        Defaults to ``gray``.
    legend_title : str, optional
        Legend title (colormap). Defaults to ``Membership values``.

    Examples
    --------
    ```bash
    CompareGraphs --in-graph1 graph1.gml --in-matrix membership_mat.npy
    --percentile 80 --in-graph2 graph2.gml
    ```
    """

    if verbose:
        logging.getLogger().setLevel(logging.INFO)
        coloredlogs.install(level=logging.INFO)

    assert_input(in_graph1, in_graph2)
    assert_output_dir_exist(overwrite, out_folder, create_dir=True)

    # Loading membership matrix.
    logging.info("Loading graphs.")
    graph1 = GraphLoader().load_graph(in_graph1)
    graph2 = GraphLoader().load_graph(in_graph2)

    # Fetch data from graph #1.
    mat = graph1.fetch_edge_data(weight=weight)

    # Extracting percentiles.
    logging.info("Extracting percentiles.")
    # Extracting the Xth percentile subjects.
    percentile_dict = extract_subject_percentile(mat.get_data().values.T,
                                                 percentile)

    # Mapping the nodes' cmap.
    nodes_cmap = creating_node_colormap(percentile_dict)

    # Creating the alpha for subject's that are not in the Xth percentile.
    if background_alpha:
        sub_alpha = []
        for i in nodes_cmap:
            if i is str:
                sub_alpha.append(0.2)
            else:
                sub_alpha.append(1)
    else:
        sub_alpha = np.array([1] * mat.values.shape[0])

    logging.info("Visualizing percentiles on the 1st graph.")
    graph1.visualize(
        output=f"{out_folder}/graph1.png",
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
        colormap=colormap,
        title="Graph Network #1",
        legend_title=legend_title,
    )

    logging.info("Visualizing percentiles on the 2nd graph.")
    graph2.visualize(
        output=f"{out_folder}/graph2.png",
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
        colormap=colormap,
        title="Graph Network #2",
        legend_title=legend_title,
    )


if __name__ == "__main__":
    app()
