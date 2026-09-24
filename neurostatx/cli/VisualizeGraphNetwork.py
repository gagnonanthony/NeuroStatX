#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import coloredlogs
import logging

from cyclopts import App, Parameter
import numpy as np
from typing import List
from typing_extensions import Annotated

from neurostatx.io.utils import (assert_input, assert_output_dir_exist)
from neurostatx.io.loader import GraphLoader
from neurostatx.network.viz import create_cmap_from_list


# Initializing the app.
app = App(default_parameter=Parameter(negative=()))
"""Cyclopts application for the VisualizeGraphNetwork command-line tool."""


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
    label_name: Annotated[
        List[str],
        Parameter(
            show_default=False,
            group="Label Options",
        ),
    ] = None,
    cohort: Annotated[
        int,
        Parameter(
            show_default=False,
            group="Essential Files Options",
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
    ] = 0.1,
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
    """Visualize a graph network generated from clustering results.

    VisualizeGraphNetwork uses NetworkX to draw the graph and can label
    specific nodes based on a condition (for example a diagnosis). Appearance
    can be customized with the visualization parameters. It is meant to work
    with
    [ComputeGraphNetwork][neurostatx.cli.ComputeGraphNetwork.ComputeGraphNetwork]
    because it requires precomputed node positions in the graph file.

    Notes
    -----

    **Labelling nodes**

    Specific nodes can be labelled from a binary node attribute via
    --label-name. Multiple labels can be provided with repeated --label-name
    flags; the script writes one graph per label name.

    **Cohort selection**

    When the graph contains data from multiple cohorts, --cohort fetches the
    specified cohort and filters --label-name. The graph must contain a cohort
    attribute; add it with
    [AddNodesAttributes][neurostatx.cli.AddNodesAttributes.AddNodesAttributes]
    if needed.

    **Graph appearance**

    Using subjects labelling will crowd the network if it contains a high
    number of nodes. Centroids are labelled by default ``c1, c2, ...`` and
    subjects ``s1, s2, ...``. The script also exports a
    ``graph_network_file.gexf`` file that can be customized further in other
    tools such as Gephi [1].

    References
    ----------
    [1] [Gephi](https://gephi.org/)

    Parameters
    ----------
    in_graph : str
        Input graph network to visualize (.gml format).
    out_folder : str, optional
        Output folder for the graph network visualization. Defaults to
        ``./visualize_network/``.
    verbose : bool, optional
        Verbose mode. Defaults to False.
    overwrite : bool, optional
        Overwrite existing files. Defaults to False.
    save_parameters : bool, optional
        Save parameters to a .txt file. Defaults to False.
    label_name : List[str], optional
        List of label names to subsequently use for labelling. Defaults to
        None.
    cohort : int, optional
        Cohort identifier. If your graph contains data from multiple cohorts,
        you can specify the cohort you want for visualization. Defaults to
        None.
    background_alpha : bool, optional
        Use background alpha for the graph. Defaults to True.
    weight : str, optional
        Weight to use for the graph network. Defaults to ``membership``.
    label_centroids : bool, optional
        Label centroids. Defaults to True.
    label_subjects : bool, optional
        Label subjects. Defaults to False.
    centroids_size : int, optional
        Size of the centroids. Defaults to 500.
    centroid_alpha : float, optional
        Alpha of the centroids. Defaults to 1.
    centroid_node_color : str, optional
        Color of the centroids. Defaults to ``white``.
    centroid_edge_color : str, optional
        Edge color of the centroids. Defaults to ``black``.
    subject_node_size : int, optional
        Size of the subjects. Defaults to 5.
    subject_node_alpha : float, optional
        Alpha of the subjects. Defaults to 0.1.
    subject_node_color : str, optional
        Color of the subjects. Defaults to ``darkgrey``.
    subject_edge_color : str, optional
        Edge color of the subjects. Defaults to None.
    colormap : str, optional
        Colormap to use for the graph network. Defaults to ``plasma``.
    title : str, optional
        Title of the graph network. Defaults to ``Network Graph of the
        clustering membership values.``.
    legend_title : str, optional
        Legend title of the graph network. Defaults to ``Membership values``.

    Examples
    --------
    ```bash
    VisualizeGraphNetwork --in-graph graph_network.gml
    --out-folder output/ --label-name diagnosis
    --weight membership -v -f -s
    ```
    """

    if verbose:
        logging.getLogger().setLevel(logging.INFO)
        coloredlogs.install(level=logging.INFO)

    assert_input(in_graph)
    assert_output_dir_exist(overwrite, out_folder, create_dir=True)

    # Saving parameters
    if save_parameters:
        parameters = list(locals().items())
        with open(f"{out_folder}/nodes_attributes_parameters.txt", "w+") as f:
            for param in parameters:
                f.writelines(str(param))

    # Loading graph.
    logging.info("Loading graph data.")
    G = GraphLoader().load_graph(in_graph)

    # Visualizing and saving network.
    logging.info("Generating graph.")
    G.visualize(
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
    if label_name is not None:

        logging.info("Constructing graph(s) with custom labels.")

        if cohort is not None:
            label_name.append('cohort')

        # Loading df.
        df = G.fetch_attributes_df(label_name).get_data()

        # Fetching data for label as array.
        for label in label_name:
            if label == 'cohort':
                continue

            # If subject is not within the specified cohort, impute 0.
            if cohort is not None:
                df.loc[:, label] = np.where(df.loc[:, label] == 1,
                                            np.where(
                                                df.loc[:, 'cohort'] == cohort,
                                                1,
                                                0),
                                            0)
            labels = df[label]

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

            G.visualize(
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
                colormap=colormap,
                title=f"{title} with {label} subjects colored.",
                legend_title=legend_title,
            )


if __name__ == "__main__":
    app()
