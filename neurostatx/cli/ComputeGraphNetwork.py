#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import coloredlogs
import logging

from cyclopts import App, Parameter
from typing_extensions import Annotated

from neurostatx.io.utils import assert_input, assert_output_dir_exist
from neurostatx.io.loader import DatasetLoader, GraphLoader
from neurostatx.network.utils import (get_nodes_and_edges,
                                      construct_attributes_dict)
from neurostatx.network.viz import (
    membership_distribution,
    NetworkLayout)


# Initializing the app.
app = App(default_parameter=Parameter(negative=()))
"""Cyclopts application for the ComputeGraphNetwork command-line tool."""


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
    method: Annotated[
        str,
        Parameter(
            show_default=True,
            group="Layout Options",
        ),
    ] = "force",
    seed: Annotated[
        int,
        Parameter(
            show_default=True,
            group="Layout Options",
        ),
    ] = 42,
    weight: Annotated[
        str,
        Parameter(
            show_default=True,
            group="Layout Options",
        ),
    ] = "membership",
):
    """Build an undirected weighted graph from a fuzzy membership matrix.

    ComputeGraphNetwork is designed to work seamlessly with
    [FuzzyClustering][neurostatx.cli.FuzzyClustering.FuzzyClustering]. Mapping
    membership matrices to a graph allows later use of graph-theory statistics
    such as shortest path and betweenness centrality.

    Notes
    -----

    **Layout algorithms**

    Node positions are determined from their connections to other nodes (and
    the weight of those connections). Those connections are also called edges
    and carry a weight in a weighted graph. Available layout algorithms are:

    * Kamada-Kawai uses the Kamada-Kawai path-length cost function. It is
      not optimal for large networks because it is computationally intensive.
    * Spectral layout determines position using the eigenvectors of the graph
      Laplacian.
    * Spring layout uses the Fruchterman-Reingold force-directed algorithm. It
      is suitable for large networks with a high number of nodes and is the
      default method.

    For large graphs (~10 000 nodes), a spring layout run can take about 5
    minutes depending on hardware. Layout details are documented in the
    NetworkX drawing reference [2].

    **Importing data**

    If --import-data is set, descriptive data are stored as node attributes in
    the .gml file. This is useful for later visualization or statistical
    analysis (see
    [AverageWeightedPath][neurostatx.cli.AverageWeightedPath.AverageWeightedPath]
    or
    [PartialLeastSquareRegression][neurostatx.cli.PartialLeastSquareRegression.PartialLeastSquareRegression])
    and reduces the chance of subject mismatch.

    The concept of this script was initially proposed in [1].

    References
    ----------
    [1] [Ariza-Jiménez, L., Villa, L. F., & Quintero, O. L. (2019). Memberships Networks for High-Dimensional Fuzzy Clustering Visualization](https://doi.org/10.1007/978-3-030-31019-6_23)

    [2] [NetworkX drawing documentation](https://networkx.org/documentation/stable/reference/drawing.html)

    Parameters
    ----------
    in_dataset : str
        Input dataset containing membership values for each cluster.
    id_column : str
        Name of the column containing the subject's ID tag. Required for
        proper handling of IDs.
    desc_columns : int
        Number of descriptive columns at the beginning of the dataset.
    out_folder : str, optional
        Path of the folder in which the results will be written. If not
        specified, current folder and default name will be used. Defaults to
        ``./graph_results/``.
    verbose : bool, optional
        If true, produce verbose output. Defaults to False.
    overwrite : bool, optional
        If true, force overwriting of existing output files. Defaults to False.
    save_parameters : bool, optional
        If true, save the parameters used in a .txt file. Defaults to False.
    plot_distribution : bool, optional
        If true, will plot the membership distribution and delta. Defaults to
        False.
    import_data : bool, optional
        If true, will import the data from the input dataset within the graph
        network file. Defaults to False.
    layout : NetworkLayout, optional
        Layout algorithm to determine the nodes position. Defaults to Spring.
    method : str, optional
        Layout method passed to the NetworkX layout algorithm. Defaults to
        ``force``.
    seed : int, optional
        Random seed used when computing node positions. Defaults to 42.
    weight : str, optional
        Name of the column containing the edge weight. Defaults to
        ``membership``.

    Examples
    --------
    ```bash
    ComputeGraphNetwork --in-dataset cluster_membership.xlsx
    --id-column subjectkey --desc-columns 1 --out-folder output/
    ```
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
    raw_df = DatasetLoader().load_data(in_dataset)
    descriptive_columns = [n for n in range(0, desc_columns)]

    # Creating the array.
    desc_data = raw_df.get_descriptive_columns(descriptive_columns)
    raw_df.drop_columns(descriptive_columns)
    raw_df.join(desc_data[desc_data.columns[0]], left=True)

    # Plotting membership distributions and delta.
    if plot_distribution:
        membership_distribution(
            raw_df.get_data().values[:, 1:],
            output=f"{out_folder}/membership_distribution.png"
        )

    # Fetching dataframe of nodes and edges.
    df, _, _ = raw_df.custom_function(
        get_nodes_and_edges,
        edge_attr=weight
    )

    # Creating network graph.
    G = GraphLoader().build_graph(
        df,
        "node1",
        "node2",
        edge_attr=weight
    )

    # Computing graph network layout.
    logging.info("Computing graph network layout and setting nodes position.")
    G.layout(layout=layout, weight=weight, method=method, seed=seed)

    if import_data:
        logging.info("Importing data within the .gml file.")
        attributes = construct_attributes_dict(desc_data,
                                               desc_data.columns[1:],
                                               id_column)
        G.add_node_attribute(attributes)

    # Saving graph as a .gexf object for easy reloading.
    G.save_graph(f"{out_folder}/network_graph_file.gml")


if __name__ == "__main__":
    app()
