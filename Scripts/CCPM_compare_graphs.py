#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import coloredlogs
import logging

import networkx as nx
import numpy as np
import typer
from typing_extensions import Annotated

from CCPM.io.utils import (assert_input,
                           assert_output_dir_exist)
from CCPM.network.utils import extract_subject_percentile
from CCPM.network.viz import (visualize_network,
                              creating_node_colormap,
                              NetworkLayout)


# Initializing the app.
app = typer.Typer(add_completion=False)

@app.command()
def main(
        in_graph1: Annotated[str, typer.Option(help='1st graph from which subjects above --percentile will be extracted and colored.',
                                               show_default=False,
                                               rich_help_panel='Essential Files Options')],
        in_matrix: Annotated[str, typer.Option(help='Numpy array containing the fuzzy clustering membership values (.npy).',
                                               show_default=False,
                                               rich_help_panel='Essential Files Options')],
        percentile: Annotated[float, typer.Option(help='Percentile value used to extract subjects.',
                                                  show_default=False,
                                                  rich_help_panel='Essential Files Options')],
        in_graph2: Annotated[str, typer.Option(help='2nd graph to color extracted subjects on.',
                                               show_default=False,
                                               rich_help_panel='Essential Files Options')],
        out_folder: Annotated[str, typer.Option(help='Path of the folder in which the results will be written. '
                                                     'If not specified, current folder and default '
                                                     'name will be used.',
                                                rich_help_panel='Essential Files Options')] = './comparison_results/',
        verbose: Annotated[bool, typer.Option('-v', '--verbose', help='If true, produce verbose output.',
                                              rich_help_panel="Optional parameters")] = False,
        overwrite: Annotated[bool, typer.Option('-f', '--overwrite', help='If true, force overwriting of existing '
                                                                          'output files.',
                                                rich_help_panel="Optional parameters")] = False,
        layout: Annotated[NetworkLayout, typer.Option(help='Layout algorithm to determine the nodes position.',
                                                      show_default=True,
                                                      show_choices=True,
                                                      rich_help_panel='Network Visualization Options')] = NetworkLayout.Spring,
        label_centroids: Annotated[bool, typer.Option(help='If true, centroids will be labelled.',
                                                      show_default=True,
                                                      rich_help_panel='Network Visualization Options')] = True,
        label_subjects: Annotated[bool, typer.Option(help='If true, will label subjects nodes.',
                                                     show_default=True, 
                                                     rich_help_panel='Network Visualization Options')] = False,
        centroids_size: Annotated[int, typer.Option(help='Size of the centroids nodes.',
                                                    show_default=True,
                                                    rich_help_panel='Network Visualization Options')] = 500,
        centroid_alpha: Annotated[float, typer.Option(help='Alpha value representing the transparency of the centroids '
                                                      'nodes.',
                                                      show_default=True,
                                                      rich_help_panel='Network Visualization Options')] = 1,
        centroid_node_color: Annotated[str, typer.Option(help='Centroids nodes color to use.',
                                                         show_default=True,
                                                         rich_help_panel='Network Visualization Options')] = 'white',
        centroid_edge_color: Annotated[str, typer.Option(help='Assign a color to the edge of the centroids nodes.',
                                                         show_default=True,
                                                         rich_help_panel='Network Visualization Options')] = 'black',
        subject_node_size: Annotated[int, typer.Option(help='Assign the size of the subjects nodes.',
                                                       show_default=True,
                                                       rich_help_panel='Network Visualization Options')] = 5,
        background_alpha: Annotated[bool, typer.Option(help='If true, will plot the background nodes with alpha = 0.2.',
                                                          show_default=True,
                                                          rich_help_panel='Network Visualization Options')] = True,
        subject_edge_color: Annotated[str, typer.Option(help='Assign a color to the edge of the subjects nodes.',
                                                        show_default=True,
                                                        rich_help_panel='Network Visualization Options')] = None,
        colormap: Annotated[str, typer.Option(help='Colormap to use when coloring the edges of the network based on the membership'
                                              'values to each clusters. Available colormap are those from plt.cm.',
                                              show_default=True,
                                              rich_help_panel='Network Visualization Options')] = 'gray',
        legend_title: Annotated[str, typer.Option(help='Legend title (colormap).',
                                                  show_default=False,
                                                  rich_help_panel='Network Visualization Options')] = 'Membership values'):

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
    CCPM_compare_graphs.py is a script that compares 2 undirected weighted graph network. As of now,
    the only comparison implemented is the extraction of the Xth percentile nodes from --in-graph1 and 
    label those nodes on --in-graph2. It is essential to provide the membership matrix used to create
    the graph #1 in order to extract the percentile data. In future release, this will be done directly
    from the graph file (hopefully).
    \b
    LAYOUT ALGORITHMS
    -----------------
    In order to generate a graph network, the nodes positions need to be determined in
    relation with their connections to other nodes (and the weigth of those connections).
    Those connections are also called edges and contain a weight in the case of a weighted
    graph network. Possible algorithms to choose from are :
    \b
    Kamada Kawai Layout: Use the Kamada-Kawai path-length cost-function. Not the optimal solution
                         for large network as it is computer intensive. For details, see [2].
    Spectral Layout: Position is determined using the eigenvectors of the graph Laplacian. 
                     For details, see [2].
    Spring Layout: Use the Fruchterman-Reingold force-directed algorithm. Suitable for large
                   network with high number of nodes. For details, see [2]. This is the default
                   method. 
    \b
    GRAPH NETWORK CUSTOMIZATION
    ---------------------------
    To customize the graph appearance, please see the Network Visualization Options below. It
    should be noted that using subjects_labelling will crowd the network if it contains a high
    number of nodes. Also, centroids are labelled by default 'c1, c2, ...' and subjects 's1, s2,
    ...'.
    \b
    EXAMPLE USAGE
    -------------
    CCPM_compare_graphs.py --in-graph1 graph1.gexf --in-matrix membership_mat.npy --percentile 80
        --in-graph2 graph2.gexf
    ** For large graphs (~10 000 nodes), it might take ~5 mins to run using the spring layout and
       depending on your hardware. **
    """
    
    if verbose:
        logging.getLogger().setLevel(logging.INFO)
        coloredlogs.install(level=logging.INFO)

    assert_input(in_matrix)
    assert_output_dir_exist(overwrite, out_folder, create_dir=True)
    
    # Loading membership matrix.
    logging.info("Loading graphs and membership matrix.")
    mat = np.load(in_matrix)
    graph1 = nx.read_gexf(in_graph1)
    graph2 = nx.read_gexf(in_graph2)
    
    # Extracting percentiles. 
    logging.info("Extracting percentiles.")
    # Extracting the Xth percentile subjects. 
    percentile_dict = extract_subject_percentile(mat, percentile)
    
    # Mapping the nodes' cmap. 
    nodes_cmap = creating_node_colormap(percentile_dict)
    
    # Creating the alpha for subject's that are not in the Xth percentile.
    if background_alpha:
        sub_alpha = []
        for i in nodes_cmap:
            if type(i) == str:
                sub_alpha.append(0.2)
            else:
                sub_alpha.append(1)
    else:
        sub_alpha = np.array([1] * mat.shape[1])
            
    logging.info('Visualizing percentiles on the 1st graph.')
    _ = visualize_network(graph1, output=f'{out_folder}/graph1.png',
                      layout=getattr(nx, layout),
                      weight='membership',
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
                      title='Graph Network #1',
                      legend_title=legend_title)
    
    logging.info('Visualizing percentiles on the 2nd graph.')
    _ = visualize_network(graph2, output=f'{out_folder}/graph2.png',
                      layout=getattr(nx, layout),
                      weight='membership',
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
                      title='Graph Network #2',
                      legend_title=legend_title)
    

if __name__ == '__main__':
    app()
    