# -*- coding: utf-8 -*-

from enum import Enum
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from CCPM.network.utils import (filter_node_centroids,
                                filter_node_subjects)


class NetworkLayout(str, Enum):
    KamadaKawai = 'kamada_kawai_layout',
    Spectral = 'spectral_layout',
    Spring = 'spring_layout'


def visualize_network(G, output, 
                      layout=nx.spring_layout, 
                      weight='weight',
                      centroids_labelling=True, 
                      subjects_labelling=False,
                      centroid_node_shape=500, 
                      centroid_alpha=1, 
                      centroid_node_color='white', 
                      centroid_edge_color='black',
                      subject_node_shape=5, 
                      subject_alpha=0.3, 
                      subject_node_color='black', 
                      subject_edge_color=None,
                      colormap='plasma',
                      title='Graph Network',
                      legend_title='Membership values'):
    
    # Computing optimal node's positions.
    pos = layout(G, weight=weight)
    
    # Fetching edges widths.
    widths = nx.get_edge_attributes(G, weight)
    
    # Sorting which nodes to label.
    labels = {}
    if centroids_labelling:
        for node in G.nodes():
            if 'c' in node:
                labels[node] = node
    elif centroids_labelling and subjects_labelling:
        for node in G.nodes():
            labels[node] = node
    else:
        for node in G.nodes():
            labels[node] = ''
    
    # Setting z-order of nodes.
    cntr_node = nx.subgraph_view(G, filter_node=filter_node_centroids)
    sub_node = nx.subgraph_view(G, filter_node=filter_node_subjects)
    
    # Centroids customization lists.
    cntr_shape = np.array([centroid_node_shape] * len(cntr_node.nodes()))
    cntr_alpha = np.array([centroid_alpha] * len(cntr_node.nodes()))
    
    # Subjects customization lists.
    sub_shape = np.array([subject_node_shape] * len(sub_node.nodes()))
    sub_alpha = np.array([subject_alpha] * len(sub_node.nodes()))
    
    # Plotting the graph.
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot()
    
    nodes1 = nx.draw_networkx_nodes(G, pos,
                                    nodelist=sub_node.nodes(),
                                    node_size=sub_shape,
                                    node_color=subject_node_color,
                                    alpha=sub_alpha,
                                    ax=ax)
    nodes2 = nx.draw_networkx_nodes(G, pos,
                                    nodelist=cntr_node.nodes(),
                                    node_size=cntr_shape,
                                    node_color=centroid_node_color,
                                    alpha=cntr_alpha,
                                    ax=ax)
    
    # Drawing edges.
    nx.draw_networkx_edges(G, pos,
                           edgelist=widths.keys(),
                           width=(list(widths.values())*10),
                           edge_color=list(widths.values()),
                           edge_cmap=getattr(plt.cm, colormap),
                           alpha=list(widths.values()),
                           ax=ax)
    
    # Setting z-order.
    nodes1.set_zorder(2)
    if subject_edge_color is not None:
        nodes1.set_edgecolor(subject_edge_color)
    nodes2.set_zorder(3)
    nodes2.set_edgecolor(centroid_edge_color)
    
    # Plotting labels if set.
    nx.draw_networkx_labels(G, pos,
                            labels=labels,
                            font_color='black',
                            ax=ax)
    
    # Adding colorbar, titles, etc.
    cmappable = ScalarMappable(Normalize(0,1), getattr(plt.cm, colormap))
    plt.colorbar(cmappable, ax=ax, location='right', shrink=0.5)
    
    plt.box(False)
    ax.set_title(title)
    #ax.legend(title=legend_title)
    
    plt.tight_layout()
    plt.savefig(output)
    