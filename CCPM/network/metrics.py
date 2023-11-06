# -*- coding: utf-8 -*-

from collections import OrderedDict

import networkx as nx


def get_metrics_ops():
    """
    Get a dictionary of all functions related to graph network metrics.

    Returns:
        OrderedDict:    Functions dictonary.
    """
    return OrderedDict(
        [
            ("eigencentrality", eigencentrality),
            ("closenesscentrality", closenesscentrality),
            ("betweennesscentrality", betweennesscentrality),
            ("informationcentrality", informationcentrality),
            ("currentflowbc", currentflowbc),
            ("loadcentrality", loadcentrality),
            ("harmoniccentrality", harmoniccentrality),
            ("eccentricity", eccentricity),
            ("clustering", clustering),
            ("constraint", constraint),
            ("effectivesize", effectivesize),
            ("closenessvitality", closenessvitality),
            ("degree", degree),
        ]
    )


def get_metrics_docs(ops: dict):
    """
    Function to fetch from a dictionary all functions and join all
    documentations.

    Args:
        ops (dict):         Dictionary of functions.
    """
    full_doc = []
    for func in ops.values():
        full_doc.append(func.__doc__)
    return "\n".join(full_doc)


def eigencentrality(graph, weight=None):
    """
    eigencentrality: GRAPH WEIGHT\n
        Will return a dictionary of the eigenvector centrality for all nodes.
    """
    return nx.eigenvector_centrality(graph, max_iter=1000, weight=weight)


def closenesscentrality(graph, weight=None):
    """
    closenesscentrality: GRAPH WEIGHT\n
        Will return a dictionary of the closeness centrality for all nodes.
    """
    return nx.closeness_centrality(graph, distance=weight, wf_improved=True)


def betweennesscentrality(graph, weight=None):
    """
    betweennesscentrality: GRAPH WEIGHT\n
        Will return a dictionary of the betweenness centrality for all nodes.
    """
    return nx.betweenness_centrality(graph, weight=weight)


def informationcentrality(graph, weight=None):
    """
    informationcentrality: GRAPH WEIGHT\n
        Will return a dictionary of the information centrality for all nodes.
    """
    return nx.information_centrality(graph, weight=weight, solver="full")


def currentflowbc(graph, weight=None):
    """
    currentflowbc: GRAPH WEIGHT\n
        Will return a dictionary of the current flow betweenness centrality
        for all nodes.
    """
    return nx.current_flow_betweenness_centrality(graph, weight=weight,
                                                  solver="full")


def loadcentrality(graph, weight=None):
    """
    loadcentrality: GRAPH WEIGHT\n
        Will return a dictionary of the load centrality for all nodes.
    """
    return nx.load_centrality(graph, weight=weight)


def harmoniccentrality(graph, weight=None):
    """
    harmoniccentrality: GRAPH WEIGHT\n
        Will return a dictionary of the harmonic centrality for all nodes.
    """
    return nx.harmonic_centrality(graph, distance=weight)


def eccentricity(graph, weight=None):
    """
    eccentricity: GRAPH WEIGHT\n
        Will return a dictionary of the eccentricity for all nodes.
    """
    return nx.eccentricity(graph, weight=weight)


def clustering(graph, weight=None):
    """
    clustering: GRAPH WEIGHT\n
        Will return a dictionary of the clustering coefficient for all nodes.
    """
    return nx.clustering(graph, weight=weight)


def constraint(graph, nodes, weight=None):
    """
    constraint: GRAPH NODES WEIGHT\n
        Will return a dictionary of the constraint for all specified nodes.
    """
    return nx.constraint(graph, nodes=[nodes], weight=weight)


def effectivesize(graph, nodes, weight=None):
    """
    effectivesize: GRAPH NODES WEIGHT\n
        Will return a dictionary of the effective size for all specified nodes.
    """
    return nx.effective_size(graph, nodes=[nodes], weight=weight)


def closenessvitality(graph, nodes, weight=None):
    """
    closenessvitality: GRAPH NODE WEIGHT\n
        Will return a dictionary of the closeness vitality for a single node.
    """
    return nx.closeness_vitality(graph, node=nodes, weight=weight)


def degree(graph, weight=None):
    """
    degree: GRAPH WEIGHT\n
        Will return the degree of the specified node.
    """
    return graph.degree(weight=weight)
