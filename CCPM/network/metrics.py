# -*- coding: utf-8 -*-

from collections import OrderedDict
from enum import Enum
import logging
import multiprocessing
import random

from functools import partial
import networkx as nx
import numpy as np
from p_tqdm import p_map


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
    return nx.constraint(graph, nodes=nodes.split(), weight=weight)


def effectivesize(graph, nodes, weight=None):
    """
    effectivesize: GRAPH NODES WEIGHT\n
        Will return a dictionary of the effective size for all specified nodes.
    """
    return nx.effective_size(graph, nodes=nodes.split(), weight=weight)


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


class PathLengthsMethods(str, Enum):
    Dijkstra = ("dijkstra",)
    BellmanFord = ("bellman-ford",)
    FloydWarshall = ("floyd-warshall",)
    FloydWarshallNumpy = "floyd-warshall-numpy"


def weightedpath(
    graph,
    df,
    label_name,
    iterations=1000,
    weight=None,
    method="dijkstra",
    distribution=None,
    processes=4,
    verbose=False,
):
    """
    Function to compute the average weighted shortest path length for a group
    of nodes. The function will also compute the p-value between the group of
    nodes and the randomly generated null distribution.

    Args:
        graph (Networkx.graph):                 Networkx graph object.
        df (pandas.DataFrame):                  Dataframe containing the nodes
        label_name (str):                       Name of the column containing
                                                the group label.
        iterations (int, optional):             Number of iterations to run.
                                                Defaults to 1000.
        weight (str, optional):                 Edge attributes to use as
                                                weight. Defaults to None.
        method (str, optional):                 Method to use for path
                                                computation. Defaults to
                                                "dijkstra".
        distribution (pd.DataFrame, optional):  Pre-computed distribution.
                                                Defaults to None.
        verbose (bool, optional):               Verbose flag. Defaults to
                                                False.

    Returns:
        avg_path_length (float):                Average path length.
        dist (list):                            Null distribution.
        pvalue (float):                         P-value.
    """
    # Setting lists.
    group_exclude = df.loc[df[label_name] == 0]
    nodes_exclude = group_exclude.index.to_list()
    nodes_include = [node for node in list(graph) if node not in nodes_exclude]

    logging.info("Computing weighted path for the set of nodes.")
    sub_G = nx.induced_subgraph(graph, nodes_include)
    avg_path_length = nx.average_shortest_path_length(
        sub_G, weight=weight, method=method
    )

    # Fetching all possible nodes.
    nodes_list = df.index.to_list()

    # Setting partial function to pass common arguments between iterations.
    if distribution is None:
        generate_null_dist = partial(
            _weightedpath,
            graph,
            nodes_list=nodes_list,
            sample_size=len(nodes_include),
            weight=weight,
            method=method,
        )

        # Opening multiprocessing pool.
        logging.info("Computing null distribution.")
        multiprocessing.set_start_method("spawn", force=True)
        pool = multiprocessing.Pool(processes=processes)

        # Initiating processing.
        if verbose:
            dist = p_map(generate_null_dist, range(0, iterations))
        else:
            dist = pool.map(generate_null_dist, range(0, iterations))
        pool.close()
        pool.join()
    else:
        dist = distribution[label_name].values

    # Compute p-value.
    pvalue = (
        (np.sum(np.array(dist) >= avg_path_length) + 1) / (iterations + 1)
    )

    return avg_path_length, dist, pvalue


def _weightedpath(
    graph, n_iter, nodes_list, sample_size, weight=None, method="dijkstra"
):
    """
    Core worker of weightedpath() function.
    """

    # Filtering nodes.
    random.shuffle(nodes_list)
    random_nodes = random.sample(nodes_list, sample_size)
    nodes_exclude = list(set(nodes_list) - set(random_nodes))

    # Copying graph.
    # orig = graph.copy()

    # Filtering original graph nodes to remove the ones that were not randomly
    # selected (easier to keep cluster centroids nodes that way.).
    # sub_G = nx.induced_subgraph(orig,
    #                            list(set(list(orig)) - set(nodes_exclude)))
    sub_G = graph.subgraph(list(set(graph) - set(nodes_exclude)))

    # Computing weighted path length.
    weighted_avg_path_length = nx.average_shortest_path_length(
        sub_G, weight=weight, method=method
    )

    return weighted_avg_path_length
