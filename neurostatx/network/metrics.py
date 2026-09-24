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
    """Return the mapping of metric names to callable implementations.

    Returns
    -------
    metrics : dict
        Ordered mapping of metric name to function.

    Examples
    --------
    >>> from neurostatx.network.metrics import get_metrics_ops
    >>> "eigencentrality" in get_metrics_ops()
    True
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


def eigencentrality(graph, weight=None):
    """Compute eigenvector centrality for every node.

    Parameters
    ----------
    graph : networkx.Graph
        Input graph.
    weight : str, optional
        Edge attribute used as weight. Defaults to None.

    Returns
    -------
    scores : dict
        Eigenvector centrality for each node.

    Examples
    --------
    >>> import networkx as nx
    >>> from neurostatx.network.metrics import eigencentrality
    >>> eigencentrality(nx.path_graph(3))
    """
    return nx.eigenvector_centrality(graph, max_iter=1000, weight=weight)


def closenesscentrality(graph, weight=None):
    """Compute closeness centrality for every node.

    Parameters
    ----------
    graph : networkx.Graph
        Input graph.
    weight : str, optional
        Edge attribute used as distance. Defaults to None.

    Returns
    -------
    scores : dict
        Closeness centrality for each node.

    Examples
    --------
    >>> import networkx as nx
    >>> from neurostatx.network.metrics import closenesscentrality
    >>> closenesscentrality(nx.path_graph(3))
    """
    return nx.closeness_centrality(graph, distance=weight, wf_improved=True)


def betweennesscentrality(graph, weight=None):
    """Compute betweenness centrality for every node.

    Parameters
    ----------
    graph : networkx.Graph
        Input graph.
    weight : str, optional
        Edge attribute used as weight. Defaults to None.

    Returns
    -------
    scores : dict
        Betweenness centrality for each node.

    Examples
    --------
    >>> import networkx as nx
    >>> from neurostatx.network.metrics import betweennesscentrality
    >>> betweennesscentrality(nx.path_graph(3))
    """
    return nx.betweenness_centrality(graph, weight=weight)


def informationcentrality(graph, weight=None):
    """Compute information centrality for every node.

    Parameters
    ----------
    graph : networkx.Graph
        Input graph.
    weight : str, optional
        Edge attribute used as weight. Defaults to None.

    Returns
    -------
    scores : dict
        Information centrality for each node.

    Examples
    --------
    >>> import networkx as nx
    >>> from neurostatx.network.metrics import informationcentrality
    >>> informationcentrality(nx.path_graph(3))
    """
    return nx.information_centrality(graph, weight=weight, solver="full")


def currentflowbc(graph, weight=None):
    """Compute current-flow betweenness centrality for every node.

    Parameters
    ----------
    graph : networkx.Graph
        Input graph.
    weight : str, optional
        Edge attribute used as weight. Defaults to None.

    Returns
    -------
    scores : dict
        Current-flow betweenness centrality for each node.

    Examples
    --------
    >>> import networkx as nx
    >>> from neurostatx.network.metrics import currentflowbc
    >>> currentflowbc(nx.path_graph(3))
    """
    return nx.current_flow_betweenness_centrality(graph, weight=weight,
                                                  solver="full")


def loadcentrality(graph, weight=None):
    """Compute load centrality for every node.

    Parameters
    ----------
    graph : networkx.Graph
        Input graph.
    weight : str, optional
        Edge attribute used as weight. Defaults to None.

    Returns
    -------
    scores : dict
        Load centrality for each node.

    Examples
    --------
    >>> import networkx as nx
    >>> from neurostatx.network.metrics import loadcentrality
    >>> loadcentrality(nx.path_graph(3))
    """
    return nx.load_centrality(graph, weight=weight)


def harmoniccentrality(graph, weight=None):
    """Compute harmonic centrality for every node.

    Parameters
    ----------
    graph : networkx.Graph
        Input graph.
    weight : str, optional
        Edge attribute used as distance. Defaults to None.

    Returns
    -------
    scores : dict
        Harmonic centrality for each node.

    Examples
    --------
    >>> import networkx as nx
    >>> from neurostatx.network.metrics import harmoniccentrality
    >>> harmoniccentrality(nx.path_graph(3))
    """
    return nx.harmonic_centrality(graph, distance=weight)


def eccentricity(graph, weight=None):
    """Compute eccentricity for every node.

    Parameters
    ----------
    graph : networkx.Graph
        Input graph.
    weight : str, optional
        Edge attribute used as weight. Defaults to None.

    Returns
    -------
    scores : dict
        Eccentricity for each node.

    Examples
    --------
    >>> import networkx as nx
    >>> from neurostatx.network.metrics import eccentricity
    >>> eccentricity(nx.path_graph(3))
    """
    return nx.eccentricity(graph, weight=weight)


def clustering(graph, weight=None):
    """Compute the clustering coefficient for every node.

    Parameters
    ----------
    graph : networkx.Graph
        Input graph.
    weight : str, optional
        Edge attribute used as weight. Defaults to None.

    Returns
    -------
    scores : dict
        Clustering coefficient for each node.

    Examples
    --------
    >>> import networkx as nx
    >>> from neurostatx.network.metrics import clustering
    >>> clustering(nx.complete_graph(4))
    """
    return nx.clustering(graph, weight=weight)


def constraint(graph, nodes, weight=None):
    """Compute Burt's constraint for the requested nodes.

    Parameters
    ----------
    graph : networkx.Graph
        Input graph.
    nodes : str
        Space-separated node identifiers.
    weight : str, optional
        Edge attribute used as weight. Defaults to None.

    Returns
    -------
    scores : dict
        Constraint for each requested node.

    Examples
    --------
    >>> import networkx as nx
    >>> from neurostatx.network.metrics import constraint
    >>> constraint(nx.path_graph(3), nodes="0 1")
    """
    return nx.constraint(graph, nodes=nodes.split(), weight=weight)


def effectivesize(graph, nodes, weight=None):
    """Compute effective size for the requested nodes.

    Parameters
    ----------
    graph : networkx.Graph
        Input graph.
    nodes : str
        Space-separated node identifiers.
    weight : str, optional
        Edge attribute used as weight. Defaults to None.

    Returns
    -------
    scores : dict
        Effective size for each requested node.

    Examples
    --------
    >>> import networkx as nx
    >>> from neurostatx.network.metrics import effectivesize
    >>> effectivesize(nx.path_graph(3), nodes="0 1")
    """
    return nx.effective_size(graph, nodes=nodes.split(), weight=weight)


def closenessvitality(graph, nodes, weight=None):
    """Compute closeness vitality for a single node.

    Parameters
    ----------
    graph : networkx.Graph
        Input graph.
    nodes : str
        Node identifier passed to NetworkX as ``node``.
    weight : str, optional
        Edge attribute used as weight. Defaults to None.

    Returns
    -------
    score : float
        Closeness vitality of ``nodes``.

    Examples
    --------
    >>> import networkx as nx
    >>> from neurostatx.network.metrics import closenessvitality
    >>> closenessvitality(nx.path_graph(3), nodes=1)
    """
    return nx.closeness_vitality(graph, node=nodes, weight=weight)


def degree(graph, weight=None):
    """Return the degree of every node.

    Parameters
    ----------
    graph : networkx.Graph
        Input graph.
    weight : str, optional
        Edge attribute used as weight. Defaults to None.

    Returns
    -------
    degrees : networkx.DegreeView
        Degree of each node.

    Examples
    --------
    >>> import networkx as nx
    >>> from neurostatx.network.metrics import degree
    >>> dict(degree(nx.path_graph(3)))
    {0: 1, 1: 2, 2: 1}
    """
    return graph.degree(weight=weight)


class PathLengthsMethods(str, Enum):
    """Shortest-path algorithms accepted by NetworkX path-length helpers."""

    Dijkstra = ("dijkstra",)
    BellmanFord = ("bellman-ford",)
    FloydWarshall = ("floyd-warshall",)
    FloydWarshallNumpy = "floyd-warshall-numpy"


def weightedpath(
    graph,
    df,
    label_name,
    cohort=None,
    iterations=1000,
    weight=None,
    method="dijkstra",
    distribution=None,
    processes=1,
    verbose=False,
):
    """Average weighted path length for a group, with a permutation p-value.

    Parameters
    ----------
    graph : networkx.Graph
        Input graph.
    df : pandas.DataFrame
        Table of nodes. The index must match graph node labels.
    label_name : str
        Column containing the group label. Nodes with value 0 are excluded.
    cohort : str, optional
        If set, restrict ``df`` to this ``cohort`` value. Defaults to None.
    iterations : int, optional
        Number of null-distribution draws. Defaults to 1000.
    weight : str, optional
        Edge attribute used as weight. Defaults to None.
    method : str, optional
        Shortest-path algorithm. Defaults to ``"dijkstra"``.
    distribution : pd.DataFrame, optional
        Precomputed null distribution with a ``label_name`` column. Defaults
        to None.
    processes : int, optional
        Number of worker processes. Defaults to 1.
    verbose : bool, optional
        If True, show a progress bar. Defaults to False.

    Returns
    -------
    avg_path_length : float
        Average shortest-path length of the selected subgraph.
    dist : list
        Null distribution of path lengths.
    pvalue : float
        One-sided permutation p-value.

    Examples
    --------
    >>> import pandas as pd
    >>> import networkx as nx
    >>> from neurostatx.network.metrics import weightedpath
    >>> G = nx.path_graph(4)
    >>> df = pd.DataFrame({"group": [1, 1, 0, 0]})
    >>> avg, dist, p = weightedpath(
    ...     G, df, "group", iterations=5, processes=1
    ... )
    """

    # Cohort selection.
    if cohort is not None:
        subset = df.loc[df['cohort'] == cohort]
    else:
        subset = df.copy()

    # Setting lists.
    group_exclude = subset.loc[subset[label_name] == 0]
    nodes_exclude = group_exclude.index.to_list()
    nodes_include = [node for node in list(subset.index) if node
                     not in nodes_exclude]
    centroid_nodes = [node for node in list(graph) if 'c' in node]
    nodes_include = nodes_include + centroid_nodes

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
