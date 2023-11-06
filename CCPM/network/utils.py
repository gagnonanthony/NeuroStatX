# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd


def get_nodes_and_edges(X):
    """
    Function to generate a dataframe containing edges' data.

    Args:
        X (Array):          Numpy array containing edges data (membership
                            matrix from clustering results).

    Returns:
        Pandas DataFrame of starting node, target node and edge weights.
    """

    center_list = [f"c{i+1}" for i in range(0, X.shape[0])]
    subject_list = [f"s{i+1}" for i in range(0, X.shape[1])]

    start_list = np.repeat(subject_list, X.shape[0])
    target_list = center_list * X.shape[1]

    # Create a DataFrame Object.
    df = pd.DataFrame(
        {"node1": start_list, "node2": target_list,
         "membership": X.T.flatten()}
    )

    return df, subject_list, center_list


def filter_node_centroids(n):
    """
    Function to filter cluster nodes from subject's nodes.

    Args:
        n (str):        Node label.

    Returns:
        True or False
    """

    return "c" in n


def filter_node_subjects(n):
    """
    Function to filter subject nodes from cluster's nodes.

    Args:
        n (str):        Node label.

    Returns:
        True or False
    """

    return "c" not in n
