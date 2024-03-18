# -*- coding: utf-8 -*-

import networkx as nx
import numpy as np
import pandas as pd


def get_nodes_and_edges(df):
    """
    Function to generate a dataframe containing edges' data.

    Args:
        df (DataFrame):          Pandas DataFrame containing edges data and
                                 ids (membership
                            matrix from clustering results).

    Returns:
        Pandas DataFrame of starting node, target node and edge weights.
    """

    center_list = [f"c{i+1}" for i in range(0, len(df.columns) - 1)]
    subject_list = df[df.columns[0]]

    start_list = np.repeat(subject_list, len(df.columns) - 1)
    target_list = center_list * len(df)

    membership_data = df.drop(df.columns[0], axis=1,
                              inplace=False).astype("float")

    # Create a DataFrame Object.
    df = pd.DataFrame(
        {
            "node1": start_list,
            "node2": target_list,
            "membership": membership_data.values.flatten(),
        }
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


def extract_subject_percentile(mat, percentile):
    """
    Function to extract subject

    Args:
        mat (Array):            Fuzzy C-partitioned membership matrix.
        percentile (float):     Percentile value.
    Return:
        label_dict:             Dictionary of binary arrays for each clusters.
    """

    # Fetching 1st and 2nd highest membership value.
    high1st = np.max(mat, axis=0)
    high2nd = np.partition(mat, -2, axis=0)[-2, :]
    delta = high1st - high2nd

    # Computing value for the Xth percentile.
    value = np.percentile(delta, percentile)

    # Labelling subjects that are over the Xth percentile value.
    label_dict = {}
    for i in range(mat.shape[0]):
        label_dict[f"c{i+1}"] = np.where(mat[i, :] > value, i + 1, 0)

    return label_dict


def construct_attributes_dict(df, labels, id_column):
    """
    Function to construct a dictionary of nodes' attributes from a DataFrame.

    Args:
        df (pd.DataFrame):      Pandas DataFrame containing nodes' attributes.
        labels (List):          List of labels to add as nodes' attributes.
        id_column (str):        Name of the column containing the subject's ID
                                tag.

    Returns:
        attributes_dict:        Dictionary of nodes' attributes.
    """

    # Set index to id_column.
    df.set_index(id_column, inplace=True)

    # Keeping only columns specified in labels.
    data_to_add = df[labels]

    # Transform to dictionary.
    attributes_dict = data_to_add.to_dict(orient="index")

    return attributes_dict


def fetch_attributes_df(G, attributes=None):
    """
    Function to fetch nodes' attributes from a graph as a DataFrame.

    Args:
        G (NetworkX Graph):     NetworkX Graph object.
        attributes (List):      List of attributes to fetch.

    Returns:
        df: pd.DataFrame        Pandas DataFrame containing nodes' attributes.
    """

    # Filter out nodes that are not subjects.
    sub_node = nx.subgraph_view(G, filter_node_subjects)
    d = {n: G.nodes[n] for n in sub_node}

    # Filter for selected attributes.
    if len(attributes) > 0:
        d = {k: {k2: v2 for k2, v2 in v.items() if k2 in attributes}
             for k, v in d.items()}
    else:
        d = {k: {k2: v2 for k2, v2 in v.items() if k2 != 'label'}
             for k, v in d.items()}

    # Create df.
    df = pd.DataFrame.from_dict(d, orient="index")

    return df


def fetch_edge_data(G, weight='membership'):
    """
    Function to fetch edge's data from a graph as a DataFrame. This method
    works only if the graph as been created via this package.

    Args:
        G (_type_): _description_
        weight (str, optional): _description_. Defaults to 'membership'.
    """

    # Fetch the number of cluster.
    cntr_node = nx.subgraph_view(G, filter_node_centroids)
    sub_node = nx.subgraph_view(G, filter_node_subjects)

    # Get adjacency matrix.
    adj = np.delete(
        nx.to_numpy_array(G, weight=weight),
        [i for i in range(1, len(cntr_node) + 1)],
        axis=0)
    df = pd.DataFrame(adj[:, 1:(len(cntr_node) + 1)], index=sub_node,
                      columns=[f'Cluster {i+1}' for i in range(len(cntr_node))]
                      )

    return df
