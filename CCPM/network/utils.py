# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd


def get_nodes_and_edges(df):
    """
    Function to generate a dataframe containing edges' data.

    Args:
        df (DataFrame):          Pandas DataFrame containing edges data and ids (membership
                            matrix from clustering results).

    Returns:
        Pandas DataFrame of starting node, target node and edge weights.
    """

    center_list = [f"c{i+1}" for i in range(0, len(df.columns)-1)]
    subject_list = df[df.columns[0]]

    start_list = np.repeat(subject_list, len(df.columns)-1)
    target_list = center_list * len(df)

    membership_data = df.drop(df.columns[0], axis=1, inplace=False).astype('float')

    # Create a DataFrame Object.
    df = pd.DataFrame(
        {"node1": start_list, "node2": target_list,
         "membership": membership_data.values.flatten()}
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
        label_dict[f'c{i+1}'] = np.where(mat[i, :] > value, i+1, 0)
    
    return label_dict
    