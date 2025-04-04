# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd


def get_nodes_and_edges(df):
    """
    Function to generate a dataframe containing edges' data.

    Parameters
    ----------
    df : DataFrame
        Pandas DataFrame containing edges data and
        ids (membership matrix from clustering results).

    Returns
    -------
    DataFrame
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


def extract_subject_percentile(mat, percentile):
    """
    Function to extract subjects that are above the Xth percentile.

    Parameters
    ----------
    mat : Array
        Fuzzy C-partitioned membership matrix.
    percentile : float
        Percentile value.

    Returns
    -------
    label_dict
        Dictionary of binary arrays for each clusters.
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

    Parameters
    ----------
    df : DataFrame
        Pandas DataFrame containing nodes' attributes.
    labels : List
        List of labels to add as nodes' attributes.
    id_column : str
        Name of the column containing the subject's ID tag.

    Returns
    -------
    attributes_dict
        Dictionary of nodes' attributes.
    """

    # Set index to id_column.
    df.set_index(id_column, inplace=True)

    # Keeping only columns specified in labels.
    data_to_add = df[labels]

    # Transform to dictionary.
    attributes_dict = data_to_add.to_dict(orient="index")

    return attributes_dict
