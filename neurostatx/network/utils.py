# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd


def get_nodes_and_edges(df, edge_attr="membership"):
    """Build an edgelist from a subject-by-cluster membership table.

    Parameters
    ----------
    df : DataFrame
        Table whose first column is subject IDs and remaining columns are
        cluster memberships.
    edge_attr : str, optional
        Name of the weight column in the returned edgelist. Defaults to
        ``"membership"``.

    Returns
    -------
    edges : DataFrame
        Edgelist with ``node1``, ``node2``, and ``edge_attr``.
    subject_list : Series
        Subject identifiers from the first column.
    center_list : list
        Cluster-centroid labels (``c1``, ``c2``, ...).

    Examples
    --------
    >>> import pandas as pd
    >>> from neurostatx.network.utils import get_nodes_and_edges
    >>> df = pd.DataFrame({"id": ["s1", "s2"], "c1": [0.8, 0.2],
    ...                    "c2": [0.2, 0.8]})
    >>> edges, subjects, centers = get_nodes_and_edges(df)
    >>> list(centers)
    ['c1', 'c2']
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
            edge_attr: membership_data.values.flatten(),
        }
    )

    return df, subject_list, center_list


def extract_subject_percentile(mat, percentile):
    """Label subjects whose membership delta exceeds a percentile.

    Parameters
    ----------
    mat : array
        Fuzzy membership matrix of shape (n_clusters, n_samples).
    percentile : float
        Percentile of the first-minus-second membership gap.

    Returns
    -------
    label_dict : dict
        Mapping of cluster keys to arrays of cluster index or 0.

    Examples
    --------
    >>> import numpy as np
    >>> from neurostatx.network.utils import extract_subject_percentile
    >>> mat = np.array([[0.9, 0.2], [0.1, 0.8]])
    >>> labels = extract_subject_percentile(mat, 50)
    >>> sorted(labels)
    ['c1', 'c2']
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
    """Build a node-attribute dictionary from selected DataFrame columns.

    Parameters
    ----------
    df : DataFrame
        Table of node attributes.
    labels : list
        Column names to store as node attributes.
    id_column : str
        Column used as the node identifier.

    Returns
    -------
    attributes_dict : dict
        Mapping of node ID to attribute dictionary.

    Examples
    --------
    >>> import pandas as pd
    >>> from neurostatx.network.utils import construct_attributes_dict
    >>> df = pd.DataFrame({"id": ["s1", "s2"], "age": [20, 30]})
    >>> construct_attributes_dict(df, ["age"], "id")["s1"]["age"]
    20
    """

    # Set index to id_column.
    df.set_index(id_column, inplace=True)

    # Keeping only columns specified in labels.
    data_to_add = df[labels]

    # Replace whitespaces in column names with underscores.
    for char in [" ", "#"]:
        data_to_add.columns = data_to_add.columns.str.replace(char, "")

    # Transform to dictionary.
    attributes_dict = data_to_add.to_dict(orient="index")

    return attributes_dict
