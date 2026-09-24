from factor_analyzer.factor_analyzer import (calculate_kmo,
                                             calculate_bartlett_sphericity)
import matplotlib.pyplot as plt
from scipy.stats import shapiro
from sklearn.decomposition import PCA
import seaborn as sns
from tqdm import tqdm


def remove_nans(df):
    """Split a table into rows with missing values and complete rows.

    Parameters
    ----------
    df : pd.DataFrame
        Input table.

    Returns
    -------
    rows_with_nans : pd.DataFrame
        Rows that contain at least one NaN.
    complete_rows : pd.DataFrame
        Rows with no missing values.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from neurostatx.utils.preprocessing import remove_nans
    >>> df = pd.DataFrame({"a": [1.0, np.nan], "b": [2.0, 3.0]})
    >>> nans, complete = remove_nans(df)
    >>> len(complete)
    1
    """
    rows_with_nans = df[df.isna().any(axis=1)]
    complete_rows = df.drop(index=rows_with_nans.index)

    return rows_with_nans, complete_rows


def rename_columns(df, old_names, new_names):
    """Rename selected columns and return a copy of the table.

    Parameters
    ----------
    df : pd.DataFrame
        Input table.
    old_names : list of str
        Current column names.
    new_names : list of str
        Replacement column names, same length as ``old_names``.

    Returns
    -------
    new_df : pd.DataFrame
        Copy of ``df`` with renamed columns.

    Examples
    --------
    >>> import pandas as pd
    >>> from neurostatx.utils.preprocessing import rename_columns
    >>> df = pd.DataFrame({"a": [1], "b": [2]})
    >>> rename_columns(df, ["a"], ["x"]).columns.tolist()
    ['x', 'b']
    """
    if len(old_names) != len(new_names):
        raise ValueError("Number of old names and new names must be the same.")

    for i, old_name in enumerate(old_names):
        if old_name not in df.columns:
            raise ValueError(f"Column {old_name} not found in DataFrame.")

    new_names_set = set(new_names)
    if len(new_names_set) != len(new_names):
        raise ValueError("New names contain duplicates.")

    new_df = df.copy()
    for i in range(len(old_names)):
        if old_names[i] in new_df.columns:
            new_df.rename(columns={old_names[i]: new_names[i]}, inplace=True)
    return new_df


def binary_to_yes_no(df, cols):
    """Recode 1/0 columns to ``Yes`` / ``No`` in place.

    Values of 1 become ``Yes``, 0 become ``No``, and other values become
    ``Don't know or missing value``.

    Parameters
    ----------
    df : pd.DataFrame
        Input table. Matching columns are modified in place.
    cols : list of str
        Column names to recode.

    Returns
    -------
    df : pd.DataFrame
        The same table after recoding.

    Examples
    --------
    >>> import pandas as pd
    >>> from neurostatx.utils.preprocessing import binary_to_yes_no
    >>> df = pd.DataFrame({"flag": [1.0, 0.0]})
    >>> binary_to_yes_no(df, ["flag"])["flag"].tolist()
    ['Yes', 'No']
    """
    for col in cols:
        if df[col].isin([0.0, 1.0, 2.0, "nan"]).any():
            df[col] = df[col].apply(
                lambda x: "Yes"
                if x == 1
                else "No"
                if x == 0
                else "Don't know or missing value"
            )
    return df


def get_column_indices(df, column_names):
    """Return integer positions for the requested column names.

    Parameters
    ----------
    df : pd.DataFrame
        Input table.
    column_names : list of str
        Column names to look up.

    Returns
    -------
    indices : list of int
        Column positions. Missing names are skipped.

    Examples
    --------
    >>> import pandas as pd
    >>> from neurostatx.utils.preprocessing import get_column_indices
    >>> df = pd.DataFrame({"a": [1], "b": [2]})
    >>> get_column_indices(df, ["b", "a"])
    [1, 0]
    """
    indices = []
    for name in column_names:
        try:
            index = df.columns.get_loc(name)
            indices.append(index)
        except KeyError:
            print(f"Column '{name}' not found in DataFrame.")
    return indices


def plot_distributions(df, out_folder, context="poster", font_scale=1):
    """Write a histogram and ECDF for each column in ``df``.

    Parameters
    ----------
    df : pd.DataFrame
        Input table.
    out_folder : str
        Directory where ``{column}.png`` files are written.
    context : str, optional
        Seaborn context. Defaults to ``"poster"``.
    font_scale : float, optional
        Seaborn font scale. Defaults to 1.

    Examples
    --------
    >>> import pandas as pd
    >>> from neurostatx.utils.preprocessing import plot_distributions
    >>> df = pd.DataFrame({"a": [1, 2, 3]})
    >>> plot_distributions(df, ".")
    """

    # Setting plotting parameters.
    plt.rcParams["figure.figsize"] = [10, 7]
    plt.rcParams["figure.autolayout"] = True
    sns.set_style("white")
    sns.set_context(f"{context}", font_scale)

    # Iterating over columns for plotting variables.
    for var in tqdm(df.columns):
        f, axes = plt.subplots(1, 2)
        sns.histplot(data=df, x=var, fill=True, kde=True, ax=axes[0])
        sns.ecdfplot(data=df, x=var, ax=axes[1])
        plt.savefig(f"{out_folder}/{var}.png")
        plt.close()


def compute_shapiro_wilk_test(df):
    """Run a Shapiro-Wilk normality test on each column.

    Parameters
    ----------
    df : pd.DataFrame
        Input table.

    Returns
    -------
    wilk : list of float
        Shapiro-Wilk W statistics.
    pvalues : list of float
        Corresponding p-values.

    Examples
    --------
    >>> import pandas as pd
    >>> from neurostatx.utils.preprocessing import compute_shapiro_wilk_test
    >>> df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
    >>> wilk, pvalues = compute_shapiro_wilk_test(df)
    >>> len(wilk)
    1
    """

    wilk = []
    pvalues = []

    for var in tqdm(df.columns):
        var_data = df[var].values
        w, pval = shapiro(var_data)
        wilk.append(w)
        pvalues.append(pval)

    return wilk, pvalues


def compute_correlation_coefficient(
    df, out_folder, context="poster", font_scale=0.2, cmap=None, annot=False
):
    """Compute a Pearson correlation matrix and write a heatmap.

    Parameters
    ----------
    df : pd.DataFrame
        Input table.
    out_folder : str
        Directory where ``correlation_heatmap.png`` is written.
    context : str, optional
        Seaborn context. Defaults to ``"poster"``.
    font_scale : float, optional
        Seaborn font scale. Defaults to 0.2.
    cmap : str, optional
        Heatmap colormap. Defaults to None.
    annot : bool, optional
        If True, write correlation values on the heatmap. Defaults to False.

    Returns
    -------
    corr_mat : pd.DataFrame
        Pearson correlation matrix.

    Examples
    --------
    >>> import pandas as pd
    >>> from neurostatx.utils.preprocessing import (
    ...     compute_correlation_coefficient)
    >>> df = pd.DataFrame({"a": [1, 2, 3], "b": [2, 4, 6]})
    >>> corr = compute_correlation_coefficient(df, ".")
    >>> corr.loc["a", "b"]
    1.0
    """

    # Setting plotting parameters.
    plt.rcParams["figure.figsize"] = [20, 15]
    plt.rcParams["figure.autolayout"] = True
    sns.set_style("white")
    sns.set_context(f"{context}", font_scale)

    corr_mat = df.corr()
    sns.heatmap(
        corr_mat,
        cmap=cmap,
        annot=annot,
        square=True,
        xticklabels=True,
        yticklabels=True,
        cbar=True,
    )
    plt.savefig(f"{out_folder}/correlation_heatmap.png")

    return corr_mat


def merge_dataframes(dict_df, index, repeated_columns=False):
    """Join several tables on a shared index column.

    Index values must be unique within each table.

    Parameters
    ----------
    dict_df : dict
        Mapping of labels to DataFrames.
    index : str
        Shared column used as the join index.
    repeated_columns : bool, optional
        If True, disambiguate overlapping column names with suffixes.
        Defaults to False.

    Returns
    -------
    out : pd.DataFrame
        Joined table.

    Examples
    --------
    >>> import pandas as pd
    >>> from neurostatx.utils.preprocessing import merge_dataframes
    >>> left = pd.DataFrame({"id": [1, 2], "a": [10, 20]})
    >>> right = pd.DataFrame({"id": [1, 2], "b": [3, 4]})
    >>> merge_dataframes({"l": left, "r": right}, "id")["a"].tolist()
    [10, 20]
    """

    keys = list(dict_df.keys())
    for k in keys:
        dict_df[k] = dict_df[k].set_index(f"{index}")

    if repeated_columns:
        out = dict_df[keys[0]]
        for k in keys[1: len(keys)]:
            out = out.join(dict_df[k], lsuffix="a", rsuffix="b")
    else:
        out = dict_df[keys[0]].join([dict_df[k] for k in keys[1: len(keys)]])

    return out


def compute_pca(X, n_components):
    """Fit a PCA and return scores, diagnostics, and the model.

    Parameters
    ----------
    X : pd.DataFrame
        Input table.
    n_components : int
        Number of components to keep.

    Returns
    -------
    scores : array
        Transformed data.
    pca : PCA
        Fitted PCA model.
    exp_var : array
        Explained variance ratio.
    components : array
        Principal axes.
    p_value : float
        Bartlett sphericity p-value.
    kmo_model : float
        Overall KMO statistic.

    Examples
    --------
    >>> import pandas as pd
    >>> from neurostatx.utils.preprocessing import compute_pca
    >>> X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0],
    ...                   "b": [2.0, 1.0, 4.0, 3.0]})
    >>> scores, pca, exp_var, components, p_value, kmo = compute_pca(X, 1)
    >>> scores.shape
    (4, 1)
    """

    chi_square_value, p_value = calculate_bartlett_sphericity(X.values)
    kmo_all, kmo_model = calculate_kmo(X.values)
    pca = PCA(n_components=n_components).fit(X.values)
    X = pca.transform(X.values)
    exp_var = pca.explained_variance_ratio_
    components = pca.components_

    return X, pca, exp_var, components, p_value, kmo_model
