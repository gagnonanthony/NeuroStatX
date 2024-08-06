from factor_analyzer.factor_analyzer import (calculate_kmo,
                                             calculate_bartlett_sphericity)
import matplotlib.pyplot as plt
from scipy.stats import shapiro
from sklearn.decomposition import PCA
import seaborn as sns
from tqdm import tqdm


def remove_nans(df):
    """
    Clean up dataset by removing all rows containing NaNs.

    Parameters
    ----------
    df : pd.DataFrame
        Pandas dataframe.

    Returns
    -------
    rows_with_nans : pd.DataFrame
        Dataframe containing rows with NaNs.
    complete_rows : pd.DataFrame
        Cleaned dataframe.
    """
    rows_with_nans = df[df.isna().any(axis=1)]
    complete_rows = df.drop(index=rows_with_nans.index)

    return rows_with_nans, complete_rows


def rename_columns(df, old_names, new_names):
    """
    Function renaming specific columns according to a list of new and old
    column names.

    Parameters
    ----------
    df : pd.DataFrame
        Pandas dataframe object.
    old_names : List[str]
        List of old column name as strings.
    new_names : List[str]
        List of new column name as strings.

    Returns
    -------
    df : pd.DataFrame
        Pandas dataframe object containing the renamed columns.
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
    """
    Function to change binary answers (1/0) to Yes or No in specific columns
    from a Pandas Dataframe.
    **Please validate that yes and no are assigned to the correct values,
    default behavior is yes = 1 and no = 0.**

    Parameters
    ----------
    df : pd.DataFrame
        Pandas dataframe object.
    cols : List[str]
        List of column names.

    Returns
    -------
    df : pd.DataFrame
        Pandas dataframe object with changed binary answers.
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
    """
    Function to extract column index based on a list of column names.

    Parameters
    ----------
    df : pd.DataFrame
        Pandas dataframe object.
    column_names : List[str]
        List of column names as strings.

    Returns
    -------
    indices : List[int]
        List of column index.
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
    """
    Script to visualize distribution plots for a complete dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Pandas dataframe.
    out_folder : str
        Path to the output folder.
    context : str, optional
        Style to apply to the plots. Defaults to 'poster'.
    font_scale : float, optional
        Font scale. Defaults to 1.
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
    """
    Function computing the normality statistic using the Shapiro Wilk's test
    for normality and outputting W and p values.

    Parameters
    ----------
    df : pd.DataFrame
        Pandas dataframe.

    Returns
    -------
    wilk : List[float]
        Shapiro-Wilk values (W).
    pvalues : List[float]
        Associated p-values.
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
    """
    Function to compute a correlation matrix for all variables in a dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Pandas dataframe.
    out_folder : str
        Path to the output folder.
    context : str, optional
        Style to apply to the plots. Defaults to 'poster'.
    font_scale : float, optional
        Font scale. Defaults to 0.2.
    cmap : str, optional
        Cmap to use in the heatmap. Defaults to None.
    annot : bool, optional
        Flag to write correlation values inside the heatmap squares.
        Defaults to False.

    Returns
    -------
    corr_mat : pd.DataFrame
        Correlation matrix with Pearson correlation coefficients.
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
    """
    Function to merge a variable number of dataframe by matching the values of
    a specific column (hereby labeled as index.) Index values must appear only
    once in the dataframe for the function to work.

    Parameters
    ----------
    dict_df : Dict[str, pd.DataFrame]
        Dictionary of pandas dataframe.
    index : str
        String of the name of the column to use as index (needs to be the same
        across all dataframes).
    repeated_columns : bool, optional
        Flag to use if column name are repeated across dataframe to merge.
        Defaults to False.

    Returns
    -------
    out : pd.DataFrame
        Joint large pandas dataframe.
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
    """
    Function compute PCA decomposition on a dataset.

    Parameters
    ----------
    X : Array
        Data array.
    n_components : int
        Number of components.

    Returns
    -------
    X : Array
        Transformed data array.
    pca : PCA
        PCA model.
    exp_var : Array
        Explained variance.
    components : Array
        Components.
    p_value : float
        Bartlett's p-value.
    kmo_model : float
        KMO model.
    """

    chi_square_value, p_value = calculate_bartlett_sphericity(X)
    kmo_all, kmo_model = calculate_kmo(X)
    pca = PCA(n_components=n_components).fit(X)
    X = pca.transform(X)
    exp_var = pca.explained_variance_ratio_
    components = pca.components_

    return X, pca, exp_var, components, p_value, kmo_model
