import pandas as pd
from sklearn.impute import KNNImputer


def KNNimputation(ref_df, df, n_neighbors=5, weights='distance',
                  metric='nan_euclidean', keep_all_features=True):
    """Impute missing values in ``df`` from a complete reference table.

    Fits sklearn ``KNNImputer`` on ``ref_df`` and applies it to ``df``. Both
    tables must share the same columns, and ``ref_df`` should contain no
    missing values.

    Parameters
    ----------
    ref_df : pd.DataFrame
        Complete reference dataset used to learn neighbor relationships.
    df : pd.DataFrame
        Dataset to impute.
    n_neighbors : int, optional
        Number of neighbors. Defaults to 5.
    weights : str, optional
        Neighbor weighting, ``"uniform"`` or ``"distance"``. Defaults to
        ``"distance"``.
    metric : str, optional
        Distance metric. Defaults to ``"nan_euclidean"``.
    keep_all_features : bool, optional
        If True, columns that are entirely missing are still imputed.
        Defaults to True.

    Returns
    -------
    out : pd.DataFrame
        Imputed dataset.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from neurostatx.statistics.utils import KNNimputation
    >>> ref = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [2.0, 4.0, 6.0]})
    >>> df = pd.DataFrame({"a": [1.5, np.nan], "b": [3.0, 5.0]})
    >>> KNNimputation(ref, df, n_neighbors=2).isna().sum().sum()
    0
    """

    # Assert columns from both datasets are named the same.
    assert ref_df.columns.all() == df.columns.all(), "Columns from both \
        datasets should be named the same. Please validate the column names."

    # Initialize the imputer.
    KNN = KNNImputer(n_neighbors=n_neighbors, metric=metric, weights=weights,
                     keep_empty_features=keep_all_features)

    # Fit the imputer to the reference dataset.
    KNN.fit(ref_df)

    # Transform data from the dataset to impute.
    out = pd.DataFrame(KNN.transform(df), columns=df.columns)

    return out


def apply_various_models(df, mod):
    """Apply a fitted factor or sklearn transformer to ``df``.

    Supports semopy models, sklearn transformers, and factor_analyzer
    models.

    Parameters
    ----------
    df : pd.DataFrame
        Data to transform.
    mod : object
        Fitted model exposing ``predict_factors`` or ``transform``.

    Returns
    -------
    y : pd.DataFrame
        Transformed factor or component scores.

    Examples
    --------
    >>> import pandas as pd
    >>> from sklearn.decomposition import PCA
    >>> from neurostatx.statistics.utils import apply_various_models
    >>> df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [2.0, 4.0, 6.0]})
    >>> pca = PCA(n_components=1).fit(df)
    >>> apply_various_models(df, pca).shape
    (3, 1)
    """

    if "semopy" in str(type(mod)):
        y = mod.predict_factors(df)
    elif "sklearn" in str(type(mod)):
        y = pd.DataFrame(mod.transform(df),
                         columns=list(mod.get_feature_names_out()))
    elif "factor_analyzer" in str(type(mod)):
        y = pd.DataFrame(mod.transform(df),
                         columns=["factor_{}".format(i)
                                  for i in range(0, mod.n_factors)])
    else:
        raise TypeError("Model of type {} currently not supported, please"
                        "open an issue on GitHub.".format(type(mod)))

    return y
