import pandas as pd
from sklearn.impute import KNNImputer


def KNNimputation(ref_df, df, n_neighbors=5, weights='distance',
                  metric='nan_euclidean', keep_all_features=True):
    """Function to impute data in a dataset based on learned relationship
    from a reference dataset.

    This function uses the KNNImputer from the sklearn library to impute
    missing values in a dataset. The imputation is based on the relationship
    learned from a reference dataset. The reference dataset is used to
    calculate the distance between samples and the missing values are
    imputed based on the n_neighbors closest samples. Useful to complete
    data from a different population and compare both of them later on.

    ** Note: The reference dataset should not contain any missing values. **
    ** Note: The reference dataset and the dataset to impute values in should
    contain the same columns. **

    Args:
        ref_df (pd.DataFrame):              Reference dataset to learn the
                                            features' relationship from.
        df (pd.DataFrame):                  Dataset to impute.
        n_neighbors (int, optional):        Number of neighbors to use.
                                            Defaults to 5.
        weights (str, optional):            Weight function to use, possible
                                            value:
                                            * 'uniform': uniform weights. All
                                            points will have equal importance.
                                            * 'distance': Weight by the inverse
                                            of their distance.
                                            Defaults to 'distance'.
        metric (str, optional):             Distance metric for searching
                                            neighbors. Defaults to
                                            'nan_euclidean'.
        keep_all_features (bool, optional): If True, even columns containing
                                            only NaNs will be imputed.
                                            Defaults to True.

    Returns:
        pd.DataFrame:                       Imputed dataset.
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
    """Function to apply various models to a dataset.

    Args:
        df (pd.DataFrame):              Dataframe to use.
        mod (Model):                    Model to use.

    Returns:
        y:                              Predicted values.
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
