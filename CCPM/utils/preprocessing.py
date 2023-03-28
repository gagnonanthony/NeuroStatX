import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import shapiro


def remove_nans(df):
    """
    Clean up dataset by removing all rows containing NaNs.
    :param df:      Pandas dataframe.
    :return:        One df containing the removed rows and one df with the complete rows.
    """
    rows_with_nans = df[df.isna().any(axis=1)]
    complete_rows = df.drop(index=rows_with_nans.index)

    return rows_with_nans, complete_rows


def plot_distributions(df, out_folder, context='poster', font_scale=1):
    """
    Script to visualize distribution plots for a complete dataframe.
    :param df:              Pandas dataframe.
    :param out_folder:      Path to the output folder.
    :param context:         Style to apply to the plots.
    :param font_scale:      Font Scale
    :return:
    """

    # Setting plotting parameters.
    plt.rcParams['figure.figsize'] = [10, 7]
    plt.rcParams['figure.autolayout'] = True
    sns.set_style('white')
    sns.set_context(f'{context}', font_scale)

    # Iterating over columns for plotting variables.
    for var in tqdm(df.columns):
        f, axes = plt.subplots(1, 2)
        sns.histplot(data=df, x=var, fill=True, kde=True, ax=axes[0])
        sns.ecdfplot(data=df, x=var, ax=axes[1])
        plt.savefig(f'{out_folder}/{var}.png')
        plt.close()


def compute_shapiro_wilk_test(df):
    """
    Function computing the normality statistic using the Shapiro Wilk's test for normality
    and outputting W and p values.
    :param df:      Pandas dataframe.
    :return:        Shapiro-Wilk values (W) and associated p-values.
    """

    wilk = []
    pvalues = []

    for var in tqdm(df.columns):
        var_data = df[var].values
        w, pval = shapiro(var_data)
        wilk.append(w)
        pvalues.append(pval)

    return wilk, pvalues


def compute_correlation_coefficient(df, out_folder, context='poster', font_scale=0.2, cmap=None,
                                    annot=False):
    """
    Function to compute a correlation matrix for all variables in a dataframe.
    :param df:              Pandas dataframe.
    :param out_folder:      Path to the output folder.
    :param context:         Style to apply to the plots.
    :param font_scale:      Font scale.
    :param cmap:            Cmap to use in the heatmap.
    :param annot:           Flag to write correlation values inside the heatmap squares.
    :return:
    Correlation matrix with pearson correlation coefficients.
    """

    # Setting plotting parameters.
    plt.rcParams['figure.figsize'] = [20, 15]
    plt.rcParams['figure.autolayout'] = True
    sns.set_style('white')
    sns.set_context(f'{context}', font_scale)

    corr_mat = df.corr()
    sns.heatmap(corr_mat, cmap=cmap, annot=annot, square=True, xticklabels=True, yticklabels=True,
                cbar=True)
    plt.savefig(f'{out_folder}/correlation_heatmap.png')

    return corr_mat


def merge_dataframes(dict_df, index, repeated_columns=False):
    """
    Function to merge a variable number of dataframe by matching the values of a specific column (hereby
    labeled as index.) Index values must appear only once in the dataframe for the function to work.
    :param dict_df:             Dictionary of pandas dataframe.
    :param index:               String of the name of the column to use as index (needs to be the same across all dataframes).
    :param repeated_columns     Flag to use if column name are repeated across dataframe to merge.
    :return:                    Joint large pandas dataframe.
    """

    keys = list(dict_df.keys())
    for k in keys:
        dict_df[k] = dict_df[k].set_index(f'{index}')

    if repeated_columns:
        out = dict_df[keys[0]]
        for k in keys[1:len(keys)]:
            out = out.join(dict_df[k], lsuffix='a', rsuffix='b')
    else:
        out = dict_df[keys[0]].join([dict_df[k] for k in keys[1:len(keys)]])

    return out
