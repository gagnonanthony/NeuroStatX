import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import shapiro


def remove_nans(df):
    """
    Clean up dataset by removing all rows containing NaNs.
    :param df:
    :return:
    """
    rows_with_nans = df[df.isna().any(axis=1)]
    complete_rows = df.drop(index=rows_with_nans.index)

    return rows_with_nans, complete_rows


def plot_distributions(df, out_folder, context='poster', font_scale=1):
    """
    Script to visualize distribution plots for a complete dataframe.
    :param df:
    :param out_folder:
    :param context:
    :param font_scale:
    :param aspect:
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
    :param df:
    :return:
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
    :param df:
    :param out_folder:
    :param context:
    :param font_scale:
    :param cmap:
    :param annot:
    :return:
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
