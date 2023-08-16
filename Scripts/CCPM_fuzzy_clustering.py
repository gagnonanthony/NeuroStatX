#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os
import sys

import pandas as pd
import typer
from typing import List
from typing_extensions import Annotated

from CCPM.io.utils import (load_df_in_any_format,
                           PDF,
                           assert_input,
                           assert_output_dir_exist)
from CCPM.clustering.fuzzy import fuzzyCmeans
from CCPM.utils.preprocessing import merge_dataframes
from CCPM.clustering.viz import plot_clustering_results, plot_dendrogram
from CCPM.clustering.metrics import compute_knee_location


# Initializing the app.
app = typer.Typer(add_completion=False)

@app.command()
def main(
        in_dataset: Annotated[List[str], typer.Option(help='Input dataset(s) to filter. If multiple files are'
                                                           'provided as input, will be merged according to '
                                                           'the subject id columns.',
                                                      show_default=False,
                                                      rich_help_panel='Essential Files Options')],
        id_column: Annotated[str, typer.Option(help="Name of the column containing the subject's ID tag. "
                                                    "Required for proper handling of IDs and "
                                                    "merging multiple datasets.",
                                               show_default=False,
                                               rich_help_panel='Essential Files Options')],
        desc_columns: Annotated[int, typer.Option(help='Number of descriptive columns at the beginning of the dataset'
                                                       ' to exclude in statistics and descriptive tables.',
                                                  show_default=False,
                                                  rich_help_panel='Essential Files Options')],
        max_cluster: Annotated[int, typer.Option(help='Maximum number of cluster to fit a model for.',
                                                 show_default=True,
                                                 rich_help_panel='Clustering Options')] = 10,
        m: Annotated[float, typer.Option(help='Exponentiation value to apply on the membership function.',
                                         show_default=True,
                                         rich_help_panel='Clustering Options')] = 2,
        error: Annotated[float, typer.Option(help='Error threshold for convergence stopping criterion.',
                                             show_default=True,
                                             rich_help_panel='Clustering Options')] = 1E-6,
        maxiter: Annotated[int, typer.Option(help='Maximum number of iterations to perform.',
                                             show_default=True,
                                             rich_help_panel='Clustering Options')] = 1000,
        init: Annotated[str, typer.Option(help='Initial fuzzy c-partitioned matrix',
                                          show_default=True,
                                          rich_help_panel='Clustering Options')] = None,
        distance: Annotated[str, typer.Option(help='Distance method to use to compute intra/inter subjects/clusters distance',
                                              show_default=True,
                                              rich_help_panel='Clustering Options')] = 'euclidean',
        out_folder: Annotated[str, typer.Option(help='Path of the folder in which the results will be written. '
                                                     'If not specified, current folder and default '
                                                     'name will be used (e.g. = ./output/).',
                                                rich_help_panel='Essential Files Options')] = './',
        verbose: Annotated[bool, typer.Option('-v', '--verbose', help='If true, produce verbose output.',
                                              rich_help_panel="Optional parameters")] = False,
        overwrite: Annotated[bool, typer.Option('-f', '--overwrite', help='If true, force overwriting of existing '
                                                                          'output files.',
                                                rich_help_panel="Optional parameters")] = False,
        report: Annotated[bool, typer.Option('-r', '--report', help='If true, will generate a pdf report named '
                                                                    'report_factor_analysis.pdf',
                                             rich_help_panel='Optional parameters')] = False):

    """
    \b
    =============================================================================
                ________    ________   ________     ____     ____
               /    ____|  /    ____| |   ___  \   |    \___/    |
              /   /       /   /       |  |__|   |  |             |
             |   |       |   |        |   _____/   |   |\___/|   |
              \   \_____  \   \_____  |  |         |   |     |   |
               \________|  \________| |__|         |___|     |___|
                  Children Cognitive Profile Mapping Toolbox©
    =============================================================================
    \b
    EVALUATION METRICS
    ------------------
    The fuzzy partition coefficient (FPC) is a metric defined between 0 and 1 with 1 representing the better score. It
    represents how well the data is described by the clustering model. Therefore, a higher FPC represents a better fitted
    model.
    \b
    The Silhouette Coefficient represents an evaluation of cluster's definition. The score is bounded (-1 to 1) with 1
    as the perfect score and -1 as not a good clustering result. A higher Silhouette Coefficient relates to a model with
    better defined clusters (therefore a better model). It tends to have higher score with cluster generated from density-
    based methods.
    \b
    The Calinski-Harabasz Index (or the Variance Ratio Criterion) can be used when no known labels are available. It 
    represents the density and separation of clusters. Although it tends to be higher for cluster generated from density-
    based methods. A higher Calinski-Harabasz Index relates to better defined clusters.  
    \b
    Davies-Bouldin Index is reported for all cluster-models. A lower DBI relates to a model with better cluster separation.
    It represents a measure of similarity between clusters and is solely based on quantities and features of the dataset.
    It also tends to be generally higher for convex clusters and it uses the centroid distance between clusters therefore
    limiting the distance metric to euclidean space.
    \b
    WSS
    \b
    GAP
    \b
    REFERENCES
    ----------
    [1]     https://pythonhosted.org/scikit-fuzzy/auto_examples/plot_cmeans.html
    [2]     https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation
    [3]     https://towardsdatascience.com/cheat-sheet-to-implementing-7-methods-for-selecting-optimal-number-of-clusters-in-python-898241e1d6ad
    [4]     https://towardsdatascience.com/how-to-determine-the-right-number-of-clusters-with-code-d58de36368b1
    
    """
    
    if verbose:
        logging.getLogger().setLevel(logging.INFO)

    assert_input(in_dataset)
    assert_output_dir_exist(overwrite, out_folder, create_dir=True)
    
    # Creating substructures for output folder.
    os.mkdir(f'{out_folder}/METRICS/')
    
    # Loading dataframe.
    logging.info('Loading dataset(s)...')
    if len(in_dataset) > 1:
        if id_column is None:
            sys.exit('Column name for index matching is required when inputting multiple dataframe.')
        dict_df = {i: load_df_in_any_format(i) for i in in_dataset}
        raw_df = merge_dataframes(dict_df, id_column)
    else:
        raw_df = load_df_in_any_format(in_dataset[0])
    descriptive_columns = [n for n in range(0, desc_columns)]
    
    # Creating the array.
    subid = raw_df[id_column]
    df_for_clust = raw_df.drop(raw_df.columns[descriptive_columns], axis=1, inplace=False).astype('float')
    X = df_for_clust.values
    
    # Plotting the dendrogram.
    sys.setrecursionlimit(50000)
    plot_dendrogram(X, f'{out_folder}/METRICS/dendrogram.png')
    
    # Computing a range of C-means clustering method. 
    cntr, u, wss, fpcs, ss, chi, dbi, gap = fuzzyCmeans(X,
                                                        max_cluster=max_cluster,
                                                        m=m,
                                                        error=error,
                                                        maxiter=maxiter,
                                                        init=init,
                                                        distance=distance,
                                                        output=out_folder)
    
    # Compute knee location on Silhouette Score. 
    elbow_wss = compute_knee_location(wss)
    
    # Creating a dataframe to export statistics. 
    stats = pd.DataFrame(data={'FPC': fpcs, 'Silhouette Score': ss, 'CHI': chi, 'DBI': dbi, 'WSS': wss, 'GAP': gap},
                         index=[f'{i}-Cluster Model' for i in range(2, len(ss)+2)])
    stats.to_excel(f'{out_folder}/validation_indices.xlsx', header=True, index=True)
    
    # Plotting results for each indicators. 
    plot_clustering_results(wss, title='Within Cluster Sum of Square Error (WSS)', metric='WSS', output=f'{out_folder}/METRICS/wss.png',
                            annotation=f'Elbow threshold (Optimal cluster nb): {elbow_wss+2}')
    fpcs_index = fpcs.index(max(fpcs))
    plot_clustering_results(fpcs, title='Fuzzy Partition Coefficient (FPC)', metric='FPC', output=f'{out_folder}/METRICS/fpc.png',
                            annotation=f'Optimal Number of Cluster: {fpcs_index+2}')
    ss_index = ss.index(max(ss))
    plot_clustering_results(ss, title='Silhouette Score Coefficient (SS)', metric='SS', output=f'{out_folder}/METRICS/ss.png',
                            annotation=f'Optimal Number of Clusters: {ss_index+2}')
    chi_index = chi.index(max(chi))
    plot_clustering_results(chi, title='Calinski-Harabasz Index (CHI)', metric='CHI', output=f'{out_folder}/METRICS/chi.png',
                            annotation=f'Optimal Number of Clusters: {chi_index+2}')
    dbi_index = dbi.index(min(dbi))
    plot_clustering_results(dbi, title='Davies-Bouldin Index (DBI)', metric='DBI', output=f'{out_folder}/METRICS/dbi.png',
                            annotation=f'Optimal Number of Clusters: {dbi_index+2}')
    gap_index = gap.index(max(gap))
    plot_clustering_results(gap, title='GAP Statistics.', metric='GAP', output=f'{out_folder}/METRICS/gap.png',
                            annotation=f'Optimal Number of Clusters: {gap_index+2}')
    
    # Selecting the best number of cluster.
    
    
if __name__ == '__main__':
    app()