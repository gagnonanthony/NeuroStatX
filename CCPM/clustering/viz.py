# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from matplotlib.pyplot import get_cmap
from matplotlib.colors import rgb2hex
import numpy as np
import pandas as pd
from pandas.plotting import parallel_coordinates
import scipy.cluster.hierarchy as shc
from scipy.stats import ttest_ind
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from statannotations.Annotator import Annotator


def plot_clustering_results(lst, title, metric, output, errorbar=None, annotation=None):
    """
    Function to plot goodness of fit indicators resulting from a clustering model. Resulting
    plot will be saved in the output folder specified in function's arguments. 

    Args:
        lst (List):                 List of values to plot.
        title (str):                Title of the plot.
        metric (str):               Metric name.
        output (str):               Output filename. 
        errorbar (List):            List of values to plot as errorbar (CI, SD, etc.).
        annotation (str, optional): Annotation to add directly on the plot. Defaults to None.
    """
    
    # Plotting data.
    fig = plt.figure(figsize=(10,7))
    axes= fig.add_subplot(111)
    
    with plt.rc_context({'font.size': 10, 'font.weight': 'bold', 'axes.titleweight': 'bold'}):

        axes.plot(range(2, len(lst)+2), lst)
        
        # Add error bar if provided.
        if errorbar is not None:
            assert len(errorbar) == len(lst), "Values to plot and error bars are not of the same length [{} and {}]".format(len(lst), len(errorbar)) 
            axes.errorbar(range(2, len(errorbar)+2), lst,
                        yerr=errorbar, 
                        ecolor='black',
                        elinewidth=1,
                        fmt="o",
                        color='black',
                        barsabove=True)

        # Set parameters.
        plt.xticks(range(2, len(lst)+2))
        plt.title(f'{title}')
        plt.xlabel('Number of clusters')
        plt.ylabel(f'{metric}')
        axes.spines[['top', 'right']].set_visible(False)
        axes.spines['bottom'].set(linewidth=1.5)
        axes.spines['left'].set(linewidth=1.5)
        
        # Annotating 
        if annotation is not None:
            if axes.get_ylim()[0] < 0:
                y_pos = axes.get_ylim()[0] * 0.95
            else:
                y_pos = axes.get_ylim()[1] * 0.95
        
            plt.text(x=axes.get_xlim()[1] * 0.5, y=y_pos, s=f'{annotation}')
        
        plt.savefig(f'{output}')
        plt.close()
    

def plot_dendrogram(X, output, title='Dendrograms', annotation=None):
    """
    Function to plot a dendrogram plot showing hierarchical clustering. Useful to visually 
    determine the appropriate number of clusters. 
    Adapted from: 
    https://towardsdatascience.com/cheat-sheet-to-implementing-7-methods-for-selecting-optimal-number-of-clusters-in-python-898241e1d6ad

    Args:
        X (DataFrame):                  Data on which clustering will be performed.
        output (str):                   Output filename and path. 
        title (str, optional):          Title for the plot. Defaults to 'Dendrograms'.
        annotation (str, optional):     Annotation to add directly on the plot. Defaults to None.
    """
    
    fig = plt.figure(figsize=(10,7))
    axes = fig.add_subplot(111)
    
    with plt.rc_context({'font.size': 10, 'font.weight': 'bold', 'axes.titleweight': 'bold'}):
        axes.set_title(f'{title}')
        
        dend = shc.dendrogram(shc.linkage(X, method='ward'))
        
        if annotation is not None:
            plt.text(x=fig.axes[0].get_xlim()[1] * 0.10, y=fig.axes[0].get_ylim()[1] * 0.85, s=f'{annotation}')
            
        axes.set_xticks([])
        axes.set_yticks([])
        axes.spines[['top', 'right', 'left']].set_visible(False)
        axes.spines['bottom'].set(linewidth=1.5)
        
        plt.savefig(f'{output}')
        plt.close()
    
    
def sort_int_labels_legend(ax, title=None):
    """
    Function automatically reorder numerically labels with matching handles in matplotlib legend. 

    Args:
        ax:                     Matplotlib Axes.
        title (str, optional):  Title of the legend.  

    Returns:
        ax.legend:              Axes legend object.
    """
    
    # Fetching handles and tags from matplotlib axes.
    handles, tags = ax.get_legend_handles_labels()

    # Converting tags to int if they are not already. 
    if type(tags[0]) is str:
        tags = [int(float(x)) for x in tags]
    
    # Sorting tags and make handles follow the same ordering.
    tag, handle = zip(*sorted(zip(tags, handles)))
    
    if title is not None: 
        return ax.legend(handle, tag, title=title)

    else:
        return ax.legend(handle, tag)

    
def plot_parallel_plot(X, labels, output, mean_values=False, title='Parallel Coordinates plot.'):
    """
    Function to plot a parallel coordinates plot to visualize differences between clusters.
    Useful to highlight significant changes between clusters and interpret them. 
    Adapted from: 
    https://towardsdatascience.com/the-art-of-effective-visualization-of-multi-dimensional-data-6c7202990c57

    Args:
        X (_type_): _description_
        labels (_type_): _description_
        output (_type_): _description_
        title (str, optional): _description_. Defaults to 'Parallel Coordinates plot.'.
        annotation (_type_, optional): _description_. Defaults to None.
    """
    
    labels = labels + 1
    
    columns = list(X.columns)
    columns.append('Clusters')
    
    ss = StandardScaler()
    scaled_df = ss.fit_transform(X)
    df = pd.DataFrame(np.vstack((scaled_df.T, labels)).T, columns=columns)
    
    if mean_values:
        # Calculating mean values for each features for each clusters. 
        final_df = pd.DataFrame()
        i = 0
        for col in X.columns:
            mean = list()
            for k in np.unique(labels):
                mean.append(df.loc[df['Clusters'] == k, col].mean())
            final_df.insert(i, col, mean)
            i += 1
        final_df.insert(i, 'Clusters', np.unique(labels))
    else:
        indexes = np.random.choice(df.shape[0], size=500, replace=False)
        final_df = df.iloc[indexes]

    # Setting color palette.
    cmap = get_cmap('plasma', len(np.unique(labels)))
    colors = [rgb2hex(cmap(i)) for i in range(cmap.N)]
    
    fig = plt.figure(figsize=(15, 7))
    ax = fig.add_subplot(111)
    
    with plt.rc_context({'font.size': 10, 'font.weight': 'bold', 'axes.titleweight': 'bold'}):
        
        parallel_coordinates(final_df, 'Clusters', ax=ax, color=colors)
        sort_int_labels_legend(ax, title='Cluster #')
        ax.set_title(f'{title}')
        ax.grid(False)
        ax.spines[['top', 'bottom']].set_visible(False)
        ax.spines['left'].set(linewidth=1.5)
        ax.spines['right'].set(linewidth=1.5)
        plt.savefig(f'{output}')
        plt.close()


def plot_grouped_barplot(X, labels, output, title='Barplot', annotation=None):
    """
    Function to plot a barplot for all features in the original dataset stratified by clusters.
    T-test between clusters' mean within a feature is also computed and annotated directly on the
    plot. When plotting a high number of clusters, plotting of significant annotation is 
    polluting the plot, will be fixed in the future. 

    Args:
        X (DataFrame):                         Input dataset of shape (S, F).    
        labels (Array):                        Array of hard membership value (S, ). 
        output (str):                          Filename of the png file. 
        title (str, optional): _description_.  Title of the plot. Defaults to 'Barplot'.
        annotation (str, optional):            Annotation. Defaults to None.
    """
    # Make labels start at 1 rather than 0, better for viz. 
    labels = labels + 1
    
    features = list(X.columns)
    features.append('Clusters')
    
    ss = StandardScaler()
    scaled_df = ss.fit_transform(X)
    df = pd.DataFrame(np.vstack((scaled_df.T, labels)).T, columns=features)
    
    # Melting the dataframe for plotting.
    viz_df = df.melt(id_vars='Clusters')
    
    # Setting up matplotlib figure. 
    fig = plt.figure(figsize=(15,8))
    axes = fig.add_subplot()
    
    # Setting parameters for the barplot.
    plotting_parameters = {
        'data': viz_df,
        'x': 'variable',
        'y': 'value',
        'hue': 'Clusters',
        'palette': 'plasma',
        'saturation': 0.5
    }
    
    with plt.rc_context({'font.size': 10, 'font.weight': 'bold', 'axes.titleweight': 'bold'}):
        
        # Plotting barplot using Seaborn.
        sns.barplot(ax=axes, **plotting_parameters)
        
        # Setting pairs for statistical testing between clusters for each feature.
        clusters = np.unique(labels)
        features = list(X.columns)
        
        pairs = [[(var, cluster1), (var, cluster2)]
                for var in features
                for i, cluster1 in enumerate(clusters)
                for cluster2 in clusters[-(i):]
                if (cluster1 != cluster2) and not (cluster1 > cluster2)]
        
        # Plotting statistical difference. 
        annotator = Annotator(axes, pairs, verbose=False, **plotting_parameters, hide_non_significant=True)
        annotator.configure(test='t-test_ind', text_format='star', show_test_name=False,
                            verbose=0)
        annotator.apply_test()
        _, results = annotator.annotate()
        
        # Customization options.
        axes.spines[['top', 'right', 'bottom']].set_visible(False)
        axes.spines['left'].set(linewidth=1.5)
        axes.legend(title='Cluster #', loc='best', title_fontsize='medium')
        axes.set_title(f'{title}')
        axes.set_ylabel('Scaled Scores', fontdict={'fontweight': 'bold'})
        axes.set_xlabel('')
        
        plt.savefig(f'{output}')
        plt.close()
    
    
    