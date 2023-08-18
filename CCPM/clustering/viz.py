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


def plot_clustering_results(lst, title, metric, output, annotation=None):
    """
    Function to plot goodness of fit indicators resulting from a clustering model. Resulting
    plot will be saved in the output folder specified in function's arguments. 

    Args:
        lst (List):                 List of values to plot.
        title (str):                Title of the plot.
        metric (str):               Metric name.
        output (str):               Output filename. 
        annotation (str, optional): Annotation to add directly on the plot. Defaults to None.
    """
    
    # Plotting data.
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111)
    ax.plot(range(2, len(lst)+2), lst)
    plt.xticks(range(2, len(lst)+2))
    plt.title(f'{title}')
    plt.xlabel('Number of clusters')
    plt.ylabel(f'{metric}')
    
    # Annotating 
    if annotation is not None:
        if ax.get_ylim()[0] < 0:
            y_pos = ax.get_ylim()[0] * 0.95
        else:
            y_pos = ax.get_ylim()[1] * 0.95
    
        plt.text(x=ax.get_xlim()[1] * 0.5, y=y_pos, s=f'{annotation}')
    
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
    plt.title(f'{title}')
    
    dend = shc.dendrogram(shc.linkage(X, method='ward'))
    
    if annotation is not None:
        plt.text(x=fig.axes[0].get_xlim()[1] * 0.10, y=fig.axes[0].get_ylim()[1] * 0.85, s=f'{annotation}')
        
    plt.xticks([])
    
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
    parallel_coordinates(final_df, 'Clusters', ax=ax, color=colors)
    sort_int_labels_legend(ax, title='Cluster #')
    ax.set_title(f'{title}')
    plt.rcParams.update({'font.size': 8})
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
        annotation (str, optional):         Annotation. Defaults to None.
    """
    # Make labels start at 1 rather than 0, better for viz. 
    labels = labels + 1
    
    columns = list(X.columns)
    columns.append('Clusters')
    
    ss = StandardScaler()
    scaled_df = ss.fit_transform(X)
    df = pd.DataFrame(np.vstack((scaled_df.T, labels)).T, columns=columns)
    
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
    final_df.set_index('Clusters', inplace=True, drop=True)
    
    # Add horizontal significance bars and asterisks
    num_bars = len(final_df)
    num_features = len(final_df.columns)
    
    bar_width = 0.1  # Adjust the width of the significance bars
    bar_centers = np.arange(num_features) + (bar_width * num_bars / 2)
    
    fig = plt.figure(figsize=(12, 7))
    axes = fig.add_subplot(111)
    
    for i, feature in enumerate(final_df.T.columns):
        axes.bar(bar_centers + bar_width * i, final_df.T[feature], width=bar_width, label=feature)
    
    # Calculate p-values for significance between clusters for each feature
    p_values = np.zeros((num_features, len(np.unique(labels)), len(np.unique(labels))))
    for f_idx, feature in enumerate(X.columns):
        for i in range(len(np.unique(labels))):
            for j in range(i + 1, len(np.unique(labels))):
                cluster_i = df[df['Clusters'] == i+1][feature]
                cluster_j = df[df['Clusters'] == j+1][feature]
                _, p_value = ttest_ind(cluster_i, cluster_j)
                p_values[f_idx, i, j] = p_value
    
    # Plot significance bar. 
    for f_idx, feature in enumerate(X.columns):
        bar_height = []
        for i in range(len(np.unique(labels))):
            for j in range(i + 1, len(np.unique(labels))):
                p_value = p_values[f_idx, i, j]
                if p_value < 0.05:  # You can adjust the significance threshold
                    y_position = max(final_df.iloc[i][f_idx], final_df.iloc[j][f_idx]) * 1.05
                    if y_position in bar_height:
                        y_position = y_position * 1.1
                    bar_height.append(y_position)
                    x_position = [bar_centers[f_idx] + bar_width * i, bar_centers[f_idx] + bar_width * j]
                    axes.plot(x_position, [y_position, y_position], color='black')
                    x_center = np.mean(x_position)
                    axes.text(x_center, y_position * 1.05, '*', verticalalignment='center')
    
    # Customizing plot.
    sort_int_labels_legend(axes)
    axes.set_xticks(bar_centers + (bar_width * num_features / 2))
    axes.set_xticklabels(final_df.columns)
    axes.legend(title='Cluster #')
    axes.set_title(f'{title}')
    plt.rcParams.update({'font.size': 12})
    
    plt.savefig(f'{output}')
    plt.close()
    
    