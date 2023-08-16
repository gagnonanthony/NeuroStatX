import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc


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
    
    # Plotting decreasing curve. 
    plot = plt.plot(range(2, len(lst)+2), lst)
    plt.xticks(range(2, len(lst)+2))
    plt.title(f'{title}')
    plt.xlabel('Number of clusters')
    plt.ylabel(f'{metric}')
    
    # Annotating 
    if annotation is not None:
        plt.text(x=len(lst)/2, y=max(lst)/2, s=f'{annotation}')
    
    plt.savefig(f'{output}')
    plt.close()
    

def plot_dendrogram(X, output, title='Dendrograms', annotation=None):
    """
    Function to plot a dendrogram plot showing hierarchical clustering. Useful to visually 
    determine the appropriate number of clusters. 
    Adapted from: https://towardsdatascience.com/cheat-sheet-to-implementing-7-methods-for-selecting-optimal-number-of-clusters-in-python-898241e1d6ad

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
    