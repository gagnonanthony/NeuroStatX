# -*- coding: utf-8 -*-

import math

import matplotlib.pyplot as plt
from matplotlib.pyplot import get_cmap
from matplotlib.colors import rgb2hex
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
import numpy as np
import pandas as pd
from pandas.plotting import parallel_coordinates
import scipy.cluster.hierarchy as shc
from scipy.stats import f_oneway


def plot_clustering_results(lst, title, metric, output, errorbar=None,
                            annotation=None):
    """
    Function to plot goodness of fit indicators resulting from a clustering
    model. Resulting plot will be saved in the output folder specified in
    function's arguments.

    Parameters
    ----------
    lst : List
        List of values to plot.
    title : str
        Title of the plot.
    metric : str
        Metric name.
    output : str
        Output filename.
    errorbar : List, optional
        List of values to plot as errorbar (CI, SD, etc.). Defaults to None.
    annotation : str, optional
        Annotation to add directly on the plot. Defaults to None.
    """

    # Plotting data.
    fig = plt.figure(figsize=(10, 7))
    axes = fig.add_subplot(111)

    with plt.rc_context(
        {"font.size": 10, "font.weight": "bold", "axes.titleweight": "bold"}
    ):
        axes.plot(range(2, len(lst) + 2), lst)

        # Add error bar if provided.
        if errorbar is not None:
            assert len(errorbar) == len(
                lst
            ), "Values to plot and error bars are not of the same length "
            "[{} and {}]".format(len(lst), len(errorbar))

            axes.errorbar(
                range(2, len(errorbar) + 2),
                lst,
                yerr=errorbar,
                ecolor="black",
                elinewidth=1,
                fmt="o",
                color="black",
                barsabove=True,
            )

        # Set parameters.
        plt.xticks(range(2, len(lst) + 2))
        plt.title(f"{title}")
        plt.xlabel("Number of clusters")
        plt.ylabel(f"{metric}")
        axes.spines[["top", "right"]].set_visible(False)
        axes.spines["bottom"].set(linewidth=1.5)
        axes.spines["left"].set(linewidth=1.5)

        # Annotating
        if annotation is not None:
            if axes.get_ylim()[0] < 0:
                y_pos = axes.get_ylim()[0] * 0.95
            else:
                y_pos = axes.get_ylim()[1] * 0.95

            plt.text(x=axes.get_xlim()[1] * 0.5, y=y_pos, s=f"{annotation}")

        plt.savefig(f"{output}")
        plt.close()


def plot_dendrogram(X, output, title="Dendrograms", annotation=None):
    """
    Function to plot a dendrogram plot showing hierarchical clustering. Useful
    to visually determine the appropriate number of clusters.
    Adapted from:
    https://towardsdatascience.com/cheat-sheet-to-implementing-7-methods-for-selecting-optimal-number-of-clusters-in-python-898241e1d6ad

    Parameters
    ----------
    X : DataFrame
        Data on which clustering will be performed.
    output : str
        Output filename and path.
    title : str, optional
        Title for the plot. Defaults to 'Dendrograms'.
    annotation : str, optional
        Annotation to add directly on the plot. Defaults to None.
    """

    fig = plt.figure(figsize=(10, 7))
    axes = fig.add_subplot(111)

    with plt.rc_context(
        {"font.size": 10, "font.weight": "bold", "axes.titleweight": "bold"}
    ):
        axes.set_title(f"{title}")

        shc.dendrogram(shc.linkage(X, method="ward"))

        if annotation is not None:
            plt.text(
                x=fig.axes[0].get_xlim()[1] * 0.10,
                y=fig.axes[0].get_ylim()[1] * 0.85,
                s=f"{annotation}",
            )

        axes.set_xticks([])
        axes.set_yticks([])
        axes.spines[["top", "right", "left"]].set_visible(False)
        axes.spines["bottom"].set(linewidth=1.5)

        plt.savefig(f"{output}")
        plt.close()


def sort_int_labels_legend(ax, title=None):
    """
    Function automatically reorder numerically labels with matching handles in
    matplotlib legend.

    Parameters
    ----------
    ax : Matplotlib Axes
        Axes object.
    title : str, optional
        Title of the legend. Defaults to None.

    Returns
    -------
    ax.legend : Axes legend object
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


def plot_parallel_plot(
    X, labels, output, mean_values=False, cmap='magma',
    title="Parallel Coordinates plot."
):
    """
    Function to plot a parallel coordinates plot to visualize differences
    between clusters. Useful to highlight significant changes between clusters
    and interpret them.
    Adapted from:
    https://towardsdatascience.com/the-art-of-effective-visualization-of-multi-dimensional-data-6c7202990c57

    Parameters
    ----------
    X : DataFrame
        Input dataset of shape (S, F).
    labels : np.array
        Array of hard membership value (S, ).
    output : str
        Filename of the png file.
    mean_values : bool, optional
        If true, will plot the mean values of each features for each clusters.
        Defaults to False.
    cmap : str, optional
        Colormap to use for the plot. Defaults to 'magma'. See
        https://matplotlib.org/stable/tutorials/colors/colormaps.html
    title : str, optional
        Title of the plot. Defaults to 'Parallel Coordinates plot.'
    """

    labels = labels + 1

    columns = list(X.columns)
    columns.append("Clusters")

    df = pd.concat([X, pd.DataFrame(labels, columns=["Clusters"])], axis=1)

    if mean_values:
        # Calculating mean values for each features for each clusters.
        final_df = pd.DataFrame()
        i = 0
        for col in X.columns:
            mean = list()
            for k in np.unique(labels):
                mean.append(df.loc[df["Clusters"] == k, col].mean())
            final_df.insert(i, col, mean)
            i += 1
        final_df.insert(i, "Clusters", np.unique(labels))
    else:
        indexes = np.random.choice(df.shape[0], size=500, replace=False)
        final_df = df.iloc[indexes]

    # Setting color palette.
    cmap = get_cmap(cmap, len(np.unique(labels)))
    colors = [rgb2hex(cmap(i)) for i in range(cmap.N)]

    fig = plt.figure(figsize=(15, 7))
    ax = fig.add_subplot(111)

    with plt.rc_context(
        {"font.size": 10, "font.weight": "bold", "axes.titleweight": "bold"}
    ):
        parallel_coordinates(final_df, "Clusters", ax=ax, color=colors)
        sort_int_labels_legend(ax, title="Cluster #")
        ax.set_title(f"{title}")
        ax.grid(False)
        ax.spines[["left", "right", "top", "bottom"]].set(linewidth=1.5)

        ax.figure.autofmt_xdate()

        plt.savefig(f"{output}")
        plt.close()


def radar_plot(X, labels, output, frame='circle', title="Radar plot",
               cmap='magma'):
    """
    Function to plot a radar plot for all features in the original dataset
    stratified by clusters. T-test between clusters' mean within a feature is
    also computed and annotated directly on the plot. When plotting a high
    number of clusters, plotting of significant annotation is polluting the
    plot, will be fixed in the future.

    Parameters
    ----------
    X : DataFrame
        Input dataset of shape (S, F).
    labels : np.array
        Array of hard membership value (S, ).
    output : str
        Filename of the png file.
    frame : str, optional
        Shape of the radar plot. Defaults to 'circle'. Choices are 'circle'
        or 'polygon'.
    title : str, optional
        Title of the plot. Defaults to 'Radar plot'.
    cmap : str, optional
        Colormap to use for the plot. Defaults to 'magma'. See
        https://matplotlib.org/stable/tutorials/colors/colormaps.html
    """

    # Setting color palette.
    cmap = get_cmap(cmap, len(np.unique(labels)))
    colors = [rgb2hex(cmap(i)) for i in range(cmap.N)]

    # Make labels start at 1 rather than 0, better for viz.
    labels = labels + 1

    # Axis labels.
    var_labels = X.columns.tolist()
    var_labels.append(var_labels[0])

    # Computing ANOVA for each features.
    anova = []
    i = 0
    for col in X.columns:
        f, p = f_oneway(*[X.loc[labels == k, col] for k in np.unique(labels)])
        anova.append(p)
        i += 1

    # Computing mean values for each features for each clusters.
    mean_df = pd.DataFrame()
    i = 0
    for col in X.columns:
        mean = list()
        for k in np.unique(labels):
            mean.append(X.loc[labels == k, col].mean())
        mean_df.insert(i, col, mean)
        i += 1

    # Computing stds for each features for each clusters.
    std_df = pd.DataFrame()
    i = 0
    for col in X.columns:
        std = list()
        for k in np.unique(labels):
            std.append(X.loc[labels == k, col].std())
        std_df.insert(i, col, std)
        i += 1
    max_val = math.ceil((mean_df + std_df).max().max())
    min_val = math.floor((mean_df - std_df).min().min())

    mean_df.insert(i, "Clusters", np.unique(labels))
    std_df.insert(i, "Clusters", np.unique(labels))

    with plt.rc_context(
        {"font.size": 12, "font.weight": "bold", "axes.titleweight": "bold",
         "font.family": "Sans Serif"}
    ):
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, polar=True)

        # Set radar plot parameters.
        theta = create_radar_plot(len(X.columns), frame=frame)

        for idx, cluster in enumerate(np.unique(labels)):
            values = mean_df.iloc[idx].drop('Clusters').values.tolist()
            values.append(values[0])
            stds = std_df.iloc[idx].drop('Clusters').values.tolist()
            stds.append(stds[0])
            stds_pos = [np.sum(x) for x in zip(values, stds)]
            stds_neg = [s - d for s, d in zip(values, stds)]
            ax.plot(theta, values, c=colors[idx], linewidth=2,
                    label=f'Cluster {cluster}', markersize=4, zorder=3)
            plot = ax.errorbar(theta, values, yerr=stds, fmt='o-',
                               color=colors[idx], linewidth=0,
                               label=f'Cluster {cluster}')
            ax.fill_between(theta, values, stds_pos, alpha=0.2,
                            color=colors[idx], edgecolor='none',
                            label='_nolegend_')
            ax.fill_between(theta, values, stds_neg, alpha=0.2,
                            color=colors[idx], edgecolor='none',
                            label='_nolegend_')

            plot[-1][0].set_color(colors[idx])

    # Add p-values to the plot.
    for i, p in enumerate(anova):
        if 0.01 < p < 0.05:
            ax.text(theta[i], max_val * 0.95, '*', fontsize=18, color='black',
                    weight='bold', rotation=math.degrees(theta[i]) + 90)
        elif 0.001 < p < 0.01:
            ax.text(theta[i], max_val * 0.95, '**', fontsize=18, color='black',
                    weight='bold', rotation=math.degrees(theta[i]) + 90)
        elif p < 0.001:
            ax.text(theta[i], max_val * 0.95, '***', fontsize=18,
                    color='black', weight='bold',
                    rotation=math.degrees(theta[i]) + 90,
                    ha='center', va='center')

    # Set legend and variables parameters.
    legend = ax.legend(np.unique(labels), loc=(0.95, 0.9), title='Profile #',
                       fontsize=14)
    frame = legend.get_frame()
    frame.set_facecolor('#eef0eb')
    frame.set_edgecolor('gray')
    ax.set_thetagrids(theta * 180 / np.pi, var_labels, zorder=1)
    ax.set_rlabel_position(-36)
    ax.set_ylim(min_val, max_val)
    yticks = np.arange(min_val, max_val, 0.5)
    ax.set_yticks(yticks)

    # Set spines and title parameters.
    for spine in ax.spines.values():
        spine.set_color('black')
    ax.grid(axis='y', color='white', linewidth=1, zorder=3)
    ax.grid(axis='x', color='white', linewidth=.5, zorder=2)
    ax.set_facecolor('#eef0eb')
    ax.set_xticklabels(var_labels)

    ax.set_title(f"{title}", weight='bold', size=16,
                 horizontalalignment='center')

    # Set the position for the labels.
    for label, angle in zip(ax.get_xticklabels(), theta):
        if angle == 0:
            label.set_horizontalalignment('left')
        elif angle == np.pi:
            label.set_horizontalalignment('right')
        elif 0 < angle < np.pi / 2 or angle > 3 * np.pi / 2:
            label.set_horizontalalignment('left')
        else:
            label.set_horizontalalignment('right')

    plt.tight_layout()
    plt.savefig(f"{output}", dpi=300)
    plt.close()


def create_radar_plot(nb_vars, frame='circle'):
    """
    Create a radar chart with `nb_vars` axes.

    Args:
        nb_vars (int):          Number of variables to plot.
        frame (str, optional):  Shape of the radar plot. Defaults to 'circle'.
                                Choices are 'circle' or 'polygon'.

    Returns:
        np.array:               Array of evenly spaced axis angles.
    """

    # Compute evenly spaced axis angles.
    theta = np.linspace(0, 2 * np.pi, nb_vars, endpoint=False)
    theta = np.concatenate((theta, [theta[0]]))

    class RadarTransform(PolarAxes.PolarTransform):

        def transform_path_non_affine(self, path):
            if path._interpolation_steps > 1:
                path = path.interpolated(nb_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):

        name = 'radar'
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Rotate plot to place the first axis at the top.
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default.

            Args:
                closed (bool, optional): _description_. Defaults to True.
            """
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """
            Override plot so that line is closed by default.
            """
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), nb_vars,
                                      radius=0.5, edgecolor='k')
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':

                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(nb_vars))
                spine.set_transform(
                    Affine2D().scale(0.5).translate(0.5, 0.5) + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)

    return theta
