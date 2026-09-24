import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def determine_layout(nb_axes):
    """Return a compact subplot grid for ``nb_axes`` panels.

    Parameters
    ----------
    nb_axes : int
        Number of axes to plot.

    Returns
    -------
    num_rows : int
        Number of rows.
    num_cols : int
        Number of columns.

    Examples
    --------
    >>> from neurostatx.io.viz import determine_layout
    >>> determine_layout(4)
    (2, 2)
    """

    num_rows = int(np.sqrt(nb_axes))
    num_cols = int(np.ceil(nb_axes / num_rows))

    return num_rows, num_cols


def flexible_barplot(
    df, nb_axes, output, cmap='magma', title='Barplot', xlabel=None,
    ylabel=None
):
    """Write a multi-panel bar plot from a dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Values to plot. The index is used as the x-axis and each column is
        drawn on its own axis.
    nb_axes : int
        Number of axes to plot.
    output : str
        Output filename.
    cmap : str, optional
        Colormap name. Defaults to ``"magma"``.
    title : str, optional
        Title of the figure. Defaults to ``"Barplot"``.
    xlabel : str, optional
        Label for the x-axis. Defaults to None.
    ylabel : str, optional
        Label for the y-axis. Defaults to None.

    Examples
    --------
    >>> import pandas as pd
    >>> from neurostatx.io.viz import flexible_barplot
    >>> df = pd.DataFrame({"a": [0.1, 0.2], "b": [0.3, 0.4]}, index=["x", "y"])
    >>> flexible_barplot(df, nb_axes=2, output="barplot.png")
    """

    # Fetch optimal number of rows and columns.
    num_rows, num_cols = determine_layout(nb_axes)

    plotting_parameters = {
        'palette': cmap,
        'saturation': 1,
        'orient': 'v',
    }

    with plt.rc_context(
        {"font.family": "Sans Serif",
         "font.size": 10, "font.weight": "normal", "axes.titleweight": "bold",
         }
    ):
        # Setting up figure and style.
        fig, axes = plt.subplots(
            nrows=num_rows, ncols=num_cols,
            figsize=(8 * num_cols, 6 * num_rows)
        )

        for i, ax in enumerate(axes.flat):
            if i < nb_axes:
                sns.barplot(
                    data=df, x=df.index, hue=df.index, y=df.columns[i],
                    legend=True,
                    ax=ax, **plotting_parameters
                )
                ax.set_title(df.columns[i])
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                ax.grid(False)
                ax.spines[['top', 'right', 'left', 'bottom']].set(linewidth=2)
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right',
                         va='top', rotation_mode='anchor')

                for bars in ax.containers:
                    ax.bar_label(bars, fmt='{:,.3f}', padding=1)

            else:
                ax.axis('off')

        fig.suptitle(title, fontsize=20, fontweight='bold')

        plt.tight_layout()
        plt.savefig(f"{output}", dpi=300, bbox_inches="tight")
        plt.close()


def generate_coef_plot(df, pval, coefname, varname, output, cmap="magma"):
    """Write a horizontal coefficient bar plot with significance stars.

    Parameters
    ----------
    df : pd.DataFrame
        Table of coefficients and variable names.
    pval : array-like
        P-values aligned with ``df``. Values below 0.05 are marked with ``*``.
    coefname : str
        Name of the column containing the coefficients.
    varname : str
        Name of the column containing the variable names.
    output : str
        Output filename.
    cmap : str, optional
        Colormap name. Defaults to ``"magma"``.

    Examples
    --------
    >>> import pandas as pd
    >>> from neurostatx.io.viz import generate_coef_plot
    >>> df = pd.DataFrame({"var": ["a", "b"], "coef": [0.4, -0.2]})
    >>> generate_coef_plot(df, [0.01, 0.2], "coef", "var", "coef.png")
    """

    coef = df[coefname]
    label = ['*' if p < 0.05 else '' for p in pval]
    y = np.arange(0, len(coef))

    plotting_parameters = {
        'data': df,
        'x': coefname,
        'y': varname,
        'palette': cmap,
        'saturation': 1,
        'orient': 'h'
    }

    with plt.rc_context(
        {"font.family": "Sans Serif",
         "font.size": 18, "font.weight": "normal", "axes.titleweight": "bold",
         }
    ):
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.barplot(ax=ax, **plotting_parameters)

        ax.spines[['left', 'right', 'bottom', 'top']].set(linewidth=2)
        ax.set_ylabel('')
        ax.set_xlabel('ß coefficients', fontdict={'fontweight': 'bold',
                                                  'fontsize': 20})

        for i in range(len(coef)):
            if coef[i] < 0:
                plt.text(coef[i], y[i] + 0.05, label[i], ha='right',
                         va='center_baseline', color='black', fontsize=35,
                         weight='bold')
            else:
                plt.text(coef[i], y[i] + 0.05, label[i], ha='left',
                         va='center_baseline', color='black', fontsize=35,
                         weight='bold')

    plt.tight_layout()
    plt.savefig(f"{output}")
    plt.close()


def flexible_hist(df, output, cmap="magma", title="Histogram",
                  xlabel=None, ylabel=None):
    """Write a single histogram of all dataframe columns.

    Parameters
    ----------
    df : pd.DataFrame
        Values to plot. Columns are treated as separate variables.
    output : str
        Output filename.
    cmap : str, optional
        Colormap name. Defaults to ``"magma"``.
    title : str, optional
        Title of the plot. Defaults to ``"Histogram"``.
    xlabel : str, optional
        Label for the x-axis. Defaults to None.
    ylabel : str, optional
        Label for the y-axis. Defaults to None.

    Examples
    --------
    >>> import pandas as pd
    >>> from neurostatx.io.viz import flexible_hist
    >>> df = pd.DataFrame({"a": [0.1, 0.2, 0.3], "b": [1.0, 1.1, 0.9]})
    >>> flexible_hist(df, "hist.png")
    """

    with plt.rc_context(
        {"font.family": "Sans Serif",
         "font.size": 12, "font.weight": "bold", "axes.titleweight": "bold"}
    ):

        # Setting up figure and style.
        fig, ax = plt.subplots(figsize=(12, 10))

        sns.histplot(data=df.melt(), x='value', hue='variable', ax=ax,
                     kde=True, palette=cmap, stat='density')

        ax.spines[['top', 'right', 'left', 'bottom']].set(linewidth=1.5)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel(xlabel, fontweight='bold')
        ax.set_ylabel(ylabel, fontweight='bold')
        ax.legend(df.columns, title='Variables', fontsize=12,
                  title_fontsize=14)

        plt.tight_layout()
        plt.savefig(f'{output}')
        plt.close()
