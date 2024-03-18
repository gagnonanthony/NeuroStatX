import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def determine_layout(nb_axes):
    """
    Returns the optimal number of rows and columns for the bar plot.

    Args:
        nb_axes (int):      Number of axes to plot.

    Returns:
        int, int:           Number of rows and columns.
    """

    num_rows = int(np.sqrt(nb_axes))
    num_cols = int(np.ceil(nb_axes / num_rows))

    return num_rows, num_cols


def flexible_barplot(
    values, num_axes, output, title='Barplot', xlabel=None, ylabel=None
):
    """
    Function to generate a bar plot with multiple axes in a publication-ready
    style.

    Args:
        values (pd.DataFrame):          Dataframe with the values to plot. The
                                        index represents the x-axis and the
                                        columns the y-axis.
        num_axes (int):                 Number of axes to plot.
        output (str):                   Output filename.
        title (str, optional):          Title of the plot.
        xlabel (str, optional):         Label for the x-axis.
        ylabel (str, optional):         Label for the y-axis.
    """

    # Fetch optimal number of rows and columns.
    num_rows, num_cols = determine_layout(num_axes)

    plotting_parameters = {
        'palette': 'magma',
        'saturation': 1,
        'orient': 'v',
    }

    with plt.rc_context(
        {"font.family": "Sans Serif",
         "font.size": 12, "font.weight": "normal", "axes.titleweight": "bold",
         }
    ):
        # Setting up figure and style.
        fig, axes = plt.subplots(
            nrows=num_rows, ncols=num_cols,
            figsize=(8 * num_cols, 6 * num_rows)
        )

        for i, ax in enumerate(axes.flat):
            if i < num_axes:
                sns.barplot(
                    data=values, x=values.index, y=values.columns[i],
                    ax=ax, **plotting_parameters
                )
                ax.set_title(values.columns[i])
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)

                ax.spines[['top', 'right', 'left', 'bottom']].set(linewidth=2)

                for bars in ax.containers:
                    ax.bar_label(bars, fmt='{:,.3f}', padding=1)

            else:
                ax.axis('off')

        fig.suptitle(title, fontsize=20, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{output}", dpi=300, bbox_inches="tight")
        plt.close()


def generate_coef_plot(df, permutation, pval, coefname, varname, output):
    """
    Function to generate a coefficient plot.

    Args:
        df (_type_): _description_
        coefname (_type_): _description_
        varname (_type_): _description_
        output (_type_): _description_
    """
    # Compute standard deviation from permutation testing.
    stdev = np.std(permutation, axis=0)

    # Creating list of colors.
    colors = []
    for p, c in zip(pval, df[coefname]):
        if p < 0.05 and c > 0:
            colors.append('green')
        elif p < 0.05 and c < 0:
            colors.append('red')
        else:
            colors.append('black')

    fig, ax = plt.subplots(figsize=(12, 7))
    bar = ax.bar(x=varname, height=coefname, data=df, color='none')
    ax.set_ylabel("Coefficient")
    ax.set_xlabel("Variables")
    ax.bar_label(bar, color='black', fontsize=15, label_type='edge',
                 labels=['*' if p < 0.05 else '' for p in pval],
                 padding=3)
    ax.axhline(y=0, color="lightgrey", linestyle="--", linewidth=1)
    ax.scatter(x=np.arange(df.shape[0]), marker="s", s=20, y=df[coefname],
               color=colors)
    ax.fill_between(df[varname], -stdev, stdev, alpha=0.2, color="lightgreen")
    ax.set_xticklabels(df[varname], fontdict={'fontsize': 5, 'rotation': 90,
                                              'horizontalalignment': 'center'})

    plt.tight_layout()
    plt.savefig(f"{output}")
    plt.close()
