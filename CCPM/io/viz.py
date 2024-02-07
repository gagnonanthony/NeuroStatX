import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from adjustText import adjust_text
from matplotlib import rc
import numpy as np


def autolabel(rects, axs):
    """
    Function to attach a text label over or under the bar in a bar graph.
    :param rects:       Graphical object from matplotlib.
    :param axs:         Axe number.
    :return:            Graphical object with annotated bars.
    """
    texts = []
    for rect in rects:
        height = rect.get_height()
        if height > 0:
            texts.append(
                axs.text(
                    rect.get_x() + rect.get_width() / 2.0,
                    (height * 1.05),
                    "%.3f" % float(height),
                    ha="center",
                    va="bottom",
                )
            )
        else:
            texts.append(
                axs.text(
                    rect.get_x() + rect.get_width() / 2.0,
                    (height * 1.05) - 0.15,
                    "%.3f" % float(height),
                    ha="center",
                    va="bottom",
                )
            )

    # adjust the position of text labels to avoid overlapping
    adjust_text(texts, ax=axs, add_objects=rects, only_move="y+")


def flexible_barplot(
    values, labels, num_axes, title, filename, xlabel=None, ylabel=None
):
    """
    Function to generate a bar plot with multiple axes in a publication-ready
    style.

    :param values:
    :param labels:
    :param num_axis:
    :param title:
    :param xlabel:
    :param ylabel:
    :param filename:
    :return:
    """

    def determine_layout(nb_axes):
        """
        Returns the optimal number of rows and columns for the bar plot.
        :param num_axes:
        :return:
        """
        num_rows = int(np.sqrt(nb_axes))
        num_cols = int(np.ceil(nb_axes / num_rows))

        return num_rows, num_cols

    # Fetch optimal number of rows and columns.
    num_rows, num_cols = determine_layout(num_axes)

    # Setting up figure and style.
    fig, axes = plt.subplots(
        nrows=num_rows, ncols=num_cols, figsize=(8 * num_cols, 6 * num_rows)
    )

    # Generate each bar plot.
    for i, ax in enumerate(axes.flat):
        if i < num_axes:
            # Set theme for plot.
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.yaxis.set_major_locator(MultipleLocator(0.5))
            ax.yaxis.set_minor_locator(MultipleLocator(0.1))
            ax.grid(False)
            rc(
                "font",
                **{"family": "sans-serif", "sans-serif": ["DejaVu Sans"],
                   "size": 10},
            )

            # Getting values
            data = values[i]

            # Set the bars.
            num_bars = len(data)
            bar_width = 0.8 / num_cols
            x_positions = np.arange(num_bars)

            # Setting limits
            ax.set_ylim(-1, 1)

            # Plot the bars.
            bars = ax.bar(
                x_positions, data, bar_width, align="center", tick_label=labels
            )

            # Adding labels to bars.
            autolabel(bars, ax)

            # Setting labels
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(f"{title} : Factor {i + 1}")

            # Setting tick labels.
            label_format = ""
            ticks_loc = ax.get_xticks().tolist()
            ax.set_xticks(ax.get_xticks().tolist())
            ax.set_xticklabels([label_format.format(x) for x in ticks_loc])
            ax.set_xticklabels(labels, rotation=45)
            plt.setp(ax.xaxis.get_majorticklabels(), ha="right")
        else:
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(f"{filename}", dpi=300, bbox_inches="tight")
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
