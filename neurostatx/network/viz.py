from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from strenum import StrEnum


class NetworkLayout(StrEnum, Enum):
    """NetworkX layout names used when placing graph nodes."""

    KamadaKawai = ("kamada_kawai_layout",)
    Spectral = ("spectral_layout",)
    Spring = "spring_layout"


def membership_distribution(mat, output="./membership_distribution.png"):
    """Write histograms of top membership values and their delta.

    Parameters
    ----------
    mat : array
        Fuzzy membership matrix with one row per subject.
    output : str, optional
        Output filename. Defaults to ``"./membership_distribution.png"``.

    Examples
    --------
    >>> import numpy as np
    >>> from neurostatx.network.viz import membership_distribution
    >>> mat = np.array([[0.8, 0.2], [0.3, 0.7]])
    >>> membership_distribution(mat, output="membership.png")
    """

    # Fetching 1st highest membership value.
    high1st = np.max(mat, axis=1)

    # Fetching 2nd highest membership value.
    high2nd = np.partition(mat, -2, axis=1)[:, -2]

    # Calculating delta.
    delta = high1st - high2nd

    # Plotting the distribution.
    with plt.rc_context(
        {"font.family": "Sans Serif",
         "font.size": 12, "font.weight": "normal", "axes.titleweight": "bold"}
    ):

        fig, ax = plt.subplots(2, 2, figsize=(16, 16))

        # Plotting 1st highest membership value.
        sns.histplot(data=high1st, stat="density", bins=50, kde=True,
                     ax=ax[0, 0], color="lightgray")
        ax[0, 0].set_xlabel(
            "1st highest membership value distribution amongst all subjects."
        )
        ax[0, 0].axvline(x=np.median(high1st), ymin=0, ymax=1, color="gray",
                         linestyle="--")

        # Plotting 2nd highest memberhsip value.
        sns.histplot(data=high2nd, stat="density", bins=50, kde=True,
                     ax=ax[0, 1], color="lightgray")
        ax[0, 1].set_xlabel(
            "2nd highest membership value distribution amongst all subjects."
        )
        ax[0, 1].axvline(x=np.median(high2nd), ymin=0, ymax=1, color="gray",
                         linestyle="--")

        # Plotting delta.
        sns.histplot(data=delta, stat="density", bins=50, kde=True,
                     ax=ax[1, 0], color="lightgray")
        ax[1, 0].set_xlabel(
            "Delta (1st highest value - 2nd highest value) distribution "
            "amongst all subjects."
        )
        ax[1, 0].axvline(x=np.median(delta), ymin=0, ymax=1, color="gray",
                         linestyle="--")

        ax[1, 1].remove()
        plt.tight_layout()
        plt.savefig(f"{output}")
        plt.close()


def creating_node_colormap(percentile_dict):
    """Map each node to a color from its highest percentile cluster.

    Parameters
    ----------
    percentile_dict : dict
        Mapping of cluster keys to per-node percentile arrays.

    Returns
    -------
    nodes_cmap : list
        Color for each node. Nodes with a max of 0 are ``"black"``.

    Examples
    --------
    >>> from neurostatx.network.viz import creating_node_colormap
    >>> colors = creating_node_colormap({"c1": [0, 1], "c2": [0, 0]})
    >>> colors[0]
    'black'
    """

    cmap = plt.cm.tab10(np.linspace(0, 1, 10))

    # Assuming it is impossible for node to have 2 main clusters.
    percentile_array = np.max(np.array(list(percentile_dict.values())), axis=0)

    nodes_cmap = []
    for i in percentile_array:
        if i == 0:
            nodes_cmap.append("black")
        else:
            nodes_cmap.append(cmap[i])

    return nodes_cmap


def create_cmap_from_list(array):
    """Map per-subject values to node colors.

    Integer zeros are drawn as ``"darkgrey"``; other integers use ``tab10``.
    Float arrays are ranked and mapped through ``plasma``.

    Parameters
    ----------
    array : np.array
        Per-subject values. Integers are treated as discrete conditions;
        floats are treated as a continuous scale.

    Returns
    -------
    nodes_cmap : list
        Color for each subject.

    Examples
    --------
    >>> import numpy as np
    >>> from neurostatx.network.viz import create_cmap_from_list
    >>> colors = create_cmap_from_list(np.array([0, 1, 2]))
    >>> colors[0]
    'darkgrey'
    """

    if array.dtype == "float64":
        cmap = plt.cm.plasma(np.linspace(0, 1, len(array)))

        ranks = array.argsort().argsort()

        nodes_cmap = []
        for i in ranks:
            nodes_cmap.append(cmap[i])

    else:
        cmap = plt.cm.tab10(np.linspace(0, 1, 10))

        nodes_cmap = []
        for i in array:
            if i == 0:
                nodes_cmap.append("darkgrey")
            else:
                nodes_cmap.append(cmap[i])

    return nodes_cmap
