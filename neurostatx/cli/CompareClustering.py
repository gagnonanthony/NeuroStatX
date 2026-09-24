#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import coloredlogs
import logging

import matplotlib.pyplot as plt
from cyclopts import App, Parameter
from typing import List
from typing_extensions import Annotated
import seaborn as sns

from neurostatx.io.utils import (assert_input, assert_output_dir_exist)
from neurostatx.io.loader import DatasetLoader
from neurostatx.clustering.metrics import compute_rand_index

# Initializing the app.
app = App(default_parameter=Parameter(negative=()))
"""Cyclopts application for the CompareClustering command-line tool."""


@app.default()
def CompareClustering(
    in_dataset: Annotated[
        List[str],
        Parameter(
            show_default=False,
            group="Essential Files Options",
        ),
    ],
    id_column: Annotated[
        str,
        Parameter(
            show_default=False,
            group="Essential Files Options",
        ),
    ],
    desc_columns: Annotated[
        int,
        Parameter(
            show_default=False,
            group="Essential Files Options",
        ),
    ],
    out_folder: Annotated[
        str,
        Parameter(
            show_default=False,
            group="Essential Files Options",
        ),
    ],
    columns_name: Annotated[
        List[str],
        Parameter(
            group="Visualization Options",
            show_default=True,
        )
    ] = [],
    cmap: Annotated[
        str,
        Parameter(
            show_default=True,
            group="Visualization Options",
        )
    ] = "magma",
    title: Annotated[
        str,
        Parameter(
            show_default=True,
            group="Visualization Options",
        )
    ] = 'Adjusted Rand Index Heatmap',
    verbose: Annotated[
        bool,
        Parameter(
            "-v",
            "--verbose",
            group="Optional parameters",
        ),
    ] = False,
    save_parameters: Annotated[
        bool,
        Parameter(
            "-s",
            "--save_parameters",
            group="Optional parameters",
        ),
    ] = False,
    overwrite: Annotated[
        bool,
        Parameter(
            "-f",
            "--overwrite",
            group="Optional parameters",
        ),
    ] = False,
):
    """Compare clustering solutions with an Adjusted Rand Index (ARI) heatmap.

    CompareClustering compares clustering results from multiple solutions using
    ARI and writes a heatmap of the pairwise scores.

    Notes
    -----

    **Adjusted Rand Index**

    The Adjusted Rand Index (ARI) measures similarity between two clustering
    results by comparing predicted labels to ground-truth labels [1]. ARI
    ranges from -1 to 1, where 1 means the two clusterings are identical, 0
    means they are independent (as good as random labelling), and -1 means they
    are completely different. ARI extends the Rand Index (RI) [2] by
    accounting for the fact that RI is expected to be higher for a large number
    of clusters. See also [3] and the scikit-learn documentation [4].

    References
    ----------
    [1] [Hubert, L., & Arabie, P. (1985). Comparing
    partitions](https://doi.org/10.1007/BF01908075)

    [2] [Rand, W. M. (1971). Objective criteria for the evaluation of clustering
    methods](https://doi.org/10.2307/2284239)

    [3] [Steinley, D. (2004). Properties of the Hubert-Arabie adjusted Rand
    index](https://psycnet.apa.org/doi/10.1037/1082-989X.9.3.386)

    [4] [scikit-learn Adjusted Rand
    Index](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html)

    Parameters
    ----------
    in_dataset : List[str]
        Input dataset(s) (at least 2 are expected to produce a comparison).
    id_column : str
        Name of the column containing the subject's ID tag. Required for
        proper handling of IDs and merging multiple datasets.
    desc_columns : int
        Number of descriptive columns at the beginning of the dataset to
        exclude in statistics and descriptive tables.
    out_folder : str
        Output folder containing the results.
    columns_name : List[str], optional
        Name given to each input dataset (needs to be in the same order as
        the input datasets). Defaults to ``[]``.
    cmap : str, optional
        Name of the colormap to use. Defaults to ``magma``. See
        https://matplotlib.org/stable/tutorials/colors/colormaps.html.
    title : str, optional
        Heatmap title. Defaults to ``Adjusted Rand Index Heatmap``.
    verbose : bool, optional
        If true, produce verbose output. Defaults to False.
    save_parameters : bool, optional
        If true, will save input parameters to .txt file. Defaults to False.
    overwrite : bool, optional
        If true, force overwriting of existing output files. Defaults to False.

    Examples
    --------
    ```bash
    CompareClustering --in_dataset dataset1.csv --in_dataset dataset2.csv
    --in_dataset dataset3.csv --id_column ID --desc_columns 1 --out_folder
    ./ --columns_name dataset1 dataset2 dataset3 --title "ARI Heatmap"
    --verbose
    ```
    """

    if verbose:
        logging.getLogger().setLevel(logging.INFO)
        coloredlogs.install(level=logging.INFO)

    assert_input(in_dataset)
    assert_output_dir_exist(overwrite, out_folder, create_dir=True)

    if save_parameters:
        parameters = list(locals().items())
        with open(f"{out_folder}/parameters.txt", "w+") as f:
            for param in parameters:
                f.writelines(str(param))

    logging.info("Loading datasets...")
    # Loading all datasets into a dictionary.
    assert len(in_dataset) >= 2, "At least 2 datasets are required for "
    "                            comparison."
    dict_df = {
        i: DatasetLoader().load_data(df).get_data() for i,
        df in enumerate(in_dataset)
    }

    # Dropping desc column.
    descriptive_columns = [n for n in range(0, desc_columns)]
    for k in dict_df.keys():
        dict_df[k] = dict_df[k].drop(
            dict_df[k].columns[descriptive_columns], axis=1)

    logging.info("Computing Adjusted Rand Index...")
    # Comparison between each clustering results.
    ari = compute_rand_index(dict_df)

    logging.info("Plotting and saving results...")
    # Exporting symmetric matrix.
    if len(columns_name) == 0:
        columns_name = [f'{i+1}' for i in range(0, len(in_dataset))]
    mat = DatasetLoader().import_data(
        ari, columns=columns_name, index=columns_name)
    mat.save_data(f'{out_folder}/ari_matrix.csv', index=True, header=True)

    # Create heatmap.
    with plt.rc_context(
        {"font.family": "Sans Serif", "font.size": 12, "font.weight": "normal",
         "axes.titleweight": "bold"}
    ):
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(mat.get_data(), annot=True, ax=ax, cmap=cmap, linewidths=1,
                    vmin=-1, vmax=1)
        ax.set_title(title)
        plt.savefig(f"{out_folder}/ari_heatmap.png")
        plt.close()


if __name__ == "__main__":
    app()
