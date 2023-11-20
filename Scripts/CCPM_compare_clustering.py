#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import coloredlogs
import logging

import matplotlib.pyplot as plt
import pandas as pd
import typer
from typing import List
from typing_extensions import Annotated
import seaborn as sns

from CCPM.io.utils import (assert_input, assert_output_dir_exist,
                           load_df_in_any_format)
from CCPM.clustering.metrics import compute_rand_index

# Initializing the app.
app = typer.Typer(add_completion=False)


@app.command()
def main(
    in_dataset: Annotated[
        List[str],
        typer.Option(
            help="Input dataset(s) (at least 2 are expected to produce a "
            "comparison)",
            show_default=False,
            rich_help_panel="Essential Files Options",
        ),
    ],
    id_column: Annotated[
        str,
        typer.Option(
            help="Name of the column containing the subject's ID tag. "
            "Required for proper handling of IDs and "
            "merging multiple datasets.",
            show_default=False,
            rich_help_panel="Essential Files Options",
        ),
    ],
    desc_columns: Annotated[
        int,
        typer.Option(
            help="Number of descriptive columns at the beginning of the "
            "dataset to exclude in statistics and descriptive tables.",
            show_default=False,
            rich_help_panel="Essential Files Options",
        ),
    ],
    out_folder: Annotated[
        str,
        typer.Option(
            help="Output folder for the predicted membership matrix.",
            show_default=False,
            rich_help_panel="Essential Files Options",
        ),
    ],
    columns_name: Annotated[
        List[str],
        typer.Option(
            help="Name given to each input dataset (needs to be in the same "
            "order as the input datasets)",
            rich_help_panel="Visualization Options",
            show_default=True,
        )
    ] = None,
    title: Annotated[
        str,
        typer.Option(
            help="Heatmap title.",
            show_default=True,
            rich_help_panel="Visualization Options",
        )
    ] = 'Adjusted Rand Index Heatmap',
    verbose: Annotated[
        bool,
        typer.Option(
            "-v",
            "--verbose",
            help="If true, produce verbose output.",
            rich_help_panel="Optional parameters",
        ),
    ] = False,
    save_parameters: Annotated[
        bool,
        typer.Option(
            "-s",
            "--save_parameters",
            help="If true, will save input parameters to .txt file.",
            rich_help_panel="Optional parameters",
        ),
    ] = False,
    overwrite: Annotated[
        bool,
        typer.Option(
            "-f",
            "--overwrite",
            help="If true, force overwriting of existing " "output files.",
            rich_help_panel="Optional parameters",
        ),
    ] = False,
):
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
    dict_df = {i: load_df_in_any_format(i) for i in in_dataset}

    # Dropping desc column.
    descriptive_columns = [n for n in range(0, desc_columns)]
    for k in dict_df.keys():
        dict_df[k] = dict_df[k].drop(
            dict_df[k].columns[descriptive_columns], axis=1)

    logging.info("Computing Adjusted Rand Index...")
    # Comparison between each clustering results.
    ari = compute_rand_index(dict_df)

    logging.info("Plotting and saving results...")
    # Create heatmap.
    fig = plt.figure(figsize=(12, 7))
    axes = fig.add_subplot(111)

    # Exporting symmetric matrix.
    if columns_name is None:
        columns_name = [i for i in range(0, len(in_dataset))]
    mat = pd.DataFrame(ari, columns=columns_name,
                       index=columns_name)
    mat.to_excel(f'{out_folder}/ari_matrix.xlsx', index=True, header=True)

    # Plotting heatmap.
    sns.heatmap(mat, annot=True, ax=axes)
    axes.set_title(title)
    plt.savefig(f"{out_folder}/ari_heatmap.png")
    plt.close()


if __name__ == "__main__":
    app()
