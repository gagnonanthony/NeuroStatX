#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import coloredlogs
import logging
import os
import sys

from enum import Enum
import pandas as pd
import typer
from typing import List
from typing_extensions import Annotated

from CCPM.io.utils import (
    load_df_in_any_format,
    assert_input,
    assert_output_dir_exist,
)
from CCPM.utils.preprocessing import (
    remove_nans,
    plot_distributions,
    compute_shapiro_wilk_test,
    compute_correlation_coefficient,
    merge_dataframes,
)


class ContextChoices(str, Enum):
    paper = "paper"
    poster = "poster"
    talk = "talk"
    notebook = "notebook"


# Initializing the app.
app = typer.Typer(add_completion=False)


@app.command()
def main(
    in_dataset: Annotated[
        List[str],
        typer.Option(
            help="Input dataset(s) to filter. If multiple files are"
            "provided as input, will be merged according to "
            "the subject id columns.",
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
            "dataset to exclude in statistics and descriptive tables. "
            "(excluding id_column)",
            show_default=False,
            rich_help_panel="Essential Files Options",
        ),
    ],
    out_folder: Annotated[
        str,
        typer.Option(
            help="Path of the folder in which the results will be written. "
            "If not specified, current folder and default "
            "name will be used (e.g. = ./output/).",
            rich_help_panel="Essential Files Options",
        ),
    ] = "./default",
    disable_plotting: Annotated[
        bool,
        typer.Option(
            "--disable_plotting",
            help="If used, will disable all plotting and simply output "
            "tables.",
            rich_help_panel="Essential Files Options",
            show_default=True,
        ),
    ] = False,
    context: Annotated[
        ContextChoices,
        typer.Option(
            help="Context to use as plot style.",
            rich_help_panel="Matplotlib Options",
            case_sensitive=False,
        ),
    ] = ContextChoices.poster,
    font_scale: Annotated[
        int,
        typer.Option(
            help="Font size to use in plots.", rich_help_panel="Matplotlib "
            "Options"
        ),
    ] = 1,
    cmap: Annotated[
        str,
        typer.Option(
            help="Cmap to be used in the heatmap plot.",
            rich_help_panel="Matplotlib Options",
        ),
    ] = "mako",
    annotate: Annotated[
        bool,
        typer.Option(
            "--annotate",
            help="If true, heatmap will show pearson correlation coefficient "
            "in the center of each square.",
            rich_help_panel="Matplotlib Options",
        ),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option(
            "-v",
            "--verbose",
            help="If true, produce verbose output.",
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
    FILTERING DATASET
    -----------------
    CCPM_filtering_dataset is designed to compute basic statistics, plotting
    distributions and correlation matrix of raw data.
    \b
    STEPS
    -----
    Filtering and evaluating steps are:
        1)  Removing rows containing NaNs. (2 dataframes will be outputted, one
            with the cleaned dataset and one with the excluded rows.) Both
            files should be validated after running this script to ensure
            correct filtering.
        2)  Computing global basic statistics for each variable. Stats
            included are : count, mean, std, min, 25%, 50%, 75%, max, Wilk and
            Wilk p-values.
        3)  Plotting distributions graph. 2 plots are outputted : histogram
            with kde and ecdf plot.
        4)  Computing correlation matrix. Correlation between all variables
            from the dataset is computed and plotted as a heatmap.
    \b
    Manual evaluation of all outputted results is recommended to ensure the
    script behaved correctly.
    \b
    When inputting multiple dataset, users must validate that the index used
    (for example subject's id) appear only once in a single dataframe.
    Otherwise, merging dataframe will not behave correctly and index will not
    be aligned.
    \b
    EXAMPLE USAGE
    -------------
    CCPM_filtering_dataset.py --in-dataset IN_DATASET --id-column subjectkey
        --desc-columns subjectkey --out-folder OUT_FOLDER
    """

    if verbose:
        logging.getLogger().setLevel(logging.INFO)
        coloredlogs.install(level=logging.INFO)

    assert_input(in_dataset)
    assert_output_dir_exist(overwrite, out_folder, create_dir=True)

    # Loading dataframe.
    logging.info("Loading dataset(s)...")
    if len(in_dataset) > 1:
        if id_column is None:
            sys.exit(
                "Column name for index matching is required when inputting "
                "multiple dataframe."
            )
        dict_df = {i: load_df_in_any_format(i) for i in in_dataset}
        raw_df = merge_dataframes(dict_df, id_column)
    else:
        raw_df = load_df_in_any_format(in_dataset[0])
    descriptive_columns = [n for n in range(0, desc_columns)]

    # Removing NaNs from dataset and saving the rows in a different file.
    # Exporting excluded rows and included rows in
    # 2 different files.
    logging.info("Filtering dataset to remove NaN values. ")
    nans, clean = remove_nans(raw_df)
    nans.to_excel(f"{out_folder}/excluded_rows.xlsx", header=True, index=True)
    clean.to_excel(f"{out_folder}/included_rows.xlsx", header=True, index=True)

    # Exporting global description statistics.
    variable_for_stats = clean.drop(
        clean.columns[descriptive_columns], axis=1, inplace=False
    ).astype("float")
    logging.info("Computing global descriptive statistics.")
    description_df = variable_for_stats.describe()
    wilk, pvalues = compute_shapiro_wilk_test(variable_for_stats)
    w_pval = pd.DataFrame(
        [wilk, pvalues], index=["Wilk", "Wilk pvalues"],
        columns=description_df.columns
    )
    description_df = pd.concat([description_df, w_pval], axis=0)
    description_df.to_excel(
        f"{out_folder}/descriptive_statistics.xlsx", header=True, index=True
    )

    # Plotting functions and/or computing correlation heatmap.
    if disable_plotting:
        logging.info("Plotting is disabled, skipping the plotting steps.")
        corr_mat = variable_for_stats.corr()

    else:
        logging.info("Plotting distributions for the complete dataframe.")
        path_plot = os.path.join(out_folder + "/" + "Distribution_Plots")
        os.makedirs(path_plot, exist_ok=True)
        variable_to_plot = clean.drop(
            clean.columns[descriptive_columns], axis=1, inplace=False
        )
        # Plotting distributions.
        plot_distributions(
            variable_to_plot, path_plot, context=context, font_scale=font_scale
        )

        # Plotting heatmap and computing correlation matrix.
        logging.info("Generating correlation matrix and heatmap.")
        if annotate:
            if len(variable_to_plot.columns) > 20:
                logging.warning(
                    "Due to high number of variables, annotating heatmap is "
                    "deactivated. Annotation is "
                    "only available for dataset with 20 or less variables."
                )
            corr_mat = compute_correlation_coefficient(
                variable_to_plot,
                out_folder,
                context=context,
                font_scale=font_scale,
                cmap=cmap,
                annot=False if len(variable_to_plot.columns) > 20 else True,
            )
        else:
            corr_mat = compute_correlation_coefficient(
                variable_to_plot,
                out_folder,
                context=context,
                font_scale=font_scale,
                cmap=cmap,
            )

    corr_mat.to_excel(
        f"{out_folder}/correlation_coefficient.xlsx", header=True, index=True
    )

    # Generating basic recommendations. TODO: Set up a easy to interpret
    # report with useful distribution infos.


#    if report:
#        logging.info('Generating the report...')
#        pdf = PDF()
#        pdf.alias_nb_pages()
#        pdf.set_font('Times', '', 12)
#
#        wilk_value = description_df.loc['Wilk',]
#        with open('basic_stats.txt', 'w') as f:
#            for i in range(0, len(wilk_value)):
#                if wilk_value[i] < args.wilk_threshold:
#                    f.write(f"Variable {wilk_value.index[i]} present a W < "
#                             "0.95 ({round(wilk_value[i], 2)}), you may "
#                            f"need to inspect the histogram and ecdf plots to"
#                             " determine the suitability "
#                            f"for parametric test. \n")
#
#        pdf.print_chapter(1, 'Normality concerns : ', 'basic_stats.txt')
#        os.remove('basic_stats.txt')
#
#        with open('correlation_stats.txt', 'w') as f:
#            corr = np.triu(corr_mat.to_numpy(), 1)
#            for i in range(0, corr.shape[0]):
#                for j in range(0, corr.shape[1]):
#                    if corr[i, j] > args.corr_threshold:
#                        f.write(f"Variables {description_df.columns[i]} and "
#                                 "{description_df.columns[j]} "
#                                f"present a high pearson correlation "
#                                 "coefficient > 0.8 ({round(corr[i, j], 3)})."
#                                 " It may"
#                                f" be interesting to evaluate the need of "
#                                 "keeping both variables since they carry "
#                                f"similar information. \n")
#
#        pdf.print_chapter(2, 'Correlation concerns : ',
#                          'correlation_stats.txt')
#        os.remove('correlation_stats.txt')
#        pdf.output(f'{out_folder}/report.pdf', 'F')


if __name__ == "__main__":
    app()
