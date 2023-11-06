#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import coloredlogs
import logging
import os
import sys

from rich.progress import Progress, SpinnerColumn, TextColumn
from tableone import TableOne
import typer
from typing import List
from typing_extensions import Annotated

from CCPM.io.utils import assert_input, assert_output, load_df_in_any_format
from CCPM.utils.preprocessing import merge_dataframes


def rename_columns(df, old_names, new_names):
    """
    Function renaming specific columns according to a list of new and old
    column names.

    :param df:              Pandas dataframe object.
    :param old_names:       List of old column name as strings.
    :param new_names:       List of new column name as strings.
    :return:
    Pandas dataframe object containing the renamed columns.
    """
    if len(old_names) != len(new_names):
        raise ValueError("Number of old names and new names must be the same.")

    for i, old_name in enumerate(old_names):
        if old_name not in df.columns:
            raise ValueError(f"Column {old_name} not found in DataFrame.")

    new_names_set = set(new_names)
    if len(new_names_set) != len(new_names):
        raise ValueError("New names contain duplicates.")

    new_df = df.copy()
    for i in range(len(old_names)):
        if old_names[i] in new_df.columns:
            new_df.rename(columns={old_names[i]: new_names[i]}, inplace=True)
    return new_df


def binary_to_yes_no(df, cols):
    """
    Function to change binary answers (1/0) to Yes or No in specific columns
    from a Pandas Dataframe.
    *** Please validate that yes and no are assigned to the correct values,
    default behavior is yes = 1 and no = 0 ***
    :param df:              Pandas' dataframe object.
    :param cols:            List of column names.
    :return:
    Pandas' dataframe object with changed binary answers.
    """
    for col in cols:
        if df[col].isin([0.0, 1.0, 2.0, "nan"]).any():
            df[col] = df[col].apply(
                lambda x: "Yes"
                if x == 1
                else "No"
                if x == 0
                else "Don't know or missing value"
            )
    return df


def get_column_indices(df, column_names):
    """
    Function to extract column index based on a list of column names.
    :param df:              Pandas dataframe object.
    :param column_names:    List of column names as strings.
    :return:
    List of column index.
    """
    indices = []
    for name in column_names:
        try:
            index = df.columns.get_loc(name)
            indices.append(index)
        except KeyError:
            print(f"Column '{name}' not found in DataFrame.")
    return indices


# Initializing the app.
app = typer.Typer(add_completion=False)


@app.command()
def main(
    in_dataset: Annotated[
        List[str],
        typer.Option(
            help="Input dataset(s) to use in for the descriptive table. "
            "If multiple files are provided as input,"
            "will be merged according to the subject id columns.",
            rich_help_panel="Essential Files Options",
            show_default=False,
        ),
    ],
    id_column: Annotated[
        str,
        typer.Option(
            help="Column name containing the subject ids. (Necessary when"
            "merging multiple dataframe.",
            rich_help_panel="Essential Files Options",
            show_default=False,
        ),
    ],
    output: Annotated[
        str,
        typer.Option(
            help="Path and filename of the output dataframe.",
            rich_help_panel="Essential Files Options",
            show_default=False,
        ),
    ],
    raw_variables: Annotated[
        List[str],
        typer.Option(
            "-r",
            "--raw_variables",
            help="Variables to include in final table (raw_name). Must"
            "include every variable from every datatype (categorical"
            " or continuous).",
            rich_help_panel="Formatting Options",
            show_default=False,
        ),
    ],
    categorical_variables: Annotated[
        List[str],
        typer.Option(
            "-c",
            "--categorical_variables",
            help="Variables containing answers as string."
            "(Variable name still needs to be stated in"
            "--raw_variables and in the same order.",
            rich_help_panel="Formatting Options",
            show_default=False,
        ),
    ],
    variable_names: Annotated[
        List[str],
        typer.Option(
            "-n",
            "--variable_names",
            help="Specify new variables' name for a clean table. Needs"
            "to be the same length as --raw_variables and in the"
            "same order. State the new names as 'Column Name' if "
            "you want to enter spaces in your column name."
            "Otherwise, the parser will interpret the next word as"
            "a new variable.",
            rich_help_panel="Formatting Options",
            show_default=False,
        ),
    ],
    apply_yes_or_no: Annotated[
        bool,
        typer.Option(
            "--apply_yes_or_no",
            help="If true, will change binary answers for specified"
            "categorical variables to yes or no (assume yes = 1 and"
            "no = 0).",
            rich_help_panel="Formatting Options",
        ),
    ] = True,
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
    DEMOGRAPHICS AND MEDICAL STATISTICS
    -----------------------------------
    CCPM_creating_medical_statistics.py is a script that creates and exports a
    demographics or medical data table. It can handle continuous, categorical
    and binary variables.
    \b
    Input can be a single or multiple files (in this case, --id_column needs
    to be specified).
    \b
    Supported output format is : csv, xlsx, html, json and tex.
    \b
    To rename variables for final clean formatting, use --variable_names
    arguments. Variable names have to be in the same order as the list provided
    in --raw_variables.
    \b
    --apply_yes_or_no assumes that variables specified in
    --categorical_variables have 1 = yes and 0 = no. Please validate that your
    dataset assumes the same values. If it is not the case, please modify your
    dataset before launching this script.
    \b
    EXAMPLE USAGE
    -------------
    CCPM_creating_medical_statistics.py --in-dataset in_dataset
    \b                                  --output out_table
    \b                                  --id-column subid -r Sex -r Age -r IQ
    \b                                  -c Sex -n Sex -n Age -n Quotient
    \b                                  --apply_yes_or_no -f
    """

    if verbose:
        logging.getLogger().setLevel(logging.INFO)
        coloredlogs.install(level=logging.INFO)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=False,
    ) as progress:
        task = progress.add_task(description="Running script...", total=None)
        assert_input(in_dataset)
        assert_output(overwrite, output)

        # Loading dataframe.
        logging.info("Loading dataset(s)...")
        if len(in_dataset) > 1:
            if id_column is None:
                sys.exit(
                    "Column name for index matching is required when "
                    "inputting multiple dataframe."
                )
            dict_df = {i: load_df_in_any_format(i) for i in in_dataset}
            raw_df = merge_dataframes(dict_df, id_column)
        else:
            raw_df = load_df_in_any_format(in_dataset[0])

        # Changing binary response for yes or no in categorical variables.
        logging.info("Changing binary values to yes or no...")
        if apply_yes_or_no:
            assert len(categorical_variables) > 0, (
                "To change values to yes or no, the argument "
                "--categorical_variables must be provided."
            )
            raw_df = binary_to_yes_no(raw_df, categorical_variables)

        # Changing column names.
        logging.info("Changing column names...")
        col_index = get_column_indices(raw_df, categorical_variables)
        new_df = rename_columns(raw_df, raw_variables, variable_names)
        new_cat_names = list(new_df.columns[col_index])

        # Creating descriptive table.
        logging.info("Creating table...")
        mytable = TableOne(new_df, columns=variable_names,
                           categorical=new_cat_names)

        # Exporting table in desired output format.
        logging.info("Exporting table...")
        _, ext = os.path.splitext(output)
        if ext == ".csv":
            mytable.to_csv(output)
        elif ext == ".xlsx":
            mytable.to_excel(output)
        elif ext == ".html":
            mytable.to_html(output)
        elif ext == ".json":
            mytable.to_json(output)
        elif ext == ".tex":
            mytable.to_latex(output)

        progress.update(
            task, completed=True, description="[green]Table generated. "
                                              "Have fun! :beer:"
        )


if __name__ == "__main__":
    app()
