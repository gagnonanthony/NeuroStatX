#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
=============================================================================
            ________    ________   ________     ____     ____
           /    ____|  /    ____| |   ___  \   |    \___/    |
          /   /       /   /       |  |__|   |  |             |
         |   |       |   |        |   _____/   |   |\___/|   |
          \   \_____  \   \_____  |  |         |   |     |   |
           \________|  \________| |__|         |___|     |___|
              Children Cognitive Profile Mapping Toolbox©
=============================================================================




"""

import argparse
import logging
import os

from tableone import TableOne

from CCPM.io.utils import (add_verbose_arg,
                           add_overwrite_arg,
                           assert_input,
                           assert_output,
                           load_df_in_any_format)
from CCPM.utils.preprocessing import (merge_dataframes)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('-i', '--in_dataset', nargs='+', required=True,
                   help="Input dataset(s). When inputting multiple datasets, all columns except the identifier column"
                        "need to be unique.")
    p.add_argument('--identifier_column',
                   help='Column name containing the subject ids. (Necessary when merging multiple dataframe.')
    p.add_argument('-o', '--output', required=True,
                   help="Path and filename of the output dataframe.")

    idx = p.add_argument_group('Formatting options.')
    idx.add_argument('--total_variables', nargs='+',
                     help="Variables to include in final table (raw_name).")
    idx.add_argument('--categorical_variables', nargs='+',
                     help='Variables containing answers as string. (variable name still needs to be stated in'
                          '--total_variables and in the same order)')
    idx.add_argument('--var_names', nargs='+',
                     help="Specify new variables' name for a clean table. Needs to be the same length as"
                          "--total_variables and in the same order. Use 'Column Name' if you want to enter spaces"
                          "in your column name. Otherwise, the parser will interpret the next word as a new variable.")
    idx.add_argument('--apply_yes_or_no', action='store_true',
                     help="If true, will change binary answers for categorical variables to yes or no"
                          "(assumes yes = 1 and no = 0)")

    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def rename_columns(df, old_names, new_names):
    """
    Function renaming specific columns according to a list of new and old column names.
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
    Function to change binary answers (1/0) to Yes or No in specific columns from a Pandas Dataframe.
    *** Please validate that yes and no are assigned to the correct values, default behavior is yes = 1
    and no = 0 ***
    :param df:              Pandas' dataframe object.
    :param cols:            List of column names.
    :return:
    Pandas' dataframe object with changed binary answers.
    """
    for col in cols:
        if df[col].isin([0., 1., 2., 'nan']).any():
            df[col] = df[col].apply(lambda x: "Yes" if x == 1 else 'No' if x == 0 else "Don't know or missing value")
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


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    assert_input(parser, args.in_dataset)
    assert_output(parser, args, args.output)

    # Loading dataframe.
    logging.info('Loading dataset(s)...')
    if len(args.in_dataset) > 1:
        if args.identifier_column is None:
            parser.error('Column name for index matching is required when inputting multiple dataframe.')
        dict_df = {i: load_df_in_any_format(i) for i in args.in_dataset}
        raw_df = merge_dataframes(dict_df, args.identifier_column)
    else:
        raw_df = load_df_in_any_format(args.in_dataset[0])

    # Changing binary response for yes or no in categorical variables.
    logging.info('Changing binary values to yes or no...')
    if args.apply_yes_or_no:
        assert len(args.categorical_variables) > 0, 'To change values to yes or no, the argument --categorical_values' \
                                                    ' must be provided.'
        raw_df = binary_to_yes_no(raw_df, args.categorical_variables)

    # Changing column names.
    logging.info('Changing column names...')
    col_index = get_column_indices(raw_df, args.categorical_variables)
    new_df = rename_columns(raw_df, args.total_variables, args.var_names)
    new_cat_names = list(new_df.columns[col_index])

    # Creating descriptive table.
    logging.info('Creating table...')
    mytable = TableOne(new_df, columns=args.var_names, categorical=new_cat_names)

    # Exporting table in desired output format.
    logging.info('Exporting table...')
    _, ext = os.path.splitext(args.output)
    if ext == '.csv':
        mytable.to_csv(args.output)
    elif ext == '.xlsx':
        mytable.to_excel(args.output)
    elif ext == '.html':
        mytable.to_html(args.output)
    elif ext == '.json':
        mytable.to_json(args.output)
    elif ext == '.tex':
        mytable.to_latex(args.output)


if __name__ == '__main__':
    main()
