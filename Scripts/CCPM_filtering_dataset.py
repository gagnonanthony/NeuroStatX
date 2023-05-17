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

CCPM_filtering_dataset is designed to compute basic statistics, plotting
distributions and correlation matrix of raw data.

Filtering and evaluating steps are:
    1)  Removing rows containing NaNs. (2 dataframes will be outputted, one
        with the cleaned dataset and one with the excluded rows.) Both files
        should be validated after running this script to ensure correct filtering.
    2)  Computing global basic statistics for each variable. Stats included are : count, mean,
        std, min, 25%, 50%, 75%, max, Wilk and Wilk p-values.
    3)  Plotting distributions graph. 2 plots are outputted : histogram with kde and
        ecdf plot.
    4)  Computing correlation matrix. Correlation between all variables from the dataset
        is computed and plotted as a heatmap.
    5)  Final pdf file with indications based on descriptive statistics is outputted (if --report
        is selected). Indications will highlight variables with W statistic < 0.95 and pair
        of variables with pearson value > 0.8.

Manual evaluation of all outputted results is recommended to ensure the script behaved
correctly.

When inputting multiple dataset, users must validate that the index used (for example
subject's id) appear only once in a single dataframe. Otherwise, merging dataframe will
not behave correctly and index will not be aligned.

EXAMPLE USAGE :
CCPM_filtering_dataset.py -i FILE -o OUT_FOLDER -n 1

"""

import argparse
import logging
import os

import numpy as np
import pandas as pd

from CCPM.io.utils import (load_df_in_any_format,
                           PDF,
                           assert_input,
                           assert_output_dir_exist,
                           add_verbose_arg,
                           add_overwrite_arg)
from CCPM.utils.preprocessing import (remove_nans,
                                      plot_distributions,
                                      compute_shapiro_wilk_test,
                                      compute_correlation_coefficient,
                                      merge_dataframes)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('-i', '--in_dataset', metavar='FILE', required=True, nargs='+',
                   help='Dataset containing raw data all variables for all subjects.')
    p.add_argument('-o', '--out_folder', required=True,
                   help='Directory in which datasets, plots and statistics will be saved.')
    p.add_argument('-n', '--nb_descriptive_columns', type=int, required=True,
                   help='Number of descriptive columns at the beginning of the dataset to exclude in \n'
                        'statistics and descriptive tables.')
    p.add_argument('--identifier_column',
                   help='Column name containing the subject ids. (Necessary when merging multiple dataframe.')
    p.add_argument('--disable_plotting', action='store_true',
                   help='If used, will disable all plotting and simply output tables.')

    plot = p.add_argument_group('Matplotlib Options')
    plot.add_argument('--context', choices={'paper', 'poster', 'talk', 'notebook'}, default='poster',
                      help='Context to use as plot style.')
    plot.add_argument('--font_scale', type=int, default=1,
                      help='Font size to use in plots.')
    plot.add_argument('--aspect_ratio', type=float, default=1.5,
                      help='Aspect ratio for the plot.')
    plot.add_argument('--cmap', default='mako',
                      help='Cmap used in the heatmap plot.')
    plot.add_argument('--annotate', default=False, action='store_true',
                      help='If true, heatmap will show pearson correlation coefficient in the center \n'
                           'of each square.')

    report = p.add_argument_group('Report Options')
    report.add_argument('--report', action='store_true', default=False,
                        help='If used, script will output a report with basic recommendations in a pdf file.')
    report.add_argument('--wilk_threshold', type=float, default=0.95,
                        help='Threshold to use for the Wilk value.')
    report.add_argument('--corr_threshold', type=float, default=0.8,
                        help='Threshold to use for correlation coefficient.')

    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    assert_input(args.in_dataset)
    assert_output_dir_exist(args.overwrite, args.out_folder, create_dir=True)

    # Loading dataframe.
    logging.info('Loading dataset(s)...')
    if len(args.in_dataset) > 1:
        if args.identifier_column is None:
            parser.error('Column name for index matching is required when inputting multiple dataframe.')
        dict_df = {i: load_df_in_any_format(i) for i in args.in_dataset}
        raw_df = merge_dataframes(dict_df, args.identifier_column)
    else:
        raw_df = load_df_in_any_format(args.in_dataset[0])
    descriptive_columns = [n for n in range(0, args.nb_descriptive_columns)]

    # Removing NaNs from dataset and saving the rows in a different file.
    logging.info('Filtering dataset to remove NaN values. ')
    nans, clean = remove_nans(raw_df)

    # Exporting global description statistics.
    variable_for_stats = clean.drop(clean.columns[descriptive_columns], axis=1, inplace=False).astype('float')
    logging.info('Computing global descriptive statistics.')
    description_df = variable_for_stats.describe()
    wilk, pvalues = compute_shapiro_wilk_test(variable_for_stats)
    w_pval = pd.DataFrame([wilk, pvalues], index=['Wilk', 'Wilk pvalues'], columns=description_df.columns)
    description_df = pd.concat([description_df, w_pval], axis=0)
    description_df.to_excel(f'{args.out_folder}/descriptive_statistics.xlsx', header=True, index=True)

    # Plotting functions and/or computing correlation heatmap.
    if args.disable_plotting:
        logging.info('Plotting is disabled, skipping the plotting steps.')
        corr_mat = variable_for_stats.corr()

    else:
        logging.info('Plotting distributions for the complete dataframe.')
        path_plot = os.path.join(args.out_folder + '/' + 'Distribution_Plots')
        os.makedirs(path_plot, exist_ok=True)
        variable_to_plot = clean.drop(clean.columns[descriptive_columns], axis=1, inplace=False)
        # Plotting distributions.
        plot_distributions(variable_to_plot, path_plot, context=args.context, font_scale=args.font_scale)

        # Plotting heatmap and computing correlation matrix.
        logging.info('Generating correlation matrix and heatmap.')
        if args.annotate:
            if len(raw_df.columns) > 10:
                logging.warning('Due to high number of variables, annotating heatmap is deactivated. Annotation is \n '
                                'only available for dataset with 10 or less variables.')
            corr_mat = compute_correlation_coefficient(variable_to_plot, args.out_folder, context=args.context,
                                                       font_scale=args.font_scale, cmap=args.cmap,
                                                       annot=False if len(raw_df.columns) > 10 else True)
        else:
            corr_mat = compute_correlation_coefficient(variable_to_plot, args.out_folder, context=args.context,
                                                       font_scale=args.font_scale, cmap=args.cmap)

    corr_mat.to_excel(f'{args.out_folder}/correlation_coefficient.xlsx', header=True, index=True)

    # Generating basic recommendations.
    if args.report:
        logging.info('Generating the report...')
        pdf = PDF()
        pdf.alias_nb_pages()
        pdf.set_font('Times', '', 12)

        wilk_value = description_df.loc['Wilk',]
        with open('basic_stats.txt', 'w') as f:
            for i in range(0, len(wilk_value)):
                if wilk_value[i] < args.wilk_threshold:
                    f.write(f"Variable {wilk_value.index[i]} present a W < 0.95 ({round(wilk_value[i], 2)}), you may "
                            f"need to inspect the histogram and ecdf plots to determine the suitability "
                            f"for parametric test. \n")

        pdf.print_chapter(1, 'Normality concerns : ', 'basic_stats.txt')
        os.remove('basic_stats.txt')

        with open('correlation_stats.txt', 'w') as f:
            corr = np.triu(corr_mat.to_numpy(), 1)
            for i in range(0, corr.shape[0]):
                for j in range(0, corr.shape[1]):
                    if corr[i, j] > args.corr_threshold:
                        f.write(f"Variables {description_df.columns[i]} and {description_df.columns[j]} "
                                f"present a high pearson correlation coefficient > 0.8 ({round(corr[i, j], 3)}). It may"
                                f" be interesting to evaluate the need of keeping both variables since they carry "
                                f"similar information. \n")

        pdf.print_chapter(2, 'Correlation concerns : ', 'correlation_stats.txt')
        os.remove('correlation_stats.txt')
        pdf.output(f'{args.out_folder}/report.pdf', 'F')


if __name__ == '__main__':
    main()
