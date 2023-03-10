#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging

import pandas as pd

from CCPM.io.utils import load_df_in_any_format
from CCPM.utils.preprocessing import (remove_nans,
                                      plot_distributions,
                                      compute_shapiro_wilk_test,
                                      compute_correlation_coefficient)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('-i', '--in_dataset', metavar='FILE', required=True,
                   help='Dataset containing raw data all variables for all subjects.')
    p.add_argument('-o', '--out_folder', required=True,
                   help='Directory in which datasets, plots and statistics will be saved.')
    p.add_argument('-n', '--nb_descriptive_columns', type=int, required=True,
                   help='Number of descriptive columns at the beginning of the dataset to exclude in \n'
                        'statistics and descriptive tables.')
    p.add_argument('--disable_plotting', action='store_true',
                   help='If used, will disable all plotting and simply output tables.')

    plot = p.add_argument_group('Plot Options')
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

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    raw_df = load_df_in_any_format(args.in_dataset)
    descriptive_columns = [n for n in range(0, args.nb_descriptive_columns)]

    # Removing NaNs from dataset and saving the rows in a different file.
    logging.debug('Filtering dataset to remove NaN values. ')
    nans, clean = remove_nans(raw_df)

    # Exporting global description statistics.
    variable_for_stats = clean.drop(clean.columns[descriptive_columns], axis=1, inplace=False)
    logging.debug('Computing global descriptive statistics.')
    description_df = variable_for_stats.describe()
    wilk, pvalues = compute_shapiro_wilk_test(variable_for_stats)
    w_pval = pd.DataFrame([wilk, pvalues], index=['Wilk', 'Wilk pvalues'], columns=description_df.columns)
    description_df = pd.concat([description_df, w_pval], axis=0)
    description_df.to_excel(f'{args.out_folder}/descriptive_statistics.xlsx', header=True, index=True)

    # Plotting functions and/or computing correlation heatmap.
    if args.disable_plotting:
        logging.debug('Plotting is disabled, skipping the plotting steps.')
        corr_mat = variable_for_stats.corr()

    else:
        logging.debug('Plotting distributions for the complete dataframe.')
        variable_to_plot = clean.drop(clean.columns[descriptive_columns], axis=1, inplace=False)
        # Plotting distributions.
        plot_distributions(variable_to_plot, args.out_folder, context=args.context, font_scale=args.font_scale)

        # Plotting heatmap and computing correlation matrix.
        logging.debug('Generating correlation matrix and heatmap.')
        corr_mat = compute_correlation_coefficient(variable_to_plot, args.out_folder, context=args.context,
                                                   font_scale=args.font_scale, cmap=args.cmap, annot=args.annotate)

    corr_mat.to_excel(f'{args.out_folder}/correlation_coefficient.xlsx', header=True, index=True)


if __name__ == '__main__':
    main()
