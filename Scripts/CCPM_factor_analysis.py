#!/usr/bin/env python
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
CCPM_factor_analysis.py is a script that can be used to perform a factor analysis
on observed data. It can handle continuous, categorical and binary variables. This
script can be used to perform a factor analysis on observed data. It will
return weights and new scores for each subject and global graphs showing loadings
for each variable.

Dataset should contain only subject's ID and variables that will be included in
factorial analysis. Rows with missing values will be removed by default, please
select the mean or median option to impute missing data (be cautious when doing
this).

Usually used to interpret psychometric or behavioral measures. The default parameters
might not be optimized for all types of data.
"""

# Import required libraries.
import argparse
import logging

from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tableone import tableone

from CCPM.io.utils import (add_verbose_arg,
                           add_overwrite_arg,
                           assert_input,
                           assert_output_dir_exist,
                           load_df_in_any_format)
from CCPM.io.viz import flexible_barplot, autolabel
from CCPM.utils.preprocessing import merge_dataframes


# Build argument parser.
def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument('-i', '--in_dataset', nargs='+', required=True,
                   help='Input dataset(s) to use in the factorial analysis. If multiple files are provided as input,'
                        'will be merged according to the subject id columns.')
    p.add_argument('-s', '--id_column', required=True,
                   help="Name of the column containing the subject's ID tag. Required for proper handling of IDs and"
                        "merging multiple datasets.")
    p.add_argument('-o', '--out_folder', required=False, default='./output',
                   help='Path of the folder in which the results will be written. If not specified, current folder \n'
                        'and default name will be used (e.g. = ./output/loadings.pdf).')
    p.add_argument('--test_name', required=False, default='',
                   help='Provide the name of the test the variables come from. Will be used in the titles if provided.')
    p.add_argument('--rotation',
                   choices=['promax', 'oblimin', 'varimax', 'oblimax', 'quartimin', 'quartimax', 'equamax'],
                   default='promax',
                   help='Select the type of rotation to apply on your data. [Default="promax"] \n'
                        'List of possible rotations: \n'
                        'varimax: Orthogonal Rotation \n'
                        'promax: Oblique Rotation \n'
                        'oblimin: Oblique Rotation \n'
                        'oblimax: Orthogonal Rotation \n'
                        'quartimin: Oblique Rotation \n'
                        'quartimax: Orthogonal Rotation \n'
                        'equamax: Orthogonal Rotation')
    p.add_argument('--method', choices=['minres', 'ml', 'principal'], default='minres',
                   help='Select the method for fitting the data. [Default="minres"] \n'
                        'List of possible methods: \n'
                        'minres: Minimal Residual \n'
                        'ml: Maximum Likelihood Factor \n'
                        'principal: Principal Component')

    impute = p.add_mutually_exclusive_group()
    impute.add_argument('--mean', action='store_true',
                        help='Impute missing values in the original dataset based on the column mean.')
    impute.add_argument('--median', action='store_true',
                        help='Impute missing values in the original dataset based on the column median.')

    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    logging.info('Validating input files and creating output folder {}'.format(args.out_folder))
    assert_input(parser, args.in_dataset)
    assert_output_dir_exist(parser, args, args.out_folder, create_dir=True)

    # Loading dataset.
    logging.info('Loading {}'.format(args.in_dataset))
    if len(args.in_dataset) > 1:
        if args.id_column is None:
            parser.error('Column name for index matching is required when inputting multiple dataframes.')
        dict_df = {i: load_df_in_any_format(i) for i in args.in_dataset}
        df = merge_dataframes(dict_df, args.identifier_column)
    else:
        df = load_df_in_any_format(args.in_dataset[0])

    # Imputing missing values (or not).
    if args.mean:
        logging.info('Imputing missing values using the mean method.')
        for column in df.columns:
            df[f"{column}"].fillna(df[f"{column}"].mean(), inplace=True)
    elif args.median:
        logging.info('Imputing missing values using the median method.')
        for column in df.columns:
            df[f"{column}"].fillna(df[f"{column}"].median(), inplace=True)
    else:
        logging.info('No methods selected for imputing missing values. Removing them.')
        df.dropna(inplace=True)

    record_id = df[args.id_column]
    df.drop([args.id_column], axis=1, inplace=True)

    # Requirement for factorial analysis.
    chi_square_value, p_value = calculate_bartlett_sphericity(df)
    kmo_all, kmo_model = calculate_kmo(df)
    logging.info("Bartlett's test of sphericity returned a p-value of {} and Keiser-Meyer-Olkin (KMO)"
                 "test returned a value of {}.".format(p_value, kmo_model))

    # Fit the data in the model
    if kmo_model > 0.6 and p_value < 0.05:
        logging.info("Dataset passed the Bartlett's test and KMO test. Proceeding with factorial analysis.")
        fa = FactorAnalyzer(rotation=None)
        fa.fit(df)
        ev, v = fa.get_eigenvalues()

        # Plot results using matplotlib
        plt.scatter(range(1, df.shape[1] + 1), ev)
        plt.plot(range(1, df.shape[1] + 1), ev)
        sns.set_style("whitegrid")
        plt.title('Scree Plot of the eigenvalues for each factor')
        plt.xlabel('Factors')
        plt.ylabel('Eigenvalues')
        plt.grid()
        plt.savefig(f"{args.out_folder}/scree_plot.pdf")
        plt.close()

        # Perform the factorial analysis.
        eigenvalues = sum(map(lambda a: a > 1, ev))
        fa_final = FactorAnalyzer(rotation=args.rotation, n_factors=eigenvalues, method=args.method)
        fa_final.fit(df)
        out = fa_final.transform(df)
        columns = [f"Factor {i}" for i in range(1, eigenvalues+1)]  # Validate if the list comprehension works.
        out = pd.DataFrame(out, index=record_id, columns=columns)
        out.to_excel(f"{args.out_folder}/transformed_data.xlsx", header=True, index=True)

        # Plot correlation matrix between all raw variables.
        corr = pd.DataFrame(fa_final.corr_, index=df.columns, columns=df.columns)
        mask = np.triu(np.ones_like(corr, dtype=bool))
        f, ax = plt.subplots(figsize=(11, 9))
        ax = sns.heatmap(corr, mask=mask, cmap='BrBG', vmax=1, vmin=-1, center=0, square=True, annot=True,
                         linewidth=.5, fmt=".1f", annot_kws={"size" : 8})
        ax.set_title('Correlation Heatmap of raw {} variables.'.format(args.test_name))
        plt.tight_layout()
        plt.savefig(f'{args.out_folder}/Heatmap.pdf')
        plt.close()

        # Plot loadings in a barplot.
        loadings = pd.DataFrame(fa_final.loadings_, columns=columns, index=df.columns)
        data_to_plot = [loadings[i].values for i in loadings.columns]
        flexible_barplot(data_to_plot, loadings.index, eigenvalues,
                         title='Loadings values', filename=f'{args.out_folder}/barplot_loadings.png',
                         ylabel='Loading')

        # Export and plot loadings for all variables
        eig, v = fa_final.get_eigenvalues()
        eigen_table = pd.DataFrame(eig, index=[f'Factor {i}' for i in range(1, len(df.columns)+1)],
                                   columns=['Eigenvalues'])
        eigen_table.to_excel(f"{args.out_folder}/eigenvalues.xlsx", header=True, index=True)
        loadings.to_excel(f"{args.out_folder}/loadings.xlsx", header=True, index=True)
        x = loadings['Factor 1'].tolist()
        y = loadings["Factor 2"].tolist()
        label = df.columns.tolist()
        plt.scatter(x, y)
        plt.title('Scatterplot of variables loadings from the first and second factor.')
        plt.xlabel('Factor 1')
        plt.ylabel('Factor 2')
        for i, txt in enumerate(label):
            plt.annotate(txt, (x[i], y[i]))
        plt.tight_layout()
        plt.savefig(f"{args.out_folder}/scatterplot_loadings.pdf")

    else:
        print(f"In order to perform a factorial analysis, the Bartlett's test p-value needs to be significant (<0.05)\n"
              f"and the Keiser-Meyer-Olkin (KMO) Test needs to return a value greater than 0.6. Current results : \n"
              f"Bartlett's p-value = {p_value} and KMO value = {kmo_model}.")


if __name__ == "__main__":
    main()
