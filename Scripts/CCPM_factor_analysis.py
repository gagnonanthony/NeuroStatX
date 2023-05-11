#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script can be used to perform a factor analysis on observed data. It will return weights
and new scores for each subjects.

Dataset should contain only subject's ID and variables that will be included in factorial analysis.
Rows with missing values will be removed by default, please select the mean or median option to impute
missing data.

Usually used to interpret psychometric or behavioral measures. Might not be optimize for other types
of data.
"""

# Import required libraries.
import argparse
import numpy as np
import pandas as pd
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo
import matplotlib.pyplot as plt
import seaborn as sns


# Build argument parser.
def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument('in_dataset',
                   help='Path of the dataset file containing the observed variable + subjects ID (.csv)')
    p.add_argument('-o', '--output', required=False, default='./',
                   help='Path of the folder in which the results will be written. If not specified, current folder \n'
                        'and default name will be used (e.g. = ./loadings.pdf).')
    p.add_argument('--rotation',
                   choices=['promax', 'oblimin', 'varimax', 'oblimax', 'quartimin', 'quartimax', 'equamax'],
                   default='promax',
                   help='Select the type of rotation to apply on your data. [Default="promax"]')
    p.add_argument('--method', choices=['minres', 'ml', 'principal'], default='minres',
                   help='Select the method for fitting the data. [Default="minres"]')
    p.add_argument('--variables', required=True,
                   help='List of variables names in .txt file.')

    impute = p.add_mutually_exclusive_group()
    impute.add_argument('--mean', action='store_true',
                        help='Impute missing values in the original dataset based on the column mean.')
    impute.add_argument('--median', action='store_true',
                        help='Impute missing values in the original dataset based on the column median.')

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Loading dataset.
    df = pd.read_csv(f'{args.in_dataset}')

    # Imputing missing values (or not).
    if args.mean:
        for column in df.columns:
            df[f"{column}"].fillna(df[f"{column}"].mean(), inplace=True)
    elif args.median:
        for column in df.columns:
            df[f"{column}"].fillna(df[f"{column}"].median(), inplace=True)
    else:
        df.dropna(inplace=True)
    record_id = df["record_id"]
    df.drop(['record_id'], axis=1, inplace=True)

    # Requirement for factorial analysis.
    chi_square_value, p_value = calculate_bartlett_sphericity(df)
    kmo_all, kmo_model = calculate_kmo(df)

    # Fit the data in the model
    if kmo_model > 0.6 and p_value < 0.05:
        fa = FactorAnalyzer(rotation=None)
        fa.fit(df)
        ev, v = fa.get_eigenvalues()

        # Plot results using matplotlib
        plt.scatter(range(1, df.shape[1] + 1), ev)
        plt.plot(range(1, df.shape[1] + 1), ev)
        plt.title('Scree Plot of the eigenvalues for each factor')
        plt.xlabel('Factors')
        plt.ylabel('Eigenvalues')
        plt.grid()
        plt.savefig(f"{args.output}scree_plot.pdf")

        # Perform the factorial analysis
        eigenvalues = sum(map(lambda a: a > 1, ev))
        fa_final = FactorAnalyzer(rotation=args.rotation, n_factors=eigenvalues, method=args.method)
        fa_final.fit(df)
        out = fa_final.transform(df)
        columns = ["Factor 2", "Factor 1"]
        out = pd.DataFrame(out, index=record_id, columns=columns)
        out.to_excel(f"{args.output}transformed_data.xlsx", header=True, index=True)

        # Plot correlation matrix between all raw variables.
        plt.clf()
        plt.cla()
        variables = open(args.variables).read().split()
        corr = pd.DataFrame(fa_final.corr_, index=variables, columns=variables)
        mask = np.triu(np.ones_like(corr, dtype=bool))
        f, ax = plt.subplots(figsize=(11, 9))
        ax = sns.heatmap(corr, mask=mask, cmap='BrBG', vmax=1, vmin=-1, center=0, square=True, annot=True)
        ax.set_title('Correlation Heatmap of raw CPT3 variables')
        plt.tight_layout()
        plt.savefig(f'{args.output}Heatmap.pdf')

        # Plot loadings in a barplot.
        plt.clf()
        plt.cla()
        loadings = pd.DataFrame(fa_final.loadings_, columns=['Factor 1', 'Factor 2'], index=df.columns)
        pos = list(range(1, len(variables) + 1))
        fig, axs = plt.subplots(2)
        axs[0].set_title('Graph of the contribution of each variables on the two main factors.')
        axs[0].bar(pos, loadings['Factor 1'], align='center', tick_label=variables)
        axs[1].bar(pos, loadings['Factor 2'], align='center', tick_label=variables)
        for ax in axs.flat:
            ax.set(ylabel='Loadings')
        for ax in axs.flat:
            ax.label_outer()
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.savefig(f'{args.output}/loadings_barplot.pdf')

        # Export and plot loadings for all variables
        plt.clf()
        plt.cla()
        eig, v = fa_final.get_eigenvalues()
        eig = pd.Series(eig)
        loadings = loadings.append(eig, ignore_index=True)
        loadings.to_excel(f"{args.output}loadings.xlsx", header=True, index=True)
        x = loadings['Factor 1'].tolist()
        y = loadings["Factor 2"].tolist()
        label = df.columns.tolist()
        plt.scatter(x, y)
        plt.title('Scatterplot of variables loadings resulting from the factorial analysis.')
        plt.xlabel('Factor 1')
        plt.ylabel('Factor 2')
        for i, txt in enumerate(label):
            plt.annotate(txt, (x[i], y[i]))
        plt.tight_layout()
        plt.savefig(f"{args.output}loadings.pdf")

    else:
        print(f"In order to perform a factorial analysis, the Bartlett's test p-value needs to be significant (<0.05)\n"
              f"and the Keiser-Meyer-Olkin (KMO) Test needs to return a value greater than 0.6. Current results : \n"
              f"Bartlett's p-value = {p_value} and KMO value = {kmo_model}.")


if __name__ == "__main__":
    main()