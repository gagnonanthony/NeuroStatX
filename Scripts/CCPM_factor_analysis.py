#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import required libraries.
import logging
import time
import warnings
from pathlib import Path

from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich.progress import Progress, SpinnerColumn, TextColumn
import seaborn as sns
from tableone import tableone
import typer
from typing import List
from typing_extensions import Annotated

from CCPM.io.utils import (assert_input,
                           assert_output_dir_exist,
                           load_df_in_any_format)
from CCPM.io.viz import flexible_barplot, autolabel
from CCPM.utils.preprocessing import merge_dataframes
from CCPM.utils.factor import RotationTypes, MethodTypes


def main(in_dataset: Annotated[list[str], typer.Option(help='Input dataset(s) to use in the factorial analysis. '
                                                            'If multiple files are provided as input,'
                                                            'will be merged according to the subject id columns.',
                                                       show_default=False,
                                                       rich_help_panel='Essential Files Options')],
         id_column: Annotated[str, typer.Option(help="Name of the column containing the subject's ID tag. "
                                                     "Required for proper handling of IDs and"
                                                     "merging multiple datasets.",
                                                show_default=False,
                                                rich_help_panel='Essential Files Options')],
         out_folder: Annotated[str, typer.Option(help='Path of the folder in which the results will be written. '
                                                      'If not specified, current folder and default'
                                                      'name will be used (e.g. = ./output/).',
                                                 rich_help_panel='Essential Files Options')] = './default',
         test_name: Annotated[str, typer.Option(help='Provide the name of the test the variables come from. Will '
                                                     'be used in the titles if provided.', show_default=False,
                                                rich_help_panel='Essential Files Options')] = "",
         rotation: Annotated[RotationTypes, typer.Option(help='Select the type of rotation to apply on your data.\n'
                                                              'List of possible rotations: \n'
                                                              'varimax: Orthogonal Rotation \n'
                                                              'promax: Oblique Rotation \n'
                                                              'oblimin: Oblique Rotation \n'
                                                              'oblimax: Orthogonal Rotation \n'
                                                              'quartimin: Oblique Rotation \n'
                                                              'quartimax: Orthogonal Rotation \n'
                                                              'equamax: Orthogonal Rotation',
                                                         rich_help_panel="Factorial Analysis parameters",
                                                         case_sensitive=False)] = RotationTypes.promax,
         method: Annotated[MethodTypes, typer.Option(help='Select the method for fitting the data. \n'
                                                          'List of possible methods: \n'
                                                          'minres: Minimal Residual \n'
                                                          'ml: Maximum Likelihood Factor \n'
                                                          'principal: Principal Component',
                                                     rich_help_panel="Factorial Analysis parameters",
                                                     case_sensitive=False)] = MethodTypes.minres,
         mean: Annotated[bool, typer.Option('--mean', help='Impute missing values in the original dataset based on the '
                                                           'column mean.',
                                            rich_help_panel="Imputing parameters")] = False,
         median: Annotated[bool, typer.Option('--median', help='Impute missing values in the original dataset based '
                                                               'on the column mean.',
                                              rich_help_panel="Imputing parameters")] = False,
         verbose: Annotated[bool, typer.Option('-v', '--verbose', help='If true, produce verbose output.',
                                               rich_help_panel="Optional parameters")] = False,
         overwrite: Annotated[bool, typer.Option('-f', '--overwrite', help='If true, force overwriting of existing '
                                                                           'output files.',
                                                 rich_help_panel="Optional parameters")] = False):

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
    CCPM_factor_analysis.py is a script that can be used to perform a factor analysis
    on observed data. It can handle continuous, categorical and binary variables. This
    script can be used to perform a factor analysis on observed data. It will
    return weights and new scores for each subject and global graphs showing loadings
    for each variable.
    \b
    Dataset should contain only subject's ID and variables that will be included in
    factorial analysis. Rows with missing values will be removed by default, please
    select the mean or median option to impute missing data (be cautious when doing
    this).
    \b
    Usually used to interpret psychometric or behavioral measures. The default parameters
    might not be optimized for all types of data.
    """
    warnings.filterwarnings('ignore', category=DeprecationWarning)

    if verbose:
        logging.getLogger().setLevel(logging.INFO)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=False,
    ) as progress:
        task = progress.add_task(description="Loading dataset(s)...", total=None)
        logging.info('Validating input files and creating output folder {}'.format(out_folder))
        assert_input(in_dataset)
        assert_output_dir_exist(overwrite, out_folder, create_dir=True)

        # Loading dataset.
        logging.info('Loading {}'.format(in_dataset))
        if len(in_dataset) > 1:
            if id_column is None:
                exit('Column name for index matching is required when inputting multiple dataframes.')
            dict_df = {i: load_df_in_any_format(i) for i in in_dataset}
            df = merge_dataframes(dict_df, id_column)
        else:
            df = load_df_in_any_format(in_dataset[0])
        progress.update(task, completed=True, description="[green]Dataset(s) loaded.")

        task = progress.add_task(description="Processing of dataset(s)...", total=None)
        # Imputing missing values (or not).
        if mean:
            logging.info('Imputing missing values using the mean method.')
            for column in df.columns:
                df[f"{column}"].fillna(df[f"{column}"].mean(), inplace=True)
        elif median:
            logging.info('Imputing missing values using the median method.')
            for column in df.columns:
                df[f"{column}"].fillna(df[f"{column}"].median(), inplace=True)
        else:
            logging.info('No methods selected for imputing missing values. Removing them.')
            df.dropna(inplace=True)

        record_id = df[id_column]
        df.drop([id_column], axis=1, inplace=True)

        # Requirement for factorial analysis.
        chi_square_value, p_value = calculate_bartlett_sphericity(df)
        kmo_all, kmo_model = calculate_kmo(df)
        logging.info("Bartlett's test of sphericity returned a p-value of {} and Keiser-Meyer-Olkin (KMO)"
                     "test returned a value of {}.".format(p_value, kmo_model))
        progress.update(task, completed=True, description="[green]Dataset(s) processed.")

        task = progress.add_task(description="Performing factorial analysis and generating plots...", total=None)
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
            plt.savefig(f"{out_folder}/scree_plot.pdf")
            plt.close()

            # Perform the factorial analysis.
            eigenvalues = sum(map(lambda a: a > 1, ev))
            fa_final = FactorAnalyzer(rotation=rotation, n_factors=eigenvalues, method=method)
            fa_final.fit(df)
            out = fa_final.transform(df)
            columns = [f"Factor {i}" for i in range(1, eigenvalues+1)]  # Validate if the list comprehension works.
            out = pd.DataFrame(out, index=record_id, columns=columns)
            out.to_excel(f"{out_folder}/transformed_data.xlsx", header=True, index=True)

            # Plot correlation matrix between all raw variables.
            corr = pd.DataFrame(fa_final.corr_, index=df.columns, columns=df.columns)
            mask = np.triu(np.ones_like(corr, dtype=bool))
            f, ax = plt.subplots(figsize=(11, 9))
            ax = sns.heatmap(corr, mask=mask, cmap='BrBG', vmax=1, vmin=-1, center=0, square=True, annot=True,
                             linewidth=.5, fmt=".1f", annot_kws={"size" : 8})
            ax.set_title('Correlation Heatmap of raw {} variables.'.format(test_name))
            plt.tight_layout()
            plt.savefig(f'{out_folder}/Heatmap.pdf')
            plt.close()

            # Plot loadings in a barplot.
            loadings = pd.DataFrame(fa_final.loadings_, columns=columns, index=df.columns)
            data_to_plot = [loadings[i].values for i in loadings.columns]
            flexible_barplot(data_to_plot, loadings.index, eigenvalues,
                             title='Loadings values', filename=f'{out_folder}/barplot_loadings.png',
                             ylabel='Loading')

            # Export and plot loadings for all variables
            eig, v = fa_final.get_eigenvalues()
            eigen_table = pd.DataFrame(eig, index=[f'Factor {i}' for i in range(1, len(df.columns)+1)],
                                       columns=['Eigenvalues'])
            eigen_table.to_excel(f"{out_folder}/eigenvalues.xlsx", header=True, index=True)
            loadings.to_excel(f"{out_folder}/loadings.xlsx", header=True, index=True)
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
            plt.savefig(f"{out_folder}/scatterplot_loadings.pdf")

        else:
            print(f"In order to perform a factorial analysis, the Bartlett's test p-value needs to be significant (<0.05)\n"
                  f"and the Keiser-Meyer-Olkin (KMO) Test needs to return a value greater than 0.6. Current results : \n"
                  f"Bartlett's p-value = {p_value} and KMO value = {kmo_model}.")

        progress.update(task, completed=True, description="[green]Analysis completed. Enjoy your results ! :beer:")


if __name__ == "__main__":
    typer.run(main)
