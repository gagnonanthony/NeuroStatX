#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import required libraries.
import coloredlogs
import dill as pickle
import logging
import sys

from cyclopts import App, Parameter
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from typing import List
from typing_extensions import Annotated

from neurostatx.io.utils import (
    assert_input,
    assert_output_dir_exist,
    load_df_in_any_format,
)
from neurostatx.io.viz import flexible_barplot
from neurostatx.utils.preprocessing import merge_dataframes
from neurostatx.utils.factor import (
    RotationTypes,
    MethodTypes,
    horn_parallel_analysis,
    efa,
)


# Initializing the app.
app = App(default_parameter=Parameter(negative=()))


@app.default()
def ExploratoryFA(
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
            group="Essential Files Options",
        ),
    ] = "./ResultsEFA/",
    rotation: Annotated[
        RotationTypes,
        Parameter(
            group="Factorial Analysis parameters",
        ),
    ] = RotationTypes.promax,
    method: Annotated[
        MethodTypes,
        Parameter(
            group="Factorial Analysis parameters",
        ),
    ] = MethodTypes.minres,
    nb_factors: Annotated[
        int,
        Parameter(
            "--nb_factors",
            group="Factorial Analysis parameters",
        ),
    ] = None,
    train_dataset_size: Annotated[
        float,
        Parameter(
            "--train_dataset_size",
            group="Factorial Analysis parameters",
        ),
    ] = 0.5,
    random_state: Annotated[
        int,
        Parameter(
            "--random_state",
            group="Factorial Analysis parameters",
        ),
    ] = 1234,
    mean: Annotated[
        bool,
        Parameter(
            "--mean",
            group="Imputing parameters",
        ),
    ] = False,
    median: Annotated[
        bool,
        Parameter(
            "--median",
            group="Imputing parameters",
        ),
    ] = False,
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
    """EXPLORATORY FACTORIAL ANALYSIS
    ------------------
    ExploratoryFA is a script that can be used to perform an
    exploratory factorial analysis (EFA).

    In the case of performing only an EFA (use the flag --use_only_efa), the
    script will use Horn's parallel analysis to determine the optimal number
    of factors to extract from the data. Then the final EFA model will be
    fitted using the provided rotation and method.

    It is also possible to perform EFA on a training dataset and export the
    test dataset to be used for further analysis (e.g. ConfirmatoryFA). The
    script will output the EFA model, the loadings, communalities, and the
    transformed dataset.

    INPUT SPECIFICATIONS
    --------------------
    Dataset can contain multiple descriptive rows before the variables of
    interest. Simply specify the number of descriptive rows using
    --desc-columns. Rows with missing values will be removed by
    default, please select the mean or median option to impute missing data
    (be cautious when doing this).

    REFERENCES
    ----------
    [1] [Costa, V., & Sarmento, R. Confirmatory Factor
    Analysis.](https://arxiv.org/ftp/arxiv/papers/1905/1905.05598.pdf)

    [2] [Whether to use EFA or CFA to predict latent variables
    scores.](https://stats.stackexchange.com/questions/346499/whether-to-use-efa-or-cfa-to-predict-latent-variables-scores)

    [3] [Comparison of factor score estimation
    methods](https://github.com/gagnonanthony/NeuroStatX/pull/11)

    EXAMPLE USAGE
    -------------
    ::

        ExploratoryFA --in-dataset df --id-column IDs --out-folder results_FA/
        --rotation promax --method ml --train_dataset_size 0.5 -v -f -s

    Parameters
    ----------
    in_dataset : List[str]
        Input dataset(s) to use in the factorial analysis. If multiple files
        are provided as input, will be merged according to the subject id
        columns. For multiple inputs, use this: --in-dataset df1 --in-dataset
        df2 [...]
    id_column : str
        Name of the column containing the subject's ID tag. Required for proper
        handling of IDs and merging multiple datasets.
    desc_columns : int
        Number of descriptive columns at the beginning of the dataset to
        exclude in statistics and descriptive tables.
    out_folder : str, optional
        Path of the folder in which the results will be written. If not
        specified, current folder and default name will be used (e.g. =
        ./output/).
    rotation : RotationTypes, optional
        Select the type of rotation to apply on your data.
    method : MethodTypes, optional
        Select the method for fitting the data.
    nb_factors : int, optional
        Specify the number of factors to extract from the data. If not
        specified, the script will use Horn's parallel analysis to determine
        the optimal number of factors.
    train_dataset_size : float, optional
        Specify the proportion of the input dataset to use as training dataset
        in the EFA. (value from 0 to 1)
    random_state : int, optional
        Random seed for reproducibility.
    mean : bool, optional
        Impute missing values in the original dataset based on the column mean.
    median : bool, optional
        Impute missing values in the original dataset based on the column
        median.
    verbose : bool, optional
        If true, produce verbose output.
    save_parameters : bool, optional
        If true, save the parameters used in the analysis in a text file.
    overwrite : bool, optional
        If true, force overwriting of existing output files.
    """

    if verbose:
        logging.getLogger().setLevel(logging.INFO)
        coloredlogs.install(level=logging.INFO)

    logging.info(
        "Validating input files and creating output folder {}"
        .format(out_folder)
    )
    assert_input(in_dataset)
    assert_output_dir_exist(overwrite, out_folder, create_dir=True)

    if save_parameters:
        parameters = list(locals().items())
        with open(f"{out_folder}/parameters.txt", "w+") as f:
            for param in parameters:
                f.writelines(str(param))

    # Loading dataset.
    logging.info("Loading {}".format(in_dataset))
    if len(in_dataset) > 1:
        if id_column is None:
            sys.exit(
                "Column name for index matching is required when inputting"
                " multiple dataframes."
            )
        dict_df = {i: load_df_in_any_format(i) for i in in_dataset}
        df = merge_dataframes(dict_df, id_column)
    else:
        df = load_df_in_any_format(in_dataset[0])
    descriptive_columns = [n for n in range(0, desc_columns)]

    # Imputing missing values (or not).
    # Disabled for now, wait until we have a solution to check only columns
    # that will be used in the model.
    # if mean:
    #    logging.info("Imputing missing values using the mean method.")
    #    for column in df.columns:
    #        df[f"{column}"].fillna(df[f"{column}"].mean(), inplace=True)
    # elif median:
    #    logging.info("Imputing missing values using the median method.")
    #    for column in df.columns:
    #        df[f"{column}"].fillna(df[f"{column}"].median(), inplace=True)
    # else:
    #    logging.info(
    #        "No methods selected for imputing missing values. "
    #        "Removing them."
    #    )
    #    df.dropna(inplace=True)

    if train_dataset_size != 1:
        logging.info("Splitting into train and test datasets. Using training "
                     "dataset for EFA.")
        train, test = train_test_split(df, train_size=train_dataset_size,
                                       random_state=random_state)
        train.reset_index(inplace=True, drop=True)
        train.to_excel(f"{out_folder}/train_dataset.xlsx", header=True,
                       index=False)
        test.reset_index(inplace=True, drop=True)
        test.to_excel(f"{out_folder}/test_dataset.xlsx", header=True,
                      index=False)
    else:
        logging.info("Using the full dataset for EFA.")
        train = df

    desc_col = train[train.columns[descriptive_columns]]
    train.drop(df.columns[descriptive_columns], axis=1, inplace=True)

    # Requirement for factorial analysis.
    chi_square_value, p_value = calculate_bartlett_sphericity(train)
    kmo_all, kmo_model = calculate_kmo(train)
    logging.info(
        "Bartlett's test of sphericity returned a p-value of {} and "
        "Keiser-Meyer-Olkin (KMO)"
        "test returned a value of {}.".format(p_value, kmo_model)
    )

    # Horn's parallel analysis.
    suggfactor, suggcomponent = horn_parallel_analysis(
        train.values, out_folder, rotation=None, method=method
    )
    if nb_factors is None:
        nb_factors = suggfactor

    efa_mod, ev, v, scores, loadings, communalities = efa(
        train, rotation=rotation, method=method, nfactors=nb_factors
    )

    # Plot scree plot to determine the optimal number of factors using
    # the Kaiser's method. (eigenvalues > 1)
    plt.scatter(range(1, train.shape[1] + 1), ev)
    plt.plot(range(1, train.shape[1] + 1), ev)
    sns.set_style("whitegrid")
    plt.title("Scree Plot of the eigenvalues for each factor")
    plt.xlabel("Factors")
    plt.ylabel("Eigenvalues")
    plt.grid()
    plt.savefig(f"{out_folder}/scree_plot.png")
    plt.close()

    columns = [f"Factor {i}" for i in range(1, nb_factors + 1)]

    # Export EFA and CFA transformed data.
    efa_out = pd.DataFrame(
        scores, columns=columns
    )
    efa_out = pd.concat([desc_col, efa_out], axis=1)
    efa_out.to_excel(f"{out_folder}/EFA_scores.xlsx", header=True,
                     index=False)

    # Plot correlation matrix between all raw variables.
    corr = pd.DataFrame(efa_mod.corr_, index=train.columns,
                        columns=train.columns)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    f, ax = plt.subplots(figsize=(11, 9))
    ax = sns.heatmap(
        corr,
        mask=mask,
        cmap="BrBG",
        vmax=1,
        vmin=-1,
        center=0,
        square=True,
        annot=True,
        linewidth=0.5,
        fmt=".1f",
        annot_kws={"size": 8},
    )
    ax.set_title("Correlation Heatmap of raw variables.")
    plt.tight_layout()
    plt.savefig(f"{out_folder}/Heatmap.png")
    plt.close()

    # Plot EFA loadings in a barplot.
    efa_loadings = pd.DataFrame(
        loadings, columns=columns, index=train.columns
    )
    efa_communalities = pd.DataFrame(
        communalities, columns=["Communalities"], index=train.columns
    )

    flexible_barplot(
        efa_loadings,
        nb_factors,
        title="Loadings values for the EFA",
        output=f"{out_folder}/barplot_loadings.png",
        ylabel="Loading value",
    )

    # Export EFA loadings for all variables.
    eigen_table = pd.DataFrame(
        ev,
        index=[f"Factor {i}" for i in range(1, len(train.columns) + 1)],
        columns=["Eigenvalues"],
    )
    eigen_table.to_excel(
        f"{out_folder}/eigenvalues.xlsx", header=True, index=True
    )
    efa_loadings.to_excel(
        f"{out_folder}/loadings.xlsx", header=True, index=True
    )
    efa_communalities.to_excel(
        f"{out_folder}/communalities.xlsx", header=True, index=True
    )

    # Export EFA model.
    with open(f"{out_folder}/EFA_model.pkl", "wb") as f:
        pickle.dump(efa_mod, f)


if __name__ == "__main__":
    app()
