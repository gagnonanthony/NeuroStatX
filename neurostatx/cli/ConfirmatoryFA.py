#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import required libraries.
import coloredlogs
import dill as pickle
import logging
import sys

from cyclopts import App, Parameter
import pandas as pd
import semopy
from typing import List
from typing_extensions import Annotated

from neurostatx.io.utils import (
    assert_input,
    assert_output_dir_exist,
    load_df_in_any_format
)
from neurostatx.utils.preprocessing import merge_dataframes
from neurostatx.utils.factor import cfa


# Initializing the app.
app = App(default_parameter=Parameter(negative=()))


@app.default()
def ConfirmatoryFA(
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
    ] = "./ResultsCFA/",
    loadings_df: Annotated[
        str,
        Parameter(
            show_default=False,
            group="Factorial Analysis parameters",
        ),
    ] = None,
    model: Annotated[
        List[str],
        Parameter(
            show_default=False,
            group="Factorial Analysis parameters",
        ),
    ] = None,
    threshold: Annotated[
        float,
        Parameter(
            "--threshold",
            group="Factorial Analysis parameters",
        ),
    ] = 0.40,
    iterations: Annotated[
        int,
        Parameter(
            "--iterations",
            group="Factorial Analysis parameters",
        ),
    ] = None,
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
    """CONFIRMATORY FACTORIAL ANALYSIS
    -------------------------------
    ConfirmatoryFA is a script that can be used to perform a confirmatory
    factorial analysis (CFA) to test a hypothesized model of the relationships
    between observed variables and latent constructs. The script will output
    factor scores and statistics of goodness of fit such as Chi-square, RMSEA,
    CFI and TLI. The script will also generate a html report containing the
    results of the analysis. A good reference to understand those metrics can
    be accessed in [1].

    USING EFA SCORES OR CFA SCORES
    ------------------------------
    Both method can be used to derive factor scores. Since there is no clear
    consensus surrounding which is preferred (see [2]) the script will output
    both factor scores. As shown in [3], both methods highly correlate with one
    another. It then comes down to the user's preference.

    INPUT SPECIFICATIONS
    --------------------
    Dataset can contain multiple descriptive rows before the variables of
    interest. Simply specify the number of descriptive rows using
    --desc-columns. Rows with missing values will be removed by
    default, please select the mean or median option to impute missing data
    (be cautious when doing this).

    REFERENCES
    ----------
    [1] Costa, V., & Sarmento, R. Confirmatory Factor Analysis.
    https://arxiv.org/ftp/arxiv/papers/1905/1905.05598.pdf

    [2]
    https://stats.stackexchange.com/questions/346499/whether-to-use-efa-or-cfa-to-predict-latent-variables-scores

    [3] https://github.com/gagnonanthony/NeuroStatX/pull/11

    EXAMPLE USAGE
    -------------
    ::

        ConfirmatoryFA --in-dataset dataset.csv --id-column ID --desc-columns 1
        --out-folder ./output/ --loadings-df loadings.csv --threshold 0.40
        --mean -v -f

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
    loadings_df : str, optional
        Filename of the dataframe containing the loadings of the EFA analysis.
        Columns must be factors and rows variables.
    model : str, optional
        Model specification for the CFA analysis. Must be provided within
        brackets. (ex: --model "factor1 =~ var1 + var2 + var3"
        --model "factor2 =~ var4 + var5")
    threshold : float, optional
        Threshold to use to determine variables to include for each factor
        in CFA analysis. (ex: if set to 0.40, only variables with loadings
        higher than 0.40 will be assigned to a factor in the CFA model).
    iterations : int, optional
        Number of iterations to perform the bootstrapping of the model.
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

    if loadings_df is None and model is None:
        sys.exit(
            "Please provide a loadings dataframe or a model specification."
        )

    if loadings_df is not None and threshold is None:
        sys.exit(
            "Please provide a threshold value to determine the variables to "
            "include in the CFA model."
        )

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
    #   logging.info(
    #        "No methods selected for imputing missing values. "
    #        "Removing them."
    #    )
    #    df.dropna(inplace=True)

    desc_col = df[df.columns[descriptive_columns]]
    df.drop(df.columns[descriptive_columns], axis=1, inplace=True)

    if loadings_df is not None:
        logging.info("Creating model's specification.")
        loadings_df = load_df_in_any_format(loadings_df)

        modeldict = {}
        for col in loadings_df.columns:
            idx = loadings_df.index[
                (loadings_df[col] >= threshold) | (loadings_df[
                    col] <= -threshold)
            ].tolist()
            modeldict[col] = idx

        mod = ""
        for key, values in modeldict.items():
            mod += f"{key} =~ {' + '.join(values)}\n"
    else:
        mod = ""
        for i in model:
            mod += f"{i}\n"

    logging.info("Performing Confirmatory Factorial Analysis (CFA) with"
                 " the following model specification:\n{}".format(mod))

    cfa_mod, scores, stats = cfa(df, mod)

    logging.info("Exporting results and statistics.")
    scores = pd.concat([desc_col, scores], axis=1)
    scores.to_excel(
        f"{out_folder}/cfa_scores.xlsx", header=True, index=False
    )

    stats.to_excel(
        f"{out_folder}/cfa_stats.xlsx", header=True, index=False
    )

    # Bootstrapping the model.
    if iterations is not None:
        logging.info("Bootstrapping the model.")
        semopy.bias_correction(cfa_mod, n=iterations)

    semopy.semplot(cfa_mod, f"{out_folder}/semplot.png",
                   plot_covs=True)
    semopy.report(cfa_mod, f"{out_folder}/CFA_report")

    # Saving the model.
    with open(f'{out_folder}/cfa_model.pkl', 'wb') as f:
        pickle.dump(cfa_mod, f)


if __name__ == "__main__":
    app()
