#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import required libraries.
import coloredlogs
import dill as pickle
import logging
import sys

from cyclopts import App, Parameter
import semopy
from typing import List
from typing_extensions import Annotated

from neurostatx.io.utils import assert_input, assert_output_dir_exist
from neurostatx.io.loader import DatasetLoader
from neurostatx.utils.factor import cfa


# Initializing the app.
app = App(default_parameter=Parameter(negative=()))
"""Cyclopts application for the ConfirmatoryFA command-line tool."""


@app.default()
def ConfirmatoryFA(
    in_dataset: Annotated[
        str,
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
    # mean: Annotated[
    #    bool,
    #    Parameter(
    #        "--mean",
    #        group="Imputing parameters",
    #    ),
    # ] = False,
    # median: Annotated[
    #    bool,
    #    Parameter(
    #        "--median",
    #        group="Imputing parameters",
    #    ),
    # ] = False,
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
    """Perform confirmatory factor analysis from loadings or a
    lavaan-style model.

    ConfirmatoryFA tests a hypothesized model of the relationships between
    observed variables and latent constructs. It writes factor scores, goodness
    of fit statistics (Chi-square, RMSEA, CFI, TLI), and an HTML report.

    Notes
    -----

    **EFA vs CFA scores**

    Both EFA and CFA scores can be used to derive factor scores. There is no
    clear consensus on which is preferred [2], so the script outputs both. The
    two methods are highly correlated [3]; the choice is a matter of
    preference. A useful reference for the fit metrics is [1].

    **Input specifications**

    The dataset can contain multiple descriptive columns before the variables
    of interest. Specify that count with --desc-columns. Rows with missing
    values are removed by default; use the mean or median option to impute
    missing data (be cautious when doing this).

    References
    ----------
    [1] [Costa, V., & Sarmento, R. Confirmatory Factor
    Analysis](https://arxiv.org/ftp/arxiv/papers/1905/1905.05598.pdf)

    [2] [Whether to use EFA or CFA to predict latent variable
    scores](https://stats.stackexchange.com/questions/346499/whether-to-use-efa-or-cfa-to-predict-latent-variables-scores)

    [3] [Comparison of factor score estimation
    methods](https://github.com/gagnonanthony/NeuroStatX/pull/11)

    Parameters
    ----------
    in_dataset : str
        Input dataset to use in the factorial analysis.
    id_column : str
        Name of the column containing the subject's ID tag. Required for proper
        handling of IDs.
    desc_columns : int
        Number of descriptive columns at the beginning of the dataset to
        exclude in statistics and descriptive tables.
    out_folder : str, optional
        Path of the folder in which the results will be written. If not
        specified, current folder and default name will be used. Defaults to
        ``./ResultsCFA/``.
    loadings_df : str, optional
        Filename of the dataframe containing the loadings of the EFA analysis.
        Columns must be factors and rows variables. Defaults to None.
    model : List[str], optional
        Model specification for the CFA analysis. Must be provided within
        brackets (ex: --model "factor1 =~ var1 + var2 + var3"
        --model "factor2 =~ var4 + var5"). Defaults to None.
    threshold : float, optional
        Threshold to use to determine variables to include for each factor
        in CFA analysis (ex: if set to 0.40, only variables with loadings
        higher than 0.40 will be assigned to a factor in the CFA model).
        Defaults to 0.40.
    iterations : int, optional
        Number of iterations to perform the bootstrapping of the model.
        Defaults to None.
    verbose : bool, optional
        If true, produce verbose output. Defaults to False.
    save_parameters : bool, optional
        If true, save the parameters used in the analysis in a text file.
        Defaults to False.
    overwrite : bool, optional
        If true, force overwriting of existing output files. Defaults to False.

    Examples
    --------
    ```bash
    ConfirmatoryFA --in-dataset dataset.csv --id-column ID --desc-columns 1
    --out-folder ./output/ --loadings-df loadings.csv --threshold 0.40
    --mean -v -f
    ```
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
    df = DatasetLoader().load_data(in_dataset)
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

    desc_col = df.get_descriptive_columns(descriptive_columns)
    df.drop_columns(descriptive_columns)

    if loadings_df is not None:
        logging.info("Creating model's specification.")
        loadings_df = DatasetLoader().load_data(loadings_df).get_data()

        modeldict = {}
        for col in loadings_df.columns:
            idx = loadings_df.index[
                (loadings_df[col] >= threshold) | (loadings_df[col] <=
                                                   -threshold)
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

    cfa_mod, scores, stats = df.custom_function(
        cfa,
        model=mod
    )

    logging.info("Exporting results and statistics.")
    scores = DatasetLoader().import_data(scores)
    scores.join(desc_col, left=True)
    scores.save_data(
        f"{out_folder}/cfa_scores.xlsx", header=True, index=False
    )

    DatasetLoader().import_data(stats).save_data(
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
