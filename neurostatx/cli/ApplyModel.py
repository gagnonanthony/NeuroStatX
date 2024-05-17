#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import required libraries.
import coloredlogs
import dill as pickle
import logging
import sys

from cyclopts import App, Parameter
import pandas as pd
from typing import List
from typing_extensions import Annotated

from neurostatx.io.utils import (
    assert_input,
    assert_output_dir_exist,
    load_df_in_any_format
)
from neurostatx.utils.preprocessing import merge_dataframes
from neurostatx.statistics.utils import apply_various_models


# Initializing the app.
app = App(default_parameter=Parameter(negative=()))


@app.default()
def ApplyModel(
    in_dataset: Annotated[
        List[str],
        Parameter(
            show_default=False,
            group="Essential Files Options",
        ),
    ],
    model: Annotated[
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
    ] = "./ApplyModel/",
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
    """APPLY MODEL
    -----------
    Apply a model to a dataset. Features in the dataset will be scaled prior
    to applying the model.

    CURRENTLY SUPPORTED MODELS
    --------------------------
    * SEMopy
    * FactorAnalyzer
    * sklearn

    EXAMPLE USAGE
    -------------
    ApplyModel --in-dataset dataset.xlsx --model model.pkl --id-column ID
    --desc-columns 1 --out-folder ./output -v -f -s

    Parameters
    ----------
    in_dataset : List[str]
        List of input datasets.
    model : str
        Path to the model to apply.
    id_column : str
        Column name to use as index.
    desc_columns : int
        Number of columns to use as descriptors.
    out_folder : str, optional
        Output folder.
    verbose : bool, optional
        Increase verbosity.
    save_parameters : bool, optional
        Save parameters to a file.
    overwrite : bool, optional
        Overwrite output folder if it already exists.
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

    # Assessing if NaNs are present in the dataset.
    if df.isnull().values.any():
        raise ValueError("NaNs are present in the dataset. Please impute "
                         "missing values prior to applying the model.")

    descriptive_columns = [n for n in range(0, desc_columns)]
    desc_data = df.iloc[:, descriptive_columns]
    df.drop(df.columns[descriptive_columns], axis=1, inplace=True)

    # Loading model.
    logging.info("Loading model")
    with open(model, "rb") as f:
        mod = pickle.load(f)

    # Applying model.
    logging.info("Applying model")
    out = apply_various_models(df, mod)

    # Saving transformed dataset.
    logging.info("Saving transformed dataset")
    columns = desc_data.columns.append(out.columns)
    out = pd.concat([desc_data, out], axis=1, ignore_index=True)
    out.columns = columns
    out.to_excel(f"{out_folder}/transformed_dataset.xlsx", index=False)


if __name__ == "__main__":
    app()
