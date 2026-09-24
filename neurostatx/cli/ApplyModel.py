#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import required libraries.
import coloredlogs
import dill as pickle
import logging

from cyclopts import App, Parameter
from typing_extensions import Annotated

from neurostatx.io.utils import (
    assert_input,
    assert_output_dir_exist
)
from neurostatx.io.loader import DatasetLoader
from neurostatx.statistics.utils import apply_various_models


# Initializing the app.
app = App(default_parameter=Parameter(negative=()))
"""Cyclopts application for the ApplyModel command-line tool."""


@app.default()
def ApplyModel(
    in_dataset: Annotated[
        str,
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
    """Apply a pickled SEMopy, FactorAnalyzer, or sklearn model to a dataset.

    Features in the dataset are scaled before the model is applied.

    Notes
    -----
    Currently supported models:

    * SEMopy
    * FactorAnalyzer
    * sklearn

    Parameters
    ----------
    in_dataset : str
        Input dataset.
    model : str
        Path to the model to apply.
    id_column : str
        Column name to use as index.
    desc_columns : int
        Number of columns to use as descriptors.
    out_folder : str, optional
        Output folder. Defaults to ``./ApplyModel/``.
    verbose : bool, optional
        Increase verbosity. Defaults to False.
    save_parameters : bool, optional
        Save parameters to a file. Defaults to False.
    overwrite : bool, optional
        Overwrite output folder if it already exists. Defaults to False.

    Examples
    --------
    ```bash
    ApplyModel --in-dataset dataset.xlsx --model model.pkl --id-column ID
    --desc-columns 1 --out-folder ./output -v -f -s
    ```
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
    df = DatasetLoader().load_data(in_dataset)

    # Assessing if NaNs are present in the dataset.
    # Disabled for now, wait until we have a solution to check only columns
    # that will be used in the model.
    # if df.isnull().values.any():
    #    raise ValueError("NaNs are present in the dataset. Please impute "
    #                     "missing values prior to applying the model.")

    descriptive_columns = [n for n in range(0, desc_columns)]
    desc_data = df.get_descriptive_columns(descriptive_columns)
    df.drop_columns(descriptive_columns)

    # Loading model.
    logging.info("Loading model")
    with open(model, "rb") as f:
        mod = pickle.load(f)

    # Applying model.
    logging.info("Applying model")
    out = df.custom_function(
        apply_various_models,
        mod=mod
    )

    # Saving transformed dataset.
    logging.info("Saving transformed dataset")
    DatasetLoader().import_data(out).join(
        desc_data, left=True
    ).save_data(
        f"{out_folder}/transformed_dataset.xlsx",
        header=True,
        index=False
    )


if __name__ == "__main__":
    app()
