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
    """Apply Model
    -----------
    Apply a model to a dataset. Features in the dataset will be scaled prior
    to applying the model.

    Currently Supported Models
    --------------------------
    * SEMopy
    * FactorAnalyzer
    * sklearn

    Example Usage
    -------------
    ApplyModel --in-dataset dataset.xlsx --model model.pkl --id-column ID
    --desc-columns 1 --out-folder ./output -v -f -s

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
