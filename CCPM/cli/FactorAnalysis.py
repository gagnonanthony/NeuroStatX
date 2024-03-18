#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import required libraries.
import coloredlogs
import logging
import os
import sys

from cyclopts import App, Parameter
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import semopy
from sklearn.preprocessing import StandardScaler
from typing import List
from typing_extensions import Annotated

from CCPM.io.utils import (
    assert_input,
    assert_output_dir_exist,
    load_df_in_any_format,
)
from CCPM.io.viz import flexible_barplot
from CCPM.utils.preprocessing import merge_dataframes
from CCPM.utils.factor import (
    RotationTypes,
    MethodTypes,
    horn_parallel_analysis,
    apply_efa_only,
    apply_efa_and_cfa,
)


# Initializing the app.
app = App(default_parameter=Parameter(negative=()))


@app.default()
def FactorAnalysis(
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
    ] = "./output/",
    test_name: Annotated[
        str,
        Parameter(
            show_default=False,
            group="Essential Files Options",
        ),
    ] = "",
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
    factor_number: Annotated[
        int,
        Parameter(
            "--factor_number",
            group="Factorial Analysis parameters",
        ),
    ] = None,
    use_horn_parallel: Annotated[
        bool,
        Parameter(
            "--use_horn_parallel",
            group="Factorial Analysis parameters",
        ),
    ] = False,
    use_only_efa: Annotated[
        bool,
        Parameter(
            "--use_only_efa",
            group="Factorial Analysis parameters",
        ),
    ] = False,
    train_dataset_size: Annotated[
        float,
        Parameter(
            "--train_dataset_size",
            group="Factorial Analysis parameters",
        ),
    ] = 0.5,
    threshold: Annotated[
        float,
        Parameter(
            "--threshold",
            group="Factorial Analysis parameters",
        ),
    ] = 0.40,
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
    overwrite: Annotated[
        bool,
        Parameter(
            "-f",
            "--overwrite",
            group="Optional parameters",
        ),
    ] = False,
):
    """FACTORIAL ANALYSIS
    ------------------
    FactorAnalysis is a script that can be used to perform an
    exploratory factorial analysis (EFA) and a confirmatory factorial analysis
    (CFA). A user can decide if he wants to perform only EFA or both
    sequentially.

    EXPLORATORY VS CONFIRMATORY FACTORIAL ANALYSIS
    ----------------------------------------------
    In the case of performing only an EFA (use the flag --use_only_efa), the
    script will use either the Kaiser's criterion or Horn's parallel analysis
    (see --use_horn_parallel) to determine the optimal number of factors to
    extract from the data. Then the final EFA model will be fitted using the
    provided rotation and method.

    If --use_only_efa is not selected, the script will perform subsequently an
    EFA and a CFA. The selection of the appropriate number of factors will be
    done on the full input dataset. However, the fitting of the EFA model will
    be performed on the selected proportion of the input data to use as a
    training data (see --train_dataset_size). The parameters surrounding the
    fitting of the final EFA is identical to what is described above.

    Following the EFA, resulting loadings will be assigned to a latent factor
    based on the supplied threshold value. A CFA model will then be build using
    this structure and fitted to the data. Statistics of goodness of fit such
    as Chi-square, RMSEA, CFI and NFI will be computed and exported as a table
    and into a html report. A good reference to understand those metrics can be
    accessed in [1].

    Both method can be used to derive factor scores. Since there is no clear
    consensus surrounding which is preferred (see [2]) the script will output
    both factor scores. As shown in [3], both methods highly correlate with one
    another. It then comes down to the user's preference.

    INPUT SPECIFICATIONS
    --------------------
    Dataset should contain only subject's ID and variables that will be
    included in factorial analysis. Rows with missing values will be removed by
    default, please select the mean or median option to impute missing data
    (be cautious when doing this).

    REFERENCES
    ----------
    [1] Costa, V., & Sarmento, R. Confirmatory Factor Analysis.
    https://arxiv.org/ftp/arxiv/papers/1905/1905.05598.pdf

    [2]
    https://stats.stackexchange.com/questions/346499/whether-to-use-efa-or-cfa-to-predict-latent-variables-scores

    [3] https://github.com/gagnonanthony/CCPM/pull/11

    EXAMPLE USAGE
    -------------
    ::

        FactorAnalysis --in-dataset df --id-column IDs --out-folder results_FA/
        --test-name EXAMPLE --rotation promax --method ml --threshold 0.4
        --train_dataset_size 0.8 -v -f

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
    test_name : str, optional
        Provide the name of the test the variables come from. Will be used in
        the titles if provided.
    rotation : RotationTypes, optional
        Select the type of rotation to apply on your data.
    method : MethodTypes, optional
        Select the method for fitting the data.
    factor_number : int, optional
        If set, the script will use this number as the final number of factors.
    use_horn_parallel : bool, optional
        If set, will use the suggested number of factors from the Horns
        parallel analysis in a case where values differ between the Kaiser
        criterion and Horns parallel analysis.
    use_only_efa : bool, optional
        If set, the script will not perform the default 2 steps factor
        analysis (exploratory factor analysis + confirmatory factor analysis
        on 2 distinct subset of the data) but will simply do an exploratory
        factor analysis on the complete dataset.
    train_dataset_size : float, optional
        Specify the proportion of the input dataset to use as training dataset
        in the EFA. (value from 0 to 1)
    threshold : float, optional
        Threshold to use to determine variables to include for each factor
        in CFA analysis. (ex: if set to 0.40, only variables with loadings
        higher than 0.40 will be assigned to a factor in the CFA.
    random_state : int, optional
        Random State value to use for reproducibility.
    mean : bool, optional
        Impute missing values in the original dataset based on the column mean.
    median : bool, optional
        Impute missing values in the original dataset based on the column
        median.
    verbose : bool, optional
        If true, produce verbose output.
    overwrite : bool, optional
        If true, force overwriting of existing output files.
    """

    if verbose:
        logging.getLogger().setLevel(logging.INFO)
        coloredlogs.install(level=logging.INFO)

    if factor_number is not None and use_horn_parallel:
        sys.exit(
            "--factor_number and --use_horn_parallel cannot be used at the "
            "same time."
        )

    logging.info(
        "Validating input files and creating output folder {}"
        .format(out_folder)
    )
    assert_input(in_dataset)
    assert_output_dir_exist(overwrite, out_folder, create_dir=True)

    if not use_only_efa:
        os.makedirs(f"{out_folder}/efa/")
        os.makedirs(f"{out_folder}/cfa/")

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
    if mean:
        logging.info("Imputing missing values using the mean method.")
        for column in df.columns:
            df[f"{column}"].fillna(df[f"{column}"].mean(), inplace=True)
    elif median:
        logging.info("Imputing missing values using the median method.")
        for column in df.columns:
            df[f"{column}"].fillna(df[f"{column}"].median(), inplace=True)
    else:
        logging.info(
            "No methods selected for imputing missing values. "
            "Removing them."
        )
        df.dropna(inplace=True)

    record_id = df[id_column]
    desc_col = df[df.columns[descriptive_columns]]
    df.drop(df.columns[descriptive_columns], axis=1, inplace=True)

    # Scaling the dataset.
    scaled_df = pd.DataFrame(StandardScaler().fit_transform(df),
                             columns=df.columns)

    # Requirement for factorial analysis.
    chi_square_value, p_value = calculate_bartlett_sphericity(scaled_df)
    kmo_all, kmo_model = calculate_kmo(scaled_df)
    logging.info(
        "Bartlett's test of sphericity returned a p-value of {} and "
        "Keiser-Meyer-Olkin (KMO)"
        "test returned a value of {}.".format(p_value, kmo_model)
    )

    # Fit the data in the model
    if kmo_model > 0.6 and p_value < 0.05:
        logging.info(
            "Dataset passed the Bartlett's test and KMO test. Proceeding "
            "with factorial analysis."
        )
        fa = FactorAnalyzer(rotation=None, method=method)
        fa.fit(scaled_df)
        ev, v = fa.get_eigenvalues()

        # Plot scree plot to determine the optimal number of factors using
        # the Kaiser's method. (eigenvalues > 1)
        plt.scatter(range(1, df.shape[1] + 1), ev)
        plt.plot(range(1, df.shape[1] + 1), ev)
        sns.set_style("whitegrid")
        plt.title("Scree Plot of the eigenvalues for each factor")
        plt.xlabel("Factors")
        plt.ylabel("Eigenvalues")
        plt.grid()
        plt.savefig(f"{out_folder}/scree_plot.png")
        plt.close()

        # Horn's parallel analysis.
        suggfactor, suggcomponent = horn_parallel_analysis(
            scaled_df.values, out_folder, rotation=None, method=method
        )

        # Validating the results from scree plot and horn's parallel
        # analysis.
        eigenvalues = sum(map(lambda a: a > 1, ev))
        if suggfactor == eigenvalues:
            logging.info(
                "Both the scree plot and horn's parallel analysis suggests"
                " the same number of factors : "
                "{} . Proceeding with this number for the final analysis."
                .format(suggfactor)
            )
            nfactors = suggfactor
        else:
            logging.info(
                "The scree plot and horn's parallel analysis returned "
                "different values : {} and {} respectively. Default is "
                "taking the value suggested from the scree plot method "
                "if the --use_horns_parallel flag is not used.".format(
                    eigenvalues, suggfactor
                )
            )
            if use_horn_parallel:
                nfactors = suggfactor
            elif factor_number is not None:
                nfactors = factor_number
            else:
                nfactors = eigenvalues

        # Perform the factorial analysis.
        if use_only_efa:
            efa = apply_efa_only(
                scaled_df, rotation=rotation, nfactors=nfactors,
                method=method
            )
        else:
            efa, cfa = apply_efa_and_cfa(
                scaled_df,
                nfactors=nfactors,
                rotation=rotation,
                method=method,
                threshold=threshold,
                random_state=random_state,
                train_size=train_dataset_size,
            )

        columns = [f"Factor {i}" for i in range(1, nfactors + 1)]

        # Export EFA and CFA transformed data.
        efa_out = pd.DataFrame(
            efa.transform(scaled_df), index=record_id, columns=columns
        )
        if use_only_efa:
            efa_out.to_excel(f"{out_folder}/scores.xlsx", header=True,
                             index=True)
        else:
            efa_out.to_excel(
                f"{out_folder}/efa/scores.xlsx", header=True, index=True
            )

            # Export scores from CFA analysis.
            cfa_scores = cfa.predict_factors(scaled_df)
            scores = pd.concat([desc_col, cfa_scores], axis=1)
            scores.to_excel(
                f"{out_folder}/cfa/scores.xlsx", header=True, index=False
            )

        # Plot correlation matrix between all raw variables.
        corr = pd.DataFrame(efa.corr_, index=df.columns,
                            columns=df.columns)
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
        ax.set_title("Correlation Heatmap of raw {} variables."
                     .format(test_name))
        plt.tight_layout()
        plt.savefig(f"{out_folder}/Heatmap.png")
        plt.close()

        # Plot EFA loadings in a barplot.
        efa_loadings = pd.DataFrame(
            efa.loadings_, columns=columns, index=df.columns
        )
        if use_only_efa:
            flexible_barplot(
                efa_loadings,
                nfactors,
                title="Loadings values for the EFA",
                output=f"{out_folder}/barplot_loadings.png",
                ylabel="Loading value",
            )
        else:
            flexible_barplot(
                efa_loadings,
                nfactors,
                title="Loadings values for the EFA",
                output=f"{out_folder}/efa/barplot_loadings.png",
                ylabel="Loading value",
            )

            # Export table with all estimate from the CFA analysis.
            stats = cfa.inspect(mode="list", what="est",
                                information="expected")
            stats.to_excel(
                f"{out_folder}/cfa/statistics.xlsx", header=True,
                index=False
            )

        # Export EFA loadings for all variables.
        eigen_table = pd.DataFrame(
            efa.get_eigenvalues()[0],
            index=[f"Factor {i}" for i in range(1, len(df.columns) + 1)],
            columns=["Eigenvalues"],
        )
        if use_only_efa:
            eigen_table.to_excel(
                f"{out_folder}/eigenvalues.xlsx", header=True, index=True
            )
            efa_loadings.to_excel(
                f"{out_folder}/loadings.xlsx", header=True, index=True
            )
        else:
            eigen_table.to_excel(
                f"{out_folder}/efa/eigenvalues.xlsx", header=True,
                index=True
            )
            efa_loadings.to_excel(
                f"{out_folder}/efa/loadings.xlsx", header=True, index=True
            )

            semopy.semplot(cfa, f"{out_folder}/cfa/semplot.png",
                           plot_covs=True)
            semopy.report(cfa, f"{out_folder}/cfa/Detailed_CFA_Report")

    else:
        print(
            f"In order to perform a factorial analysis, the Bartlett's "
            f"test p-value needs to be significant \n"
            f" (<0.05) and the Keiser-Meyer-Olkin (KMO) Test needs to "
            f"return a value greater than 0.6. Current\n "
            f" results : Bartlett's p-value = {p_value} and KMO value = "
            f"{kmo_model}."
        )


if __name__ == "__main__":
    app()
