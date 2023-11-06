# -*- coding: utf-8 -*-

from enum import Enum

import pandas as pd
from factor_analyzer import FactorAnalyzer
import matplotlib.pyplot as plt
import numpy as np
import semopy
from sklearn.model_selection import train_test_split


class RotationTypes(str, Enum):
    promax = "promax"
    oblimin = "oblimin"
    varimax = "varimax"
    oblimax = "oblimax"
    quartimin = "quartimin"
    quartimax = "quartimax"
    equamax = "equamax"


class MethodTypes(str, Enum):
    minres = "minres"
    ml = "ml"
    principal = "principal"


class FormattedTextPrompt(str):
    heatmap = "Evaluation of inter-variable correlation."
    screeplot = "Factor to keep are the ones with an eigenvalues over 1 (>1)."
    loadings = (
        "Plot displaying the contribution of each variable to all"
        "factors with an eigenvalues > 1."
    )
    scatterplot = (
        "Visual representation of the contribution of each variable to the"
        "first 2 factors. Scatterplot for the other factors are not produced."
    )
    semplot = (
        "Visual representation of the relationship between latent"
        "variables and indicators."
    )


def horn_parallel_analysis(
    x, output_folder, method="minres", rotation=None, nfactors=1, niter=20
):
    """
    This function is mimicking the function from the psych R package
    fa.parallel to compute the horn's parallel analysis to determine the
    appropriate number of factors to use in factorial analysis.

    Portion of this code comes from this post on stackoverflow :
    https://stackoverflow.com/questions/62303782/is-there-a-way-to-conduct-a-parallel-analysis-in-python
    and from the translation of the original function fa.parallel in the psych
    R package :
    https://github.com/cran/psych/blob/ee72f0cc2aa7c85a844e3ef63c8629096f22c35d/R/fa.parallel.R

    Results have been compared between the original R code and this function
    and no difference have been observed between the two (see pull request #11,
    https://github.com/gagnonanthony/CCPM/pull/11)

    :param x:                   Numpy array containing the data.
    :param output_folder:       Folder in which the plot will be outputted.
    :param method:              Method to use in factorial analysis.
    :param rotation:            Rotation to use in factorial analysis.
    :param nfactors:            Number of factors.
    :param niter:               Number of iterations for the simulated data.
    :return:                    Suggested number of factors and suggested
                                number of components.
    """

    # Getting input data dimension.
    n_sub, n_variables = x.shape

    # Initiating the factor analysis object.
    fa = FactorAnalyzer(
        n_factors=nfactors, method=method, rotation=rotation, use_smc=True
    )

    def fitting_random_data(k, n, m, sumdata):
        fa.fit(np.random.normal(size=(n, m)))
        sumdata["compeigens"] = (sumdata["compeigens"]
                                 + fa.get_eigenvalues()[0])
        sumdata["factoreigens"] = (sumdata["factoreigens"]
                                   + fa.get_eigenvalues()[1])

        return sumdata

    # Starting the iterations over random data.
    sumdata = {"compeigens": 0, "factoreigens": 0}
    for k in range(0, niter):
        sumdata = fitting_random_data(k, n_sub, n_variables, sumdata)

    sumdata["compeigens"] = sumdata["compeigens"] / niter
    sumdata["factoreigens"] = sumdata["factoreigens"] / niter

    # Fitting the real data.
    fa_values_x = np.array(fa.fit(x).get_eigenvalues())

    # Finding the optimal number of factors/components.
    suggfactors = sum((fa_values_x[1] - sumdata["factoreigens"]) > 0)
    suggcomponents = sum((fa_values_x[0] - sumdata["compeigens"]) > 0)

    # Setting up the scree plot.
    plt.figure(figsize=(10, 8))

    # Plot the eigenvalues over the number of variables.
    plt.plot([0, n_variables + 1], [1, 1], "k--", alpha=0.3)
    plt.plot(
        range(1, n_variables + 1),
        sumdata["compeigens"],
        "b",
        label="PC - random",
        alpha=0.4,
    )
    plt.scatter(range(1, n_variables + 1), fa_values_x[0], c="b", marker="o")
    plt.plot(range(1, n_variables + 1), fa_values_x[0], "b", label="PC - data")
    plt.plot(
        range(1, n_variables + 1),
        sumdata["factoreigens"],
        "g",
        label="FA - random",
        alpha=0.4,
    )
    plt.scatter(range(1, n_variables + 1), fa_values_x[1], c="g", marker="o")
    plt.plot(range(1, n_variables + 1), fa_values_x[1], "g", label="FA - data")
    plt.title("Horn's Parallel Analysis Scree Plots", {"fontsize": 16})
    plt.xlabel("Factors/Components", {"fontsize": 12})
    plt.xticks(ticks=range(1, n_variables + 1),
               labels=range(1, n_variables + 1))
    plt.ylabel("Eigenvalue", {"fontsize": 12})
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_folder}/horns_parallel_screeplot.png")
    plt.close()

    return suggfactors, suggcomponents


def apply_efa_only(df, method, nfactors, rotation):
    """
    Legacy function to compute a simple exploratory factor analysis (EFA)
    using the factor_analyzer package. The function fit a EFA model to the
    data with the specified number of factors, method and rotation while
    returning the complete model object.

    :param df:          Input dataset with only variables to include in the
                        EFA.
    :param method:      Method used to fit the model.
                        List of possible methods:
                            minres: Minimal Residual
                            ml: Maximum Likelihood Factor
                            principal: Principal Component
    :param nfactors:    Number of factors (latent variables) to extract from
                        the data.
    :param rotation:    Rotation method to apply to the factor loadings:
                        List of possible rotations:
                            varimax: Orthogonal Rotation
                            promax: Oblique Rotation
                            oblimin: Oblique Rotation
                            oblimax: Orthogonal Rotation
                            quartimin: Oblique Rotation
                            quartimax: Orthogonal Rotation
                            equamax: Orthogonal Rotation
    :return:            Fitted FactorAnalyzer object.
    """

    # Instantiating and fitting the exploratory factorial analysis.
    efa = FactorAnalyzer(rotation=rotation, n_factors=nfactors, method=method)
    efa.fit(df)

    # Return every possible information about the model and factors.
    return efa


def apply_efa_and_cfa(
    df, method, nfactors, rotation, train_size=0.5, threshold=0.40,
    random_state=None
):
    """
    Used to compute subsequently an exploratory factor analysis (EFA) to
    determine the original loadings and a confirmatory factor analysis (CFA)
    to evaluate the goodness of fit of the model.

    This function uses the factor_analyzer to compute the EFA
    (https://factor-analyzer.readthedocs.io/en/latest/index.html) and then,
    leverage semopy package to evaluate the goodness of fit of the proposed
    model (https://semopy.com/).

    Supplied dataset will be split into 2 datasets : train and test sets. The
    train set will be used in the EFA whereas the test set will be used in
    the CFA.

    The function then returned both model object classes for further plotting,
    exporting statistics,
    etc.

    :param df:              Input dataset with only variables to include in
                            the EFA.
    :param method:          Method used to fit the model.
                            List of possible methods:
                                minres: Minimal Residual
                                ml: Maximum Likelihood Factor
                                principal: Principal Component
    :param nfactors:        Number of factors (latent variables) to extract
                            from the data.
    :param rotation:        Rotation method to apply to the factor loadings:
                            List of possible rotations:
                                varimax: Orthogonal Rotation
                                promax: Oblique Rotation
                                oblimin: Oblique Rotation
                                oblimax: Orthogonal Rotation
                                quartimin: Oblique Rotation
                                quartimax: Orthogonal Rotation
                                equamax: Orthogonal Rotation
    :param train_size:      Proportion of supplied dataset to use for training
                            (a.k.a EFA).
                            Test dataset size will be derived from this simple
                            equation :
                            test_size = 1 - train_size.
    :param threshold:       Threshold value to consider a loading to be
                            sufficiently enough to be included in the model to
                            test with CFA.
    :param random_state:    Random hash to use for reproducible results.
    :return:                Both FactorAnalyzer class object and semopy. Model
                            class object.
    """

    # Splitting the dataset randomly in 2 subsets.
    test_size = 1 - train_size
    train_efa, test_cfa = train_test_split(
        df, train_size=train_size, test_size=test_size,
        random_state=random_state
    )

    # Run traditional efa analysis on training dataset (simple random half of
    # the original dataset).
    efa = FactorAnalyzer(rotation=rotation, n_factors=nfactors, method=method)
    efa.fit(train_efa)

    # Sort model specification using the semopy synthax.
    columns = [f"F{i}" for i in range(1, nfactors + 1)]
    loadings_df = pd.DataFrame(efa.loadings_, columns=columns,
                               index=df.columns)
    modeldict = {}
    for col in loadings_df.columns:
        idx = loadings_df.index[
            (loadings_df[col] >= threshold) | (loadings_df[col] <= -threshold)
        ].tolist()
        modeldict[col] = idx

    mod = ""
    for key, values in modeldict.items():
        mod += f"{key} =~ {' + '.join(values)}\n"

    contributing_col = {x for v in modeldict.values() for x in v}
    test_cfa = test_cfa.filter(contributing_col)

    # Run confirmatory factor analysis.
    cfa = semopy.Model(mod)
    cfa.fit(test_cfa)

    return efa, cfa
