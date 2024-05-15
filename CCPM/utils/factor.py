# -*- coding: utf-8 -*-

from enum import Enum

from factor_analyzer import FactorAnalyzer
import matplotlib.pyplot as plt
import numpy as np
import semopy
from strenum import StrEnum


class RotationTypes(StrEnum, Enum):
    promax = "promax"
    oblimin = "oblimin"
    varimax = "varimax"
    oblimax = "oblimax"
    quartimin = "quartimin"
    quartimax = "quartimax"
    equamax = "equamax"


class MethodTypes(StrEnum, Enum):
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

    Args:
        x (np.array):               Input dataset with only variables to
                                    include in the EFA.
        method (str):               Method used to fit the model.
                                    List of possible methods:
                                        minres: Minimal Residual
                                        ml: Maximum Likelihood Factor
                                        principal: Principal Component
        rotation (str):             Rotation method to apply to the factor
                                    loadings:
                                    List of possible rotations:
                                        varimax: Orthogonal Rotation
                                        promax: Oblique Rotation
                                        oblimin: Oblique Rotation
                                        oblimax: Orthogonal Rotation
                                        quartimin: Oblique Rotation
                                        quartimax: Orthogonal Rotation
                                        equamax: Orthogonal Rotation
        nfactors (int):             Number of factors (latent variables) to
                                    extract from the data. Default is 1.
        niter (int):                Number of iterations to perform the
                                    parallel analysis. Default is 20.
    Returns:
        suggfactors (int):          Suggested number of factors to use in the
                                    factorial analysis.
        suggcomponents (int):       Suggested number of components to use in
                                    the factorial analysis.
    """

    # Getting input data dimension.
    n_sub, n_variables = x.shape

    # Initiating the factor analysis object.
    fa = FactorAnalyzer(
        n_factors=nfactors, method=method, rotation=rotation, use_smc=True
    )

    def fitting_random_data(k, n, m, sumdata):
        fa.fit(np.random.normal(size=(n, m)))
        sumdata["compeigens"] = (
            sumdata["compeigens"] + fa.get_eigenvalues()[0])
        sumdata["factoreigens"] = (
            sumdata["factoreigens"] + fa.get_eigenvalues()[1])

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


def efa(df, method, rotation, nfactors=1):
    """
    Function to compute a simple exploratory factor analysis (EFA)
    using the factor_analyzer package.

    Args:
        df (pd.DataFrame):          Input dataset with only variables to
                                    include in the EFA.
        method (str):               Method used to fit the model.
                                    List of possible methods:
                                        minres: Minimal Residual
                                        ml: Maximum Likelihood Factor
                                        principal: Principal Component
        rotation (str):             Rotation method to apply to the factor
                                    loadings:
                                    List of possible rotations:
                                        varimax: Orthogonal Rotation
                                        promax: Oblique Rotation
                                        oblimin: Oblique Rotation
                                        oblimax: Orthogonal Rotation
                                        quartimin: Oblique Rotation
                                        quartimax: Orthogonal Rotation
                                        equamax: Orthogonal Rotation
        nfactors (int):             Number of factors (latent variables) to
                                    extract from the data. Default is 1.

    Returns:
        FactorAnalyzer:             FactorAnalyzer object containing the model.
        ev (np.array):              Eigenvalues of the model.
        v (np.array):               Eigenvalues of the model.
        scores (np.array):          Factor scores of the model.
        loadings (np.array):        Loadings of the model.
        communalities (np.array):   Communalities of the model.
    """

    # Instantiating and fitting the exploratory factorial analysis.
    efa = FactorAnalyzer(rotation=rotation, method=method,
                         n_factors=nfactors)
    efa.fit(df)

    ev, v = efa.get_eigenvalues()
    scores = efa.transform(df)
    loadings = efa.loadings_
    communalities = efa.get_communalities()

    # Return every possible information about the model and factors.
    return efa, ev, v, scores, loadings, communalities


def cfa(
    df, model
):
    """
    Used to compute a confirmatory factor analysis (CFA) to evaluate the
    goodness of fit of the model.

    This function uses the semopy package to evaluate the goodness of fit of
    the proposed model (https://semopy.com/).

    Args:
        df (pd.DataFrame):          Input dataset with only variables to
                                    include in the CFA.
        model (str):                Model description for the CFA.
    Returns:
        semopy.Model:               Model object containing the CFA.
        scores (pd.DataFrame):      Factor scores of the model.
        stats (pd.DataFrame):       Statistics of the model.
    """

    cfa = semopy.Model(model)
    cfa.fit(df)

    scores = cfa.predict_factors(df)
    stats = cfa.inspect(mode="list", what="est",
                        information="expected")

    return cfa, scores, stats
