# -*- coding: utf-8 -*-

from enum import Enum

from factor_analyzer import FactorAnalyzer
import factor_analyzer.factor_analyzer as _factor_analyzer_mod
import matplotlib.pyplot as plt
import numpy as np
import semopy
from strenum import StrEnum


def _patch_factor_analyzer_sklearn_compat() -> None:
    """Accept sklearn 1.8+ with factor-analyzer 0.5.1 from PyPI.

    That release still passes ``force_all_finite`` into ``check_array``,
    which scikit-learn removed in 1.8. The upstream fix lives only on
    git, which PyPI will not accept as a dependency.
    """
    orig = _factor_analyzer_mod.check_array
    if getattr(orig, "_neurostatx_patched", False):
        return

    def check_array(*args, **kwargs):
        if "force_all_finite" in kwargs and "ensure_all_finite" not in kwargs:
            kwargs["ensure_all_finite"] = kwargs.pop("force_all_finite")
        else:
            kwargs.pop("force_all_finite", None)
        return orig(*args, **kwargs)

    check_array._neurostatx_patched = True
    _factor_analyzer_mod.check_array = check_array


_patch_factor_analyzer_sklearn_compat()


class RotationTypes(StrEnum, Enum):
    """Factor rotation methods for exploratory factor analysis."""

    promax = "promax"
    oblimin = "oblimin"
    varimax = "varimax"
    oblimax = "oblimax"
    quartimin = "quartimin"
    quartimax = "quartimax"
    equamax = "equamax"


class MethodTypes(StrEnum, Enum):
    """Factor extraction methods for exploratory factor analysis."""

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
    """Estimate the number of factors and components with Horn's method.

    Compares observed eigenvalues to those from random data, following
    psych's ``fa.parallel``.

    Parameters
    ----------
    x : np.array
        Variables to include in the analysis.
    output_folder : str
        Directory where ``horns_parallel_screeplot.png`` is written.
    method : str, optional
        Extraction method: ``"minres"``, ``"ml"``, or ``"principal"``.
        Defaults to ``"minres"``.
    rotation : str, optional
        Rotation applied when fitting. Defaults to None.
    nfactors : int, optional
        Number of factors extracted while computing eigenvalues.
        Defaults to 1.
    niter : int, optional
        Number of random-data iterations. Defaults to 20.

    Returns
    -------
    suggfactors : int
        Suggested number of factors.
    suggcomponents : int
        Suggested number of components.

    Examples
    --------
    >>> import numpy as np
    >>> from neurostatx.utils.factor import horn_parallel_analysis
    >>> rng = np.random.RandomState(0)
    >>> x = rng.rand(40, 5)
    >>> suggfactors, suggcomponents = horn_parallel_analysis(
    ...     x, output_folder=".", niter=2
    ... )
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
    """Fit an exploratory factor analysis with factor_analyzer.

    Parameters
    ----------
    df : pd.DataFrame
        Variables to include in the EFA.
    method : str
        Extraction method: ``"minres"``, ``"ml"``, or ``"principal"``.
    rotation : str
        Rotation applied to the loadings, or None.
    nfactors : int, optional
        Number of factors to extract. Defaults to 1.

    Returns
    -------
    model : FactorAnalyzer
        Fitted factor analyzer.
    ev : np.array
        Original eigenvalues.
    v : np.array
        Common-factor eigenvalues.
    scores : np.array
        Factor scores.
    loadings : np.array
        Factor loadings.
    communalities : np.array
        Communalities.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from neurostatx.utils.factor import efa
    >>> rng = np.random.RandomState(0)
    >>> df = pd.DataFrame(rng.rand(40, 4), columns=list("abcd"))
    >>> model, ev, v, scores, loadings, communalities = efa(
    ...     df, method="minres", rotation="varimax", nfactors=1
    ... )
    >>> scores.shape[0]
    40
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
    """Fit a confirmatory factor analysis model with semopy.

    Parameters
    ----------
    df : pd.DataFrame
        Observed indicators included in the CFA.
    model : str
        semopy model specification.

    Returns
    -------
    fitted : semopy.Model
        Fitted CFA model.
    scores : pd.DataFrame
        Predicted factor scores.
    stats : pd.DataFrame
        Parameter estimates from ``inspect``.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from neurostatx.utils.factor import cfa
    >>> rng = np.random.RandomState(0)
    >>> f = rng.randn(40)
    >>> df = pd.DataFrame({"x1": f + 0.1 * rng.randn(40),
    ...                    "x2": f + 0.1 * rng.randn(40),
    ...                    "x3": f + 0.1 * rng.randn(40)})
    >>> fitted, scores, stats = cfa(df, "F =~ x1 + x2 + x3")
    >>> "F" in scores.columns
    True
    """

    cfa = semopy.Model(model)
    cfa.fit(df)

    scores = cfa.predict_factors(df)
    stats = cfa.inspect(mode="list", what="est",
                        information="expected")

    return cfa, scores, stats
