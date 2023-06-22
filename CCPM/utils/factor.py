# -*- coding: utf-8 -*-

from enum import Enum
from factor_analyzer import FactorAnalyzer
import matplotlib.pyplot as plt
import numpy as np


class RotationTypes(str, Enum):
    promax = 'promax'
    oblimin = 'oblimin'
    varimax = 'varimax'
    oblimax = 'oblimax'
    quartimin = 'quartimin'
    quartimax = 'quartimax'
    equamax = 'equamax'


class MethodTypes(str, Enum):
    minres = 'minres'
    ml = 'ml'
    principal = 'principal'


class FormattedTextPrompt(str):
    heatmap = "Evaluation of inter-variable correlation."
    screeplot = "Factor to keep are the ones with an eigenvalues over 1 (>1)."
    loadings = "Plot displaying the contribution of each variable to all factors with an eigenvalues > 1."
    scatterplot = "Visual representation of the contribution of each variable to the first 2 factors." \
                  " Scatterplot for the other factors are not produced."


def horn_parallel_analysis(x, output_folder, method="minres", rotation=None, nfactors=1, niter=20):
    """
    This function is mimicking the function from the psych R package fa.parallel to compute the
    horn's parallel analysis to determine the appropriate number of factors to use in factorial
    analysis.

    Portion of this code comes from this post on stackoverflow :
    https://stackoverflow.com/questions/62303782/is-there-a-way-to-conduct-a-parallel-analysis-in-python
    and from the translation of the original function fa.parallel in the psych R package :
    https://github.com/cran/psych/blob/ee72f0cc2aa7c85a844e3ef63c8629096f22c35d/R/fa.parallel.R

    Results have been compared between the original R code and this function and no difference
    have been observed between the two (see pull request #11, https://github.com/gagnonanthony/CCPM/pull/11)

    :param x:                   Numpy array containing the data.
    :param output_folder:       Folder in which the plot will be outputted.
    :param method:              Method to use in factorial analysis.
    :param rotation:            Rotation to use in factorial analysis.
    :param nfactors:            Number of factors.
    :param niter:               Number of iterations for the simulated data.
    :return:                    Suggested number of factors and suggested number of components.
    """

    # Getting input data dimension.
    n_sub, n_variables = x.shape

    # Initiating the factor analysis object.
    fa = FactorAnalyzer(n_factors=nfactors, method=method, rotation=rotation, use_smc=True)

    def fitting_random_data(k, n, m, sumdata):
        fa.fit(np.random.normal(size=(n, m)))
        sumdata['compeigens'] = sumdata['compeigens'] + fa.get_eigenvalues()[0]
        sumdata['factoreigens'] = sumdata['factoreigens'] + fa.get_eigenvalues()[1]

        return sumdata

    # Starting the iterations over random data.
    sumdata = {'compeigens': 0,
               'factoreigens': 0}
    for k in range(0, niter):
        sumdata = fitting_random_data(k, n_sub, n_variables, sumdata)

    sumdata['compeigens'] = sumdata['compeigens'] / niter
    sumdata['factoreigens'] = sumdata['factoreigens'] / niter

    # Fitting the real data.
    fa_values_x = np.array(fa.fit(x).get_eigenvalues())

    # Finding the optimal number of factors/components.
    suggfactors = sum((fa_values_x[1] - sumdata['factoreigens']) > 0)
    suggcomponents = sum((fa_values_x[0] - sumdata['compeigens']) > 0)

    # Setting up the scree plot.
    plt.figure(figsize=(10, 8))

    # Plot the eigenvalues over the number of variables.
    plt.plot([0, n_variables+1], [1, 1], 'k--', alpha=0.3)
    plt.plot(range(1, n_variables+1), sumdata['compeigens'], 'b', label='PC - random', alpha=0.4)
    plt.scatter(range(1, n_variables+1), fa_values_x[0], c='b', marker='o')
    plt.plot(range(1, n_variables+1), fa_values_x[0], 'b', label='PC - data')
    plt.plot(range(1, n_variables+1), sumdata['factoreigens'], 'g', label='FA - random', alpha=0.4)
    plt.scatter(range(1, n_variables+1), fa_values_x[1], c='g', marker='o')
    plt.plot(range(1, n_variables+1), fa_values_x[1], 'g', label='FA - data')
    plt.title("Horn's Parallel Analysis Scree Plots", {'fontsize': 16})
    plt.xlabel('Factors/Components', {'fontsize': 12})
    plt.xticks(ticks=range(1, n_variables+1), labels=range(1, n_variables+1))
    plt.ylabel('Eigenvalue', {'fontsize': 12})
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_folder}/horns_parallel_screeplot.png')
    plt.close()

    return suggfactors, suggcomponents
