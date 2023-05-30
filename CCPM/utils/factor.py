# -*- coding: utf-8 -*-
from enum import Enum


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
