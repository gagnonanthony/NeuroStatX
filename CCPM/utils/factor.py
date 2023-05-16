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
