# -*- coding: utf-8 -*-

from enum import Enum
from strenum import StrEnum


class DistanceMetrics(StrEnum, Enum):
    braycurtis = ("braycurtis",)
    canberra = ("canberra",)
    chebyshev = ("chebyshev",)
    cityblock = ("cityblock",)
    correlation = ("correlation",)
    cosine = ("cosine",)
    dice = ("dice",)
    euclidean = ("euclidean",)
    hamming = ("hamming",)
    jaccard = ("jaccard",)
    jensenshannon = ("jensenshannon",)
    kulczynski1 = ("kulczynski1",)
    mahalanobis = ("mahalanobis",)
    matching = ("matching",)
    minkowski = ("minkowski",)
    rogerstanimoto = ("rogerstanimoto",)
    russellrao = ("russellrao",)
    seuclidean = ("seuclidean",)
    sokalmichener = ("sokalmichener",)
    sokalsneath = ("sokalsneath",)
    sqeuclidean = ("sqeuclidean",)
    yule = "yule"
