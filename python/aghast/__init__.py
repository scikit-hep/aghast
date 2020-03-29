#!/usr/bin/env python

# BSD 3-Clause License; see https://github.com/scikit-hep/aghast/blob/master/LICENSE

from aghast.interface import frombuffer
from aghast.interface import fromarray
from aghast.interface import fromfile


def to_numpy(obj):
    import aghast._connect._numpy

    return aghast._connect._numpy.to_numpy(obj)


def from_numpy(obj):
    import aghast._connect._numpy

    return aghast._connect._numpy.from_numpy(obj)


def to_pandas(obj):
    import aghast._connect._pandas

    return aghast._connect._pandas.to_pandas(obj)


def from_pandas(obj):
    import aghast._connect._pandas

    return aghast._connect._pandas.from_pandas(obj)


def to_root(obj, name):
    import aghast._connect._root

    return aghast._connect._root.to_root(obj, name)


def from_root(obj, collection=False):
    import aghast._connect._root

    return aghast._connect._root.from_root(obj, collection=collection)


def to_fnalhist(obj):
    import aghast._connect._fnalhist

    return aghast._connect._fnalhist.to_fnalhist(obj)


def from_fnalhist(obj):
    import aghast._connect._fnalhist

    return aghast._connect._fnalhist.from_fnalhist(obj)


from aghast.interface import Metadata
from aghast.interface import Decoration
from aghast.interface import Buffer
from aghast.interface import ExternalBuffer
from aghast.interface import Interpretation
from aghast.interface import InterpretedBuffer
from aghast.interface import RawInlineBuffer
from aghast.interface import RawExternalBuffer
from aghast.interface import InterpretedInlineBuffer
from aghast.interface import InterpretedInlineInt64Buffer
from aghast.interface import InterpretedInlineFloat64Buffer
from aghast.interface import InterpretedExternalBuffer
from aghast.interface import StatisticFilter
from aghast.interface import Moments
from aghast.interface import Extremes
from aghast.interface import Quantiles
from aghast.interface import Modes
from aghast.interface import Statistics
from aghast.interface import Covariance
from aghast.interface import BinLocation
from aghast.interface import IntegerBinning
from aghast.interface import RealInterval
from aghast.interface import RealOverflow
from aghast.interface import RegularBinning
from aghast.interface import HexagonalBinning
from aghast.interface import EdgesBinning
from aghast.interface import IrregularBinning
from aghast.interface import CategoryBinning
from aghast.interface import SparseRegularBinning
from aghast.interface import FractionBinning
from aghast.interface import PredicateBinning
from aghast.interface import Assignment
from aghast.interface import Variation
from aghast.interface import VariationBinning
from aghast.interface import Axis
from aghast.interface import Profile
from aghast.interface import UnweightedCounts
from aghast.interface import WeightedCounts
from aghast.interface import Parameter
from aghast.interface import ParameterizedFunction
from aghast.interface import EvaluatedFunction
from aghast.interface import BinnedEvaluatedFunction
from aghast.interface import Histogram
from aghast.interface import Page
from aghast.interface import ColumnChunk
from aghast.interface import Chunk
from aghast.interface import Column
from aghast.interface import NtupleInstance
from aghast.interface import Ntuple
from aghast.interface import Collection
