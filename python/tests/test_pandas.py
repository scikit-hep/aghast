#!/usr/bin/env python

# BSD 3-Clause License; see https://github.com/scikit-hep/aghast/blob/master/LICENSE

import unittest

import pytest
import numpy

import aghast
from aghast import *

pandas = pytest.importorskip("pandas")
pytest.importorskip("aghast._connect._pandas")

class Test(unittest.TestCase):
    def runTest(self):
        pass

    def test_pandas(self):
        a = Histogram([Axis(IntegerBinning(5, 9)), Axis(IntegerBinning(-5, 5))], WeightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(55)), sumw2=InterpretedInlineBuffer.fromarray(numpy.arange(55))), profile=[Profile("qqq", Statistics(moments=[Moments(InterpretedInlineBuffer.fromarray(numpy.arange(55)), n=2, weightpower=2), Moments(InterpretedInlineBuffer.fromarray(numpy.arange(55)), n=0)], quantiles=[Quantiles(InterpretedInlineBuffer.fromarray(numpy.arange(55)), p=0.5)], max=Extremes(InterpretedInlineBuffer.fromarray(numpy.arange(55))), min=Extremes(InterpretedInlineBuffer.fromarray(numpy.arange(55))), mode=Modes(InterpretedInlineBuffer.fromarray(numpy.arange(55))))), Profile("zzz", Statistics(moments=[Moments(InterpretedInlineBuffer.fromarray(numpy.arange(55)), n=2, weightpower=2), Moments(InterpretedInlineBuffer.fromarray(numpy.arange(55)), n=0)], quantiles=[Quantiles(InterpretedInlineBuffer.fromarray(numpy.arange(55)), p=1)], max=Extremes(InterpretedInlineBuffer.fromarray(numpy.arange(55))), min=Extremes(InterpretedInlineBuffer.fromarray(numpy.arange(55))), mode=Modes(InterpretedInlineBuffer.fromarray(numpy.arange(55)))))])
        aghast.from_pandas(aghast.to_pandas(a)[:25])
