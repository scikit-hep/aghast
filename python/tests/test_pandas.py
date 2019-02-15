#!/usr/bin/env python

# Copyright (c) 2019, IRIS-HEP
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import unittest

import pytest
import numpy

from aghast import *

pandas = pytest.importorskip("pandas")
connect_pandas = pytest.importorskip("aghast.connect.pandas")

class Test(unittest.TestCase):
    def runTest(self):
        pass

    def test_pandas(self):
        a = Histogram([Axis(IntegerBinning(5, 9)), Axis(IntegerBinning(-5, 5))], WeightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(55)), sumw2=InterpretedInlineBuffer.fromarray(numpy.arange(55))), profile=[Profile("qqq", Statistics(moments=[Moments(InterpretedInlineBuffer.fromarray(numpy.arange(55)), n=2, weightpower=2), Moments(InterpretedInlineBuffer.fromarray(numpy.arange(55)), n=0)], quantiles=[Quantiles(InterpretedInlineBuffer.fromarray(numpy.arange(55)), p=0.5)], max=Extremes(InterpretedInlineBuffer.fromarray(numpy.arange(55))), min=Extremes(InterpretedInlineBuffer.fromarray(numpy.arange(55))), mode=Modes(InterpretedInlineBuffer.fromarray(numpy.arange(55))))), Profile("zzz", Statistics(moments=[Moments(InterpretedInlineBuffer.fromarray(numpy.arange(55)), n=2, weightpower=2), Moments(InterpretedInlineBuffer.fromarray(numpy.arange(55)), n=0)], quantiles=[Quantiles(InterpretedInlineBuffer.fromarray(numpy.arange(55)), p=1)], max=Extremes(InterpretedInlineBuffer.fromarray(numpy.arange(55))), min=Extremes(InterpretedInlineBuffer.fromarray(numpy.arange(55))), mode=Modes(InterpretedInlineBuffer.fromarray(numpy.arange(55)))))])
        connect_pandas.frompandas(connect_pandas.topandas(a)[:25])
