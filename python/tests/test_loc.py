#!/usr/bin/env python

# Copyright (c) 2018, DIANA-HEP
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

import numpy

from stagg import *

class Test(unittest.TestCase):
    def runTest(self):
        pass

    def test_loc_simple(self):
        a = Histogram([Axis(IntegerBinning(-5, 5))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.ones(11, dtype=int)*10)))
        assert a.axis[0].binning.toCategoryBinning().categories == ["-5", "-4", "-3", "-2", "-1", "0", "1", "2", "3", "4", "5"]
        assert a.counts.counts.array.tolist() == [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
        aloc = a.loc[-3:2]
        assert aloc.axis[0].binning.toCategoryBinning().categories == ["-3", "-2", "-1", "0", "1", "2", "(-inf, -4]", "[3, +inf)"]
        assert aloc.counts.counts.array.tolist() == [10, 10, 10, 10, 10, 10, 20, 30]
        aloc = a.iloc[2:8]
        assert aloc.axis[0].binning.toCategoryBinning().categories == ["-3", "-2", "-1", "0", "1", "2", "(-inf, -4]", "[3, +inf)"]
        assert aloc.counts.counts.array.tolist() == [10, 10, 10, 10, 10, 10, 20, 30]

        a = Histogram([Axis(IntegerBinning(-5, 5, loc_overflow=IntegerBinning.below1))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.array([900, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]))))
        assert a.axis[0].binning.toCategoryBinning().categories == ["[6, +inf)", "-5", "-4", "-3", "-2", "-1", "0", "1", "2", "3", "4", "5"]
        assert a.counts.counts.array.tolist() == [900, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
        aloc = a.loc[-3:2]
        assert aloc.axis[0].binning.toCategoryBinning().categories == ["-3", "-2", "-1", "0", "1", "2", "(-inf, -4]", "[3, +inf)"]
        assert aloc.counts.counts.array.tolist() == [10, 10, 10, 10, 10, 10, 20, 930]
