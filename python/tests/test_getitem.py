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

import sys
import unittest

import numpy

from aghast import *

class Test(unittest.TestCase):
    def runTest(self):
        pass

    def test_getitem_twodim(self):
        a = Histogram([Axis(IntegerBinning(0, 3)), Axis(IntegerBinning(0, 2))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.array([[10, 100, 1000], [20, 200, 2000], [30, 300, 3000], [40, 400, 4000]]))))
        a.checkvalid()
        assert a.axis[0].binning.toCategoryBinning().categories == ["0", "1", "2", "3"]
        assert a.axis[1].binning.toCategoryBinning().categories == ["0", "1", "2"]
        assert a.counts.counts.array.tolist() == [[10, 100, 1000], [20, 200, 2000], [30, 300, 3000], [40, 400, 4000]]

        assert a.counts[None, None] == sum([10, 100, 1000, 20, 200, 2000, 30, 300, 3000, 40, 400, 4000])
        assert a.counts[None, :].tolist() == [100, 1000, 10000]
        assert a.counts[None].tolist() == [100, 1000, 10000]
        assert a.counts[:, None].tolist() == [1110, 2220, 3330, 4440]
        assert a.counts[None, 1] == 1000
        assert a.counts[1, None] == 2220
        assert a.counts[None, 1:].tolist() == [1000, 10000]
        assert a.counts[1:, None].tolist() == [2220, 3330, 4440]
        assert a.counts[None, [2, 1, 1, 0]].tolist() == [10000, 1000, 1000, 100]
        assert a.counts[[3, 2, 2, 0], None].tolist() == [4440, 3330, 3330, 1110]
        assert a.counts[None, [True, False, True]].tolist() == [100, 10000]
        assert a.counts[[False, True, True, False], None].tolist() == [2220, 3330]

        assert a.counts[:, :].tolist() == [[10, 100, 1000], [20, 200, 2000], [30, 300, 3000], [40, 400, 4000]]
        assert a.counts[:].tolist() == [[10, 100, 1000], [20, 200, 2000], [30, 300, 3000], [40, 400, 4000]]
        assert a.counts[1:, :].tolist() == [[20, 200, 2000], [30, 300, 3000], [40, 400, 4000]]
        assert a.counts[1:].tolist() == [[20, 200, 2000], [30, 300, 3000], [40, 400, 4000]]
        assert a.counts[:, 1:].tolist() == [[100, 1000], [200, 2000], [300, 3000], [400, 4000]]
        assert a.counts[2:, 1:].tolist() == [[300, 3000], [400, 4000]]
        assert a.counts[:, 1].tolist() == [100, 200, 300, 400]
        assert a.counts[1, :].tolist() == [20, 200, 2000]
        assert a.counts[1].tolist() == [20, 200, 2000]
        assert a.counts[2:, 1].tolist() == [300, 400]
        assert a.counts[1, 2:].tolist() == [2000]
        assert a.counts[:, [2, 0]].tolist() == [[1000, 10], [2000, 20], [3000, 30], [4000, 40]]
        assert a.counts[[2, 0], :].tolist() == [[30, 300, 3000], [10, 100, 1000]]
        assert a.counts[1:, [2, 0]].tolist() == [[2000, 20], [3000, 30], [4000, 40]]
        assert a.counts[[2, 0], 1:].tolist() == [[300, 3000], [100, 1000]]
        assert a.counts[:, [True, False, True]].tolist() == [[10, 1000], [20, 2000], [30, 3000], [40, 4000]]
        assert a.counts[[False, True, True, False], :].tolist() == [[20, 200, 2000], [30, 300, 3000]]
        assert a.counts[1:, [True, False, True]].tolist() == [[20, 2000], [30, 3000], [40, 4000]]
        assert a.counts[[False, True, True, False], 1:].tolist() == [[200, 2000], [300, 3000]]

        assert a.counts[1, 2] == 2000
        assert a.counts[1, [2, 2, 0]].tolist() == [2000, 2000, 20]
        assert a.counts[[2, 2, 0], 1].tolist() == [300, 300, 100]
        assert a.counts[1, [True, False, True]].tolist() == [20, 2000]
        assert a.counts[[False, True, True, False], 1].tolist() == [200, 300]

        assert a.counts[[1, 2], [2, 0]].tolist() == [[2000, 20], [3000, 30]]
        assert a.counts[[False, True, True, False], [2, 0]].tolist() == [[2000, 20], [3000, 30]]
        assert a.counts[[False, True, True, False], [True, False, True]].tolist() == [[20, 2000], [30, 3000]]

        assert a.counts[[2, 0], [2, 2, 0]].tolist() == [[3000, 3000, 30], [1000, 1000, 10]]
        assert a.counts[[2, 0], [True, False, True]].tolist() == [[30, 3000], [10, 1000]]
        assert a.counts[[True, False, True, False], [True, False, True]].tolist() == [[10, 1000], [30, 3000]]

    def test_getitem_IntegerBinning(self):
        a = Histogram([Axis(IntegerBinning(-5, 5))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(11, dtype=int))))
        assert a.axis[0].binning.toCategoryBinning().categories == ["-5", "-4", "-3", "-2", "-1", "0", "1", "2", "3", "4", "5"]
        assert a.counts.counts.array.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        assert a.counts[None] == 55
        assert a.counts[:].tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        assert a.counts[5:].tolist() == [5, 6, 7, 8, 9, 10]
        assert a.counts[5] == 5
        assert a.counts[[7, 4, 7, 5, -1]].tolist() == [7, 4, 7, 5, 10]
        assert a.counts[numpy.array([7, 4, 7, 5, -1])].tolist() == [7, 4, 7, 5, 10]
        assert a.counts[[True, False, True, False, True, False, True, False, True, False, True]].tolist() == [0, 2, 4, 6, 8, 10]
        assert a.counts[numpy.array([True, False, True, False, True, False, True, False, True, False, True])].tolist() == [0, 2, 4, 6, 8, 10]

        a = Histogram([Axis(IntegerBinning(-5, 5, loc_overflow=IntegerBinning.above1))], UnweightedCounts(InterpretedInlineBuffer.fromarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 999])))
        assert a.axis[0].binning.toCategoryBinning().categories == ["-5", "-4", "-3", "-2", "-1", "0", "1", "2", "3", "4", "5", "[6, +inf)"]
        assert a.counts.counts.array.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 999]

        assert a.counts[None] == 55 + 999
        assert a.counts[:].tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        assert a.counts[5:].tolist() == [5, 6, 7, 8, 9, 10]
        assert a.counts[5:numpy.inf].tolist() == [5, 6, 7, 8, 9, 10, 999]
        assert a.counts[5] == 5
        assert a.counts[numpy.inf] == 999
        assert a.counts[[7, 4, 7, 5, -1]].tolist() == [7, 4, 7, 5, 10]
        assert a.counts[[7, 4, 7, numpy.inf, 5, -1]].tolist() == [7, 4, 7, 999, 5, 10]
        assert a.counts[[True, False, True, False, True, False, True, False, True, False, True]].tolist() == [0, 2, 4, 6, 8, 10]

        a = Histogram([Axis(IntegerBinning(-5, 5, loc_overflow=IntegerBinning.below1))], UnweightedCounts(InterpretedInlineBuffer.fromarray([999, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])))
        assert a.axis[0].binning.toCategoryBinning().categories == ["[6, +inf)", "-5", "-4", "-3", "-2", "-1", "0", "1", "2", "3", "4", "5"]
        assert a.counts.counts.array.tolist() == [999, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        assert a.counts[None] == 55 + 999
        assert a.counts[:].tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        assert a.counts[5:].tolist() == [5, 6, 7, 8, 9, 10]
        assert a.counts[5:numpy.inf].tolist() == [5, 6, 7, 8, 9, 10, 999]
        assert a.counts[5] == 5
        assert a.counts[numpy.inf] == 999
        assert a.counts[[7, 4, 7, 5, -1]].tolist() == [7, 4, 7, 5, 10]
        assert a.counts[[7, 4, 7, numpy.inf, 5, -1]].tolist() == [7, 4, 7, 999, 5, 10]
        assert a.counts[[True, False, True, False, True, False, True, False, True, False, True]].tolist() == [0, 2, 4, 6, 8, 10]

        a = Histogram([Axis(IntegerBinning(-5, 5, loc_underflow=IntegerBinning.below2, loc_overflow=IntegerBinning.below1))], UnweightedCounts(InterpretedInlineBuffer.fromarray([123, 999, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])))
        assert a.axis[0].binning.toCategoryBinning().categories == ["(-inf, -6]", "[6, +inf)", "-5", "-4", "-3", "-2", "-1", "0", "1", "2", "3", "4", "5"]
        assert a.counts.counts.array.tolist() == [123, 999, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        assert a.counts[None] == 55 + 123 + 999
        assert a.counts[:].tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        assert a.counts[5:].tolist() == [5, 6, 7, 8, 9, 10]
        assert a.counts[5:numpy.inf].tolist() == [5, 6, 7, 8, 9, 10, 999]
        assert a.counts[-numpy.inf:5].tolist() == [123, 0, 1, 2, 3, 4]
        assert a.counts[-numpy.inf:numpy.inf].tolist() == [123, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 999]
        assert a.counts[:numpy.inf].tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 999]
        assert a.counts[-numpy.inf:].tolist() == [123, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        assert a.counts[5] == 5
        assert a.counts[-numpy.inf] == 123
        assert a.counts[numpy.inf] == 999
        assert a.counts[[7, 4, 7, 5, -1]].tolist() == [7, 4, 7, 5, 10]
        assert a.counts[[7, 4, 7, numpy.inf, 5, -1]].tolist() == [7, 4, 7, 999, 5, 10]
        assert a.counts[[7, 4, 7, numpy.inf, 5, -numpy.inf, -1]].tolist() == [7, 4, 7, 999, 5, 123, 10]
        assert a.counts[[7, -numpy.inf, 4, 7, numpy.inf, 5, -numpy.inf, -1]].tolist() == [7, 123, 4, 7, 999, 5, 123, 10]
        assert a.counts[[True, False, True, False, True, False, True, False, True, False, True]].tolist() == [0, 2, 4, 6, 8, 10]

        a = Histogram([Axis(IntegerBinning(-5, 5, loc_underflow=IntegerBinning.above1, loc_overflow=IntegerBinning.below1))], UnweightedCounts(InterpretedInlineBuffer.fromarray([999, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 123])))
        assert a.axis[0].binning.toCategoryBinning().categories == ["[6, +inf)", "-5", "-4", "-3", "-2", "-1", "0", "1", "2", "3", "4", "5", "(-inf, -6]"]
        assert a.counts.counts.array.tolist() == [999, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 123]

        assert a.counts[None] == 55 + 123 + 999
        assert a.counts[:].tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        assert a.counts[5:].tolist() == [5, 6, 7, 8, 9, 10]
        assert a.counts[5:numpy.inf].tolist() == [5, 6, 7, 8, 9, 10, 999]
        assert a.counts[-numpy.inf:5].tolist() == [123, 0, 1, 2, 3, 4]
        assert a.counts[-numpy.inf:numpy.inf].tolist() == [123, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 999]
        assert a.counts[:numpy.inf].tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 999]
        assert a.counts[-numpy.inf:].tolist() == [123, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        assert a.counts[5] == 5
        assert a.counts[-numpy.inf] == 123
        assert a.counts[numpy.inf] == 999
        assert a.counts[[7, 4, 7, 5, -1]].tolist() == [7, 4, 7, 5, 10]
        assert a.counts[[7, 4, 7, numpy.inf, 5, -1]].tolist() == [7, 4, 7, 999, 5, 10]
        assert a.counts[[7, 4, 7, numpy.inf, 5, -numpy.inf, -1]].tolist() == [7, 4, 7, 999, 5, 123, 10]
        assert a.counts[[7, -numpy.inf, 4, 7, numpy.inf, 5, -numpy.inf, -1]].tolist() == [7, 123, 4, 7, 999, 5, 123, 10]
        assert a.counts[[True, False, True, False, True, False, True, False, True, False, True]].tolist() == [0, 2, 4, 6, 8, 10]

    def test_getitem_RegularBinning(self):
        a = Histogram([Axis(RegularBinning(10, RealInterval(-5, 5)))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(10, dtype=int))))
        assert a.axis[0].binning.toCategoryBinning().categories == ["[-5, -4)", "[-4, -3)", "[-3, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5)"]
        assert a.counts.counts.array.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        assert a.counts[None] == sum([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        assert a.counts[:].tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        assert a.counts[5:].tolist() == [5, 6, 7, 8, 9]
        assert a.counts[5] == 5
        assert a.counts[[7, 4, 7, 5, -1]].tolist() == [7, 4, 7, 5, 9]
        assert a.counts[numpy.array([7, 4, 7, 5, -1])].tolist() == [7, 4, 7, 5, 9]
        assert a.counts[[True, False, True, False, True, False, True, False, True, False]].tolist() == [0, 2, 4, 6, 8]
        assert a.counts[numpy.array([True, False, True, False, True, False, True, False, True, False])].tolist() == [0, 2, 4, 6, 8]

        a = Histogram([Axis(RegularBinning(10, RealInterval(-5, 5), overflow=RealOverflow(loc_overflow=RealOverflow.above1)))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 999], dtype=int))))
        assert a.axis[0].binning.toCategoryBinning().categories == ["[-5, -4)", "[-4, -3)", "[-3, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5)", "[5, +inf]"]
        assert a.counts.counts.array.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 999]

        assert a.counts[None] == sum([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) + 999
        assert a.counts[:].tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        assert a.counts[:numpy.inf].tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 999]
        assert a.counts[5:].tolist() == [5, 6, 7, 8, 9]
        assert a.counts[5:numpy.inf].tolist() == [5, 6, 7, 8, 9, 999]
        assert a.counts[5] == 5
        assert a.counts[[7, numpy.inf, 4, 7, 5, numpy.inf, -1]].tolist() == [7, 999, 4, 7, 5, 999, 9]
        assert a.counts[numpy.array([7, numpy.inf, 4, 7, 5, numpy.inf, -1])].tolist() == [7, 999, 4, 7, 5, 999, 9]
        assert a.counts[[True, False, True, False, True, False, True, False, True, False]].tolist() == [0, 2, 4, 6, 8]
        assert a.counts[numpy.array([True, False, True, False, True, False, True, False, True, False])].tolist() == [0, 2, 4, 6, 8]

        a = Histogram([Axis(RegularBinning(10, RealInterval(-5, 5), overflow=RealOverflow(loc_overflow=RealOverflow.below1)))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.array([999, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int))))
        assert a.axis[0].binning.toCategoryBinning().categories == ["[5, +inf]", "[-5, -4)", "[-4, -3)", "[-3, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5)"]
        assert a.counts.counts.array.tolist() == [999, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        assert a.counts[None] == sum([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) + 999
        assert a.counts[:].tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        assert a.counts[:numpy.inf].tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 999]
        assert a.counts[5:].tolist() == [5, 6, 7, 8, 9]
        assert a.counts[5:numpy.inf].tolist() == [5, 6, 7, 8, 9, 999]
        assert a.counts[5] == 5
        assert a.counts[numpy.inf] == 999
        assert a.counts[[7, numpy.inf, 4, 7, 5, numpy.inf, -1]].tolist() == [7, 999, 4, 7, 5, 999, 9]
        assert a.counts[numpy.array([7, numpy.inf, 4, 7, 5, numpy.inf, -1])].tolist() == [7, 999, 4, 7, 5, 999, 9]
        assert a.counts[[True, False, True, False, True, False, True, False, True, False]].tolist() == [0, 2, 4, 6, 8]
        assert a.counts[numpy.array([True, False, True, False, True, False, True, False, True, False])].tolist() == [0, 2, 4, 6, 8]

        a = Histogram([Axis(RegularBinning(10, RealInterval(-5, 5), overflow=RealOverflow(loc_overflow=RealOverflow.below2, loc_nanflow=RealOverflow.below1)))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.array([999, 123, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int))))
        assert a.axis[0].binning.toCategoryBinning().categories == ["[5, +inf]", "{nan}", "[-5, -4)", "[-4, -3)", "[-3, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5)"]
        assert a.counts.counts.array.tolist() == [999, 123, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        assert a.counts[None] == sum([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) + 999 + 123
        assert a.counts[:].tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        assert a.counts[:numpy.inf].tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 999]
        assert a.counts[5:].tolist() == [5, 6, 7, 8, 9]
        assert a.counts[5:numpy.inf].tolist() == [5, 6, 7, 8, 9, 999]
        assert a.counts[5] == 5
        assert a.counts[numpy.inf] == 999
        assert a.counts[numpy.nan] == 123
        assert a.counts[[7, numpy.inf, 4, 7, 5, numpy.nan, -1]].tolist() == [7, 999, 4, 7, 5, 123, 9]
        if sys.version_info[0] >= 3:
            exec("assert a.counts[[numpy.inf, ..., numpy.nan]].tolist() == [999, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 123]")
        assert a.counts[numpy.array([7, numpy.inf, 4, 7, 5, numpy.nan, -1])].tolist() == [7, 999, 4, 7, 5, 123, 9]
        assert a.counts[[True, False, True, False, True, False, True, False, True, False]].tolist() == [0, 2, 4, 6, 8]
        assert a.counts[numpy.array([True, False, True, False, True, False, True, False, True, False])].tolist() == [0, 2, 4, 6, 8]

        a = Histogram([Axis(RegularBinning(10, RealInterval(-5, 5), overflow=RealOverflow(loc_overflow=RealOverflow.above1, loc_nanflow=RealOverflow.below1)))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.array([123, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 999], dtype=int))))
        assert a.axis[0].binning.toCategoryBinning().categories == ["{nan}", "[-5, -4)", "[-4, -3)", "[-3, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5)", "[5, +inf]"]
        assert a.counts.counts.array.tolist() == [123, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 999]

        assert a.counts[None] == sum([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) + 999 + 123
        assert a.counts[:].tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        assert a.counts[:numpy.inf].tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 999]
        assert a.counts[5:].tolist() == [5, 6, 7, 8, 9]
        assert a.counts[5:numpy.inf].tolist() == [5, 6, 7, 8, 9, 999]
        assert a.counts[5] == 5
        assert a.counts[numpy.inf] == 999
        assert a.counts[numpy.nan] == 123
        assert a.counts[[7, numpy.inf, 4, 7, 5, numpy.nan, -1]].tolist() == [7, 999, 4, 7, 5, 123, 9]
        assert a.counts[numpy.array([7, numpy.inf, 4, 7, 5, numpy.nan, -1])].tolist() == [7, 999, 4, 7, 5, 123, 9]
        assert a.counts[[True, False, True, False, True, False, True, False, True, False]].tolist() == [0, 2, 4, 6, 8]
        assert a.counts[numpy.array([True, False, True, False, True, False, True, False, True, False])].tolist() == [0, 2, 4, 6, 8]
