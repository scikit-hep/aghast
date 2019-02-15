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

import numpy

from aghast import *

class Test(unittest.TestCase):
    def runTest(self):
        pass

    def test_loc_twodim(self):
        a = Histogram([Axis(IntegerBinning(5, 9)), Axis(IntegerBinning(-5, 5))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.ones(55, dtype=int)*10)))
        assert a.axis[0].binning.toCategoryBinning().categories == ["5", "6", "7", "8", "9"]
        assert a.axis[1].binning.toCategoryBinning().categories == ["-5", "-4", "-3", "-2", "-1", "0", "1", "2", "3", "4", "5"]
        assert a.counts.counts.array.tolist() == (numpy.ones(55, dtype=int)*10).reshape((5, 11)).tolist()
        aloc = a.loc[7:, -3:2]
        assert aloc.axis[0].binning.toCategoryBinning().categories == ["7", "8", "9", "(-inf, 6]"]
        assert aloc.axis[1].binning.toCategoryBinning().categories == ["-3", "-2", "-1", "0", "1", "2", "(-inf, -4]", "[3, +inf)"]
        assert aloc.counts.counts.array.tolist() == [[10, 10, 10, 10, 10, 10, 20, 30], [10, 10, 10, 10, 10, 10, 20, 30], [10, 10, 10, 10, 10, 10, 20, 30], [20, 20, 20, 20, 20, 20, 40, 60]]
        assert a.axis[0].binning.toCategoryBinning().categories == ["5", "6", "7", "8", "9"]
        assert a.axis[1].binning.toCategoryBinning().categories == ["-5", "-4", "-3", "-2", "-1", "0", "1", "2", "3", "4", "5"]
        assert a.counts.counts.array.tolist() == (numpy.ones(55, dtype=int)*10).reshape((5, 11)).tolist()

        a = Histogram([Axis(IntegerBinning(5, 9)), Axis(IntegerBinning(-5, 5))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.ones(55, dtype=int)*10)))
        assert a.axis[0].binning.toCategoryBinning().categories == ["5", "6", "7", "8", "9"]
        assert a.axis[1].binning.toCategoryBinning().categories == ["-5", "-4", "-3", "-2", "-1", "0", "1", "2", "3", "4", "5"]
        assert a.counts.counts.array.tolist() == (numpy.ones(55, dtype=int)*10).reshape((5, 11)).tolist()
        aloc = a.loc[..., -3:2]
        assert aloc.axis[0].binning.toCategoryBinning().categories == ["5", "6", "7", "8", "9"]
        assert aloc.axis[1].binning.toCategoryBinning().categories == ["-3", "-2", "-1", "0", "1", "2", "(-inf, -4]", "[3, +inf)"]
        assert aloc.counts.counts.array.tolist() == [[10, 10, 10, 10, 10, 10, 20, 30], [10, 10, 10, 10, 10, 10, 20, 30], [10, 10, 10, 10, 10, 10, 20, 30], [10, 10, 10, 10, 10, 10, 20, 30], [10, 10, 10, 10, 10, 10, 20, 30]]
        assert a.axis[0].binning.toCategoryBinning().categories == ["5", "6", "7", "8", "9"]
        assert a.axis[1].binning.toCategoryBinning().categories == ["-5", "-4", "-3", "-2", "-1", "0", "1", "2", "3", "4", "5"]
        assert a.counts.counts.array.tolist() == (numpy.ones(55, dtype=int)*10).reshape((5, 11)).tolist()

        a = Histogram([Axis(IntegerBinning(5, 9)), Axis(IntegerBinning(-5, 5))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.ones(55, dtype=int)*10)))
        assert a.axis[0].binning.toCategoryBinning().categories == ["5", "6", "7", "8", "9"]
        assert a.axis[1].binning.toCategoryBinning().categories == ["-5", "-4", "-3", "-2", "-1", "0", "1", "2", "3", "4", "5"]
        assert a.counts.counts.array.tolist() == (numpy.ones(55, dtype=int)*10).reshape((5, 11)).tolist()
        aloc = a.loc[:, -3:2]
        assert aloc.axis[0].binning.toCategoryBinning().categories == ["5", "6", "7", "8", "9"]
        assert aloc.axis[1].binning.toCategoryBinning().categories == ["-3", "-2", "-1", "0", "1", "2", "(-inf, -4]", "[3, +inf)"]
        assert aloc.counts.counts.array.tolist() == [[10, 10, 10, 10, 10, 10, 20, 30], [10, 10, 10, 10, 10, 10, 20, 30], [10, 10, 10, 10, 10, 10, 20, 30], [10, 10, 10, 10, 10, 10, 20, 30], [10, 10, 10, 10, 10, 10, 20, 30]]
        assert a.axis[0].binning.toCategoryBinning().categories == ["5", "6", "7", "8", "9"]
        assert a.axis[1].binning.toCategoryBinning().categories == ["-5", "-4", "-3", "-2", "-1", "0", "1", "2", "3", "4", "5"]
        assert a.counts.counts.array.tolist() == (numpy.ones(55, dtype=int)*10).reshape((5, 11)).tolist()

        a = Histogram([Axis(IntegerBinning(5, 9)), Axis(IntegerBinning(-5, 5))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.ones(55, dtype=int)*10)))
        assert a.axis[0].binning.toCategoryBinning().categories == ["5", "6", "7", "8", "9"]
        assert a.axis[1].binning.toCategoryBinning().categories == ["-5", "-4", "-3", "-2", "-1", "0", "1", "2", "3", "4", "5"]
        assert a.counts.counts.array.tolist() == (numpy.ones(55, dtype=int)*10).reshape((5, 11)).tolist()
        aloc = a.loc[5:9, -3:2]
        assert aloc.axis[0].binning.toCategoryBinning().categories == ["5", "6", "7", "8", "9"]
        assert aloc.axis[1].binning.toCategoryBinning().categories == ["-3", "-2", "-1", "0", "1", "2", "(-inf, -4]", "[3, +inf)"]
        assert aloc.counts.counts.array.tolist() == [[10, 10, 10, 10, 10, 10, 20, 30], [10, 10, 10, 10, 10, 10, 20, 30], [10, 10, 10, 10, 10, 10, 20, 30], [10, 10, 10, 10, 10, 10, 20, 30], [10, 10, 10, 10, 10, 10, 20, 30]]
        assert a.axis[0].binning.toCategoryBinning().categories == ["5", "6", "7", "8", "9"]
        assert a.axis[1].binning.toCategoryBinning().categories == ["-5", "-4", "-3", "-2", "-1", "0", "1", "2", "3", "4", "5"]
        assert a.counts.counts.array.tolist() == (numpy.ones(55, dtype=int)*10).reshape((5, 11)).tolist()

        a = Histogram([Axis(IntegerBinning(5, 9)), Axis(IntegerBinning(-5, 5))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.ones(55, dtype=int)*10)))
        assert a.axis[0].binning.toCategoryBinning().categories == ["5", "6", "7", "8", "9"]
        assert a.axis[1].binning.toCategoryBinning().categories == ["-5", "-4", "-3", "-2", "-1", "0", "1", "2", "3", "4", "5"]
        assert a.counts.counts.array.tolist() == (numpy.ones(55, dtype=int)*10).reshape((5, 11)).tolist()
        aloc = a.loc[7:]
        assert aloc.axis[0].binning.toCategoryBinning().categories == ["7", "8", "9", "(-inf, 6]"]
        assert aloc.axis[1].binning.toCategoryBinning().categories == ["-5", "-4", "-3", "-2", "-1", "0", "1", "2", "3", "4", "5"]
        assert aloc.counts.counts.array.tolist() == [[10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10], [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10], [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10], [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20]]
        assert a.axis[0].binning.toCategoryBinning().categories == ["5", "6", "7", "8", "9"]
        assert a.axis[1].binning.toCategoryBinning().categories == ["-5", "-4", "-3", "-2", "-1", "0", "1", "2", "3", "4", "5"]
        assert a.counts.counts.array.tolist() == (numpy.ones(55, dtype=int)*10).reshape((5, 11)).tolist()

        a = Histogram([Axis(IntegerBinning(5, 9)), Axis(IntegerBinning(-5, 5))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.ones(55, dtype=int)*10)))
        assert a.axis[0].binning.toCategoryBinning().categories == ["5", "6", "7", "8", "9"]
        assert a.axis[1].binning.toCategoryBinning().categories == ["-5", "-4", "-3", "-2", "-1", "0", "1", "2", "3", "4", "5"]
        assert a.counts.counts.array.tolist() == (numpy.ones(55, dtype=int)*10).reshape((5, 11)).tolist()
        aloc = a.loc[None, -3:2]
        assert len(aloc.axis) == 1
        assert aloc.axis[0].binning.toCategoryBinning().categories == ["-3", "-2", "-1", "0", "1", "2", "(-inf, -4]", "[3, +inf)"]
        assert aloc.counts.counts.array.tolist() == [50, 50, 50, 50, 50, 50, 100, 150]
        assert a.axis[0].binning.toCategoryBinning().categories == ["5", "6", "7", "8", "9"]
        assert a.axis[1].binning.toCategoryBinning().categories == ["-5", "-4", "-3", "-2", "-1", "0", "1", "2", "3", "4", "5"]
        assert a.counts.counts.array.tolist() == (numpy.ones(55, dtype=int)*10).reshape((5, 11)).tolist()

        a = Histogram([Axis(IntegerBinning(5, 9)), Axis(IntegerBinning(-5, 5))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.ones(55, dtype=int)*10)))
        assert a.axis[0].binning.toCategoryBinning().categories == ["5", "6", "7", "8", "9"]
        assert a.axis[1].binning.toCategoryBinning().categories == ["-5", "-4", "-3", "-2", "-1", "0", "1", "2", "3", "4", "5"]
        assert a.counts.counts.array.tolist() == (numpy.ones(55, dtype=int)*10).reshape((5, 11)).tolist()
        aloc = a.loc[7:, None]
        assert len(aloc.axis) == 1
        assert aloc.axis[0].binning.toCategoryBinning().categories == ["7", "8", "9", "(-inf, 6]"]
        assert aloc.counts.counts.array.tolist() == [110, 110, 110, 220]
        assert a.axis[0].binning.toCategoryBinning().categories == ["5", "6", "7", "8", "9"]
        assert a.axis[1].binning.toCategoryBinning().categories == ["-5", "-4", "-3", "-2", "-1", "0", "1", "2", "3", "4", "5"]
        assert a.counts.counts.array.tolist() == (numpy.ones(55, dtype=int)*10).reshape((5, 11)).tolist()

    def test_loc_IntegerBinning(self):
        a = Histogram([Axis(IntegerBinning(-5, 5))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.ones(11, dtype=int)*10)))
        assert a.axis[0].binning.toCategoryBinning().categories == ["-5", "-4", "-3", "-2", "-1", "0", "1", "2", "3", "4", "5"]
        assert a.counts.counts.array.tolist() == [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
        aloc = a.loc[-3:2]
        assert aloc.axis[0].binning.toCategoryBinning().categories == ["-3", "-2", "-1", "0", "1", "2", "(-inf, -4]", "[3, +inf)"]
        assert aloc.counts.counts.array.tolist() == [10, 10, 10, 10, 10, 10, 20, 30]
        assert a.axis[0].binning.toCategoryBinning().categories == ["-5", "-4", "-3", "-2", "-1", "0", "1", "2", "3", "4", "5"]
        assert a.counts.counts.array.tolist() == [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
        aloc = a.iloc[2:8]
        assert aloc.axis[0].binning.toCategoryBinning().categories == ["-3", "-2", "-1", "0", "1", "2", "(-inf, -4]", "[3, +inf)"]
        assert aloc.counts.counts.array.tolist() == [10, 10, 10, 10, 10, 10, 20, 30]
        assert a.axis[0].binning.toCategoryBinning().categories == ["-5", "-4", "-3", "-2", "-1", "0", "1", "2", "3", "4", "5"]
        assert a.counts.counts.array.tolist() == [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]

        a = Histogram([Axis(IntegerBinning(-5, 5, loc_overflow=IntegerBinning.below1))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.array([900, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]))))
        assert a.axis[0].binning.toCategoryBinning().categories == ["[6, +inf)", "-5", "-4", "-3", "-2", "-1", "0", "1", "2", "3", "4", "5"]
        assert a.counts.counts.array.tolist() == [900, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
        aloc = a.loc[-3:2]
        assert aloc.axis[0].binning.toCategoryBinning().categories == ["-3", "-2", "-1", "0", "1", "2", "(-inf, -4]", "[3, +inf)"]
        assert aloc.counts.counts.array.tolist() == [10, 10, 10, 10, 10, 10, 20, 930]
        assert a.axis[0].binning.toCategoryBinning().categories == ["[6, +inf)", "-5", "-4", "-3", "-2", "-1", "0", "1", "2", "3", "4", "5"]
        assert a.counts.counts.array.tolist() == [900, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]

        a = Histogram([Axis(IntegerBinning(-5, 5))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.ones(11, dtype=int)*10)))
        assert a.axis[0].binning.toCategoryBinning().categories == ["-5", "-4", "-3", "-2", "-1", "0", "1", "2", "3", "4", "5"]
        assert a.counts.counts.array.tolist() == [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
        aloc = a.loc[2]
        assert aloc.axis[0].binning.toCategoryBinning().categories == ["2", "(-inf, 1]", "[3, +inf)"]
        assert aloc.counts.counts.array.tolist() == [10, 70, 30]
        assert a.axis[0].binning.toCategoryBinning().categories == ["-5", "-4", "-3", "-2", "-1", "0", "1", "2", "3", "4", "5"]
        assert a.counts.counts.array.tolist() == [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]

        a = Histogram([Axis(IntegerBinning(-5, 5))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.ones(11, dtype=int)*10)))
        assert a.axis[0].binning.toCategoryBinning().categories == ["-5", "-4", "-3", "-2", "-1", "0", "1", "2", "3", "4", "5"]
        assert a.counts.counts.array.tolist() == [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
        aloc = a.loc[None]
        assert len(aloc.axis) == 1
        assert aloc.axis[0].binning is None
        assert aloc.counts.counts.array.tolist() == [110]

    def test_loc_RegularBinning(self):
        a = Histogram([Axis(RegularBinning(20, RealInterval(-5, 5)))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]))))
        assert a.axis[0].binning.toCategoryBinning().categories == ["[-5, -4.5)", "[-4.5, -4)", "[-4, -3.5)", "[-3.5, -3)", "[-3, -2.5)", "[-2.5, -2)", "[-2, -1.5)", "[-1.5, -1)", "[-1, -0.5)", "[-0.5, 0)", "[0, 0.5)", "[0.5, 1)", "[1, 1.5)", "[1.5, 2)", "[2, 2.5)", "[2.5, 3)", "[3, 3.5)", "[3.5, 4)", "[4, 4.5)", "[4.5, 5)"]
        assert a.counts.counts.array.tolist() == [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]

        aloc = a.iloc[::2]
        assert aloc.axis[0].binning.toCategoryBinning().categories == ['[-5, -4)', '[-4, -3)', '[-3, -2)', '[-2, -1)', '[-1, 0)', '[0, 1)', '[1, 2)', '[2, 3)', '[3, 4)', '[4, 5)']
        assert aloc.counts.counts.array.tolist() == [20, 20, 20, 20, 20, 20, 20, 20, 20, 20]
        aloc = a.iloc[0:10:2]
        assert aloc.axis[0].binning.toCategoryBinning().categories == ['[-5, -4)', '[-4, -3)', '[-3, -2)', '[-2, -1)', '[-1, 0)', '[0, +inf)']
        assert aloc.counts.counts.array.tolist() == [20, 20, 20, 20, 20, 100]
        aloc = a.iloc[10:20:2]
        assert aloc.axis[0].binning.toCategoryBinning().categories == ['[0, 1)', '[1, 2)', '[2, 3)', '[3, 4)', '[4, 5)', '(-inf, 0)']
        assert aloc.counts.counts.array.tolist() == [20, 20, 20, 20, 20, 100]

        aloc = a.loc[-3.5::0.5]
        assert aloc.axis[0].binning.toCategoryBinning().categories == ["[-3.5, -3)", "[-3, -2.5)", "[-2.5, -2)", "[-2, -1.5)", "[-1.5, -1)", "[-1, -0.5)", "[-0.5, 0)", "[0, 0.5)", "[0.5, 1)", "[1, 1.5)", "[1.5, 2)", "[2, 2.5)", "[2.5, 3)", "[3, 3.5)", "[3.5, 4)", "[4, 4.5)", "[4.5, 5)", "(-inf, -3.5)"]
        assert aloc.counts.counts.array.tolist() == [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 30]
        aloc = a.loc[::1]
        assert aloc.axis[0].binning.toCategoryBinning().categories == ['[-5, -4)', '[-4, -3)', '[-3, -2)', '[-2, -1)', '[-1, 0)', '[0, 1)', '[1, 2)', '[2, 3)', '[3, 4)', '[4, 5)']
        assert aloc.counts.counts.array.tolist() == [20, 20, 20, 20, 20, 20, 20, 20, 20, 20]
        aloc = a.loc[-100:0:1]
        assert aloc.axis[0].binning.toCategoryBinning().categories == ['[-5, -4)', '[-4, -3)', '[-3, -2)', '[-2, -1)', '[-1, 0)', '[0, +inf)']
        assert aloc.counts.counts.array.tolist() == [20, 20, 20, 20, 20, 100]
        aloc = a.loc[0:100:1]
        assert aloc.axis[0].binning.toCategoryBinning().categories == ['[0, 1)', '[1, 2)', '[2, 3)', '[3, 4)', '[4, 5)', '(-inf, 0)']
        assert aloc.counts.counts.array.tolist() == [20, 20, 20, 20, 20, 100]

        a = Histogram([Axis(RegularBinning(19, RealInterval(-5, 4.5)))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]))))
        assert a.axis[0].binning.toCategoryBinning().categories == ["[-5, -4.5)", "[-4.5, -4)", "[-4, -3.5)", "[-3.5, -3)", "[-3, -2.5)", "[-2.5, -2)", "[-2, -1.5)", "[-1.5, -1)", "[-1, -0.5)", "[-0.5, 0)", "[0, 0.5)", "[0.5, 1)", "[1, 1.5)", "[1.5, 2)", "[2, 2.5)", "[2.5, 3)", "[3, 3.5)", "[3.5, 4)", "[4, 4.5)"]
        assert a.counts.counts.array.tolist() == [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]

        aloc = a.iloc[::2]
        assert aloc.axis[0].binning.toCategoryBinning().categories == ['[-5, -4)', '[-4, -3)', '[-3, -2)', '[-2, -1)', '[-1, 0)', '[0, 1)', '[1, 2)', '[2, 3)', '[3, 4)', '[4, +inf)']
        assert aloc.counts.counts.array.tolist() == [20, 20, 20, 20, 20, 20, 20, 20, 20, 10]

    def test_loc_EdgesBinning(self):
        a = Histogram([Axis(EdgesBinning([3.3, 5.5, 10.0, 55.5, 100.0]))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.array([10, 10, 10, 10]))))
        assert a.axis[0].binning.toCategoryBinning().categories == ["[3.3, 5.5)", "[5.5, 10)", "[10, 55.5)", "[55.5, 100)"]
        assert a.counts.counts.array.tolist() == [10, 10, 10, 10]

        aloc = a.iloc[::2]
        assert aloc.axis[0].binning.toCategoryBinning().categories == ["[3.3, 10)", "[10, 100)"]
        assert aloc.counts.counts.array.tolist() == [20, 20]

        aloc = a.loc[5.4:]
        assert aloc.axis[0].binning.toCategoryBinning().categories == ["[3.3, 5.5)", "[5.5, 10)", "[10, 55.5)", "[55.5, 100)"]
        assert aloc.counts.counts.array.tolist() == [10, 10, 10, 10]

        aloc = a.loc[5.5:]
        assert aloc.axis[0].binning.toCategoryBinning().categories == ["[5.5, 10)", "[10, 55.5)", "[55.5, 100)", "(-inf, 5.5)"]
        assert aloc.counts.counts.array.tolist() == [10, 10, 10, 10]

        aloc = a.loc[5.6:]
        assert aloc.axis[0].binning.toCategoryBinning().categories == ["[5.5, 10)", "[10, 55.5)", "[55.5, 100)", "(-inf, 5.5)"]
        assert aloc.counts.counts.array.tolist() == [10, 10, 10, 10]

        aloc = a.loc[:55.6]
        assert aloc.axis[0].binning.toCategoryBinning().categories == ["[3.3, 5.5)", "[5.5, 10)", "[10, 55.5)", "[55.5, 100)"]
        assert aloc.counts.counts.array.tolist() == [10, 10, 10, 10]

        aloc = a.loc[:55.5]
        assert aloc.axis[0].binning.toCategoryBinning().categories == ["[3.3, 5.5)", "[5.5, 10)", "[10, 55.5)", "[55.5, +inf)"]
        assert aloc.counts.counts.array.tolist() == [10, 10, 10, 10]

        aloc = a.loc[:55.4]
        assert aloc.axis[0].binning.toCategoryBinning().categories == ["[3.3, 5.5)", "[5.5, 10)", "[10, 55.5)", "[55.5, +inf)"]
        assert aloc.counts.counts.array.tolist() == [10, 10, 10, 10]

        a = Histogram([Axis(EdgesBinning([3.3, 5.5, 10.0, 55.5]))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.array([10, 10, 10]))))
        assert a.axis[0].binning.toCategoryBinning().categories == ["[3.3, 5.5)", "[5.5, 10)", "[10, 55.5)"]
        assert a.counts.counts.array.tolist() == [10, 10, 10]

        aloc = a.iloc[::2]
        assert aloc.axis[0].binning.toCategoryBinning().categories == ["[3.3, 10)", "[10, +inf)"]
        assert aloc.counts.counts.array.tolist() == [20, 10]

    def test_loc_IrregularBinning(self):
        a = Histogram([Axis(IrregularBinning([RealInterval(5, 10), RealInterval(8, 18), RealInterval(0, 1)]))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.array([10, 10, 10]))))
        assert a.axis[0].binning.toCategoryBinning().categories == ["[5, 10)", "[8, 18)", "[0, 1)"]
        assert a.counts.counts.array.tolist() == [10, 10, 10]

        aloc = a.iloc[::2]
        assert aloc.axis[0].binning.toCategoryBinning().categories == ["[5, 10)", "[0, 1)"]
        assert aloc.counts.counts.array.tolist() == [10, 10]

        aloc = a.loc[5:18]
        assert aloc.axis[0].binning.toCategoryBinning().categories == ["[5, 10)", "[8, 18)", "(-inf, 5)"]
        assert aloc.counts.counts.array.tolist() == [10, 10, 10]

        aloc = a.loc[6:17]
        assert aloc.axis[0].binning.toCategoryBinning().categories == ["[5, 10)", "[8, 18)", "(-inf, 5)"]
        assert aloc.counts.counts.array.tolist() == [10, 10, 10]

        aloc = a.loc[8:17]
        assert aloc.axis[0].binning.toCategoryBinning().categories == ["[5, 10)", "[8, 18)", "(-inf, 5)"]
        assert aloc.counts.counts.array.tolist() == [10, 10, 10]

        aloc = a.loc[10:17]
        assert aloc.axis[0].binning.toCategoryBinning().categories == ["[8, 18)", "(-inf, 8)"]
        assert aloc.counts.counts.array.tolist() == [10, 20]

    def test_loc_CategoryBinning(self):
        a = Histogram([Axis(CategoryBinning(["one", "two", "three", "four", "five"]))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.array([10, 10, 10, 10, 10]))))
        assert a.axis[0].binning.loc_overflow == CategoryBinning.nonexistent
        assert a.counts.counts.array.tolist() == [10, 10, 10, 10, 10]

        aloc = a.iloc[::2]
        assert aloc.axis[0].binning.loc_overflow == CategoryBinning.above1
        assert aloc.axis[0].binning.categories == ["one", "three", "five"]
        assert aloc.counts.counts.array.tolist() == [10, 10, 10, 20]

        aloc = a.iloc[3:]
        assert aloc.axis[0].binning.loc_overflow == CategoryBinning.above1
        assert aloc.axis[0].binning.categories == ["four", "five"]
        assert aloc.counts.counts.array.tolist() == [10, 10, 30]

        a = Histogram([Axis(CategoryBinning(["one", "two", "three", "four", "five"], loc_overflow=CategoryBinning.below1))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.array([100, 10, 10, 10, 10, 10]))))
        assert a.axis[0].binning.loc_overflow == CategoryBinning.below1
        assert a.counts.counts.array.tolist() == [100, 10, 10, 10, 10, 10]

        aloc = a.iloc[::2]
        assert aloc.axis[0].binning.loc_overflow == CategoryBinning.above1
        assert aloc.axis[0].binning.categories == ["one", "three", "five"]
        assert aloc.counts.counts.array.tolist() == [10, 10, 10, 120]

        a = Histogram([Axis(CategoryBinning(["one", "two", "three", "four", "five"]))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.array([1, 2, 3, 4, 5]))))
        assert a.axis[0].binning.loc_overflow == CategoryBinning.nonexistent
        assert a.counts.counts.array.tolist() == [1, 2, 3, 4, 5]

        aloc = a.loc[["four", "two", "three"]]
        assert aloc.axis[0].binning.loc_overflow == CategoryBinning.above1
        assert aloc.counts.counts.array.tolist() == [4, 2, 3, 6]

        a = Histogram([Axis(CategoryBinning(["one", "two", "three", "four", "five"], loc_overflow=CategoryBinning.below1))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.array([100, 1, 2, 3, 4, 5]))))
        assert a.axis[0].binning.loc_overflow == CategoryBinning.below1
        assert a.counts.counts.array.tolist() == [100, 1, 2, 3, 4, 5]

        aloc = a.loc[["four", "two", "three"]]
        assert aloc.axis[0].binning.loc_overflow == CategoryBinning.above1
        assert aloc.counts.counts.array.tolist() == [4, 2, 3, 106]

    def test_loc_SparseRegularBinning(self):
        a = Histogram([Axis(SparseRegularBinning([5, 3, -2, 8, -5], 10.0))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.array([1, 2, 3, 4, 5]))))
        assert a.axis[0].binning.toCategoryBinning().categories == ["[50, 60)", "[30, 40)", "[-20, -10)", "[80, 90)", "[-50, -40)"]
        assert a.counts.counts.array.tolist() == [1, 2, 3, 4, 5]

        aloc = a.iloc[2:]
        assert aloc.axis[0].binning.toCategoryBinning().categories == ["[-20, -10)", "[80, 90)", "[-50, -40)"]
        assert aloc.counts.counts.array.tolist() == [3, 4, 5]
        assert a.axis[0].binning.toCategoryBinning().categories == ["[50, 60)", "[30, 40)", "[-20, -10)", "[80, 90)", "[-50, -40)"]
        assert a.counts.counts.array.tolist() == [1, 2, 3, 4, 5]

        aloc = a.loc[0]
        assert aloc.axis[0].binning.toCategoryBinning().categories == ["[0, 10)", "(-inf, 0)", "[0, +inf)"]
        assert aloc.counts.counts.array.tolist() == [0, 8, 7]

        aloc = a.loc[30:]
        assert aloc.axis[0].binning.toCategoryBinning().categories == ["[50, 60)", "[30, 40)", "[80, 90)", "(-inf, 30)"]
        assert aloc.counts.counts.array.tolist() == [1, 2, 4, 8]
        assert a.axis[0].binning.toCategoryBinning().categories == ["[50, 60)", "[30, 40)", "[-20, -10)", "[80, 90)", "[-50, -40)"]
        assert a.counts.counts.array.tolist() == [1, 2, 3, 4, 5]

        aloc = a.loc[:30]
        assert aloc.axis[0].binning.toCategoryBinning().categories == ["[-20, -10)", "[-50, -40)", "[30, +inf)"]
        assert aloc.counts.counts.array.tolist() == [3, 5, 7]
        assert a.axis[0].binning.toCategoryBinning().categories == ["[50, 60)", "[30, 40)", "[-20, -10)", "[80, 90)", "[-50, -40)"]
        assert a.counts.counts.array.tolist() == [1, 2, 3, 4, 5]

        aloc = a.loc[30::20]
        assert aloc.axis[0].binning.toCategoryBinning().categories == ["[50, 70)", "[30, 50)", "[70, 90)", "(-inf, 30)"]
        assert aloc.counts.counts.array.tolist() == [1, 2, 4, 8]
        assert a.axis[0].binning.toCategoryBinning().categories == ["[50, 60)", "[30, 40)", "[-20, -10)", "[80, 90)", "[-50, -40)"]
        assert a.counts.counts.array.tolist() == [1, 2, 3, 4, 5]

        aloc = a.loc[::20]
        assert aloc.axis[0].binning.toCategoryBinning().categories == ["[40, 60)", "[20, 40)", "[-20, 0)", "[80, 100)", "[-60, -40)"]
        assert aloc.counts.counts.array.tolist() == [1, 2, 3, 4, 5]
        assert a.axis[0].binning.toCategoryBinning().categories == ["[50, 60)", "[30, 40)", "[-20, -10)", "[80, 90)", "[-50, -40)"]
        assert a.counts.counts.array.tolist() == [1, 2, 3, 4, 5]

