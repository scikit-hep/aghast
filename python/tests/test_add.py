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

from stagg import *

class Test(unittest.TestCase):
    def runTest(self):
        pass

    def test_add_same_same(self):
        a = Histogram([Axis(IntegerBinning(1, 2)), Axis(IntegerBinning(10, 19))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(20))))
        b = Histogram([Axis(IntegerBinning(1, 2)), Axis(IntegerBinning(10, 19))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.full(20, 100))))
        assert a.counts.counts.array.tolist() == [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]]
        assert b.counts.counts.array.tolist() == [[100, 100, 100, 100, 100, 100, 100, 100, 100, 100], [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]]
        assert (a + b).counts.counts.array.tolist() == [[100, 101, 102, 103, 104, 105, 106, 107, 108, 109], [110, 111, 112, 113, 114, 115, 116, 117, 118, 119]]
        assert a.counts.counts.array.tolist() == [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]]
        assert b.counts.counts.array.tolist() == [[100, 100, 100, 100, 100, 100, 100, 100, 100, 100], [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]]
        assert (b + a).counts.counts.array.tolist() == [[100, 101, 102, 103, 104, 105, 106, 107, 108, 109], [110, 111, 112, 113, 114, 115, 116, 117, 118, 119]]
        assert a.counts.counts.array.tolist() == [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]]
        assert b.counts.counts.array.tolist() == [[100, 100, 100, 100, 100, 100, 100, 100, 100, 100], [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]]

    def test_add_same_different(self):
        a = Histogram([Axis(IntegerBinning(1, 2)), Axis(IntegerBinning(9, 20))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(2*12))))
        b = Histogram([Axis(IntegerBinning(1, 2)), Axis(IntegerBinning(10, 19))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.full(20, 100))))
        assert a.counts.counts.array.tolist() == [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]]
        assert b.counts.counts.array.tolist() == [[100, 100, 100, 100, 100, 100, 100, 100, 100, 100], [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]]
        assert (a + b).counts.counts.array.tolist() == [[0, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 11], [12, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 23]]
        assert a.counts.counts.array.tolist() == [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]]
        assert b.counts.counts.array.tolist() == [[100, 100, 100, 100, 100, 100, 100, 100, 100, 100], [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]]
        assert (b + a).counts.counts.array.tolist() == [[0, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 11], [12, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 23]]
        assert a.counts.counts.array.tolist() == [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]]
        assert b.counts.counts.array.tolist() == [[100, 100, 100, 100, 100, 100, 100, 100, 100, 100], [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]]

    def test_add_different_same(self):
        a = Histogram([Axis(IntegerBinning(0, 3)), Axis(IntegerBinning(10, 19))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(4*10))))
        b = Histogram([Axis(IntegerBinning(1, 2)), Axis(IntegerBinning(10, 19))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.full(20, 100)))); 
        assert a.counts.counts.array.tolist() == [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 11, 12, 13, 14, 15, 16, 17, 18, 19], [20, 21, 22, 23, 24, 25, 26, 27, 28, 29], [30, 31, 32, 33, 34, 35, 36, 37, 38, 39]]
        assert b.counts.counts.array.tolist() == [[100, 100, 100, 100, 100, 100, 100, 100, 100, 100], [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]]
        assert (a + b).counts.counts.array.tolist() == [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [110, 111, 112, 113, 114, 115, 116, 117, 118, 119], [120, 121, 122, 123, 124, 125, 126, 127, 128, 129], [30, 31, 32, 33, 34, 35, 36, 37, 38, 39]]

    def test_add_different_different(self):
        a = Histogram([Axis(IntegerBinning(0, 3)), Axis(IntegerBinning(9, 20))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(4*12))))
        b = Histogram([Axis(IntegerBinning(1, 2)), Axis(IntegerBinning(10, 19))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.full(20, 100))))
        assert a.counts.counts.array.tolist() == [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23], [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35], [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]]
        assert b.counts.counts.array.tolist() == [[100, 100, 100, 100, 100, 100, 100, 100, 100, 100], [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]]
        assert (a + b).counts.counts.array.tolist() == [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [12, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 23], [24, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 35], [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]]
        assert a.counts.counts.array.tolist() == [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23], [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35], [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]]
        assert b.counts.counts.array.tolist() == [[100, 100, 100, 100, 100, 100, 100, 100, 100, 100], [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]]
        assert (b + a).counts.counts.array.tolist() == [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [12, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 23], [24, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 35], [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]]
        assert a.counts.counts.array.tolist() == [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23], [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35], [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]]
        assert b.counts.counts.array.tolist() == [[100, 100, 100, 100, 100, 100, 100, 100, 100, 100], [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]]

    def test_add_unweighted_unweighted(self):
        a = Histogram([Axis(IntegerBinning(10, 19))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(10))))
        b = Histogram([Axis(IntegerBinning(10, 19))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.full(10, 100))))
        assert a.counts.counts.array.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        assert b.counts.counts.array.tolist() == [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
        assert (a + b).counts.counts.array.tolist() == [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
        assert a.counts.counts.array.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        assert b.counts.counts.array.tolist() == [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
        assert (b + a).counts.counts.array.tolist() == [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
        assert a.counts.counts.array.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        assert b.counts.counts.array.tolist() == [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]

    def test_add_sumw_unweighted(self):
        a = Histogram([Axis(IntegerBinning(10, 19))], WeightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(10))))
        b = Histogram([Axis(IntegerBinning(10, 19))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.full(10, 100))))
        assert a.counts.sumw.array.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        assert b.counts.counts.array.tolist() == [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
        ab = a + b
        assert isinstance(ab.counts, WeightedCounts)
        assert ab.counts.sumw.array.tolist() == [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
        assert ab.counts.sumw2 is None
        assert ab.counts.unweighted is None
        assert a.counts.sumw.array.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        assert b.counts.counts.array.tolist() == [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
        ab = b + a
        assert isinstance(ab.counts, WeightedCounts)
        assert ab.counts.sumw.array.tolist() == [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
        assert ab.counts.sumw2 is None
        assert ab.counts.unweighted is None
        assert a.counts.sumw.array.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        assert b.counts.counts.array.tolist() == [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]

    def test_add_sumwsumw2_unweighted(self):
        a = Histogram([Axis(IntegerBinning(10, 19))], WeightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(10)), InterpretedInlineBuffer.fromarray(numpy.arange(10, 20))))
        b = Histogram([Axis(IntegerBinning(10, 19))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.full(10, 100))))
        assert a.counts.sumw.array.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        assert a.counts.sumw2.array.tolist() == [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        assert b.counts.counts.array.tolist() == [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
        ab = a + b
        assert isinstance(ab.counts, WeightedCounts)
        assert ab.counts.sumw.array.tolist() == [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
        assert ab.counts.sumw2 is None
        assert ab.counts.unweighted is None
        assert a.counts.sumw.array.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        assert a.counts.sumw2.array.tolist() == [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        assert b.counts.counts.array.tolist() == [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
        ab = b + a
        assert isinstance(ab.counts, WeightedCounts)
        assert ab.counts.sumw.array.tolist() == [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
        assert ab.counts.sumw2 is None
        assert ab.counts.unweighted is None
        assert a.counts.sumw.array.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        assert a.counts.sumw2.array.tolist() == [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        assert b.counts.counts.array.tolist() == [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]

    def test_add_sumwsumw2counts_unweighted(self):
        a = Histogram([Axis(IntegerBinning(10, 19))], WeightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(10)), InterpretedInlineBuffer.fromarray(numpy.arange(10, 20)), UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(20, 30)))))
        b = Histogram([Axis(IntegerBinning(10, 19))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.full(10, 100))))
        assert a.counts.sumw.array.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        assert a.counts.sumw2.array.tolist() == [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        assert a.counts.unweighted.counts.array.tolist() == [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
        assert b.counts.counts.array.tolist() == [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
        ab = a + b
        assert isinstance(ab.counts, WeightedCounts)
        assert ab.counts.sumw.array.tolist() == [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
        assert ab.counts.sumw2 is None
        assert ab.counts.unweighted.counts.array.tolist() == [120, 121, 122, 123, 124, 125, 126, 127, 128, 129]
        assert a.counts.sumw.array.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        assert a.counts.sumw2.array.tolist() == [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        assert a.counts.unweighted.counts.array.tolist() == [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
        assert b.counts.counts.array.tolist() == [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
        ab = b + a
        assert isinstance(ab.counts, WeightedCounts)
        assert ab.counts.sumw.array.tolist() == [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
        assert ab.counts.sumw2 is None
        assert ab.counts.unweighted.counts.array.tolist() == [120, 121, 122, 123, 124, 125, 126, 127, 128, 129]
        assert a.counts.sumw.array.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        assert a.counts.sumw2.array.tolist() == [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        assert a.counts.unweighted.counts.array.tolist() == [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
        assert b.counts.counts.array.tolist() == [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]

    def test_add_sumwsumw2counts_sumw(self):
        a = Histogram([Axis(IntegerBinning(10, 19))], WeightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(10)), InterpretedInlineBuffer.fromarray(numpy.arange(10, 20)), UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(20, 30)))))
        b = Histogram([Axis(IntegerBinning(10, 19))], WeightedCounts(InterpretedInlineBuffer.fromarray(numpy.full(10, 100))))
        assert a.counts.sumw.array.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        assert a.counts.sumw2.array.tolist() == [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        assert a.counts.unweighted.counts.array.tolist() == [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
        assert b.counts.sumw.array.tolist() == [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
        assert b.counts.sumw2 is None
        assert b.counts.unweighted is None
        ab = a + b
        assert isinstance(ab.counts, WeightedCounts)
        assert ab.counts.sumw.array.tolist() == [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
        assert ab.counts.sumw2 is None
        assert ab.counts.unweighted is None
        assert a.counts.sumw.array.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        assert a.counts.sumw2.array.tolist() == [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        assert a.counts.unweighted.counts.array.tolist() == [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
        assert b.counts.sumw.array.tolist() == [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
        assert b.counts.sumw2 is None
        assert b.counts.unweighted is None
        ab = b + a
        assert isinstance(ab.counts, WeightedCounts)
        assert ab.counts.sumw.array.tolist() == [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
        assert ab.counts.sumw2 is None
        assert ab.counts.unweighted is None
        assert a.counts.sumw.array.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        assert a.counts.sumw2.array.tolist() == [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        assert a.counts.unweighted.counts.array.tolist() == [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
        assert b.counts.sumw.array.tolist() == [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
        assert b.counts.sumw2 is None
        assert b.counts.unweighted is None

    def test_add_sumwsumw2counts_sumwsumw2(self):
        a = Histogram([Axis(IntegerBinning(10, 19))], WeightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(10)), InterpretedInlineBuffer.fromarray(numpy.arange(10, 20)), UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(20, 30)))))
        b = Histogram([Axis(IntegerBinning(10, 19))], WeightedCounts(InterpretedInlineBuffer.fromarray(numpy.full(10, 100)), InterpretedInlineBuffer.fromarray(numpy.full(10, 200))))
        assert a.counts.sumw.array.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        assert a.counts.sumw2.array.tolist() == [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        assert a.counts.unweighted.counts.array.tolist() == [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
        assert b.counts.sumw.array.tolist() == [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
        assert b.counts.sumw2.array.tolist() == [200, 200, 200, 200, 200, 200, 200, 200, 200, 200]
        assert b.counts.unweighted is None
        ab = a + b
        assert isinstance(ab.counts, WeightedCounts)
        assert ab.counts.sumw.array.tolist() == [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
        assert ab.counts.sumw2.array.tolist() == [210, 211, 212, 213, 214, 215, 216, 217, 218, 219]
        assert ab.counts.unweighted is None
        assert a.counts.sumw.array.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        assert a.counts.sumw2.array.tolist() == [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        assert a.counts.unweighted.counts.array.tolist() == [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
        assert b.counts.sumw.array.tolist() == [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
        assert b.counts.sumw2.array.tolist() == [200, 200, 200, 200, 200, 200, 200, 200, 200, 200]
        assert b.counts.unweighted is None
        ab = b + a
        assert isinstance(ab.counts, WeightedCounts)
        assert ab.counts.sumw.array.tolist() == [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
        assert ab.counts.sumw2.array.tolist() == [210, 211, 212, 213, 214, 215, 216, 217, 218, 219]
        assert ab.counts.unweighted is None
        assert a.counts.sumw.array.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        assert a.counts.sumw2.array.tolist() == [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        assert a.counts.unweighted.counts.array.tolist() == [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
        assert b.counts.sumw.array.tolist() == [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
        assert b.counts.sumw2.array.tolist() == [200, 200, 200, 200, 200, 200, 200, 200, 200, 200]
        assert b.counts.unweighted is None

    def test_add_sumwsumw2counts_sumwsumw2counts(self):
        a = Histogram([Axis(IntegerBinning(10, 19))], WeightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(10)), InterpretedInlineBuffer.fromarray(numpy.arange(10, 20)), UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(20, 30)))))
        b = Histogram([Axis(IntegerBinning(10, 19))], WeightedCounts(InterpretedInlineBuffer.fromarray(numpy.full(10, 100)), InterpretedInlineBuffer.fromarray(numpy.full(10, 200)), UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.full(10, 300)))))
        assert a.counts.sumw.array.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        assert a.counts.sumw2.array.tolist() == [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        assert a.counts.unweighted.counts.array.tolist() == [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
        assert b.counts.sumw.array.tolist() == [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
        assert b.counts.sumw2.array.tolist() == [200, 200, 200, 200, 200, 200, 200, 200, 200, 200]
        assert b.counts.unweighted.counts.array.tolist() == [300, 300, 300, 300, 300, 300, 300, 300, 300, 300]
        ab = a + b
        assert isinstance(ab.counts, WeightedCounts)
        assert ab.counts.sumw.array.tolist() == [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
        assert ab.counts.sumw2.array.tolist() == [210, 211, 212, 213, 214, 215, 216, 217, 218, 219]
        assert ab.counts.unweighted.counts.array.tolist() == [320, 321, 322, 323, 324, 325, 326, 327, 328, 329]
        assert a.counts.sumw.array.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        assert a.counts.sumw2.array.tolist() == [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        assert a.counts.unweighted.counts.array.tolist() == [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
        assert b.counts.sumw.array.tolist() == [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
        assert b.counts.sumw2.array.tolist() == [200, 200, 200, 200, 200, 200, 200, 200, 200, 200]
        assert b.counts.unweighted.counts.array.tolist() == [300, 300, 300, 300, 300, 300, 300, 300, 300, 300]
        ab = b + a
        assert isinstance(ab.counts, WeightedCounts)
        assert ab.counts.sumw.array.tolist() == [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
        assert ab.counts.sumw2.array.tolist() == [210, 211, 212, 213, 214, 215, 216, 217, 218, 219]
        assert ab.counts.unweighted.counts.array.tolist() == [320, 321, 322, 323, 324, 325, 326, 327, 328, 329]
        assert a.counts.sumw.array.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        assert a.counts.sumw2.array.tolist() == [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        assert a.counts.unweighted.counts.array.tolist() == [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
        assert b.counts.sumw.array.tolist() == [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
        assert b.counts.sumw2.array.tolist() == [200, 200, 200, 200, 200, 200, 200, 200, 200, 200]
        assert b.counts.unweighted.counts.array.tolist() == [300, 300, 300, 300, 300, 300, 300, 300, 300, 300]

    def test_add_IntegerBinning_perfect(self):
        a = Histogram([Axis(IntegerBinning(-5, 5))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(11))))
        b = Histogram([Axis(IntegerBinning(-5, 5))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.full(11, 100))))
        assert a.axis[0].binning.toCategoryBinning().categories == ["-5", "-4", "-3", "-2", "-1", "0", "1", "2", "3", "4", "5"]
        assert a.counts.counts.array.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        assert b.axis[0].binning.toCategoryBinning().categories == ["-5", "-4", "-3", "-2", "-1", "0", "1", "2", "3", "4", "5"]
        assert b.counts.counts.array.tolist() == [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
        ab = a + b
        assert ab.axis[0].binning.toCategoryBinning().categories == ["-5", "-4", "-3", "-2", "-1", "0", "1", "2", "3", "4", "5"]
        assert ab.counts.counts.array.tolist() == [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
        assert a.axis[0].binning.toCategoryBinning().categories == ["-5", "-4", "-3", "-2", "-1", "0", "1", "2", "3", "4", "5"]
        assert a.counts.counts.array.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        assert b.axis[0].binning.toCategoryBinning().categories == ["-5", "-4", "-3", "-2", "-1", "0", "1", "2", "3", "4", "5"]
        assert b.counts.counts.array.tolist() == [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
        ab = b + a
        assert ab.axis[0].binning.toCategoryBinning().categories == ["-5", "-4", "-3", "-2", "-1", "0", "1", "2", "3", "4", "5"]
        assert ab.counts.counts.array.tolist() == [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
        assert a.axis[0].binning.toCategoryBinning().categories == ["-5", "-4", "-3", "-2", "-1", "0", "1", "2", "3", "4", "5"]
        assert a.counts.counts.array.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        assert b.axis[0].binning.toCategoryBinning().categories == ["-5", "-4", "-3", "-2", "-1", "0", "1", "2", "3", "4", "5"]
        assert b.counts.counts.array.tolist() == [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]

    def test_add_IntegerBinning_offset(self):
        a = Histogram([Axis(IntegerBinning(-5, 5))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(11))))
        b = Histogram([Axis(IntegerBinning(-2, 8))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.full(11, 100))))
        assert a.axis[0].binning.toCategoryBinning().categories == ["-5", "-4", "-3", "-2", "-1", "0", "1", "2", "3", "4", "5"]
        assert a.counts.counts.array.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        assert b.axis[0].binning.toCategoryBinning().categories == ["-2", "-1", "0", "1", "2", "3", "4", "5", "6", "7", "8"]
        assert b.counts.counts.array.tolist() == [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
        ab = a + b
        assert ab.axis[0].binning.toCategoryBinning().categories == ["-5", "-4", "-3", "-2", "-1", "0", "1", "2", "3", "4", "5", "6", "7", "8"]
        assert ab.counts.counts.array.tolist() == [0, 1, 2, 103, 104, 105, 106, 107, 108, 109, 110, 100, 100, 100]
        assert a.axis[0].binning.toCategoryBinning().categories == ["-5", "-4", "-3", "-2", "-1", "0", "1", "2", "3", "4", "5"]
        assert a.counts.counts.array.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        assert b.axis[0].binning.toCategoryBinning().categories == ["-2", "-1", "0", "1", "2", "3", "4", "5", "6", "7", "8"]
        assert b.counts.counts.array.tolist() == [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
        ab = b + a
        assert ab.axis[0].binning.toCategoryBinning().categories == ["-5", "-4", "-3", "-2", "-1", "0", "1", "2", "3", "4", "5", "6", "7", "8"]
        assert ab.counts.counts.array.tolist() == [0, 1, 2, 103, 104, 105, 106, 107, 108, 109, 110, 100, 100, 100]
        assert a.axis[0].binning.toCategoryBinning().categories == ["-5", "-4", "-3", "-2", "-1", "0", "1", "2", "3", "4", "5"]
        assert a.counts.counts.array.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        assert b.axis[0].binning.toCategoryBinning().categories == ["-2", "-1", "0", "1", "2", "3", "4", "5", "6", "7", "8"]
        assert b.counts.counts.array.tolist() == [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]

    def test_add_IntegerBinning_disjoint(self):
        a = Histogram([Axis(IntegerBinning(-5, 5))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(11))))
        b = Histogram([Axis(IntegerBinning(8, 18))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.full(11, 100))))
        assert a.axis[0].binning.toCategoryBinning().categories == ["-5", "-4", "-3", "-2", "-1", "0", "1", "2", "3", "4", "5"]
        assert a.counts.counts.array.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        assert b.axis[0].binning.toCategoryBinning().categories == ["8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18"]
        assert b.counts.counts.array.tolist() == [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
        ab = a + b
        assert ab.axis[0].binning.toCategoryBinning().categories == ["-5", "-4", "-3", "-2", "-1", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18"]
        assert ab.counts.counts.array.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 0, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
        assert a.axis[0].binning.toCategoryBinning().categories == ["-5", "-4", "-3", "-2", "-1", "0", "1", "2", "3", "4", "5"]
        assert a.counts.counts.array.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        assert b.axis[0].binning.toCategoryBinning().categories == ["8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18"]
        assert b.counts.counts.array.tolist() == [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
        ab = b + a
        assert ab.axis[0].binning.toCategoryBinning().categories == ["-5", "-4", "-3", "-2", "-1", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18"]
        assert ab.counts.counts.array.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 0, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
        assert a.axis[0].binning.toCategoryBinning().categories == ["-5", "-4", "-3", "-2", "-1", "0", "1", "2", "3", "4", "5"]
        assert a.counts.counts.array.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        assert b.axis[0].binning.toCategoryBinning().categories == ["8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18"]
        assert b.counts.counts.array.tolist() == [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]

    def test_add_RegularBinning_same(self):
        a = Histogram([Axis(RegularBinning(10, RealInterval(-5, 5)))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(10))))
        b = Histogram([Axis(RegularBinning(10, RealInterval(-5, 5)))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(100, 110))))
        assert a.axis[0].binning.toCategoryBinning().categories == ["[-5, -4)", "[-4, -3)", "[-3, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5)"]
        assert a.counts.counts.array.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        assert b.axis[0].binning.toCategoryBinning().categories == ["[-5, -4)", "[-4, -3)", "[-3, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5)"]
        assert b.counts.counts.array.tolist() == [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
        ab = a + b
        assert ab.axis[0].binning.toCategoryBinning().categories == ["[-5, -4)", "[-4, -3)", "[-3, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5)"]
        assert ab.counts.counts.array.tolist() == [100, 102, 104, 106, 108, 110, 112, 114, 116, 118]
        assert a.axis[0].binning.toCategoryBinning().categories == ["[-5, -4)", "[-4, -3)", "[-3, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5)"]
        assert a.counts.counts.array.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        assert b.axis[0].binning.toCategoryBinning().categories == ["[-5, -4)", "[-4, -3)", "[-3, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5)"]
        assert b.counts.counts.array.tolist() == [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
        ab = b + a
        assert ab.axis[0].binning.toCategoryBinning().categories == ["[-5, -4)", "[-4, -3)", "[-3, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5)"]
        assert ab.counts.counts.array.tolist() == [100, 102, 104, 106, 108, 110, 112, 114, 116, 118]
        assert a.axis[0].binning.toCategoryBinning().categories == ["[-5, -4)", "[-4, -3)", "[-3, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5)"]
        assert a.counts.counts.array.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        assert b.axis[0].binning.toCategoryBinning().categories == ["[-5, -4)", "[-4, -3)", "[-3, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5)"]
        assert b.counts.counts.array.tolist() == [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]

        a = Histogram([Axis(RegularBinning(10, RealInterval(-5, 5), overflow=RealOverflow(loc_underflow=RealOverflow.below1, loc_overflow=RealOverflow.above1, loc_nanflow=RealOverflow.above2)))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(13))))
        b = Histogram([Axis(RegularBinning(10, RealInterval(-5, 5), overflow=RealOverflow(loc_underflow=RealOverflow.below1, loc_overflow=RealOverflow.above1, loc_nanflow=RealOverflow.above2)))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(100, 113))))
        assert a.axis[0].binning.toCategoryBinning().categories == ["[-inf, -5)", "[-5, -4)", "[-4, -3)", "[-3, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5)", "[5, +inf]", "{nan}"]
        assert a.counts.counts.array.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        assert b.axis[0].binning.toCategoryBinning().categories == ["[-inf, -5)", "[-5, -4)", "[-4, -3)", "[-3, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5)", "[5, +inf]", "{nan}"]
        assert b.counts.counts.array.tolist() == [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112]
        ab = a + b
        assert ab.axis[0].binning.toCategoryBinning().categories == ["[-inf, -5)", "[-5, -4)", "[-4, -3)", "[-3, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5)", "[5, +inf]", "{nan}"]
        assert ab.counts.counts.array.tolist() == [100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124]
        assert a.axis[0].binning.toCategoryBinning().categories == ["[-inf, -5)", "[-5, -4)", "[-4, -3)", "[-3, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5)", "[5, +inf]", "{nan}"]
        assert a.counts.counts.array.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        assert b.axis[0].binning.toCategoryBinning().categories == ["[-inf, -5)", "[-5, -4)", "[-4, -3)", "[-3, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5)", "[5, +inf]", "{nan}"]
        assert b.counts.counts.array.tolist() == [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112]
        ab = b + a
        assert ab.axis[0].binning.toCategoryBinning().categories == ["[-inf, -5)", "[-5, -4)", "[-4, -3)", "[-3, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5)", "[5, +inf]", "{nan}"]
        assert ab.counts.counts.array.tolist() == [100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124]
        assert a.axis[0].binning.toCategoryBinning().categories == ["[-inf, -5)", "[-5, -4)", "[-4, -3)", "[-3, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5)", "[5, +inf]", "{nan}"]
        assert a.counts.counts.array.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        assert b.axis[0].binning.toCategoryBinning().categories == ["[-inf, -5)", "[-5, -4)", "[-4, -3)", "[-3, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5)", "[5, +inf]", "{nan}"]
        assert b.counts.counts.array.tolist() == [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112]

    def test_add_RegularBinning_different(self):
        a = Histogram([Axis(RegularBinning(10, RealInterval(-5, 5), overflow=RealOverflow(loc_underflow=RealOverflow.above1, loc_overflow=RealOverflow.above2, loc_nanflow=RealOverflow.above3)))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(13))))
        b = Histogram([Axis(RegularBinning(10, RealInterval(-5, 5)))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(100, 110))))
        assert a.axis[0].binning.toCategoryBinning().categories == ["[-5, -4)", "[-4, -3)", "[-3, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5)", "[-inf, -5)", "[5, +inf]", "{nan}"]
        assert a.counts.counts.array.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        assert b.axis[0].binning.toCategoryBinning().categories == ["[-5, -4)", "[-4, -3)", "[-3, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5)"]
        assert b.counts.counts.array.tolist() == [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
        ab = a + b
        assert ab.axis[0].binning.toCategoryBinning().categories == ["[-5, -4)", "[-4, -3)", "[-3, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5)", "[-inf, -5)", "[5, +inf]", "{nan}"]
        assert ab.counts.counts.array.tolist() == [100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 10, 11, 12]
        assert a.axis[0].binning.toCategoryBinning().categories == ["[-5, -4)", "[-4, -3)", "[-3, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5)", "[-inf, -5)", "[5, +inf]", "{nan}"]
        assert a.counts.counts.array.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        assert b.axis[0].binning.toCategoryBinning().categories == ["[-5, -4)", "[-4, -3)", "[-3, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5)"]
        assert b.counts.counts.array.tolist() == [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
        ab = b + a
        assert ab.axis[0].binning.toCategoryBinning().categories == ["[-5, -4)", "[-4, -3)", "[-3, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5)", "[-inf, -5)", "[5, +inf]", "{nan}"]
        assert ab.counts.counts.array.tolist() == [100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 10, 11, 12]
        assert a.axis[0].binning.toCategoryBinning().categories == ["[-5, -4)", "[-4, -3)", "[-3, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5)", "[-inf, -5)", "[5, +inf]", "{nan}"]
        assert a.counts.counts.array.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        assert b.axis[0].binning.toCategoryBinning().categories == ["[-5, -4)", "[-4, -3)", "[-3, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5)"]
        assert b.counts.counts.array.tolist() == [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]

        a = Histogram([Axis(RegularBinning(10, RealInterval(-5, 5), overflow=RealOverflow(loc_underflow=RealOverflow.below1, loc_overflow=RealOverflow.above1, loc_nanflow=RealOverflow.above2)))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(13))))
        b = Histogram([Axis(RegularBinning(10, RealInterval(-5, 5)))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(100, 110))))
        assert a.axis[0].binning.toCategoryBinning().categories == ["[-inf, -5)", "[-5, -4)", "[-4, -3)", "[-3, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5)", "[5, +inf]", "{nan}"]
        assert a.counts.counts.array.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        assert b.axis[0].binning.toCategoryBinning().categories == ["[-5, -4)", "[-4, -3)", "[-3, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5)"]
        assert b.counts.counts.array.tolist() == [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
        ab = a + b
        assert ab.axis[0].binning.toCategoryBinning().categories == ["[-5, -4)", "[-4, -3)", "[-3, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5)", "[-inf, -5)", "[5, +inf]", "{nan}"]
        assert ab.counts.counts.array.tolist() == [101, 103, 105, 107, 109, 111, 113, 115, 117, 119, 0, 11, 12]
        assert a.axis[0].binning.toCategoryBinning().categories == ["[-inf, -5)", "[-5, -4)", "[-4, -3)", "[-3, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5)", "[5, +inf]", "{nan}"]
        assert a.counts.counts.array.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        assert b.axis[0].binning.toCategoryBinning().categories == ["[-5, -4)", "[-4, -3)", "[-3, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5)"]
        assert b.counts.counts.array.tolist() == [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
        ab = b + a
        assert ab.axis[0].binning.toCategoryBinning().categories == ["[-5, -4)", "[-4, -3)", "[-3, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5)", "[-inf, -5)", "[5, +inf]", "{nan}"]
        assert ab.counts.counts.array.tolist() == [101, 103, 105, 107, 109, 111, 113, 115, 117, 119, 0, 11, 12]
        assert a.axis[0].binning.toCategoryBinning().categories == ["[-inf, -5)", "[-5, -4)", "[-4, -3)", "[-3, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5)", "[5, +inf]", "{nan}"]
        assert a.counts.counts.array.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        assert b.axis[0].binning.toCategoryBinning().categories == ["[-5, -4)", "[-4, -3)", "[-3, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5)"]
        assert b.counts.counts.array.tolist() == [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]

        a = Histogram([Axis(RegularBinning(10, RealInterval(-5, 5), overflow=RealOverflow(loc_underflow=RealOverflow.below1, loc_overflow=RealOverflow.above1, loc_nanflow=RealOverflow.above2)))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(13))))
        b = Histogram([Axis(RegularBinning(10, RealInterval(-5, 5), overflow=RealOverflow(loc_underflow=RealOverflow.above1, loc_overflow=RealOverflow.below1, loc_nanflow=RealOverflow.above2)))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(100, 113))))
        assert a.axis[0].binning.toCategoryBinning().categories == ["[-inf, -5)", "[-5, -4)", "[-4, -3)", "[-3, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5)", "[5, +inf]", "{nan}"]
        assert a.counts.counts.array.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        assert b.axis[0].binning.toCategoryBinning().categories == ["[5, +inf]", "[-5, -4)", "[-4, -3)", "[-3, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5)", "[-inf, -5)", "{nan}"]
        assert b.counts.counts.array.tolist() == [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112]
        ab = a + b
        assert ab.axis[0].binning.toCategoryBinning().categories == ["[-5, -4)", "[-4, -3)", "[-3, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5)", "[-inf, -5)", "[5, +inf]", "{nan}"]
        assert ab.counts.counts.array.tolist() == [102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 111, 111, 124]
        assert a.axis[0].binning.toCategoryBinning().categories == ["[-inf, -5)", "[-5, -4)", "[-4, -3)", "[-3, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5)", "[5, +inf]", "{nan}"]
        assert a.counts.counts.array.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        assert b.axis[0].binning.toCategoryBinning().categories == ["[5, +inf]", "[-5, -4)", "[-4, -3)", "[-3, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5)", "[-inf, -5)", "{nan}"]
        assert b.counts.counts.array.tolist() == [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112]
        ab = b + a
        assert ab.axis[0].binning.toCategoryBinning().categories == ["[-5, -4)", "[-4, -3)", "[-3, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5)", "[-inf, -5)", "[5, +inf]", "{nan}"]
        assert ab.counts.counts.array.tolist() == [102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 111, 111, 124]
        assert a.axis[0].binning.toCategoryBinning().categories == ["[-inf, -5)", "[-5, -4)", "[-4, -3)", "[-3, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5)", "[5, +inf]", "{nan}"]
        assert a.counts.counts.array.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        assert b.axis[0].binning.toCategoryBinning().categories == ["[5, +inf]", "[-5, -4)", "[-4, -3)", "[-3, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5)", "[-inf, -5)", "{nan}"]
        assert b.counts.counts.array.tolist() == [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112]

    def test_add_EdgesBinning_same(self):
        a = Histogram([Axis(RegularBinning(10, RealInterval(-5, 5)).toEdgesBinning())], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(10))))
        b = Histogram([Axis(RegularBinning(10, RealInterval(-5, 5)).toEdgesBinning())], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(100, 110))))
        assert a.axis[0].binning.toCategoryBinning().categories == ["[-5, -4)", "[-4, -3)", "[-3, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5)"]
        assert a.counts.counts.array.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        assert b.axis[0].binning.toCategoryBinning().categories == ["[-5, -4)", "[-4, -3)", "[-3, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5)"]
        assert b.counts.counts.array.tolist() == [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
        ab = a + b
        assert ab.axis[0].binning.toCategoryBinning().categories == ["[-5, -4)", "[-4, -3)", "[-3, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5)"]
        assert ab.counts.counts.array.tolist() == [100, 102, 104, 106, 108, 110, 112, 114, 116, 118]
        assert a.axis[0].binning.toCategoryBinning().categories == ["[-5, -4)", "[-4, -3)", "[-3, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5)"]
        assert a.counts.counts.array.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        assert b.axis[0].binning.toCategoryBinning().categories == ["[-5, -4)", "[-4, -3)", "[-3, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5)"]
        assert b.counts.counts.array.tolist() == [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
        ab = b + a
        assert ab.axis[0].binning.toCategoryBinning().categories == ["[-5, -4)", "[-4, -3)", "[-3, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5)"]
        assert ab.counts.counts.array.tolist() == [100, 102, 104, 106, 108, 110, 112, 114, 116, 118]
        assert a.axis[0].binning.toCategoryBinning().categories == ["[-5, -4)", "[-4, -3)", "[-3, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5)"]
        assert a.counts.counts.array.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        assert b.axis[0].binning.toCategoryBinning().categories == ["[-5, -4)", "[-4, -3)", "[-3, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5)"]
        assert b.counts.counts.array.tolist() == [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]

        a = Histogram([Axis(RegularBinning(10, RealInterval(-5, 5), overflow=RealOverflow(loc_underflow=RealOverflow.below1, loc_overflow=RealOverflow.above1, loc_nanflow=RealOverflow.above2)).toEdgesBinning())], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(13))))
        b = Histogram([Axis(RegularBinning(10, RealInterval(-5, 5), overflow=RealOverflow(loc_underflow=RealOverflow.below1, loc_overflow=RealOverflow.above1, loc_nanflow=RealOverflow.above2)).toEdgesBinning())], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(100, 113))))
        assert a.axis[0].binning.toCategoryBinning().categories == ["[-inf, -5)", "[-5, -4)", "[-4, -3)", "[-3, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5)", "[5, +inf]", "{nan}"]
        assert a.counts.counts.array.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        assert b.axis[0].binning.toCategoryBinning().categories == ["[-inf, -5)", "[-5, -4)", "[-4, -3)", "[-3, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5)", "[5, +inf]", "{nan}"]
        assert b.counts.counts.array.tolist() == [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112]
        ab = a + b
        assert ab.axis[0].binning.toCategoryBinning().categories == ["[-inf, -5)", "[-5, -4)", "[-4, -3)", "[-3, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5)", "[5, +inf]", "{nan}"]
        assert ab.counts.counts.array.tolist() == [100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124]
        assert a.axis[0].binning.toCategoryBinning().categories == ["[-inf, -5)", "[-5, -4)", "[-4, -3)", "[-3, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5)", "[5, +inf]", "{nan}"]
        assert a.counts.counts.array.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        assert b.axis[0].binning.toCategoryBinning().categories == ["[-inf, -5)", "[-5, -4)", "[-4, -3)", "[-3, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5)", "[5, +inf]", "{nan}"]
        assert b.counts.counts.array.tolist() == [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112]
        ab = b + a
        assert ab.axis[0].binning.toCategoryBinning().categories == ["[-inf, -5)", "[-5, -4)", "[-4, -3)", "[-3, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5)", "[5, +inf]", "{nan}"]
        assert ab.counts.counts.array.tolist() == [100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124]
        assert a.axis[0].binning.toCategoryBinning().categories == ["[-inf, -5)", "[-5, -4)", "[-4, -3)", "[-3, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5)", "[5, +inf]", "{nan}"]
        assert a.counts.counts.array.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        assert b.axis[0].binning.toCategoryBinning().categories == ["[-inf, -5)", "[-5, -4)", "[-4, -3)", "[-3, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5)", "[5, +inf]", "{nan}"]
        assert b.counts.counts.array.tolist() == [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112]

    def test_add_EdgesBinning_different(self):
        a = Histogram([Axis(RegularBinning(10, RealInterval(-5, 5), overflow=RealOverflow(loc_underflow=RealOverflow.above1, loc_overflow=RealOverflow.above2, loc_nanflow=RealOverflow.above3)).toEdgesBinning())], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(13))))
        b = Histogram([Axis(RegularBinning(10, RealInterval(-5, 5)).toEdgesBinning())], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(100, 110))))
        assert a.axis[0].binning.toCategoryBinning().categories == ["[-5, -4)", "[-4, -3)", "[-3, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5)", "[-inf, -5)", "[5, +inf]", "{nan}"]
        assert a.counts.counts.array.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        assert b.axis[0].binning.toCategoryBinning().categories == ["[-5, -4)", "[-4, -3)", "[-3, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5)"]
        assert b.counts.counts.array.tolist() == [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
        ab = a + b
        assert ab.axis[0].binning.toCategoryBinning().categories == ["[-5, -4)", "[-4, -3)", "[-3, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5)", "[-inf, -5)", "[5, +inf]", "{nan}"]
        assert ab.counts.counts.array.tolist() == [100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 10, 11, 12]
        assert a.axis[0].binning.toCategoryBinning().categories == ["[-5, -4)", "[-4, -3)", "[-3, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5)", "[-inf, -5)", "[5, +inf]", "{nan}"]
        assert a.counts.counts.array.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        assert b.axis[0].binning.toCategoryBinning().categories == ["[-5, -4)", "[-4, -3)", "[-3, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5)"]
        assert b.counts.counts.array.tolist() == [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
        ab = b + a
        assert ab.axis[0].binning.toCategoryBinning().categories == ["[-5, -4)", "[-4, -3)", "[-3, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5)", "[-inf, -5)", "[5, +inf]", "{nan}"]
        assert ab.counts.counts.array.tolist() == [100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 10, 11, 12]
        assert a.axis[0].binning.toCategoryBinning().categories == ["[-5, -4)", "[-4, -3)", "[-3, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5)", "[-inf, -5)", "[5, +inf]", "{nan}"]
        assert a.counts.counts.array.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        assert b.axis[0].binning.toCategoryBinning().categories == ["[-5, -4)", "[-4, -3)", "[-3, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5)"]
        assert b.counts.counts.array.tolist() == [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]

        a = Histogram([Axis(RegularBinning(10, RealInterval(-5, 5), overflow=RealOverflow(loc_underflow=RealOverflow.below1, loc_overflow=RealOverflow.above1, loc_nanflow=RealOverflow.above2)).toEdgesBinning())], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(13))))
        b = Histogram([Axis(RegularBinning(10, RealInterval(-5, 5)).toEdgesBinning())], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(100, 110))))
        assert a.axis[0].binning.toCategoryBinning().categories == ["[-inf, -5)", "[-5, -4)", "[-4, -3)", "[-3, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5)", "[5, +inf]", "{nan}"]
        assert a.counts.counts.array.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        assert b.axis[0].binning.toCategoryBinning().categories == ["[-5, -4)", "[-4, -3)", "[-3, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5)"]
        assert b.counts.counts.array.tolist() == [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
        ab = a + b
        assert ab.axis[0].binning.toCategoryBinning().categories == ["[-5, -4)", "[-4, -3)", "[-3, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5)", "[-inf, -5)", "[5, +inf]", "{nan}"]
        assert ab.counts.counts.array.tolist() == [101, 103, 105, 107, 109, 111, 113, 115, 117, 119, 0, 11, 12]
        assert a.axis[0].binning.toCategoryBinning().categories == ["[-inf, -5)", "[-5, -4)", "[-4, -3)", "[-3, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5)", "[5, +inf]", "{nan}"]
        assert a.counts.counts.array.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        assert b.axis[0].binning.toCategoryBinning().categories == ["[-5, -4)", "[-4, -3)", "[-3, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5)"]
        assert b.counts.counts.array.tolist() == [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
        ab = b + a
        assert ab.axis[0].binning.toCategoryBinning().categories == ["[-5, -4)", "[-4, -3)", "[-3, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5)", "[-inf, -5)", "[5, +inf]", "{nan}"]
        assert ab.counts.counts.array.tolist() == [101, 103, 105, 107, 109, 111, 113, 115, 117, 119, 0, 11, 12]
        assert a.axis[0].binning.toCategoryBinning().categories == ["[-inf, -5)", "[-5, -4)", "[-4, -3)", "[-3, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5)", "[5, +inf]", "{nan}"]
        assert a.counts.counts.array.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        assert b.axis[0].binning.toCategoryBinning().categories == ["[-5, -4)", "[-4, -3)", "[-3, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5)"]
        assert b.counts.counts.array.tolist() == [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]

        a = Histogram([Axis(RegularBinning(10, RealInterval(-5, 5), overflow=RealOverflow(loc_underflow=RealOverflow.below1, loc_overflow=RealOverflow.above1, loc_nanflow=RealOverflow.above2)).toEdgesBinning())], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(13))))
        b = Histogram([Axis(RegularBinning(10, RealInterval(-5, 5), overflow=RealOverflow(loc_underflow=RealOverflow.above1, loc_overflow=RealOverflow.below1, loc_nanflow=RealOverflow.above2)).toEdgesBinning())], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(100, 113))))
        assert a.axis[0].binning.toCategoryBinning().categories == ["[-inf, -5)", "[-5, -4)", "[-4, -3)", "[-3, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5)", "[5, +inf]", "{nan}"]
        assert a.counts.counts.array.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        assert b.axis[0].binning.toCategoryBinning().categories == ["[5, +inf]", "[-5, -4)", "[-4, -3)", "[-3, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5)", "[-inf, -5)", "{nan}"]
        assert b.counts.counts.array.tolist() == [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112]
        ab = a + b
        assert ab.axis[0].binning.toCategoryBinning().categories == ["[-5, -4)", "[-4, -3)", "[-3, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5)", "[-inf, -5)", "[5, +inf]", "{nan}"]
        assert ab.counts.counts.array.tolist() == [102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 111, 111, 124]
        assert a.axis[0].binning.toCategoryBinning().categories == ["[-inf, -5)", "[-5, -4)", "[-4, -3)", "[-3, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5)", "[5, +inf]", "{nan}"]
        assert a.counts.counts.array.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        assert b.axis[0].binning.toCategoryBinning().categories == ["[5, +inf]", "[-5, -4)", "[-4, -3)", "[-3, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5)", "[-inf, -5)", "{nan}"]
        assert b.counts.counts.array.tolist() == [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112]
        ab = b + a
        assert ab.axis[0].binning.toCategoryBinning().categories == ["[-5, -4)", "[-4, -3)", "[-3, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5)", "[-inf, -5)", "[5, +inf]", "{nan}"]
        assert ab.counts.counts.array.tolist() == [102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 111, 111, 124]
        assert a.axis[0].binning.toCategoryBinning().categories == ["[-inf, -5)", "[-5, -4)", "[-4, -3)", "[-3, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5)", "[5, +inf]", "{nan}"]
        assert a.counts.counts.array.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        assert b.axis[0].binning.toCategoryBinning().categories == ["[5, +inf]", "[-5, -4)", "[-4, -3)", "[-3, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5)", "[-inf, -5)", "{nan}"]
        assert b.counts.counts.array.tolist() == [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112]

    def test_add_SparseRegularBinning_same(self):
        a = Histogram([Axis(SparseRegularBinning([44, 77, 22, 33], 0.1))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.array([4, 5, 6, 7]))))
        b = Histogram([Axis(SparseRegularBinning([44, 77, 22, 33], 0.1))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.array([10, 20, 30, 40]))))
        assert a.axis[0].binning.toCategoryBinning().categories == ["[4.4, 4.5)", "[7.7, 7.8)", "[2.2, 2.3)", "[3.3, 3.4)"]
        assert a.counts.counts.array.tolist() == [4, 5, 6, 7]
        assert b.axis[0].binning.toCategoryBinning().categories == ["[4.4, 4.5)", "[7.7, 7.8)", "[2.2, 2.3)", "[3.3, 3.4)"]
        assert b.counts.counts.array.tolist() == [10, 20, 30, 40]
        ab = a + b
        assert ab.axis[0].binning.toCategoryBinning().categories == ["[4.4, 4.5)", "[7.7, 7.8)", "[2.2, 2.3)", "[3.3, 3.4)"]
        assert ab.counts.counts.array.tolist() == [14, 25, 36, 47]
        assert a.axis[0].binning.toCategoryBinning().categories == ["[4.4, 4.5)", "[7.7, 7.8)", "[2.2, 2.3)", "[3.3, 3.4)"]
        assert a.counts.counts.array.tolist() == [4, 5, 6, 7]
        assert b.axis[0].binning.toCategoryBinning().categories == ["[4.4, 4.5)", "[7.7, 7.8)", "[2.2, 2.3)", "[3.3, 3.4)"]
        assert b.counts.counts.array.tolist() == [10, 20, 30, 40]
        ab = b + a
        assert ab.axis[0].binning.toCategoryBinning().categories == ["[4.4, 4.5)", "[7.7, 7.8)", "[2.2, 2.3)", "[3.3, 3.4)"]
        assert ab.counts.counts.array.tolist() == [14, 25, 36, 47]
        assert a.axis[0].binning.toCategoryBinning().categories == ["[4.4, 4.5)", "[7.7, 7.8)", "[2.2, 2.3)", "[3.3, 3.4)"]
        assert a.counts.counts.array.tolist() == [4, 5, 6, 7]
        assert b.axis[0].binning.toCategoryBinning().categories == ["[4.4, 4.5)", "[7.7, 7.8)", "[2.2, 2.3)", "[3.3, 3.4)"]
        assert b.counts.counts.array.tolist() == [10, 20, 30, 40]

    def test_add_SparseRegularBinning_same_overflow(self):
        a = Histogram([Axis(SparseRegularBinning([44, 77, 22, 33], 0.1, overflow=RealOverflow(loc_nanflow=RealOverflow.below1)))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.array([999, 4, 5, 6, 7]))))
        b = Histogram([Axis(SparseRegularBinning([44, 77, 22, 33], 0.1))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.array([10, 20, 30, 40]))))
        assert a.axis[0].binning.toCategoryBinning().categories == ["{nan}", "[4.4, 4.5)", "[7.7, 7.8)", "[2.2, 2.3)", "[3.3, 3.4)"]
        assert a.counts.counts.array.tolist() == [999, 4, 5, 6, 7]
        assert b.axis[0].binning.toCategoryBinning().categories == ["[4.4, 4.5)", "[7.7, 7.8)", "[2.2, 2.3)", "[3.3, 3.4)"]
        assert b.counts.counts.array.tolist() == [10, 20, 30, 40]
        ab = a + b
        assert ab.axis[0].binning.toCategoryBinning().categories == ["[4.4, 4.5)", "[7.7, 7.8)", "[2.2, 2.3)", "[3.3, 3.4)", "{nan}"]
        assert ab.counts.counts.array.tolist() == [14, 25, 36, 47, 999]
        assert a.axis[0].binning.toCategoryBinning().categories == ["{nan}", "[4.4, 4.5)", "[7.7, 7.8)", "[2.2, 2.3)", "[3.3, 3.4)"]
        assert a.counts.counts.array.tolist() == [999, 4, 5, 6, 7]
        assert b.axis[0].binning.toCategoryBinning().categories == ["[4.4, 4.5)", "[7.7, 7.8)", "[2.2, 2.3)", "[3.3, 3.4)"]
        assert b.counts.counts.array.tolist() == [10, 20, 30, 40]
        ab = b + a
        assert ab.axis[0].binning.toCategoryBinning().categories == ["[4.4, 4.5)", "[7.7, 7.8)", "[2.2, 2.3)", "[3.3, 3.4)", "{nan}"]
        assert ab.counts.counts.array.tolist() == [14, 25, 36, 47, 999]
        assert a.axis[0].binning.toCategoryBinning().categories == ["{nan}", "[4.4, 4.5)", "[7.7, 7.8)", "[2.2, 2.3)", "[3.3, 3.4)"]
        assert a.counts.counts.array.tolist() == [999, 4, 5, 6, 7]
        assert b.axis[0].binning.toCategoryBinning().categories == ["[4.4, 4.5)", "[7.7, 7.8)", "[2.2, 2.3)", "[3.3, 3.4)"]
        assert b.counts.counts.array.tolist() == [10, 20, 30, 40]

    def test_add_SparseRegularBinning_permuted(self):
        a = Histogram([Axis(SparseRegularBinning([44, 77, 22, 33], 0.1))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.array([4, 5, 6, 7]))))
        b = Histogram([Axis(SparseRegularBinning([33, 22, 77, 44], 0.1))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.array([10, 20, 30, 40]))))
        assert a.axis[0].binning.toCategoryBinning().categories == ["[4.4, 4.5)", "[7.7, 7.8)", "[2.2, 2.3)", "[3.3, 3.4)"]
        assert a.counts.counts.array.tolist() == [4, 5, 6, 7]
        assert b.axis[0].binning.toCategoryBinning().categories == ["[3.3, 3.4)", "[2.2, 2.3)", "[7.7, 7.8)", "[4.4, 4.5)"]
        assert b.counts.counts.array.tolist() == [10, 20, 30, 40]
        ab = a + b
        assert ab.axis[0].binning.toCategoryBinning().categories == ["[4.4, 4.5)", "[7.7, 7.8)", "[2.2, 2.3)", "[3.3, 3.4)"]
        assert ab.counts.counts.array.tolist() == [44, 35, 26, 17]
        assert a.axis[0].binning.toCategoryBinning().categories == ["[4.4, 4.5)", "[7.7, 7.8)", "[2.2, 2.3)", "[3.3, 3.4)"]
        assert a.counts.counts.array.tolist() == [4, 5, 6, 7]
        assert b.axis[0].binning.toCategoryBinning().categories == ["[3.3, 3.4)", "[2.2, 2.3)", "[7.7, 7.8)", "[4.4, 4.5)"]
        assert b.counts.counts.array.tolist() == [10, 20, 30, 40]
        ab = b + a
        assert ab.axis[0].binning.toCategoryBinning().categories == ["[3.3, 3.4)", "[2.2, 2.3)", "[7.7, 7.8)", "[4.4, 4.5)"]
        assert ab.counts.counts.array.tolist() == [17, 26, 35, 44]
        assert a.axis[0].binning.toCategoryBinning().categories == ["[4.4, 4.5)", "[7.7, 7.8)", "[2.2, 2.3)", "[3.3, 3.4)"]
        assert a.counts.counts.array.tolist() == [4, 5, 6, 7]
        assert b.axis[0].binning.toCategoryBinning().categories == ["[3.3, 3.4)", "[2.2, 2.3)", "[7.7, 7.8)", "[4.4, 4.5)"]
        assert b.counts.counts.array.tolist() == [10, 20, 30, 40]

    def test_add_SparseRegularBinning_permuted_overflow(self):
        a = Histogram([Axis(SparseRegularBinning([44, 77, 22, 33], 0.1, overflow=RealOverflow(loc_nanflow=RealOverflow.below1)))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.array([999, 4, 5, 6, 7]))))
        b = Histogram([Axis(SparseRegularBinning([33, 22, 77, 44], 0.1))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.array([10, 20, 30, 40]))))
        assert a.axis[0].binning.toCategoryBinning().categories == ["{nan}", "[4.4, 4.5)", "[7.7, 7.8)", "[2.2, 2.3)", "[3.3, 3.4)"]
        assert a.counts.counts.array.tolist() == [999, 4, 5, 6, 7]
        assert b.axis[0].binning.toCategoryBinning().categories == ["[3.3, 3.4)", "[2.2, 2.3)", "[7.7, 7.8)", "[4.4, 4.5)"]
        assert b.counts.counts.array.tolist() == [10, 20, 30, 40]
        ab = a + b
        assert ab.axis[0].binning.toCategoryBinning().categories == ["[4.4, 4.5)", "[7.7, 7.8)", "[2.2, 2.3)", "[3.3, 3.4)", "{nan}"]
        assert ab.counts.counts.array.tolist() == [44, 35, 26, 17, 999]
        assert a.axis[0].binning.toCategoryBinning().categories == ["{nan}", "[4.4, 4.5)", "[7.7, 7.8)", "[2.2, 2.3)", "[3.3, 3.4)"]
        assert a.counts.counts.array.tolist() == [999, 4, 5, 6, 7]
        assert b.axis[0].binning.toCategoryBinning().categories == ["[3.3, 3.4)", "[2.2, 2.3)", "[7.7, 7.8)", "[4.4, 4.5)"]
        assert b.counts.counts.array.tolist() == [10, 20, 30, 40]
        ab = b + a
        assert ab.axis[0].binning.toCategoryBinning().categories == ["[3.3, 3.4)", "[2.2, 2.3)", "[7.7, 7.8)", "[4.4, 4.5)", "{nan}"]
        assert ab.counts.counts.array.tolist() == [17, 26, 35, 44, 999]
        assert a.axis[0].binning.toCategoryBinning().categories == ["{nan}", "[4.4, 4.5)", "[7.7, 7.8)", "[2.2, 2.3)", "[3.3, 3.4)"]
        assert a.counts.counts.array.tolist() == [999, 4, 5, 6, 7]
        assert b.axis[0].binning.toCategoryBinning().categories == ["[3.3, 3.4)", "[2.2, 2.3)", "[7.7, 7.8)", "[4.4, 4.5)"]
        assert b.counts.counts.array.tolist() == [10, 20, 30, 40]

    def test_add_SparseRegularBinning_different(self):
        a = Histogram([Axis(SparseRegularBinning([44, 77, 22, 33], 0.1))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(4, 8))))
        b = Histogram([Axis(SparseRegularBinning([66, 22, 77, 99, 55], 0.1))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(10, 60, 10))))
        assert a.axis[0].binning.toCategoryBinning().categories == ["[4.4, 4.5)", "[7.7, 7.8)", "[2.2, 2.3)", "[3.3, 3.4)"]
        assert a.counts.counts.array.tolist() == [4, 5, 6, 7]
        assert b.axis[0].binning.toCategoryBinning().categories == ["[6.6, 6.7)", "[2.2, 2.3)", "[7.7, 7.8)", "[9.9, 10)", "[5.5, 5.6)"]
        assert b.counts.counts.array.tolist() == [10, 20, 30, 40, 50]
        ab = a + b
        assert ab.axis[0].binning.toCategoryBinning().categories == ["[4.4, 4.5)", "[7.7, 7.8)", "[2.2, 2.3)", "[3.3, 3.4)", "[6.6, 6.7)", "[9.9, 10)", "[5.5, 5.6)"]
        assert ab.counts.counts.array.tolist() == [4, 35, 26, 7, 10, 40, 50]
        assert a.axis[0].binning.toCategoryBinning().categories == ["[4.4, 4.5)", "[7.7, 7.8)", "[2.2, 2.3)", "[3.3, 3.4)"]
        assert a.counts.counts.array.tolist() == [4, 5, 6, 7]
        assert b.axis[0].binning.toCategoryBinning().categories == ["[6.6, 6.7)", "[2.2, 2.3)", "[7.7, 7.8)", "[9.9, 10)", "[5.5, 5.6)"]
        assert b.counts.counts.array.tolist() == [10, 20, 30, 40, 50]
        ab = b + a
        assert ab.axis[0].binning.toCategoryBinning().categories == ["[6.6, 6.7)", "[2.2, 2.3)", "[7.7, 7.8)", "[9.9, 10)", "[5.5, 5.6)", "[4.4, 4.5)", "[3.3, 3.4)"]
        assert ab.counts.counts.array.tolist() == [10, 26, 35, 40, 50, 4, 7]
        assert a.axis[0].binning.toCategoryBinning().categories == ["[4.4, 4.5)", "[7.7, 7.8)", "[2.2, 2.3)", "[3.3, 3.4)"]
        assert a.counts.counts.array.tolist() == [4, 5, 6, 7]
        assert b.axis[0].binning.toCategoryBinning().categories == ["[6.6, 6.7)", "[2.2, 2.3)", "[7.7, 7.8)", "[9.9, 10)", "[5.5, 5.6)"]
        assert b.counts.counts.array.tolist() == [10, 20, 30, 40, 50]

    def test_add_SparseRegularBinning_different_overflow(self):
        a = Histogram([Axis(SparseRegularBinning([44, 77, 22, 33], 0.1, overflow=RealOverflow(loc_nanflow=RealOverflow.below1)))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.array([999, 4, 5, 6, 7]))))
        b = Histogram([Axis(SparseRegularBinning([66, 22, 77, 99, 55], 0.1))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(10, 60, 10))))
        assert a.axis[0].binning.toCategoryBinning().categories == ["{nan}", "[4.4, 4.5)", "[7.7, 7.8)", "[2.2, 2.3)", "[3.3, 3.4)"]
        assert a.counts.counts.array.tolist() == [999, 4, 5, 6, 7]
        assert b.axis[0].binning.toCategoryBinning().categories == ["[6.6, 6.7)", "[2.2, 2.3)", "[7.7, 7.8)", "[9.9, 10)", "[5.5, 5.6)"]
        assert b.counts.counts.array.tolist() == [10, 20, 30, 40, 50]
        ab = a + b
        assert ab.axis[0].binning.toCategoryBinning().categories == ["[4.4, 4.5)", "[7.7, 7.8)", "[2.2, 2.3)", "[3.3, 3.4)", "[6.6, 6.7)", "[9.9, 10)", "[5.5, 5.6)", "{nan}"]
        assert ab.counts.counts.array.tolist() == [4, 35, 26, 7, 10, 40, 50, 999]
        assert a.axis[0].binning.toCategoryBinning().categories == ["{nan}", "[4.4, 4.5)", "[7.7, 7.8)", "[2.2, 2.3)", "[3.3, 3.4)"]
        assert a.counts.counts.array.tolist() == [999, 4, 5, 6, 7]
        assert b.axis[0].binning.toCategoryBinning().categories == ["[6.6, 6.7)", "[2.2, 2.3)", "[7.7, 7.8)", "[9.9, 10)", "[5.5, 5.6)"]
        assert b.counts.counts.array.tolist() == [10, 20, 30, 40, 50]
        ab = b + a
        assert ab.axis[0].binning.toCategoryBinning().categories == ["[6.6, 6.7)", "[2.2, 2.3)", "[7.7, 7.8)", "[9.9, 10)", "[5.5, 5.6)", "[4.4, 4.5)", "[3.3, 3.4)", "{nan}"]
        assert ab.counts.counts.array.tolist() == [10, 26, 35, 40, 50, 4, 7, 999]
        assert a.axis[0].binning.toCategoryBinning().categories == ["{nan}", "[4.4, 4.5)", "[7.7, 7.8)", "[2.2, 2.3)", "[3.3, 3.4)"]
        assert a.counts.counts.array.tolist() == [999, 4, 5, 6, 7]
        assert b.axis[0].binning.toCategoryBinning().categories == ["[6.6, 6.7)", "[2.2, 2.3)", "[7.7, 7.8)", "[9.9, 10)", "[5.5, 5.6)"]
        assert b.counts.counts.array.tolist() == [10, 20, 30, 40, 50]

    def test_add_CategoryBinning_same(self):
        a = Histogram([Axis(CategoryBinning(["one", "two", "three"]))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.array([4, 5, 6]))))
        b = Histogram([Axis(CategoryBinning(["one", "two", "three"]))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.array([10, 20, 30]))))
        assert a.axis[0].binning.categories == ["one", "two", "three"]
        assert a.counts.counts.array.tolist() == [4, 5, 6]
        assert b.axis[0].binning.categories == ["one", "two", "three"]
        assert b.counts.counts.array.tolist() == [10, 20, 30]
        ab = a + b
        assert ab.axis[0].binning.categories == ["one", "two", "three"]
        assert ab.counts.counts.array.tolist() == [14, 25, 36]
        assert a.axis[0].binning.categories == ["one", "two", "three"]
        assert a.counts.counts.array.tolist() == [4, 5, 6]
        assert b.axis[0].binning.categories == ["one", "two", "three"]
        assert b.counts.counts.array.tolist() == [10, 20, 30]
        ab = b + a
        assert ab.axis[0].binning.categories == ["one", "two", "three"]
        assert ab.counts.counts.array.tolist() == [14, 25, 36]
        assert a.axis[0].binning.categories == ["one", "two", "three"]
        assert a.counts.counts.array.tolist() == [4, 5, 6]
        assert b.axis[0].binning.categories == ["one", "two", "three"]
        assert b.counts.counts.array.tolist() == [10, 20, 30]

    def test_add_CategoryBinning_same_overflow(self):
        a = Histogram([Axis(CategoryBinning(["one", "two", "three"], loc_overflow=CategoryBinning.below1))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.array([999, 4, 5, 6]))))
        b = Histogram([Axis(CategoryBinning(["one", "two", "three"]))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.array([10, 20, 30]))))
        assert a.axis[0].binning.categories == ["one", "two", "three"]
        assert a.counts.counts.array.tolist() == [999, 4, 5, 6]
        assert b.axis[0].binning.categories == ["one", "two", "three"]
        assert b.counts.counts.array.tolist() == [10, 20, 30]
        ab = a + b
        assert ab.axis[0].binning.categories == ["one", "two", "three"]
        assert ab.counts.counts.array.tolist() == [14, 25, 36, 999]
        assert a.axis[0].binning.categories == ["one", "two", "three"]
        assert a.counts.counts.array.tolist() == [999, 4, 5, 6]
        assert b.axis[0].binning.categories == ["one", "two", "three"]
        assert b.counts.counts.array.tolist() == [10, 20, 30]
        ab = b + a
        assert ab.axis[0].binning.categories == ["one", "two", "three"]
        assert ab.counts.counts.array.tolist() == [14, 25, 36, 999]
        assert a.axis[0].binning.categories == ["one", "two", "three"]
        assert a.counts.counts.array.tolist() == [999, 4, 5, 6]
        assert b.axis[0].binning.categories == ["one", "two", "three"]
        assert b.counts.counts.array.tolist() == [10, 20, 30]

    def test_add_CategoryBinning_permuted(self):
        a = Histogram([Axis(CategoryBinning(["one", "two", "three"]))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.array([4, 5, 6]))))
        b = Histogram([Axis(CategoryBinning(["three", "two", "one"]))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.array([10, 20, 30]))))
        assert a.axis[0].binning.categories == ["one", "two", "three"]
        assert a.counts.counts.array.tolist() == [4, 5, 6]
        assert b.axis[0].binning.categories == ["three", "two", "one"]
        assert b.counts.counts.array.tolist() == [10, 20, 30]
        ab = a + b
        assert ab.axis[0].binning.categories == ["one", "two", "three"]
        assert ab.counts.counts.array.tolist() == [34, 25, 16]
        assert a.axis[0].binning.categories == ["one", "two", "three"]
        assert a.counts.counts.array.tolist() == [4, 5, 6]
        assert b.axis[0].binning.categories == ["three", "two", "one"]
        assert b.counts.counts.array.tolist() == [10, 20, 30]
        ab = b + a
        assert ab.axis[0].binning.categories == ["three", "two", "one"]
        assert ab.counts.counts.array.tolist() == [16, 25, 34]
        assert a.axis[0].binning.categories == ["one", "two", "three"]
        assert a.counts.counts.array.tolist() == [4, 5, 6]
        assert b.axis[0].binning.categories == ["three", "two", "one"]
        assert b.counts.counts.array.tolist() == [10, 20, 30]

    def test_add_CategoryBinning_permuted_overflow(self):
        a = Histogram([Axis(CategoryBinning(["one", "two", "three"], loc_overflow=CategoryBinning.below1))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.array([999, 4, 5, 6]))))
        b = Histogram([Axis(CategoryBinning(["three", "two", "one"]))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.array([10, 20, 30]))))
        assert a.axis[0].binning.categories == ["one", "two", "three"]
        assert a.counts.counts.array.tolist() == [999, 4, 5, 6]
        assert b.axis[0].binning.categories == ["three", "two", "one"]
        assert b.counts.counts.array.tolist() == [10, 20, 30]
        ab = a + b
        assert ab.axis[0].binning.categories == ["one", "two", "three"]
        assert ab.counts.counts.array.tolist() == [34, 25, 16, 999]
        assert a.axis[0].binning.categories == ["one", "two", "three"]
        assert a.counts.counts.array.tolist() == [999, 4, 5, 6]
        assert b.axis[0].binning.categories == ["three", "two", "one"]
        assert b.counts.counts.array.tolist() == [10, 20, 30]
        ab = b + a
        assert ab.axis[0].binning.categories == ["three", "two", "one"]
        assert ab.counts.counts.array.tolist() == [16, 25, 34, 999]
        assert a.axis[0].binning.categories == ["one", "two", "three"]
        assert a.counts.counts.array.tolist() == [999, 4, 5, 6]
        assert b.axis[0].binning.categories == ["three", "two", "one"]
        assert b.counts.counts.array.tolist() == [10, 20, 30]

    def test_add_CategoryBinning_different(self):
        a = Histogram([Axis(CategoryBinning(["one", "two", "three"]))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.array([4, 5, 6]))))
        b = Histogram([Axis(CategoryBinning(["three", "four", "one", "five"]))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.array([10, 20, 30, 40]))))
        assert a.axis[0].binning.categories == ["one", "two", "three"]
        assert a.counts.counts.array.tolist() == [4, 5, 6]
        assert b.axis[0].binning.categories == ["three", "four", "one", "five"]
        assert b.counts.counts.array.tolist() == [10, 20, 30, 40]
        ab = a + b
        assert ab.axis[0].binning.categories == ["one", "two", "three", "four", "five"]
        assert ab.counts.counts.array.tolist() == [34, 5, 16, 20, 40]
        assert a.axis[0].binning.categories == ["one", "two", "three"]
        assert a.counts.counts.array.tolist() == [4, 5, 6]
        assert b.axis[0].binning.categories == ["three", "four", "one", "five"]
        assert b.counts.counts.array.tolist() == [10, 20, 30, 40]
        ab = b + a
        assert ab.axis[0].binning.categories == ["three", "four", "one", "five", "two"]
        assert ab.counts.counts.array.tolist() == [16, 20, 34, 40, 5]
        assert a.axis[0].binning.categories == ["one", "two", "three"]
        assert a.counts.counts.array.tolist() == [4, 5, 6]
        assert b.axis[0].binning.categories == ["three", "four", "one", "five"]
        assert b.counts.counts.array.tolist() == [10, 20, 30, 40]

    def test_add_CategoryBinning_different_overflow(self):
        a = Histogram([Axis(CategoryBinning(["one", "two", "three"], loc_overflow=CategoryBinning.below1))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.array([999, 4, 5, 6]))))
        b = Histogram([Axis(CategoryBinning(["three", "four", "one", "five"]))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.array([10, 20, 30, 40]))))
        assert a.axis[0].binning.categories == ["one", "two", "three"]
        assert a.counts.counts.array.tolist() == [999, 4, 5, 6]
        assert b.axis[0].binning.categories == ["three", "four", "one", "five"]
        assert b.counts.counts.array.tolist() == [10, 20, 30, 40]
        ab = a + b
        assert ab.axis[0].binning.categories == ["one", "two", "three", "four", "five"]
        assert ab.counts.counts.array.tolist() == [34, 5, 16, 20, 40, 999]
        assert a.axis[0].binning.categories == ["one", "two", "three"]
        assert a.counts.counts.array.tolist() == [999, 4, 5, 6]
        assert b.axis[0].binning.categories == ["three", "four", "one", "five"]
        assert b.counts.counts.array.tolist() == [10, 20, 30, 40]
        ab = b + a
        assert ab.axis[0].binning.categories == ["three", "four", "one", "five", "two"]
        assert ab.counts.counts.array.tolist() == [16, 20, 34, 40, 5, 999]
        assert a.axis[0].binning.categories == ["one", "two", "three"]
        assert a.counts.counts.array.tolist() == [999, 4, 5, 6]
        assert b.axis[0].binning.categories == ["three", "four", "one", "five"]
        assert b.counts.counts.array.tolist() == [10, 20, 30, 40]

    def test_add_IrregularBinning_same(self):
        a = Histogram([Axis(IrregularBinning([RealInterval(1, 1.1), RealInterval(2, 2.2), RealInterval(3, 3.3)]))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.array([4, 5, 6]))))
        b = Histogram([Axis(IrregularBinning([RealInterval(1, 1.1), RealInterval(2, 2.2), RealInterval(3, 3.3)]))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.array([10, 20, 30]))))
        assert a.axis[0].binning.toCategoryBinning().categories == ["[1, 1.1)", "[2, 2.2)", "[3, 3.3)"]
        assert a.counts.counts.array.tolist() == [4, 5, 6]
        assert b.axis[0].binning.toCategoryBinning().categories == ["[1, 1.1)", "[2, 2.2)", "[3, 3.3)"]
        assert b.counts.counts.array.tolist() == [10, 20, 30]
        ab = a + b
        assert ab.axis[0].binning.toCategoryBinning().categories == ["[1, 1.1)", "[2, 2.2)", "[3, 3.3)"]
        assert ab.counts.counts.array.tolist() == [14, 25, 36]
        assert a.axis[0].binning.toCategoryBinning().categories == ["[1, 1.1)", "[2, 2.2)", "[3, 3.3)"]
        assert a.counts.counts.array.tolist() == [4, 5, 6]
        assert b.axis[0].binning.toCategoryBinning().categories == ["[1, 1.1)", "[2, 2.2)", "[3, 3.3)"]
        assert b.counts.counts.array.tolist() == [10, 20, 30]
        ab = b + a
        assert ab.axis[0].binning.toCategoryBinning().categories == ["[1, 1.1)", "[2, 2.2)", "[3, 3.3)"]
        assert ab.counts.counts.array.tolist() == [14, 25, 36]
        assert a.axis[0].binning.toCategoryBinning().categories == ["[1, 1.1)", "[2, 2.2)", "[3, 3.3)"]
        assert a.counts.counts.array.tolist() == [4, 5, 6]
        assert b.axis[0].binning.toCategoryBinning().categories == ["[1, 1.1)", "[2, 2.2)", "[3, 3.3)"]
        assert b.counts.counts.array.tolist() == [10, 20, 30]

    def test_add_IrregularBinning_same_overflow(self):
        a = Histogram([Axis(IrregularBinning([RealInterval(1, 1.1), RealInterval(2, 2.2), RealInterval(3, 3.3)], overflow=RealOverflow(loc_nanflow=RealOverflow.below1)))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.array([999, 4, 5, 6]))))
        b = Histogram([Axis(IrregularBinning([RealInterval(1, 1.1), RealInterval(2, 2.2), RealInterval(3, 3.3)]))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.array([10, 20, 30]))))
        assert a.axis[0].binning.toCategoryBinning().categories == ["{nan}", "[1, 1.1)", "[2, 2.2)", "[3, 3.3)"]
        assert a.counts.counts.array.tolist() == [999, 4, 5, 6]
        assert b.axis[0].binning.toCategoryBinning().categories == ["[1, 1.1)", "[2, 2.2)", "[3, 3.3)"]
        assert b.counts.counts.array.tolist() == [10, 20, 30]
        ab = a + b
        assert ab.axis[0].binning.toCategoryBinning().categories == ["[1, 1.1)", "[2, 2.2)", "[3, 3.3)", "{nan}"]
        assert ab.counts.counts.array.tolist() == [14, 25, 36, 999]
        assert a.axis[0].binning.toCategoryBinning().categories == ["{nan}", "[1, 1.1)", "[2, 2.2)", "[3, 3.3)"]
        assert a.counts.counts.array.tolist() == [999, 4, 5, 6]
        assert b.axis[0].binning.toCategoryBinning().categories == ["[1, 1.1)", "[2, 2.2)", "[3, 3.3)"]
        assert b.counts.counts.array.tolist() == [10, 20, 30]
        ab = b + a
        assert ab.axis[0].binning.toCategoryBinning().categories == ["[1, 1.1)", "[2, 2.2)", "[3, 3.3)", "{nan}"]
        assert ab.counts.counts.array.tolist() == [14, 25, 36, 999]
        assert a.axis[0].binning.toCategoryBinning().categories == ["{nan}", "[1, 1.1)", "[2, 2.2)", "[3, 3.3)"]
        assert a.counts.counts.array.tolist() == [999, 4, 5, 6]
        assert b.axis[0].binning.toCategoryBinning().categories == ["[1, 1.1)", "[2, 2.2)", "[3, 3.3)"]
        assert b.counts.counts.array.tolist() == [10, 20, 30]

    def test_add_IrregularBinning_permuted(self):
        a = Histogram([Axis(IrregularBinning([RealInterval(1, 1.1), RealInterval(2, 2.2), RealInterval(3, 3.3)]))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.array([4, 5, 6]))))
        b = Histogram([Axis(IrregularBinning([RealInterval(3, 3.3), RealInterval(2, 2.2), RealInterval(1, 1.1)]))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.array([10, 20, 30]))))
        assert a.axis[0].binning.toCategoryBinning().categories == ["[1, 1.1)", "[2, 2.2)", "[3, 3.3)"]
        assert a.counts.counts.array.tolist() == [4, 5, 6]
        assert b.axis[0].binning.toCategoryBinning().categories == ["[3, 3.3)", "[2, 2.2)", "[1, 1.1)"]
        assert b.counts.counts.array.tolist() == [10, 20, 30]
        ab = a + b
        assert ab.axis[0].binning.toCategoryBinning().categories == ["[1, 1.1)", "[2, 2.2)", "[3, 3.3)"]
        assert ab.counts.counts.array.tolist() == [34, 25, 16]
        assert a.axis[0].binning.toCategoryBinning().categories == ["[1, 1.1)", "[2, 2.2)", "[3, 3.3)"]
        assert a.counts.counts.array.tolist() == [4, 5, 6]
        assert b.axis[0].binning.toCategoryBinning().categories == ["[3, 3.3)", "[2, 2.2)", "[1, 1.1)"]
        assert b.counts.counts.array.tolist() == [10, 20, 30]
        ab = b + a
        assert ab.axis[0].binning.toCategoryBinning().categories == ["[3, 3.3)", "[2, 2.2)", "[1, 1.1)"]
        assert ab.counts.counts.array.tolist() == [16, 25, 34]
        assert a.axis[0].binning.toCategoryBinning().categories == ["[1, 1.1)", "[2, 2.2)", "[3, 3.3)"]
        assert a.counts.counts.array.tolist() == [4, 5, 6]
        assert b.axis[0].binning.toCategoryBinning().categories == ["[3, 3.3)", "[2, 2.2)", "[1, 1.1)"]
        assert b.counts.counts.array.tolist() == [10, 20, 30]

    def test_add_IrregularBinning_permuted_overflow(self):
        a = Histogram([Axis(IrregularBinning([RealInterval(1, 1.1), RealInterval(2, 2.2), RealInterval(3, 3.3)], overflow=RealOverflow(loc_nanflow=RealOverflow.below1)))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.array([999, 4, 5, 6]))))
        b = Histogram([Axis(IrregularBinning([RealInterval(3, 3.3), RealInterval(2, 2.2), RealInterval(1, 1.1)]))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.array([10, 20, 30]))))
        assert a.axis[0].binning.toCategoryBinning().categories == ["{nan}", "[1, 1.1)", "[2, 2.2)", "[3, 3.3)"]
        assert a.counts.counts.array.tolist() == [999, 4, 5, 6]
        assert b.axis[0].binning.toCategoryBinning().categories == ["[3, 3.3)", "[2, 2.2)", "[1, 1.1)"]
        assert b.counts.counts.array.tolist() == [10, 20, 30]
        ab = a + b
        assert ab.axis[0].binning.toCategoryBinning().categories == ["[1, 1.1)", "[2, 2.2)", "[3, 3.3)", "{nan}"]
        assert ab.counts.counts.array.tolist() == [34, 25, 16, 999]
        assert a.axis[0].binning.toCategoryBinning().categories == ["{nan}", "[1, 1.1)", "[2, 2.2)", "[3, 3.3)"]
        assert a.counts.counts.array.tolist() == [999, 4, 5, 6]
        assert b.axis[0].binning.toCategoryBinning().categories == ["[3, 3.3)", "[2, 2.2)", "[1, 1.1)"]
        assert b.counts.counts.array.tolist() == [10, 20, 30]
        ab = b + a
        assert ab.axis[0].binning.toCategoryBinning().categories == ["[3, 3.3)", "[2, 2.2)", "[1, 1.1)", "{nan}"]
        assert ab.counts.counts.array.tolist() == [16, 25, 34, 999]
        assert a.axis[0].binning.toCategoryBinning().categories == ["{nan}", "[1, 1.1)", "[2, 2.2)", "[3, 3.3)"]
        assert a.counts.counts.array.tolist() == [999, 4, 5, 6]
        assert b.axis[0].binning.toCategoryBinning().categories == ["[3, 3.3)", "[2, 2.2)", "[1, 1.1)"]
        assert b.counts.counts.array.tolist() == [10, 20, 30]

    def test_add_IrregularBinning_different(self):
        a = Histogram([Axis(IrregularBinning([RealInterval(1, 1.1), RealInterval(2, 2.2), RealInterval(3, 3.3)]))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.array([4, 5, 6]))))
        b = Histogram([Axis(IrregularBinning([RealInterval(3, 3.3), RealInterval(4, 4.4), RealInterval(1, 1.1), RealInterval(5, 5.5)]))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.array([10, 20, 30, 40]))))
        assert a.axis[0].binning.toCategoryBinning().categories == ["[1, 1.1)", "[2, 2.2)", "[3, 3.3)"]
        assert a.counts.counts.array.tolist() == [4, 5, 6]
        assert b.axis[0].binning.toCategoryBinning().categories == ["[3, 3.3)", "[4, 4.4)", "[1, 1.1)", "[5, 5.5)"]
        assert b.counts.counts.array.tolist() == [10, 20, 30, 40]
        ab = a + b
        assert ab.axis[0].binning.toCategoryBinning().categories == ["[1, 1.1)", "[2, 2.2)", "[3, 3.3)", "[4, 4.4)", "[5, 5.5)"]
        assert ab.counts.counts.array.tolist() == [34, 5, 16, 20, 40]
        assert a.axis[0].binning.toCategoryBinning().categories == ["[1, 1.1)", "[2, 2.2)", "[3, 3.3)"]
        assert a.counts.counts.array.tolist() == [4, 5, 6]
        assert b.axis[0].binning.toCategoryBinning().categories == ["[3, 3.3)", "[4, 4.4)", "[1, 1.1)", "[5, 5.5)"]
        assert b.counts.counts.array.tolist() == [10, 20, 30, 40]
        ab = b + a
        assert ab.axis[0].binning.toCategoryBinning().categories == ["[3, 3.3)", "[4, 4.4)", "[1, 1.1)", "[5, 5.5)", "[2, 2.2)"]
        assert ab.counts.counts.array.tolist() == [16, 20, 34, 40, 5]
        assert a.axis[0].binning.toCategoryBinning().categories == ["[1, 1.1)", "[2, 2.2)", "[3, 3.3)"]
        assert a.counts.counts.array.tolist() == [4, 5, 6]
        assert b.axis[0].binning.toCategoryBinning().categories == ["[3, 3.3)", "[4, 4.4)", "[1, 1.1)", "[5, 5.5)"]
        assert b.counts.counts.array.tolist() == [10, 20, 30, 40]

    def test_add_IrregularBinning_different_overflow(self):
        a = Histogram([Axis(IrregularBinning([RealInterval(1, 1.1), RealInterval(2, 2.2), RealInterval(3, 3.3)], overflow=RealOverflow(loc_nanflow=RealOverflow.below1)))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.array([999, 4, 5, 6]))))
        b = Histogram([Axis(IrregularBinning([RealInterval(3, 3.3), RealInterval(4, 4.4), RealInterval(1, 1.1), RealInterval(5, 5.5)]))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.array([10, 20, 30, 40]))))
        assert a.axis[0].binning.toCategoryBinning().categories == ["{nan}", "[1, 1.1)", "[2, 2.2)", "[3, 3.3)"]
        assert a.counts.counts.array.tolist() == [999, 4, 5, 6]
        assert b.axis[0].binning.toCategoryBinning().categories == ["[3, 3.3)", "[4, 4.4)", "[1, 1.1)", "[5, 5.5)"]
        assert b.counts.counts.array.tolist() == [10, 20, 30, 40]
        ab = a + b
        assert ab.axis[0].binning.toCategoryBinning().categories == ["[1, 1.1)", "[2, 2.2)", "[3, 3.3)", "[4, 4.4)", "[5, 5.5)", "{nan}"]
        assert ab.counts.counts.array.tolist() == [34, 5, 16, 20, 40, 999]
        assert a.axis[0].binning.toCategoryBinning().categories == ["{nan}", "[1, 1.1)", "[2, 2.2)", "[3, 3.3)"]
        assert a.counts.counts.array.tolist() == [999, 4, 5, 6]
        assert b.axis[0].binning.toCategoryBinning().categories == ["[3, 3.3)", "[4, 4.4)", "[1, 1.1)", "[5, 5.5)"]
        assert b.counts.counts.array.tolist() == [10, 20, 30, 40]
        ab = b + a
        assert ab.axis[0].binning.toCategoryBinning().categories == ["[3, 3.3)", "[4, 4.4)", "[1, 1.1)", "[5, 5.5)", "[2, 2.2)", "{nan}"]
        assert ab.counts.counts.array.tolist() == [16, 20, 34, 40, 5, 999]
        assert a.axis[0].binning.toCategoryBinning().categories == ["{nan}", "[1, 1.1)", "[2, 2.2)", "[3, 3.3)"]
        assert a.counts.counts.array.tolist() == [999, 4, 5, 6]
        assert b.axis[0].binning.toCategoryBinning().categories == ["[3, 3.3)", "[4, 4.4)", "[1, 1.1)", "[5, 5.5)"]
        assert b.counts.counts.array.tolist() == [10, 20, 30, 40]

    def test_add_collection_same(self):
        a = Collection({"x": Histogram([Axis(IntegerBinning(-5, 5))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(11)))),
                        "y": Histogram([Axis(RegularBinning(10, RealInterval(-5, 5)))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(10))))})
        b = Collection({"x": Histogram([Axis(IntegerBinning(-5, 5))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(11)))),
                        "y": Histogram([Axis(RegularBinning(10, RealInterval(-5, 5)))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(10))))})
        assert set(a.objects) == set(["x", "y"])
        assert a.objects["x"].axis[0].binning.toCategoryBinning().categories == ["-5", "-4", "-3", "-2", "-1", "0", "1", "2", "3", "4", "5"]
        assert a.objects["x"].counts.counts.array.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        assert a.objects["y"].axis[0].binning.toCategoryBinning().categories == ["[-5, -4)", "[-4, -3)", "[-3, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5)"]
        assert a.objects["y"].counts.counts.array.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        assert set(b.objects) == set(["x", "y"])
        assert b.objects["x"].axis[0].binning.toCategoryBinning().categories == ["-5", "-4", "-3", "-2", "-1", "0", "1", "2", "3", "4", "5"]
        assert b.objects["x"].counts.counts.array.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        assert b.objects["y"].axis[0].binning.toCategoryBinning().categories == ["[-5, -4)", "[-4, -3)", "[-3, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5)"]
        assert b.objects["y"].counts.counts.array.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        ab = a + b
        assert set(ab.objects) == set(["x", "y"])
        assert ab.objects["x"].axis[0].binning.toCategoryBinning().categories == ["-5", "-4", "-3", "-2", "-1", "0", "1", "2", "3", "4", "5"]
        assert ab.objects["x"].counts.counts.array.tolist() == [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
        assert ab.objects["y"].axis[0].binning.toCategoryBinning().categories == ["[-5, -4)", "[-4, -3)", "[-3, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5)"]
        assert ab.objects["y"].counts.counts.array.tolist() == [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
        assert set(a.objects) == set(["x", "y"])
        assert a.objects["x"].axis[0].binning.toCategoryBinning().categories == ["-5", "-4", "-3", "-2", "-1", "0", "1", "2", "3", "4", "5"]
        assert a.objects["x"].counts.counts.array.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        assert a.objects["y"].axis[0].binning.toCategoryBinning().categories == ["[-5, -4)", "[-4, -3)", "[-3, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5)"]
        assert a.objects["y"].counts.counts.array.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        assert set(b.objects) == set(["x", "y"])
        assert b.objects["x"].axis[0].binning.toCategoryBinning().categories == ["-5", "-4", "-3", "-2", "-1", "0", "1", "2", "3", "4", "5"]
        assert b.objects["x"].counts.counts.array.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        assert b.objects["y"].axis[0].binning.toCategoryBinning().categories == ["[-5, -4)", "[-4, -3)", "[-3, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5)"]
        assert b.objects["y"].counts.counts.array.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        ab = b + a
        assert set(ab.objects) == set(["x", "y"])
        assert ab.objects["x"].axis[0].binning.toCategoryBinning().categories == ["-5", "-4", "-3", "-2", "-1", "0", "1", "2", "3", "4", "5"]
        assert ab.objects["x"].counts.counts.array.tolist() == [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
        assert ab.objects["y"].axis[0].binning.toCategoryBinning().categories == ["[-5, -4)", "[-4, -3)", "[-3, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5)"]
        assert ab.objects["y"].counts.counts.array.tolist() == [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
        assert set(a.objects) == set(["x", "y"])
        assert a.objects["x"].axis[0].binning.toCategoryBinning().categories == ["-5", "-4", "-3", "-2", "-1", "0", "1", "2", "3", "4", "5"]
        assert a.objects["x"].counts.counts.array.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        assert a.objects["y"].axis[0].binning.toCategoryBinning().categories == ["[-5, -4)", "[-4, -3)", "[-3, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5)"]
        assert a.objects["y"].counts.counts.array.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        assert set(b.objects) == set(["x", "y"])
        assert b.objects["x"].axis[0].binning.toCategoryBinning().categories == ["-5", "-4", "-3", "-2", "-1", "0", "1", "2", "3", "4", "5"]
        assert b.objects["x"].counts.counts.array.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        assert b.objects["y"].axis[0].binning.toCategoryBinning().categories == ["[-5, -4)", "[-4, -3)", "[-3, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5)"]
        assert b.objects["y"].counts.counts.array.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    def test_add_collection_different(self):
        a = Collection({"x": Histogram([Axis(IntegerBinning(-5, 5))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(11)))),
                        "y": Histogram([Axis(RegularBinning(10, RealInterval(-5, 5)))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(10))))})
        b = Collection({"x": Histogram([Axis(IntegerBinning(-5, 5))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(11))))})
        assert set(a.objects) == set(["x", "y"])
        assert a.objects["x"].axis[0].binning.toCategoryBinning().categories == ["-5", "-4", "-3", "-2", "-1", "0", "1", "2", "3", "4", "5"]
        assert a.objects["x"].counts.counts.array.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        assert a.objects["y"].axis[0].binning.toCategoryBinning().categories == ["[-5, -4)", "[-4, -3)", "[-3, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5)"]
        assert a.objects["y"].counts.counts.array.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        assert set(b.objects) == set(["x"])
        assert b.objects["x"].axis[0].binning.toCategoryBinning().categories == ["-5", "-4", "-3", "-2", "-1", "0", "1", "2", "3", "4", "5"]
        assert b.objects["x"].counts.counts.array.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        ab = a + b
        assert set(ab.objects) == set(["x", "y"])
        assert ab.objects["x"].axis[0].binning.toCategoryBinning().categories == ["-5", "-4", "-3", "-2", "-1", "0", "1", "2", "3", "4", "5"]
        assert ab.objects["x"].counts.counts.array.tolist() == [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
        assert ab.objects["y"].axis[0].binning.toCategoryBinning().categories == ["[-5, -4)", "[-4, -3)", "[-3, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5)"]
        assert ab.objects["y"].counts.counts.array.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        assert set(a.objects) == set(["x", "y"])
        assert a.objects["x"].axis[0].binning.toCategoryBinning().categories == ["-5", "-4", "-3", "-2", "-1", "0", "1", "2", "3", "4", "5"]
        assert a.objects["x"].counts.counts.array.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        assert a.objects["y"].axis[0].binning.toCategoryBinning().categories == ["[-5, -4)", "[-4, -3)", "[-3, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5)"]
        assert a.objects["y"].counts.counts.array.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        assert set(b.objects) == set(["x"])
        assert b.objects["x"].axis[0].binning.toCategoryBinning().categories == ["-5", "-4", "-3", "-2", "-1", "0", "1", "2", "3", "4", "5"]
        assert b.objects["x"].counts.counts.array.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        ab = b + a
        assert set(ab.objects) == set(["x", "y"])
        assert ab.objects["x"].axis[0].binning.toCategoryBinning().categories == ["-5", "-4", "-3", "-2", "-1", "0", "1", "2", "3", "4", "5"]
        assert ab.objects["x"].counts.counts.array.tolist() == [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
        assert ab.objects["y"].axis[0].binning.toCategoryBinning().categories == ["[-5, -4)", "[-4, -3)", "[-3, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5)"]
        assert ab.objects["y"].counts.counts.array.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        assert set(a.objects) == set(["x", "y"])
        assert a.objects["x"].axis[0].binning.toCategoryBinning().categories == ["-5", "-4", "-3", "-2", "-1", "0", "1", "2", "3", "4", "5"]
        assert a.objects["x"].counts.counts.array.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        assert a.objects["y"].axis[0].binning.toCategoryBinning().categories == ["[-5, -4)", "[-4, -3)", "[-3, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5)"]
        assert a.objects["y"].counts.counts.array.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        assert set(b.objects) == set(["x"])
        assert b.objects["x"].axis[0].binning.toCategoryBinning().categories == ["-5", "-4", "-3", "-2", "-1", "0", "1", "2", "3", "4", "5"]
        assert b.objects["x"].counts.counts.array.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    def test_add_collection_axis(self):
        a = Collection({"x": Histogram([Axis(IntegerBinning(-5, 5))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(22))))}, axis=[Axis(CategoryBinning(["one", "two"]))])
        b = Collection({"x": Histogram([Axis(IntegerBinning(-5, 5))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.full(11, 100))))}, axis=[Axis(CategoryBinning(["one"]))])
        assert a.objects["x"].counts.counts.array.tolist() == [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]]
        assert b.objects["x"].counts.counts.array.tolist() == [[100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]]
        ab = a + b
        assert set(ab.objects) == set(["x"])
        assert ab.axis[0].binning.categories == ["one", "two"]
        assert ab.objects["x"].counts.counts.array.tolist() == [[100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110], [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]]
        assert a.objects["x"].counts.counts.array.tolist() == [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]]
        assert b.objects["x"].counts.counts.array.tolist() == [[100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]]
        ab = b + a
        assert set(ab.objects) == set(["x"])
        assert ab.axis[0].binning.categories == ["one", "two"]
        assert ab.objects["x"].counts.counts.array.tolist() == [[100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110], [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]]
        assert a.objects["x"].counts.counts.array.tolist() == [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]]
        assert b.objects["x"].counts.counts.array.tolist() == [[100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]]

        a = Collection({"x": Histogram([Axis(IntegerBinning(-5, 5))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(22))))}, axis=[Axis(CategoryBinning(["one", "two"]))])
        b = Collection({"x": Histogram([Axis(IntegerBinning(-5, 5))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.full(11, 100))))}, axis=[Axis(CategoryBinning(["two"]))])
        assert a.objects["x"].counts.counts.array.tolist() == [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]]
        assert b.objects["x"].counts.counts.array.tolist() == [[100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]]
        ab = a + b
        assert set(ab.objects) == set(["x"])
        assert ab.axis[0].binning.categories == ["one", "two"]
        assert ab.objects["x"].counts.counts.array.tolist() == [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121]]
        assert a.objects["x"].counts.counts.array.tolist() == [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]]
        assert b.objects["x"].counts.counts.array.tolist() == [[100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]]
        ab = b + a
        assert set(ab.objects) == set(["x"])
        assert ab.axis[0].binning.categories == ["two", "one"]
        assert ab.objects["x"].counts.counts.array.tolist() == [[111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
        assert a.objects["x"].counts.counts.array.tolist() == [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]]
        assert b.objects["x"].counts.counts.array.tolist() == [[100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]]

    def test_add_collection_fromwithin(self):
        a = Collection({"x": Histogram([Axis(IntegerBinning(-5, 5))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(22))))}, axis=[Axis(CategoryBinning(["one", "two"]))])
        b = Collection({"x": Histogram([Axis(IntegerBinning(-5, 5))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.full(11, 100))))}, axis=[Axis(CategoryBinning(["one"]))])
        assert a.objects["x"].counts.counts.array.tolist() == [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]]
        assert b.objects["x"].counts.counts.array.tolist() == [[100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]]
        ab = a.objects["x"] + b.objects["x"]
        assert isinstance(ab, Histogram)
        assert len(ab.axis) == 1
        assert ab.axis[0].binning.toCategoryBinning().categories == ["-5", "-4", "-3", "-2", "-1", "0", "1", "2", "3", "4", "5"]
        assert ab.counts.counts.buffer.tostring() == numpy.array([[100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110], [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]]).reshape(-1).tostring()
        assert a.objects["x"].counts.counts.array.tolist() == [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]]
        assert b.objects["x"].counts.counts.array.tolist() == [[100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]]
        ab = b.objects["x"] + a.objects["x"]
        assert isinstance(ab, Histogram)
        assert len(ab.axis) == 1
        assert ab.axis[0].binning.toCategoryBinning().categories == ["-5", "-4", "-3", "-2", "-1", "0", "1", "2", "3", "4", "5"]
        assert ab.counts.counts.buffer.tostring() == numpy.array([[100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110], [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]]).reshape(-1).tostring()
        assert a.objects["x"].counts.counts.array.tolist() == [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]]
        assert b.objects["x"].counts.counts.array.tolist() == [[100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]]
