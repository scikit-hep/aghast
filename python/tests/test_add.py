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

    # def test_add_Metadata(self):
    #     h = Collection({}, metadata=Metadata("""{"one": 1, "two": 2}""", language=Metadata.json))
    #     (h + h).checkvalid()
    #     (frombuffer(h.tobuffer()) + h).checkvalid()
    #     h = Collection({}, metadata=Metadata("""{"one": 1, "two": 2}"""))
    #     (h + h).checkvalid()
    #     (frombuffer(h.tobuffer()) + h).checkvalid()

    # def test_add_Decoration(self):
    #     h = Collection({}, decoration=Decoration("""points { color: red }""", language=Decoration.css))
    #     (h + h).checkvalid()
    #     (frombuffer(h.tobuffer()) + h).checkvalid()
    #     h = Collection({}, decoration=Decoration("""points { color: red }"""))
    #     (h + h).checkvalid()
    #     (frombuffer(h.tobuffer()) + h).checkvalid()

    # def test_add_RawInlineBuffer(self):
    #     h = Ntuple([Column("one", Column.int32)], [NtupleInstance([Chunk([ColumnChunk([Page(RawInlineBuffer(numpy.zeros(1, dtype=numpy.int32)))], [0, 1])])])])
    #     (h + h).checkvalid()
    #     (frombuffer(h.tobuffer()) + h).checkvalid()
    #     h = Ntuple([Column("one", Column.int32)], [NtupleInstance([Chunk([ColumnChunk([Page(RawInlineBuffer(b"\x05\x00\x00\x00"))], [0, 1])])])])
    #     (h + h).checkvalid()
    #     (frombuffer(h.tobuffer()) + h).checkvalid()

    # def test_add_RawExternalBuffer(self):
    #     buf = numpy.zeros(1, dtype=numpy.int32)
    #     h = Ntuple([Column("one", Column.int32)], [NtupleInstance([Chunk([ColumnChunk([Page(RawExternalBuffer(buf.ctypes.data, buf.nbytes))], [0, 1])])])])
    #     (h + h).checkvalid()
    #     (frombuffer(h.tobuffer()) + h).checkvalid()
    #     buf = numpy.array([3.14], dtype=numpy.float64)
    #     h = Ntuple([Column("one", Column.float64)], [NtupleInstance([Chunk([ColumnChunk([Page(RawExternalBuffer(buf.ctypes.data, buf.nbytes))], [0, 1])])])])
    #     (h + h).checkvalid()
    #     (frombuffer(h.tobuffer()) + h).checkvalid()

    # def test_add_InterpretedInlineBuffer(self):
    #     h = BinnedEvaluatedFunction([Axis()], InterpretedInlineBuffer(numpy.zeros(1, dtype=numpy.int32), dtype=InterpretedInlineBuffer.int32))
    #     (h + h).checkvalid()
    #     (frombuffer(h.tobuffer()) + h).checkvalid()
    #     h = BinnedEvaluatedFunction([Axis()], InterpretedInlineBuffer(b"\x07\x00\x00\x00", dtype=InterpretedInlineBuffer.int32))
    #     (h + h).checkvalid()
    #     (frombuffer(h.tobuffer()) + h).checkvalid()

    # def test_add_InterpretedExternalBuffer(self):
    #     buf = numpy.zeros(1, dtype=numpy.float64)
    #     h = BinnedEvaluatedFunction([Axis()], InterpretedExternalBuffer(buf.ctypes.data, buf.nbytes, dtype=InterpretedInlineBuffer.float64))
    #     (h + h).checkvalid()
    #     (frombuffer(h.tobuffer()) + h).checkvalid()
    #     buf = numpy.array([3.14], dtype=numpy.float64)
    #     h = BinnedEvaluatedFunction([Axis()], InterpretedExternalBuffer(buf.ctypes.data, buf.nbytes, dtype=InterpretedInlineBuffer.float64))
    #     (h + h).checkvalid()
    #     (frombuffer(h.tobuffer()) + h).checkvalid()

    # def test_add_IntegerBinning(self):
    #     h = Histogram([Axis(IntegerBinning(10, 20))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(11))))
    #     (h + h).checkvalid()
    #     (frombuffer(h.tobuffer()) + h).checkvalid()
    #     h = Histogram([Axis(IntegerBinning(10, 20, loc_underflow=IntegerBinning.above1))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(12))))
    #     (h + h).checkvalid()
    #     (frombuffer(h.tobuffer()) + h).checkvalid()
    #     h = Histogram([Axis(IntegerBinning(10, 20, loc_underflow=IntegerBinning.above1, loc_overflow=IntegerBinning.above2))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(13))))
    #     (h + h).checkvalid()
    #     (frombuffer(h.tobuffer()) + h).checkvalid()

    # def test_add_RealInterval(self):
    #     h = BinnedEvaluatedFunction([Axis(RegularBinning(10, RealInterval(-5, 5)))], InterpretedInlineBuffer(numpy.zeros(10), dtype=InterpretedInlineBuffer.float64))
    #     (h + h).checkvalid()
    #     (frombuffer(h.tobuffer()) + h).checkvalid()

    # def test_add_RealOverflow(self):
    #     h = BinnedEvaluatedFunction([Axis(RegularBinning(10, RealInterval(-5, 5), RealOverflow()))], InterpretedInlineBuffer(numpy.zeros(10), dtype=InterpretedInlineBuffer.float64))
    #     (h + h).checkvalid()
    #     (frombuffer(h.tobuffer()) + h).checkvalid()
    #     h = BinnedEvaluatedFunction([Axis(RegularBinning(10, RealInterval(-5, 5), RealOverflow(loc_underflow=RealOverflow.below1, loc_overflow=RealOverflow.nonexistent, loc_nanflow=RealOverflow.nonexistent)))], InterpretedInlineBuffer(numpy.zeros(11), dtype=InterpretedInlineBuffer.float64))
    #     (h + h).checkvalid()
    #     (frombuffer(h.tobuffer()) + h).checkvalid()
    #     h = BinnedEvaluatedFunction([Axis(RegularBinning(10, RealInterval(-5, 5), RealOverflow(loc_underflow=RealOverflow.below1, loc_overflow=RealOverflow.below2, loc_nanflow=RealOverflow.nonexistent)))], InterpretedInlineBuffer(numpy.zeros(12), dtype=InterpretedInlineBuffer.float64))
    #     (h + h).checkvalid()
    #     (frombuffer(h.tobuffer()) + h).checkvalid()
    #     h = BinnedEvaluatedFunction([Axis(RegularBinning(10, RealInterval(-5, 5), RealOverflow(loc_underflow=RealOverflow.below1, loc_overflow=RealOverflow.below2, loc_nanflow=RealOverflow.below3)))], InterpretedInlineBuffer(numpy.zeros(13), dtype=InterpretedInlineBuffer.float64))
    #     (h + h).checkvalid()
    #     (frombuffer(h.tobuffer()) + h).checkvalid()
    #     h = BinnedEvaluatedFunction([Axis(RegularBinning(10, RealInterval(-5, 5), RealOverflow(minf_mapping=RealOverflow.in_overflow)))], InterpretedInlineBuffer(numpy.zeros(10), dtype=InterpretedInlineBuffer.float64))
    #     (h + h).checkvalid()
    #     (frombuffer(h.tobuffer()) + h).checkvalid()
    #     h = BinnedEvaluatedFunction([Axis(RegularBinning(10, RealInterval(-5, 5), RealOverflow(pinf_mapping=RealOverflow.in_underflow)))], InterpretedInlineBuffer(numpy.zeros(10), dtype=InterpretedInlineBuffer.float64))
    #     (h + h).checkvalid()
    #     (frombuffer(h.tobuffer()) + h).checkvalid()
    #     h = BinnedEvaluatedFunction([Axis(RegularBinning(10, RealInterval(-5, 5), RealOverflow(nan_mapping=RealOverflow.missing)))], InterpretedInlineBuffer(numpy.zeros(10), dtype=InterpretedInlineBuffer.float64))
    #     (h + h).checkvalid()
    #     (frombuffer(h.tobuffer()) + h).checkvalid()

    # def test_add_RegularBinning(self):
    #     h = BinnedEvaluatedFunction([Axis(RegularBinning(10, RealInterval(-5, 5)))], InterpretedInlineBuffer(numpy.zeros(10), dtype=InterpretedInlineBuffer.float64))
    #     (h + h).checkvalid()
    #     (frombuffer(h.tobuffer()) + h).checkvalid()

    # def test_add_HexagonalBinning(self):
    #     h = BinnedEvaluatedFunction([Axis(HexagonalBinning(3, 5, -5, -4))], InterpretedInlineBuffer(numpy.array([[0.0] * 2] * 3), dtype=InterpretedInlineBuffer.float64))
    #     (h + h).checkvalid()
    #     (frombuffer(h.tobuffer()) + h).checkvalid()
    #     h = BinnedEvaluatedFunction([Axis(HexagonalBinning(3, 5, -5, -4, qoverflow=RealOverflow(loc_nanflow=RealOverflow.above1)))], InterpretedInlineBuffer(numpy.array([[0.0] * 2] * 4), dtype=InterpretedInlineBuffer.float64))
    #     (h + h).checkvalid()
    #     (frombuffer(h.tobuffer()) + h).checkvalid()
    #     h = BinnedEvaluatedFunction([Axis(HexagonalBinning(3, 5, -5, -4, roverflow=RealOverflow(loc_nanflow=RealOverflow.above1)))], InterpretedInlineBuffer(numpy.array([[0.0] * 3] * 3), dtype=InterpretedInlineBuffer.float64))
    #     (h + h).checkvalid()
    #     (frombuffer(h.tobuffer()) + h).checkvalid()
    #     h = BinnedEvaluatedFunction([Axis(HexagonalBinning(3, 5, -5, -4, qoverflow=RealOverflow(loc_nanflow=RealOverflow.above1), roverflow=RealOverflow(loc_nanflow=RealOverflow.above1)))], InterpretedInlineBuffer(numpy.array([[0.0] * 3] * 4), dtype=InterpretedInlineBuffer.float64))
    #     (h + h).checkvalid()
    #     (frombuffer(h.tobuffer()) + h).checkvalid()

    # def test_add_EdgesBinning(self):
    #     h = BinnedEvaluatedFunction([Axis(EdgesBinning([3.3], overflow=RealOverflow(loc_underflow=RealOverflow.above1, loc_overflow=RealOverflow.above2)))], InterpretedInlineBuffer(numpy.array([0.0, 0.0]), dtype=InterpretedInlineBuffer.float64))
    #     (h + h).checkvalid()
    #     (frombuffer(h.tobuffer()) + h).checkvalid()
    #     h = BinnedEvaluatedFunction([Axis(EdgesBinning([1.1, 2.2, 3.3]))], InterpretedInlineBuffer(numpy.array([0.0, 0.0]), dtype=InterpretedInlineBuffer.float64))
    #     (h + h).checkvalid()
    #     (frombuffer(h.tobuffer()) + h).checkvalid()

    # def test_add_IrregularBinning(self):
    #     h = BinnedEvaluatedFunction([Axis(IrregularBinning([RealInterval(0.5, 1.5)]))], InterpretedInlineBuffer(numpy.array([0.0]), dtype=InterpretedInlineBuffer.float64))
    #     (h + h).checkvalid()
    #     (frombuffer(h.tobuffer()) + h).checkvalid()
    #     h = BinnedEvaluatedFunction([Axis(IrregularBinning([RealInterval(0.5, 1.5), RealInterval(1.5, 1.5), RealInterval(0.0, 10.0)]))], InterpretedInlineBuffer(numpy.array([0.0, 0.0, 0.0]), dtype=InterpretedInlineBuffer.float64))
    #     (h + h).checkvalid()
    #     (frombuffer(h.tobuffer()) + h).checkvalid()

    # def test_add_CategoryBinning(self):
    #     h = BinnedEvaluatedFunction([Axis(CategoryBinning(["one", "two", "three"]))], InterpretedInlineBuffer(numpy.array([0.0, 0.0, 0.0]), dtype=InterpretedInlineBuffer.float64))
    #     (h + h).checkvalid()
    #     (frombuffer(h.tobuffer()) + h).checkvalid()
    #     h = BinnedEvaluatedFunction([Axis(CategoryBinning(["one", "two", "three"], loc_overflow=CategoryBinning.above1))], InterpretedInlineBuffer(numpy.array([0.0, 0.0, 0.0, 0.0]), dtype=InterpretedInlineBuffer.float64))
    #     (h + h).checkvalid()
    #     (frombuffer(h.tobuffer()) + h).checkvalid()

    # def test_add_SparseRegularBinning(self):
    #     h = BinnedEvaluatedFunction([Axis(SparseRegularBinning([-5, -3, 10, 1000], 0.1))], InterpretedInlineBuffer(numpy.array([0.0, 0.0, 0.0, 0.0]), dtype=InterpretedInlineBuffer.float64))
    #     (h + h).checkvalid()
    #     (frombuffer(h.tobuffer()) + h).checkvalid()

    # def test_add_FractionBinning(self):
    #     h = BinnedEvaluatedFunction([Axis(FractionBinning())], InterpretedInlineBuffer(numpy.array([0.0, 0.0]), dtype=InterpretedInlineBuffer.float64))
    #     (h + h).checkvalid()
    #     (frombuffer(h.tobuffer()) + h).checkvalid()
    #     h = BinnedEvaluatedFunction([Axis(FractionBinning()), Axis(RegularBinning(10, RealInterval(-5, 5)))], InterpretedInlineBuffer(numpy.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]), dtype=InterpretedInlineBuffer.float64))
    #     (h + h).checkvalid()
    #     (frombuffer(h.tobuffer()) + h).checkvalid()
    #     h = BinnedEvaluatedFunction([Axis(RegularBinning(10, RealInterval(-5, 5))), Axis(FractionBinning())], InterpretedInlineBuffer(numpy.array([[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]]), dtype=InterpretedInlineBuffer.float64))
    #     (h + h).checkvalid()
    #     (frombuffer(h.tobuffer()) + h).checkvalid()

    # def test_add_PredicateBinning(self):
    #     h = Histogram([Axis(PredicateBinning(["p", "q"]))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.array([0.0, 0.0]))))
    #     (h + h).checkvalid()
    #     (frombuffer(h.tobuffer()) + h).checkvalid()

    # def test_add_Assignments(self):
    #     h = Histogram([Axis(VariationBinning([Variation([Assignment("x", "1"), Assignment("y", "2"), Assignment("z", "3")])]))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.array([0.0]))))
    #     (h + h).checkvalid()
    #     (frombuffer(h.tobuffer()) + h).checkvalid()

    # def test_add_Variation(self):
    #     h = Histogram([Axis(VariationBinning([Variation([Assignment("x", "1")]), Variation([Assignment("x", "2")])]))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.array([0.0, 0.0]))))
    #     (h + h).checkvalid()
    #     (frombuffer(h.tobuffer()) + h).checkvalid()

    # def test_add_VariationBinning(self):
    #     h = Histogram([Axis(VariationBinning([Variation([Assignment("x", "1")]), Variation([Assignment("x", "2")]), Variation([Assignment("x", "3")])]))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.array([0.0, 0.0, 0.0]))))
    #     (h + h).checkvalid()
    #     (frombuffer(h.tobuffer()) + h).checkvalid()

    # def test_add_Axis(self):
    #     h = BinnedEvaluatedFunction([Axis(expression="x", title="wow")], InterpretedInlineBuffer(numpy.array([0.0]), dtype=InterpretedInlineBuffer.float64))
    #     (h + h).checkvalid()
    #     (frombuffer(h.tobuffer()) + h).checkvalid()

    # def test_add_UnweightedCounts(self):
    #     h = Histogram([Axis(RegularBinning(10, RealInterval(-5, 5)))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(10))))
    #     (h + h).checkvalid()
    #     (frombuffer(h.tobuffer()) + h).checkvalid()

    # def test_add_WeightedCounts(self):
    #     h = Histogram([Axis(RegularBinning(10, RealInterval(-5, 5)))], WeightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(10))))
    #     (h + h).checkvalid()
    #     (frombuffer(h.tobuffer()) + h).checkvalid()
    #     h = Histogram([Axis(RegularBinning(10, RealInterval(-5, 5)))], WeightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(10)), sumw2=InterpretedInlineBuffer.fromarray(numpy.arange(10)**2)))
    #     (h + h).checkvalid()
    #     (frombuffer(h.tobuffer()) + h).checkvalid()
    #     h = Histogram([Axis(RegularBinning(10, RealInterval(-5, 5)))], WeightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(10)), sumw2=InterpretedInlineBuffer.fromarray(numpy.arange(10)**2), unweighted=UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(10)))))
    #     (h + h).checkvalid()
    #     (frombuffer(h.tobuffer()) + h).checkvalid()

    # def test_add_StatisticFilter(self):
    #     h = Histogram([Axis(RegularBinning(10, RealInterval(-5, 5)), statistics=Statistics())], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(10))))
    #     (h + h).checkvalid()
    #     (frombuffer(h.tobuffer()) + h).checkvalid()
    #     h = Histogram([Axis(RegularBinning(10, RealInterval(-5, 5)), statistics=Statistics(moments=[Moments(InterpretedInlineBuffer.fromarray(numpy.array([0.0])), 1, filter=StatisticFilter(excludes_nan=False))]))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(10))))
    #     (h + h).checkvalid()
    #     (frombuffer(h.tobuffer()) + h).checkvalid()

    # def test_add_Moments(self):
    #     h = Histogram([Axis(RegularBinning(10, RealInterval(-5, 5)), statistics=Statistics(moments=[Moments(InterpretedInlineBuffer.fromarray(numpy.array([0.0])), 1), Moments(InterpretedInlineBuffer.fromarray(numpy.array([0.0])), 2)]))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(10))))
    #     (h + h).checkvalid()
    #     (frombuffer(h.tobuffer()) + h).checkvalid()
    #     h = Histogram([Axis(RegularBinning(10, RealInterval(-5, 5)), statistics=Statistics(moments=[Moments(InterpretedInlineBuffer.fromarray(numpy.array([0.0])), 0, weighted=False), Moments(InterpretedInlineBuffer.fromarray(numpy.array([0.0])), 0, weighted=True)]))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(10))))
    #     (h + h).checkvalid()
    #     (frombuffer(h.tobuffer()) + h).checkvalid()

    # def test_add_Extremes(self):
    #     h = Histogram([Axis(RegularBinning(10, RealInterval(-5, 5)), statistics=Statistics(min=Extremes(InterpretedInlineBuffer.fromarray(numpy.array([0.0])))))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(10))))
    #     (h + h).checkvalid()
    #     (frombuffer(h.tobuffer()) + h).checkvalid()
    #     h = Histogram([Axis(RegularBinning(10, RealInterval(-5, 5)), statistics=Statistics(min=Extremes(InterpretedInlineBuffer.fromarray(numpy.array([0.0]))), max=Extremes(InterpretedInlineBuffer.fromarray(numpy.array([0.0])))))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(10))))
    #     (h + h).checkvalid()
    #     (frombuffer(h.tobuffer()) + h).checkvalid()

    # def test_add_Quantiles(self):
    #     h = Histogram([Axis(RegularBinning(10, RealInterval(-5, 5)), statistics=Statistics(quantiles=[Quantiles(InterpretedInlineBuffer.fromarray(numpy.array([0.0])), 0.25), Quantiles(InterpretedInlineBuffer.fromarray(numpy.array([0.0]))), Quantiles(InterpretedInlineBuffer.fromarray(numpy.array([0.0])), 0.75)]))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(10))))
    #     (h + h).checkvalid()
    #     (frombuffer(h.tobuffer()) + h).checkvalid()
    #     h = Histogram([Axis(RegularBinning(10, RealInterval(-5, 5)), statistics=Statistics(quantiles=[Quantiles(InterpretedInlineBuffer.fromarray(numpy.array([0.0])), weighted=False), Quantiles(InterpretedInlineBuffer.fromarray(numpy.array([0.0])), weighted=True)]))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(10))))
    #     (h + h).checkvalid()
    #     (frombuffer(h.tobuffer()) + h).checkvalid()

    # def test_add_Modes(self):
    #     h = Histogram([Axis(RegularBinning(10, RealInterval(-5, 5)), statistics=Statistics(mode=Modes(InterpretedInlineBuffer.fromarray(numpy.array([0.0])))))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(10))))
    #     (h + h).checkvalid()
    #     (frombuffer(h.tobuffer()) + h).checkvalid()

    # def test_add_Statistics(self):
    #     h = Histogram([Axis(RegularBinning(10, RealInterval(-5, 5)), statistics=Statistics()), Axis(RegularBinning(10, RealInterval(-5, 5)), statistics=Statistics())], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(100))))
    #     (h + h).checkvalid()
    #     (frombuffer(h.tobuffer()) + h).checkvalid()
    #     h = Ntuple([Column("one", Column.int32), Column("two", Column.int16)], [NtupleInstance([Chunk([ColumnChunk([Page(RawInlineBuffer(b"\x05\x00\x00\x00"))], [0, 1]), ColumnChunk([Page(RawInlineBuffer(b"\x03\x00"))], [0, 1])])])], column_statistics=[Statistics(moments=[Moments(InterpretedInlineBuffer.fromarray(numpy.array([0.0])), 1), Moments(InterpretedInlineBuffer.fromarray(numpy.array([0.0])), 2)])])
    #     (h + h).checkvalid()
    #     (frombuffer(h.tobuffer()) + h).checkvalid()

    # def test_add_Covariance(self):
    #     h = Histogram([Axis(RegularBinning(10, RealInterval(-5, 5))), Axis(RegularBinning(10, RealInterval(-5, 5)))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(100))), axis_covariances=[Covariance(0, 1, InterpretedInlineBuffer.fromarray(numpy.arange(1)))])
    #     (h + h).checkvalid()
    #     (frombuffer(h.tobuffer()) + h).checkvalid()
    #     h = Histogram([Axis(RegularBinning(10, RealInterval(-5, 5))), Axis(RegularBinning(10, RealInterval(-5, 5))), Axis(RegularBinning(10, RealInterval(-5, 5)))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(1000))), axis_covariances=[Covariance(0, 1, InterpretedInlineBuffer.fromarray(numpy.arange(1))), Covariance(0, 2, InterpretedInlineBuffer.fromarray(numpy.arange(1))), Covariance(1, 2, InterpretedInlineBuffer.fromarray(numpy.arange(1)))])
    #     (h + h).checkvalid()
    #     (frombuffer(h.tobuffer()) + h).checkvalid()
    #     h = Histogram([Axis(RegularBinning(10, RealInterval(-5, 5)))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(10))), [Profile("", Statistics([Moments(InterpretedInlineBuffer.fromarray(numpy.zeros(10)), 1), Moments(InterpretedInlineBuffer.fromarray(numpy.zeros(10)), 2)])), Profile("", Statistics([Moments(InterpretedInlineBuffer.fromarray(numpy.zeros(10)), 1), Moments(InterpretedInlineBuffer.fromarray(numpy.zeros(10)), 2)]))], profile_covariances=[Covariance(0, 1, InterpretedInlineBuffer.fromarray(numpy.arange(1)))])
    #     (h + h).checkvalid()
    #     (frombuffer(h.tobuffer()) + h).checkvalid()
    #     h = Ntuple([Column("one", Column.int32), Column("two", Column.int16)], [NtupleInstance([Chunk([ColumnChunk([Page(RawInlineBuffer(b"\x05\x00\x00\x00"))], [0, 1]), ColumnChunk([Page(RawInlineBuffer(b"\x03\x00"))], [0, 1])])])], column_covariances=[Covariance(0, 1, InterpretedInlineBuffer.fromarray(numpy.arange(1)))])
    #     (h + h).checkvalid()
    #     (frombuffer(h.tobuffer()) + h).checkvalid()
    #     h = Ntuple([Column("one", Column.int32), Column("two", Column.int16)], [NtupleInstance([Chunk([ColumnChunk([Page(RawInlineBuffer(b"\x05\x00\x00\x00"))], [0, 1]), ColumnChunk([Page(RawInlineBuffer(b"\x03\x00"))], [0, 1])])])], column_covariances=[Covariance(0, 1, InterpretedInlineBuffer.fromarray(numpy.arange(1)), weighted=True), Covariance(0, 1, InterpretedInlineBuffer.fromarray(numpy.arange(1)), weighted=False)])
    #     (h + h).checkvalid()
    #     (frombuffer(h.tobuffer()) + h).checkvalid()

    # def test_add_Profile(self):
    #     h = Histogram([Axis(RegularBinning(10, RealInterval(-5, 5)))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(10))), [Profile("", Statistics([Moments(InterpretedInlineBuffer.fromarray(numpy.zeros(10)), 1), Moments(InterpretedInlineBuffer.fromarray(numpy.zeros(10)), 2)]))])
    #     (h + h).checkvalid()
    #     (frombuffer(h.tobuffer()) + h).checkvalid()
    #     h = Histogram([Axis(RegularBinning(10, RealInterval(-5, 5))), Axis(RegularBinning(10, RealInterval(-5, 5)))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(100))), [Profile("", Statistics([Moments(InterpretedInlineBuffer.fromarray(numpy.zeros(100)), 1), Moments(InterpretedInlineBuffer.fromarray(numpy.zeros(100)), 2)]))])
    #     (h + h).checkvalid()
    #     (frombuffer(h.tobuffer()) + h).checkvalid()

    # def test_add_Histogram(self):
    #     h = Histogram([Axis()], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(1))))
    #     (h + h).checkvalid()
    #     (frombuffer(h.tobuffer()) + h).checkvalid()

    # def test_add_Parameter(self):
    #     h = ParameterizedFunction("x**2", [Parameter("x", InterpretedInlineBuffer.fromarray(numpy.array([5]))), Parameter("y", InterpretedInlineBuffer.fromarray(numpy.array([6])))])
    #     (h + h).checkvalid()
    #     (frombuffer(h.tobuffer()) + h).checkvalid()

    # def test_add_ParameterizedFunction(self):
    #     h = ParameterizedFunction("x**2")
    #     (h + h).checkvalid()
    #     (frombuffer(h.tobuffer()) + h).checkvalid()
    #     h = Histogram([Axis(RegularBinning(10, RealInterval(-5, 5))), Axis(RegularBinning(10, RealInterval(-5, 5)))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(100))), functions={"f": ParameterizedFunction("x**2", [Parameter("x", InterpretedInlineBuffer.fromarray(numpy.arange(100)))])})
    #     (h + h).checkvalid()
    #     (frombuffer(h.tobuffer()) + h).checkvalid()

    # def test_add_EvaluatedFunction(self):
    #     h = Histogram([Axis(RegularBinning(10, RealInterval(-5, 5))), Axis(RegularBinning(10, RealInterval(-5, 5)))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(100))), functions={"f": EvaluatedFunction(InterpretedInlineBuffer.fromarray(numpy.arange(100)))})
    #     (h + h).checkvalid()
    #     (frombuffer(h.tobuffer()) + h).checkvalid()
    #     h = Histogram([Axis(RegularBinning(10, RealInterval(-5, 5))), Axis(RegularBinning(10, RealInterval(-5, 5)))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(100))), functions={"id": EvaluatedFunction(InterpretedInlineBuffer.fromarray(numpy.arange(100)), InterpretedInlineBuffer.fromarray(numpy.arange(100)))})
    #     (h + h).checkvalid()
    #     (frombuffer(h.tobuffer()) + h).checkvalid()
    #     h = Histogram([Axis(RegularBinning(10, RealInterval(-5, 5))), Axis(RegularBinning(10, RealInterval(-5, 5)))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(100))), functions={"id": EvaluatedFunction(InterpretedInlineBuffer.fromarray(numpy.arange(100)), InterpretedInlineBuffer.fromarray(numpy.arange(100)), [Quantiles(InterpretedInlineBuffer.fromarray(numpy.zeros(100)), 0.25), Quantiles(InterpretedInlineBuffer.fromarray(numpy.zeros(100))), Quantiles(InterpretedInlineBuffer.fromarray(numpy.zeros(100)), 0.75)])})
    #     (h + h).checkvalid()
    #     (frombuffer(h.tobuffer()) + h).checkvalid()

    # def test_add_BinnedEvaluatedFunction(self):
    #     h = BinnedEvaluatedFunction([Axis()], InterpretedInlineBuffer(numpy.array([0.0]), dtype=InterpretedInlineBuffer.float64))
    #     (h + h).checkvalid()
    #     (frombuffer(h.tobuffer()) + h).checkvalid()
    #     h = BinnedEvaluatedFunction([Axis(), Axis()], InterpretedInlineBuffer(numpy.array([0.0]), dtype=InterpretedInlineBuffer.float64))
    #     (h + h).checkvalid()
    #     (frombuffer(h.tobuffer()) + h).checkvalid()

    # def test_add_Page(self):
    #     h = Ntuple([Column("one", Column.int32)], [NtupleInstance([Chunk([ColumnChunk([Page(RawInlineBuffer(b"\x05\x00\x00\x00"))], [0, 1])])])])
    #     (h + h).checkvalid()
    #     (frombuffer(h.tobuffer()) + h).checkvalid()
    #     h = Ntuple([Column("one", Column.int32)], [NtupleInstance([Chunk([ColumnChunk([], [0])])])])
    #     (h + h).checkvalid()
    #     (frombuffer(h.tobuffer()) + h).checkvalid()
    #     h = Ntuple([Column("one", Column.int32)], [NtupleInstance([Chunk([ColumnChunk([Page(RawInlineBuffer(b"\x05\x00\x00\x00"))], [0, 1])])])])
    #     (h + h).checkvalid()
    #     (frombuffer(h.tobuffer()) + h).checkvalid()
    #     h = Ntuple([Column("one", Column.int32)], [NtupleInstance([Chunk([ColumnChunk([Page(RawInlineBuffer(b"\x05\x00\x00\x00")), Page(RawInlineBuffer(b"\x04\x00\x00\x00\x03\x00\x00\x00"))], [0, 1, 3])])])])
    #     (h + h).checkvalid()
    #     (frombuffer(h.tobuffer()) + h).checkvalid()

    # def test_add_Chunk(self):
    #     h = Ntuple([Column("one", Column.float64)], [NtupleInstance([Chunk([ColumnChunk([], [0])])])])
    #     (h + h).checkvalid()
    #     (frombuffer(h.tobuffer()) + h).checkvalid()
    #     h = Ntuple([Column("one", Column.int32)], [NtupleInstance([])])
    #     (h + h).checkvalid()
    #     (frombuffer(h.tobuffer()) + h).checkvalid()
    #     h = Ntuple([Column("one", Column.int32)], [NtupleInstance([Chunk([ColumnChunk([Page(RawInlineBuffer(b"\x05\x00\x00\x00"))], [0, 1])])])])
    #     (h + h).checkvalid()
    #     (frombuffer(h.tobuffer()) + h).checkvalid()
    #     h = Ntuple([Column("one", Column.int32)], [NtupleInstance([Chunk([ColumnChunk([Page(RawInlineBuffer(b"\x05\x00\x00\x00"))], [0, 1])]), Chunk([ColumnChunk([Page(RawInlineBuffer(b"\x05\x00\x00\x00"))], [0, 1])])])])
    #     (h + h).checkvalid()
    #     (frombuffer(h.tobuffer()) + h).checkvalid()

    # def test_add_Column(self):
    #     h = Ntuple([Column("one", Column.float64), Column("two", Column.int32)], [NtupleInstance([])])
    #     (h + h).checkvalid()
    #     (frombuffer(h.tobuffer()) + h).checkvalid()
    #     h = Ntuple([Column("one", Column.int32), Column("two", Column.int16)], [NtupleInstance([Chunk([ColumnChunk([Page(RawInlineBuffer(b"\x05\x00\x00\x00"))], [0, 1]), ColumnChunk([Page(RawInlineBuffer(b"\x03\x00"))], [0, 1])])])])
    #     (h + h).checkvalid()
    #     (frombuffer(h.tobuffer()) + h).checkvalid()

    # def test_add_Ntuple(self):
    #     h = Ntuple([Column("one", Column.float64)], [NtupleInstance([])])
    #     (h + h).checkvalid()
    #     (frombuffer(h.tobuffer()) + h).checkvalid()

    # def test_add_collection(self):
    #     h = Collection()
    #     (h + h).checkvalid()
    #     (frombuffer(h.tobuffer()) + h).checkvalid()
    #     h = Collection({"b": Collection(), "c": Collection()})
    #     (h + h).checkvalid()
    #     (frombuffer(h.tobuffer()) + h).checkvalid()
    #     h = Collection({"id": Histogram([Axis(RegularBinning(10, RealInterval(-5, 5)))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(30)))), "id2": Histogram([Axis(RegularBinning(100, RealInterval(-5, 5)))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(300))))}, axis=[Axis(RegularBinning(3, RealInterval(-1, 1)))])
    #     (h + h).checkvalid()
    #     (frombuffer(h.tobuffer()) + h).checkvalid()
    #     h = Collection({"b": Collection({"c": Histogram([Axis(RegularBinning(10, RealInterval(-5, 5)))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(60)))), "d": Histogram([Axis(RegularBinning(100, RealInterval(-5, 5)))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(600))))}, axis=[Axis(FractionBinning())])}, axis=[Axis(RegularBinning(3, RealInterval(-1, 1)))])
    #     (h + h).checkvalid()
    #     (frombuffer(h.tobuffer()) + h).checkvalid()
