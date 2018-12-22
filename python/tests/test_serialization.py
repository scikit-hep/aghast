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

from portally import *

class Test(unittest.TestCase):
    def runTest(self):
        pass

    def test_Metadata(self):
        h = Collection({}, metadata=Metadata("""{"one": 1, "two": 2}""", language=Metadata.json))
        assert h == frombuffer(h.tobuffer())

    def test_Decoration(self):
        h = Collection({}, decoration=Decoration("""points { color: red }""", language=Decoration.css))
        assert h == frombuffer(h.tobuffer())

    def test_RawInlineBuffer(self):
        pass
    #     h = Ntuple("id", [Column("one", Column.int32)], [NtupleInstance([Chunk([ColumnChunk([Page(RawInlineBuffer(numpy.zeros(1, dtype=numpy.int32)))], [0, 1])])])])
    #     h.checkvalid()
    #     assert len(h.instances[0].chunks[0].column_chunks[0].pages[0].array) == 1

    #     h = Ntuple("id", [Column("one", Column.int32)], [NtupleInstance([Chunk([ColumnChunk([Page(RawInlineBuffer(b"\x05\x00\x00\x00"))], [0, 1])])])])
    #     h.checkvalid()
    #     assert h.instances[0].chunks[0].column_chunks[0].pages[0].array.tolist() == [5]

    def test_RawExternalBuffer(self):
        pass
    #     buf = numpy.zeros(1, dtype=numpy.int32)
    #     h = Ntuple("id", [Column("one", Column.int32)], [NtupleInstance([Chunk([ColumnChunk([Page(RawExternalBuffer(buf.ctypes.data, buf.nbytes))], [0, 1])])])])
    #     h.checkvalid()
    #     assert len(h.instances[0].chunks[0].column_chunks[0].pages[0].array) == 1

    #     buf = numpy.array([3.14], dtype=numpy.float64)
    #     h = Ntuple("id", [Column("one", Column.float64)], [NtupleInstance([Chunk([ColumnChunk([Page(RawExternalBuffer(buf.ctypes.data, buf.nbytes))], [0, 1])])])])
    #     h.checkvalid()
    #     assert h.instances[0].chunks[0].column_chunks[0].pages[0].array.tolist() == [3.14]

    def test_InterpretedInlineBuffer(self):
        pass
    #     h = BinnedEvaluatedFunction("id", [Axis()], InterpretedInlineBuffer(numpy.zeros(1, dtype=numpy.int32), dtype=InterpretedInlineBuffer.int32))
    #     h.checkvalid()
    #     assert h.values.array.tolist() == [0]

    #     h = BinnedEvaluatedFunction("id", [Axis()], InterpretedInlineBuffer(b"\x07\x00\x00\x00", dtype=InterpretedInlineBuffer.int32))
    #     h.checkvalid()
    #     assert h.values.array.tolist() == [7]

    def test_InterpretedExternalBuffer(self):
        pass
    #     buf = numpy.zeros(1, dtype=numpy.float64)
    #     h = BinnedEvaluatedFunction("id", [Axis()], InterpretedExternalBuffer(buf.ctypes.data, buf.nbytes, dtype=InterpretedInlineBuffer.float64))
    #     h.checkvalid()
    #     assert h.values.array.tolist() == [0.0]

    #     buf = numpy.array([3.14], dtype=numpy.float64)
    #     h = BinnedEvaluatedFunction("id", [Axis()], InterpretedExternalBuffer(buf.ctypes.data, buf.nbytes, dtype=InterpretedInlineBuffer.float64))
    #     h.checkvalid()
    #     assert h.values.array.tolist() == [3.14]

    def test_IntegerBinning(self):
        pass
    #     h = BinnedEvaluatedFunction("id", [Axis(IntegerBinning(10, 20))], InterpretedInlineBuffer(numpy.zeros(11), dtype=InterpretedInlineBuffer.float64))
    #     h.checkvalid()
    #     h = BinnedEvaluatedFunction("id", [Axis(IntegerBinning(20, 10))], InterpretedInlineBuffer(numpy.zeros(11), dtype=InterpretedInlineBuffer.float64))
    #     assert not h.isvalid
    #     h = BinnedEvaluatedFunction("id", [Axis(IntegerBinning(10, 20, pos_underflow=IntegerBinning.nonexistent, pos_overflow=IntegerBinning.nonexistent))], InterpretedInlineBuffer(numpy.zeros(11), dtype=InterpretedInlineBuffer.float64))
    #     h.checkvalid()
    #     assert h.values.array.tolist() == [0.0] * 11
    #     h = BinnedEvaluatedFunction("id", [Axis(IntegerBinning(10, 20, pos_underflow=IntegerBinning.below1, pos_overflow=IntegerBinning.nonexistent))], InterpretedInlineBuffer(numpy.zeros(12), dtype=InterpretedInlineBuffer.float64))
    #     h.checkvalid()
    #     assert h.values.array.tolist() == [0.0] * 12
    #     h = BinnedEvaluatedFunction("id", [Axis(IntegerBinning(10, 20, pos_underflow=IntegerBinning.nonexistent, pos_overflow=IntegerBinning.above1))], InterpretedInlineBuffer(numpy.zeros(12), dtype=InterpretedInlineBuffer.float64))
    #     h.checkvalid()
    #     assert h.values.array.tolist() == [0.0] * 12
    #     h = BinnedEvaluatedFunction("id", [Axis(IntegerBinning(10, 20, pos_underflow=IntegerBinning.below1, pos_overflow=IntegerBinning.above1))], InterpretedInlineBuffer(numpy.zeros(13), dtype=InterpretedInlineBuffer.float64))
    #     h.checkvalid()
    #     assert h.values.array.tolist() == [0.0] * 13

    def test_RealInterval(self):
        pass
    #     h = BinnedEvaluatedFunction("id", [Axis(RegularBinning(10, RealInterval(-5, 5)))], InterpretedInlineBuffer(numpy.zeros(10), dtype=InterpretedInlineBuffer.float64))
    #     h.checkvalid()
    #     h = BinnedEvaluatedFunction("id", [Axis(RegularBinning(10, RealInterval(5, -5)))], InterpretedInlineBuffer(numpy.zeros(10), dtype=InterpretedInlineBuffer.float64))
    #     assert not h.isvalid

    def test_RealOverflow(self):
        pass
    #     h = BinnedEvaluatedFunction("id", [Axis(RegularBinning(10, RealInterval(-5, 5), RealOverflow(pos_underflow=RealOverflow.nonexistent, pos_overflow=RealOverflow.nonexistent, pos_nanflow=RealOverflow.nonexistent)))], InterpretedInlineBuffer(numpy.zeros(10), dtype=InterpretedInlineBuffer.float64))
    #     h.checkvalid()
    #     assert h.values.array.tolist() == [0.0] * 10
    #     h = BinnedEvaluatedFunction("id", [Axis(RegularBinning(10, RealInterval(-5, 5), RealOverflow(pos_underflow=RealOverflow.above1, pos_overflow=RealOverflow.nonexistent, pos_nanflow=RealOverflow.nonexistent)))], InterpretedInlineBuffer(numpy.zeros(11), dtype=InterpretedInlineBuffer.float64))
    #     h.checkvalid()
    #     assert h.values.array.tolist() == [0.0] * 11
    #     h = BinnedEvaluatedFunction("id", [Axis(RegularBinning(10, RealInterval(-5, 5), RealOverflow(pos_underflow=RealOverflow.nonexistent, pos_overflow=RealOverflow.above1, pos_nanflow=RealOverflow.nonexistent)))], InterpretedInlineBuffer(numpy.zeros(11), dtype=InterpretedInlineBuffer.float64))
    #     h.checkvalid()
    #     assert h.values.array.tolist() == [0.0] * 11
    #     h = BinnedEvaluatedFunction("id", [Axis(RegularBinning(10, RealInterval(-5, 5), RealOverflow(pos_underflow=RealOverflow.nonexistent, pos_overflow=RealOverflow.nonexistent, pos_nanflow=RealOverflow.above1)))], InterpretedInlineBuffer(numpy.zeros(11), dtype=InterpretedInlineBuffer.float64))
    #     h.checkvalid()
    #     assert h.values.array.tolist() == [0.0] * 11
    #     h = BinnedEvaluatedFunction("id", [Axis(RegularBinning(10, RealInterval(-5, 5), RealOverflow(pos_underflow=RealOverflow.above1, pos_overflow=RealOverflow.nonexistent, pos_nanflow=RealOverflow.above2)))], InterpretedInlineBuffer(numpy.zeros(12), dtype=InterpretedInlineBuffer.float64))
    #     h.checkvalid()
    #     assert h.values.array.tolist() == [0.0] * 12
    #     h = BinnedEvaluatedFunction("id", [Axis(RegularBinning(10, RealInterval(-5, 5), RealOverflow(pos_underflow=RealOverflow.above1, pos_overflow=RealOverflow.above2, pos_nanflow=RealOverflow.above3)))], InterpretedInlineBuffer(numpy.zeros(13), dtype=InterpretedInlineBuffer.float64))
    #     h.checkvalid()
    #     assert h.values.array.tolist() == [0.0] * 13

    def test_RegularBinning(self):
        pass
    #     h = BinnedEvaluatedFunction("id", [Axis(RegularBinning(10, RealInterval(-5, 5)))], InterpretedInlineBuffer(numpy.zeros(10), dtype=InterpretedInlineBuffer.float64))
    #     h.checkvalid()

    def test_TicTacToeOverflowBinning(self):
        pass
    #     h = BinnedEvaluatedFunction("id", [Axis(TicTacToeOverflowBinning(2, 2, RealInterval(-10, 10), RealInterval(-10, 10), RealOverflow(RealOverflow.above1, RealOverflow.above2, RealOverflow.above3), RealOverflow(RealOverflow.above1, RealOverflow.above2, RealOverflow.above3)))], InterpretedInlineBuffer(numpy.array([[0.0, 0.0, 0.0, 0.0, 0.0]] * 5), dtype=InterpretedInlineBuffer.float64))
    #     h.checkvalid()
    #     assert h.values.array.tolist() == [[0.0, 0.0, 0.0, 0.0, 0.0]] * 5
    #     h = BinnedEvaluatedFunction("id", [Axis(TicTacToeOverflowBinning(2, 2, RealInterval(-10, 10), RealInterval(-10, 10)))], InterpretedInlineBuffer(numpy.array([[0.0, 0.0], [0.0, 0.0]]), dtype=InterpretedInlineBuffer.float64))
    #     h.checkvalid()
    #     assert h.values.array.tolist() == [[0.0, 0.0], [0.0, 0.0]]

    def test_HexagonalBinning(self):
        pass
    #     h = BinnedEvaluatedFunction("id", [Axis(HexagonalBinning(3, 5, -5, -4))], InterpretedInlineBuffer(numpy.array([[0.0] * 2] * 3), dtype=InterpretedInlineBuffer.float64))
    #     h.checkvalid()
    #     assert h.values.array.tolist() == [[0.0] * 2] * 3
    #     h = BinnedEvaluatedFunction("id", [Axis(HexagonalBinning(3, 5, -5, -4, qoverflow=RealOverflow(pos_nanflow=RealOverflow.above1)))], InterpretedInlineBuffer(numpy.array([[0.0] * 2] * 4), dtype=InterpretedInlineBuffer.float64))
    #     h.checkvalid()
    #     assert h.values.array.tolist() == [[0.0] * 2] * 4
    #     h = BinnedEvaluatedFunction("id", [Axis(HexagonalBinning(3, 5, -5, -4, roverflow=RealOverflow(pos_nanflow=RealOverflow.above1)))], InterpretedInlineBuffer(numpy.array([[0.0] * 3] * 3), dtype=InterpretedInlineBuffer.float64))
    #     h.checkvalid()
    #     assert h.values.array.tolist() == [[0.0] * 3] * 3
    #     h = BinnedEvaluatedFunction("id", [Axis(HexagonalBinning(3, 5, -5, -4, qoverflow=RealOverflow(pos_nanflow=RealOverflow.above1), roverflow=RealOverflow(pos_nanflow=RealOverflow.above1)))], InterpretedInlineBuffer(numpy.array([[0.0] * 3] * 4), dtype=InterpretedInlineBuffer.float64))
    #     h.checkvalid()
    #     assert h.values.array.tolist() == [[0.0] * 3] * 4

    def test_EdgesBinning(self):
        pass
    #     h = BinnedEvaluatedFunction("id", [Axis(EdgesBinning([3.3]))], InterpretedInlineBuffer(numpy.array([]), dtype=InterpretedInlineBuffer.float64))
    #     h.checkvalid()
    #     assert h.values.array.tolist() == []
    #     h = BinnedEvaluatedFunction("id", [Axis(EdgesBinning([1.1, 2.2, 3.3]))], InterpretedInlineBuffer(numpy.array([0.0, 0.0]), dtype=InterpretedInlineBuffer.float64))
    #     h.checkvalid()
    #     assert h.values.array.tolist() == [0.0, 0.0]

    def test_IrregularBinning(self):
        pass
    #     h = BinnedEvaluatedFunction("id", [Axis(IrregularBinning([RealInterval(0.5, 1.5)]))], InterpretedInlineBuffer(numpy.array([0.0]), dtype=InterpretedInlineBuffer.float64))
    #     h.checkvalid()
    #     assert h.values.array.tolist() == [0.0]
    #     h = BinnedEvaluatedFunction("id", [Axis(IrregularBinning([RealInterval(0.5, 1.5), RealInterval(1.5, 1.5), RealInterval(0.0, 10.0)]))], InterpretedInlineBuffer(numpy.array([0.0, 0.0, 0.0]), dtype=InterpretedInlineBuffer.float64))
    #     h.checkvalid()
    #     assert h.values.array.tolist() == [0.0, 0.0, 0.0]

    def test_CategoryBinning(self):
        pass
    #     h = BinnedEvaluatedFunction("id", [Axis(CategoryBinning(["one", "two", "three"]))], InterpretedInlineBuffer(numpy.array([0.0, 0.0, 0.0]), dtype=InterpretedInlineBuffer.float64))
    #     h.checkvalid()
    #     assert h.values.array.tolist() == [0.0, 0.0, 0.0]
    #     h = BinnedEvaluatedFunction("id", [Axis(CategoryBinning(["one", "two", "three"], pos_overflow=CategoryBinning.above1))], InterpretedInlineBuffer(numpy.array([0.0, 0.0, 0.0, 0.0]), dtype=InterpretedInlineBuffer.float64))
    #     h.checkvalid()
    #     assert h.values.array.tolist() == [0.0, 0.0, 0.0, 0.0]

    def test_SparseRegularBinning(self):
        pass
    #     h = BinnedEvaluatedFunction("id", [Axis(SparseRegularBinning([-5, -3, 10, 1000], 0.1))], InterpretedInlineBuffer(numpy.array([0.0, 0.0, 0.0, 0.0]), dtype=InterpretedInlineBuffer.float64))
    #     h.checkvalid()
    #     assert h.values.array.tolist() == [0.0, 0.0, 0.0, 0.0]
    #     h = BinnedEvaluatedFunction("id", [Axis(SparseRegularBinning([-5, -3, 10, 1000], 0.1, pos_nanflow=SparseRegularBinning.above1))], InterpretedInlineBuffer(numpy.array([0.0, 0.0, 0.0, 0.0, 0.0]), dtype=InterpretedInlineBuffer.float64))
    #     h.checkvalid()
    #     assert h.values.array.tolist() == [0.0, 0.0, 0.0, 0.0, 0.0]

    def test_FractionBinning(self):
        pass
    #     h = BinnedEvaluatedFunction("id", [Axis(FractionBinning())], InterpretedInlineBuffer(numpy.array([0.0, 0.0]), dtype=InterpretedInlineBuffer.float64))
    #     h.checkvalid()
    #     assert h.values.array.tolist() == [0.0, 0.0]
    #     assert h.axis[0].binning.error_method == FractionBinning.normal
    #     h = BinnedEvaluatedFunction("id", [Axis(FractionBinning()), Axis(RegularBinning(10, RealInterval(-5, 5)))], InterpretedInlineBuffer(numpy.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]), dtype=InterpretedInlineBuffer.float64))
    #     h.checkvalid()
    #     assert h.values.array.tolist() == [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    #     h = BinnedEvaluatedFunction("id", [Axis(RegularBinning(10, RealInterval(-5, 5))), Axis(FractionBinning())], InterpretedInlineBuffer(numpy.array([[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]]), dtype=InterpretedInlineBuffer.float64))
    #     h.checkvalid()
    #     assert h.values.array.tolist() == [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]

    def test_PredicateBinning(self):
        pass
    #     h = Histogram("id", [Axis(PredicateBinning(["p", "q"]))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.array([0.0, 0.0]))))
    #     h.checkvalid()

    def test_Assignments(self):
        pass
    #     h = Histogram("id", [Axis(VariationBinning([Variation([Assignment("x", "1"), Assignment("y", "2"), Assignment("z", "3")])]))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.array([0.0]))))
    #     h.checkvalid()
    #     assert h.axis[0].binning.variations[0]["y"].expression == "2"

    def test_Variation(self):
        pass
    #     h = Histogram("id", [Axis(VariationBinning([Variation([Assignment("x", "1")]), Variation([Assignment("x", "2")])]))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.array([0.0, 0.0]))))
    #     h.checkvalid()

    def test_VariationBinning(self):
        pass
    #     h = Histogram("id", [Axis(VariationBinning([Variation([Assignment("x", "1")]), Variation([Assignment("x", "2")]), Variation([Assignment("x", "3")])]))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.array([0.0, 0.0, 0.0]))))
    #     h.checkvalid()

    def test_Axis(self):
        pass
    #     h = BinnedEvaluatedFunction("id", [Axis(expression="x", title="wow")], InterpretedInlineBuffer(numpy.array([0.0]), dtype=InterpretedInlineBuffer.float64))
    #     h.checkvalid()
    #     assert h.axis[0].expression == "x"
    #     assert h.values.array.tolist() == [0.0]

    def test_UnweightedCounts(self):
        pass
    #     h = Histogram("id", [Axis(RegularBinning(10, RealInterval(-5, 5)))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(10))))
    #     h.checkvalid()

    def test_WeightedCounts(self):
        pass
    #     h = Histogram("id", [Axis(RegularBinning(10, RealInterval(-5, 5)))], WeightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(10))))
    #     h.checkvalid()
    #     h = Histogram("id", [Axis(RegularBinning(10, RealInterval(-5, 5)))], WeightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(10)), sumw2=InterpretedInlineBuffer.fromarray(numpy.arange(10)**2)))
    #     h.checkvalid()
    #     h = Histogram("id", [Axis(RegularBinning(10, RealInterval(-5, 5)))], WeightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(10)), sumw2=InterpretedInlineBuffer.fromarray(numpy.arange(10)**2), unweighted=UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(10)))))
    #     h.checkvalid()

    def test_StatisticFilter(self):
        pass
    #     h = Histogram("id", [Axis(RegularBinning(10, RealInterval(-5, 5)), statistics=Statistics(moments=[Moments(InterpretedInlineBuffer.fromarray(numpy.array([0.0])), 1, filter=StatisticFilter(excludes_nan=False))]))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(10))))
    #     h.checkvalid()

    def test_Moments(self):
        pass
    #     h = Histogram("id", [Axis(RegularBinning(10, RealInterval(-5, 5)), statistics=Statistics(moments=[Moments(InterpretedInlineBuffer.fromarray(numpy.array([0.0])), 1), Moments(InterpretedInlineBuffer.fromarray(numpy.array([0.0])), 2)]))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(10))))
    #     h.checkvalid()
    #     h = Histogram("id", [Axis(RegularBinning(10, RealInterval(-5, 5)), statistics=Statistics(moments=[Moments(InterpretedInlineBuffer.fromarray(numpy.array([0.0])), 0, weighted=False), Moments(InterpretedInlineBuffer.fromarray(numpy.array([0.0])), 0, weighted=True)]))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(10))))
    #     h.checkvalid()

    def test_Extremes(self):
        pass
    #     h = Histogram("id", [Axis(RegularBinning(10, RealInterval(-5, 5)), statistics=Statistics(minima=Extremes(InterpretedInlineBuffer.fromarray(numpy.array([0.0])))))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(10))))
    #     h.checkvalid()
    #     h = Histogram("id", [Axis(RegularBinning(10, RealInterval(-5, 5)), statistics=Statistics(minima=Extremes(InterpretedInlineBuffer.fromarray(numpy.array([0.0]))), maxima=Extremes(InterpretedInlineBuffer.fromarray(numpy.array([0.0])))))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(10))))
    #     h.checkvalid()

    def test_Quantiles(self):
        pass
    #     h = Histogram("id", [Axis(RegularBinning(10, RealInterval(-5, 5)), statistics=Statistics(quantiles=[Quantiles(InterpretedInlineBuffer.fromarray(numpy.array([0.0])), 0.25), Quantiles(InterpretedInlineBuffer.fromarray(numpy.array([0.0]))), Quantiles(InterpretedInlineBuffer.fromarray(numpy.array([0.0])), 0.75)]))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(10))))
    #     h.checkvalid()
    #     h = Histogram("id", [Axis(RegularBinning(10, RealInterval(-5, 5)), statistics=Statistics(quantiles=[Quantiles(InterpretedInlineBuffer.fromarray(numpy.array([0.0])), weighted=False), Quantiles(InterpretedInlineBuffer.fromarray(numpy.array([0.0])), weighted=True)]))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(10))))
    #     h.checkvalid()
        
    def test_Modes(self):
        pass
    #     h = Histogram("id", [Axis(RegularBinning(10, RealInterval(-5, 5)), statistics=Statistics(modes=Modes(InterpretedInlineBuffer.fromarray(numpy.array([0.0])))))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(10))))
    #     h.checkvalid()

    def test_Statistics(self):
        pass
    #     h = Histogram("id", [Axis(RegularBinning(10, RealInterval(-5, 5)), statistics=Statistics()), Axis(RegularBinning(10, RealInterval(-5, 5)), statistics=Statistics())], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(100))))
    #     h.checkvalid()
    #     h = Ntuple("id", [Column("one", Column.int32), Column("two", Column.int16)], [NtupleInstance([Chunk([ColumnChunk([Page(RawInlineBuffer(b"\x05\x00\x00\x00"))], [0, 1]), ColumnChunk([Page(RawInlineBuffer(b"\x03\x00"))], [0, 1])])])], column_statistics=[Statistics(moments=[Moments(InterpretedInlineBuffer.fromarray(numpy.array([0.0])), 1), Moments(InterpretedInlineBuffer.fromarray(numpy.array([0.0])), 2)])])
    #     h.checkvalid()

    def test_Correlations(self):
        pass
    #     h = Histogram("id", [Axis(RegularBinning(10, RealInterval(-5, 5))), Axis(RegularBinning(10, RealInterval(-5, 5)))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(100))), axis_correlations=[Correlations(0, 1, InterpretedInlineBuffer.fromarray(numpy.arange(1)))])
    #     h.checkvalid()
    #     h = Histogram("id", [Axis(RegularBinning(10, RealInterval(-5, 5))), Axis(RegularBinning(10, RealInterval(-5, 5))), Axis(RegularBinning(10, RealInterval(-5, 5)))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(1000))), axis_correlations=[Correlations(0, 1, InterpretedInlineBuffer.fromarray(numpy.arange(1))), Correlations(0, 2, InterpretedInlineBuffer.fromarray(numpy.arange(1))), Correlations(1, 2, InterpretedInlineBuffer.fromarray(numpy.arange(1)))])
    #     h.checkvalid()
    #     h = Histogram("id", [Axis(RegularBinning(10, RealInterval(-5, 5)))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(10))), [Profile("", Statistics([Moments(InterpretedInlineBuffer.fromarray(numpy.zeros(10)), 1), Moments(InterpretedInlineBuffer.fromarray(numpy.zeros(10)), 2)])), Profile("", Statistics([Moments(InterpretedInlineBuffer.fromarray(numpy.zeros(10)), 1), Moments(InterpretedInlineBuffer.fromarray(numpy.zeros(10)), 2)]))], profile_correlations=[Correlations(0, 1, InterpretedInlineBuffer.fromarray(numpy.arange(1)))])
    #     h.checkvalid()
    #     h = Ntuple("id", [Column("one", Column.int32), Column("two", Column.int16)], [NtupleInstance([Chunk([ColumnChunk([Page(RawInlineBuffer(b"\x05\x00\x00\x00"))], [0, 1]), ColumnChunk([Page(RawInlineBuffer(b"\x03\x00"))], [0, 1])])])], column_correlations=[Correlations(0, 1, InterpretedInlineBuffer.fromarray(numpy.arange(1)))])
    #     h.checkvalid()
    #     h = Ntuple("id", [Column("one", Column.int32), Column("two", Column.int16)], [NtupleInstance([Chunk([ColumnChunk([Page(RawInlineBuffer(b"\x05\x00\x00\x00"))], [0, 1]), ColumnChunk([Page(RawInlineBuffer(b"\x03\x00"))], [0, 1])])])], column_correlations=[Correlations(0, 1, InterpretedInlineBuffer.fromarray(numpy.arange(1)), weighted=True), Correlations(0, 1, InterpretedInlineBuffer.fromarray(numpy.arange(1)), weighted=False)])
    #     h.checkvalid()

    def test_Profile(self):
        pass
    #     h = Histogram("id", [Axis(RegularBinning(10, RealInterval(-5, 5)))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(10))), [Profile("", Statistics([Moments(InterpretedInlineBuffer.fromarray(numpy.zeros(10)), 1), Moments(InterpretedInlineBuffer.fromarray(numpy.zeros(10)), 2)]))])
    #     h.checkvalid()
    #     h = Histogram("id", [Axis(RegularBinning(10, RealInterval(-5, 5))), Axis(RegularBinning(10, RealInterval(-5, 5)))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(100))), [Profile("", Statistics([Moments(InterpretedInlineBuffer.fromarray(numpy.zeros(100)), 1), Moments(InterpretedInlineBuffer.fromarray(numpy.zeros(100)), 2)]))])
    #     h.checkvalid()
        
    def test_Histogram(self):
        pass
    #     h = Histogram("id", [Axis(RegularBinning(10, RealInterval(-5, 5))), Axis(RegularBinning(10, RealInterval(-5, 5)))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(100))))
    #     h.checkvalid()

    def test_Parameter(self):
        pass
    #     h = ParameterizedFunction("id", "x**2", [Parameter("x", InterpretedInlineBuffer.fromarray(numpy.array([5]))), Parameter("y", InterpretedInlineBuffer.fromarray(numpy.array([6])))])
    #     h.checkvalid()
    #     assert h["y"].values.array.tolist() == [6]

    def test_ParameterizedFunction(self):
        pass
    #     h = ParameterizedFunction("id", "x**2")
    #     h.checkvalid()
    #     h = Histogram("id", [Axis(RegularBinning(10, RealInterval(-5, 5))), Axis(RegularBinning(10, RealInterval(-5, 5)))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(100))), functions=[ParameterizedFunction("id", "x**2", [Parameter("x", InterpretedInlineBuffer.fromarray(numpy.arange(100)))])])
    #     h.checkvalid()

    def test_EvaluatedFunction(self):
        pass
    #     h = Histogram("id", [Axis(RegularBinning(10, RealInterval(-5, 5))), Axis(RegularBinning(10, RealInterval(-5, 5)))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(100))), functions=[EvaluatedFunction("f", InterpretedInlineBuffer.fromarray(numpy.arange(100)))])
    #     h.checkvalid()
    #     assert h["f"].values.array.tolist() == numpy.arange(100).reshape((10, 10)).tolist()
    #     h = Histogram("id", [Axis(RegularBinning(10, RealInterval(-5, 5))), Axis(RegularBinning(10, RealInterval(-5, 5)))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(100))), functions=[EvaluatedFunction("id", InterpretedInlineBuffer.fromarray(numpy.arange(100)), InterpretedInlineBuffer.fromarray(numpy.arange(100)))])
    #     h.checkvalid()
    #     h = Histogram("id", [Axis(RegularBinning(10, RealInterval(-5, 5))), Axis(RegularBinning(10, RealInterval(-5, 5)))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(100))), functions=[EvaluatedFunction("id", InterpretedInlineBuffer.fromarray(numpy.arange(100)), InterpretedInlineBuffer.fromarray(numpy.arange(100)), [Quantiles(InterpretedInlineBuffer.fromarray(numpy.zeros(100)), 0.25), Quantiles(InterpretedInlineBuffer.fromarray(numpy.zeros(100))), Quantiles(InterpretedInlineBuffer.fromarray(numpy.zeros(100)), 0.75)])])
    #     h.checkvalid()

    def test_BinnedEvaluatedFunction(self):
        # h = BinnedEvaluatedFunction("id", [Axis()], InterpretedInlineBuffer(numpy.array([0.0]), dtype=InterpretedInlineBuffer.float64))
        # assert h == frombuffer(h.tobuffer())
        pass

    #     h = BinnedEvaluatedFunction("id", [Axis()], InterpretedInlineBuffer(numpy.array([0.0]), dtype=InterpretedInlineBuffer.float64))
    #     h.checkvalid()
    #     assert h.values.array.tolist() == [0.0]
    #     h = BinnedEvaluatedFunction("id", [Axis(), Axis()], InterpretedInlineBuffer(numpy.array([[0.0]]), dtype=InterpretedInlineBuffer.float64))
    #     h.checkvalid()
    #     assert h.values.array.tolist() == [[0.0]]

    def test_Page(self):
        pass
    #     h = Ntuple("id", [Column("one", Column.int32)], [NtupleInstance([Chunk([ColumnChunk([Page(RawInlineBuffer(b"\x05\x00\x00\x00"))], [0, 1])])])])
    #     h.checkvalid()
    #     assert h.instances[0].chunks[0].column_chunks[0].pages[0].array.tolist() == [5]

    #     h = Ntuple("id", [Column("one", Column.int32)], [NtupleInstance([Chunk([ColumnChunk([], [0])])])])
    #     h.checkvalid()
    #     assert h.instances[0].chunks[0].column_chunks[0].array.tolist() == []
    #     assert {n: x.tolist() for n, x in h.instances[0].chunks[0].arrays.items()} == {"one": []}
    #     for arrays in h.instances[0].arrays: pass
    #     assert {n: x.tolist() for n, x in arrays.items()} == {"one": []}

    #     h = Ntuple("id", [Column("one", Column.int32)], [NtupleInstance([Chunk([ColumnChunk([Page(RawInlineBuffer(b"\x05\x00\x00\x00"))], [0, 1])])])])
    #     h.checkvalid()
    #     assert h.instances[0].chunks[0].column_chunks[0].array.tolist() == [5]
    #     assert {n: x.tolist() for n, x in h.instances[0].chunks[0].arrays.items()} == {"one": [5]}
    #     for arrays in h.instances[0].arrays: pass
    #     assert {n: x.tolist() for n, x in arrays.items()} == {"one": [5]}

    #     h = Ntuple("id", [Column("one", Column.int32)], [NtupleInstance([Chunk([ColumnChunk([Page(RawInlineBuffer(b"\x05\x00\x00\x00")), Page(RawInlineBuffer(b"\x04\x00\x00\x00\x03\x00\x00\x00"))], [0, 1, 3])])])])
    #     h.checkvalid()
    #     assert h.instances[0].chunks[0].column_chunks[0].array.tolist() == [5, 4, 3]
    #     assert {n: x.tolist() for n, x in h.instances[0].chunks[0].arrays.items()} == {"one": [5, 4, 3]}
    #     for arrays in h.instances[0].arrays: pass
    #     assert {n: x.tolist() for n, x in arrays.items()} == {"one": [5, 4, 3]}

    def test_Chunk(self):
        pass
    #     h = Ntuple("id", [Column("one", Column.float64)], [NtupleInstance([Chunk([ColumnChunk([], [0])])])])
    #     h.checkvalid()

    #     h = Ntuple("id", [Column("one", Column.int32)], [NtupleInstance([])])
    #     h.checkvalid()
    #     for arrays in h.instances[0].arrays:
    #         assert False

    #     h = Ntuple("id", [Column("one", Column.int32)], [NtupleInstance([Chunk([ColumnChunk([Page(RawInlineBuffer(b"\x05\x00\x00\x00"))], [0, 1])])])])
    #     h.checkvalid()
    #     for arrays in h.instances[0].arrays: pass
    #     assert {n: x.tolist() for n, x in arrays.items()} == {"one": [5]}

    #     h = Ntuple("id", [Column("one", Column.int32)], [NtupleInstance([Chunk([ColumnChunk([Page(RawInlineBuffer(b"\x05\x00\x00\x00"))], [0, 1])]), Chunk([ColumnChunk([Page(RawInlineBuffer(b"\x05\x00\x00\x00"))], [0, 1])])])])
    #     h.checkvalid()
    #     for arrays in h.instances[0].arrays:
    #         assert {n: x.tolist() for n, x in arrays.items()} == {"one": [5]}

    def test_Column(self):
        pass
    #     h = Ntuple("id", [Column("one", Column.float64), Column("two", Column.int32)], [NtupleInstance([])])
    #     h.checkvalid()

    #     h = Ntuple("id", [Column("one", Column.int32), Column("two", Column.int16)], [NtupleInstance([Chunk([ColumnChunk([Page(RawInlineBuffer(b"\x05\x00\x00\x00"))], [0, 1]), ColumnChunk([Page(RawInlineBuffer(b"\x03\x00"))], [0, 1])])])])
    #     h.checkvalid()
    #     for arrays in h.instances[0].arrays: pass
    #     assert {n: x.tolist() for n, x in arrays.items()} == {"one": [5], "two": [3]}

    def test_Ntuple(self):
        pass
    #     h = Ntuple("id", [Column("one", Column.float64)], [NtupleInstance([])])
    #     h.checkvalid()

    def test_collection(self):
        h = Collection()
        assert h == frombuffer(h.tobuffer())
        h = Collection({"b": Collection(), "c": Collection()})
        assert h == frombuffer(h.tobuffer())
    #     h.checkvalid()
    #     assert h.identifier == "id"
    #     h = Collection("id", [Histogram("id", [Axis(RegularBinning(10, RealInterval(-5, 5)))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(30)))), Histogram("id2", [Axis(RegularBinning(100, RealInterval(-5, 5)))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(300))))], axis=[Axis(RegularBinning(3, RealInterval(-1, 1)))])
    #     h.checkvalid()
    #     h = Collection("a", [Collection("b", [Histogram("c", [Axis(RegularBinning(10, RealInterval(-5, 5)))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(60)))), Histogram("d", [Axis(RegularBinning(100, RealInterval(-5, 5)))], UnweightedCounts(InterpretedInlineBuffer.fromarray(numpy.arange(600))))], axis=[Axis(FractionBinning())])], axis=[Axis(RegularBinning(3, RealInterval(-1, 1)))])
    #     h.checkvalid()
    #     assert h["b", "c"].counts.counts.array.tolist() == numpy.arange(60).reshape((3, 2, 10)).tolist()
