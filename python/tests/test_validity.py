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

from histos import *

class Test(unittest.TestCase):
    def runTest(self):
        pass

    def test_Metadata(self):
        h = Collection("id", [], metadata=Metadata("""{"one": 1, "two": 2}""", language=Metadata.json))
        assert h.isvalid
        assert h.metadata.data == """{"one": 1, "two": 2}"""
        assert h.metadata.language == Metadata.json

    def test_Decoration(self):
        h = Collection("id", [], decoration=Decoration("""points { color: red }""", language=Decoration.css))
        assert h.isvalid
        assert h.decoration.data == """points { color: red }"""
        assert h.decoration.css == Decoration.css

    def test_RawInlineBuffer(self):
        pass

    def test_RawExternalBuffer(self):
        pass

    def test_InterpretedInlineBuffer(self):
        pass

    def test_InterpretedExternalBuffer(self):
        pass

    def test_FractionalBinning(self):
        h = Collection("id", [BinnedEvaluatedFunction("id", [Axis(FractionalBinning())], InterpretedInlineBuffer())])
        assert h.isvalid
        assert h["id"].axis[0].binning.error_method == FractionalBinning.normal

    def test_IntegerBinning(self):
        h = Collection("id", [BinnedEvaluatedFunction("id", [Axis(IntegerBinning(10, 20))], InterpretedInlineBuffer())])
        assert h.isvalid
        h = Collection("id", [BinnedEvaluatedFunction("id", [Axis(IntegerBinning(20, 10))], InterpretedInlineBuffer())])
        assert not h.isvalid
        h = Collection("id", [BinnedEvaluatedFunction("id", [Axis(IntegerBinning(10, 20, has_underflow=False, has_overflow=False))], InterpretedInlineBuffer(dtype=InterpretedInlineBuffer.float64))])
        assert h["id"].values.numpy_array.tolist() == [0.0] * 11
        h = Collection("id", [BinnedEvaluatedFunction("id", [Axis(IntegerBinning(10, 20, has_underflow=True, has_overflow=False))], InterpretedInlineBuffer(dtype=InterpretedInlineBuffer.float64))])
        assert h["id"].values.numpy_array.tolist() == [0.0] * 12
        h = Collection("id", [BinnedEvaluatedFunction("id", [Axis(IntegerBinning(10, 20, has_underflow=False, has_overflow=True))], InterpretedInlineBuffer(dtype=InterpretedInlineBuffer.float64))])
        assert h["id"].values.numpy_array.tolist() == [0.0] * 12
        h = Collection("id", [BinnedEvaluatedFunction("id", [Axis(IntegerBinning(10, 20, has_underflow=True, has_overflow=True))], InterpretedInlineBuffer(dtype=InterpretedInlineBuffer.float64))])
        assert h["id"].values.numpy_array.tolist() == [0.0] * 13

    def test_RealInterval(self):
        h = Collection("id", [BinnedEvaluatedFunction("id", [Axis(RegularBinning(10, RealInterval(-5, 5)))], InterpretedInlineBuffer())])
        assert h.isvalid
        h = Collection("id", [BinnedEvaluatedFunction("id", [Axis(RegularBinning(10, RealInterval(5, -5)))], InterpretedInlineBuffer())])
        assert not h.isvalid

    def test_RealOverflow(self):
        h = Collection("id", [BinnedEvaluatedFunction("id", [Axis(RegularBinning(10, RealInterval(-5, 5), RealOverflow(has_underflow=False, has_overflow=False, has_nanflow=False)))], InterpretedInlineBuffer(dtype=InterpretedInlineBuffer.float64))])
        assert h["id"].values.numpy_array.tolist() == [0.0] * 10
        h = Collection("id", [BinnedEvaluatedFunction("id", [Axis(RegularBinning(10, RealInterval(-5, 5), RealOverflow(has_underflow=True, has_overflow=False, has_nanflow=False)))], InterpretedInlineBuffer(dtype=InterpretedInlineBuffer.float64))])
        assert h["id"].values.numpy_array.tolist() == [0.0] * 11
        h = Collection("id", [BinnedEvaluatedFunction("id", [Axis(RegularBinning(10, RealInterval(-5, 5), RealOverflow(has_underflow=False, has_overflow=True, has_nanflow=False)))], InterpretedInlineBuffer(dtype=InterpretedInlineBuffer.float64))])
        assert h["id"].values.numpy_array.tolist() == [0.0] * 11
        h = Collection("id", [BinnedEvaluatedFunction("id", [Axis(RegularBinning(10, RealInterval(-5, 5), RealOverflow(has_underflow=False, has_overflow=False, has_nanflow=True)))], InterpretedInlineBuffer(dtype=InterpretedInlineBuffer.float64))])
        assert h["id"].values.numpy_array.tolist() == [0.0] * 11
        h = Collection("id", [BinnedEvaluatedFunction("id", [Axis(RegularBinning(10, RealInterval(-5, 5), RealOverflow(has_underflow=True, has_overflow=False, has_nanflow=True)))], InterpretedInlineBuffer(dtype=InterpretedInlineBuffer.float64))])
        assert h["id"].values.numpy_array.tolist() == [0.0] * 12
        h = Collection("id", [BinnedEvaluatedFunction("id", [Axis(RegularBinning(10, RealInterval(-5, 5), RealOverflow(has_underflow=True, has_overflow=True, has_nanflow=True)))], InterpretedInlineBuffer(dtype=InterpretedInlineBuffer.float64))])
        assert h["id"].values.numpy_array.tolist() == [0.0] * 13

    def test_RegularBinning(self):
        h = Collection("id", [BinnedEvaluatedFunction("id", [Axis(RegularBinning(10, RealInterval(-5, 5)))], InterpretedInlineBuffer())])
        assert h.isvalid

    def test_TicTacToeOverflowBinning(self):
        h = Collection("id", [BinnedEvaluatedFunction("id", [Axis(TicTacToeOverflowBinning(2, 2, RealInterval(-10, 10), RealInterval(-10, 10)))], InterpretedInlineBuffer(dtype=InterpretedInlineBuffer.float64))])
        assert h["id"].values.numpy_array.tolist() == [[0.0, 0.0, 0.0, 0.0, 0.0]] * 5
        h = Collection("id", [BinnedEvaluatedFunction("id", [Axis(TicTacToeOverflowBinning(2, 2, RealInterval(-10, 10), RealInterval(-10, 10), RealOverflow(False, False, False), RealOverflow(False, False, False)))], InterpretedInlineBuffer(dtype=InterpretedInlineBuffer.float64))])
        assert h["id"].values.numpy_array.tolist() == [[0.0, 0.0], [0.0, 0.0]]

    def test_HexagonalBinning(self):
        h = Collection("id", [BinnedEvaluatedFunction("id", [Axis(HexagonalBinning(IntegerBinning(3, 5), IntegerBinning(-5, -4)))], InterpretedInlineBuffer(dtype=InterpretedInlineBuffer.float64))])
        assert h.isvalid
        assert h["id"].values.numpy_array.tolist() == [[0.0] * 5] * 6
        h = Collection("id", [BinnedEvaluatedFunction("id", [Axis(HexagonalBinning(IntegerBinning(3, 5), IntegerBinning(-5, -4), q_has_nanflow=False))], InterpretedInlineBuffer(dtype=InterpretedInlineBuffer.float64))])
        assert h["id"].values.numpy_array.tolist() == [[0.0] * 5] * 5
        h = Collection("id", [BinnedEvaluatedFunction("id", [Axis(HexagonalBinning(IntegerBinning(3, 5), IntegerBinning(-5, -4), r_has_nanflow=False))], InterpretedInlineBuffer(dtype=InterpretedInlineBuffer.float64))])
        assert h["id"].values.numpy_array.tolist() == [[0.0] * 4] * 6
        h = Collection("id", [BinnedEvaluatedFunction("id", [Axis(HexagonalBinning(IntegerBinning(3, 5), IntegerBinning(-5, -4), q_has_nanflow=False, r_has_nanflow=False))], InterpretedInlineBuffer(dtype=InterpretedInlineBuffer.float64))])
        assert h["id"].values.numpy_array.tolist() == [[0.0] * 4] * 5

    def test_EdgesBinning(self):
        h = Collection("id", [BinnedEvaluatedFunction("id", [Axis(EdgesBinning([3.3]))], InterpretedInlineBuffer(dtype=InterpretedInlineBuffer.float64))])
        assert h.isvalid
        assert h["id"].values.numpy_array.tolist() == [0.0, 0.0, 0.0]
        h = Collection("id", [BinnedEvaluatedFunction("id", [Axis(EdgesBinning([1.1, 2.2, 3.3]))], InterpretedInlineBuffer(dtype=InterpretedInlineBuffer.float64))])
        assert h["id"].values.numpy_array.tolist() == [0.0, 0.0, 0.0, 0.0, 0.0]

    def test_IrregularBinning(self):
        h = Collection("id", [BinnedEvaluatedFunction("id", [Axis(IrregularBinning([RealInterval(0.5, 1.5)]))], InterpretedInlineBuffer(dtype=InterpretedInlineBuffer.float64))])
        assert h.isvalid
        assert h["id"].values.numpy_array.tolist() == [0.0, 0.0, 0.0, 0.0]
        h = Collection("id", [BinnedEvaluatedFunction("id", [Axis(IrregularBinning([RealInterval(0.5, 1.5), RealInterval(1.5, 1.5), RealInterval(0.0, 10.0)]))], InterpretedInlineBuffer(dtype=InterpretedInlineBuffer.float64))])
        assert h["id"].values.numpy_array.tolist() == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def test_CategoryBinning(self):
        pass

    def test_SparseRegularBinning(self):
        pass

    def test_Axis(self):
        h = Collection("id", [BinnedEvaluatedFunction("id", [Axis(expression="x", title="wow")], InterpretedInlineBuffer())])
        assert h.isvalid
        assert h["id"].axis[0].expression == "x"

    def test_Counts(self):
        pass

    def test_UnweightedCounts(self):
        pass

    def test_WeightedCounts(self):
        pass

    def test_Correlation(self):
        pass

    def test_Extremes(self):
        pass

    def test_Moments(self):
        pass

    def test_Quantiles(self):
        pass

    def test_GenericErrors(self):
        pass

    def test_DistributionStats(self):
        pass

    def test_Distribution(self):
        pass

    def test_Profile(self):
        pass

    def test_Histogram(self):
        pass

    def test_Parameter(self):
        h = Collection("id", [ParameterizedFunction("id", "x**2", [Parameter("x", 5), Parameter("y", 6)])])
        assert h.isvalid

    def test_ParameterizedFunction(self):
        h = Collection("id", [ParameterizedFunction("id", "x**2", [], contours=[1.1, 2.2, 3.3])])
        assert h.isvalid

    def test_EvaluatedFunction(self):
        pass

    def test_BinnedEvaluatedFunction(self):
        h = Collection("id", [BinnedEvaluatedFunction("id", [Axis()], InterpretedInlineBuffer(dtype=InterpretedInlineBuffer.float64))])
        assert h.isvalid
        assert h["id"].values.numpy_array.tolist() == [0.0]

        h = Collection("id", [BinnedEvaluatedFunction("id", [Axis(), Axis()], InterpretedInlineBuffer(dtype=InterpretedInlineBuffer.float64))])
        assert h["id"].values.numpy_array.tolist() == [[0.0]]

    def test_Page(self):
        pass

    def test_ColumnChunk(self):
        pass

    def test_Chunk(self):
        pass

    def test_Column(self):
        pass

    def test_Ntuple(self):
        pass

    def test_Region(self):
        pass

    def test_BinnedRegion(self):
        pass

    def test_Assignment(self):
        pass

    def test_Variation(self):
        pass

    def test_collection(self):
        h = Collection("id", [])
        assert h.isvalid
        assert h.identifier == "id"
