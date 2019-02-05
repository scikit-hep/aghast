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

import pytest
import numpy

from stagg import *

ROOT = pytest.importorskip("ROOT")
connect_root = pytest.importorskip("stagg.connect.root")

data = [-0.319, -1.949, -1.511, 1.176, 0.695, -0.286, 0.392, -0.094, -0.714, 1.223, 1.811, -2.57, -0.014, 0.328, -0.084, -0.13, -0.751, 0.47, 1.558, 0.714, -0.999, 0.336, -2.51, -0.532, 2.495, -0.085, 0.165, 0.764, 0.865, -0.604, -1.083, 2.501, -1.074, -0.853, -0.241, -0.346, 0.472, -0.017, 1.013, 0.887, 2.154, -1.354, 0.966, -1.118, -1.374, -0.928, -0.664, 2.803, -0.133, 1.814, 0.844, -0.719, 1.239, 0.49, 0.333, 0.518, 0.079, -1.33, -0.048, 1.335, -2.108, -1.772, -0.68, 0.151, -0.479, 0.749, 0.589, -1.048, -0.491, 1.125, 1.064, 0.4, 0.349, 0.193, -0.645, 0.038, 0.536, -0.675, 0.732, -1.442, -0.889, -0.976, 0.889, 1.296, 1.231, 0.934, -1.359, 1.602, 0.186, -0.622, -0.08, -0.887, 0.109, 0.418, 0.945, -0.081, -0.32, -1.309, -0.497, 0.346]

def check1d(before, after):
    assert before.GetNbinsX() == after.GetNbinsX()
    for i in range(before.GetNbinsX() + 2):
        assert before.GetBinContent(i) == after.GetBinContent(i)
        assert before.GetBinError(i) == after.GetBinError(i)
    assert before.GetEntries() == after.GetEntries()
    assert before.GetMean() == after.GetMean()
    assert before.GetStdDev() == after.GetStdDev()
    assert before.GetTitle() == after.GetTitle()
    assert before.GetXaxis().GetTitle() == after.GetXaxis().GetTitle()
    assert bool(before.GetXaxis().GetLabels()) == bool(after.GetXaxis().GetLabels())
    assert before.GetXaxis().IsVariableBinSize() == after.GetXaxis().IsVariableBinSize()
    if before.GetXaxis().GetLabels():
        assert list(before.GetXaxis().GetLabels()) == list(after.GetXaxis().GetLabels())
    elif before.GetXaxis().IsVariableBinSize():
        beforeedges = numpy.full(before.GetNbinsX() + 1, 999, dtype=numpy.float64)
        before.GetXaxis().GetLowEdge(beforeedges)
        beforeedges[-1] = before.GetXaxis().GetBinUpEdge(before.GetNbinsX())
        afteredges = numpy.full(after.GetNbinsX() + 1, 123, dtype=numpy.float64)
        after.GetXaxis().GetLowEdge(afteredges)
        afteredges[-1] = after.GetXaxis().GetBinUpEdge(after.GetNbinsX())
        assert numpy.array_equal(before, after)
    else:
        assert before.GetXaxis().GetBinLowEdge(1) == after.GetXaxis().GetBinLowEdge(1)
        assert before.GetXaxis().GetBinUpEdge(before.GetNbinsX()) == after.GetXaxis().GetBinUpEdge(after.GetNbinsX())

num = 0
def rootname():
    global num
    num += 1
    return "name-{0}".format(num)

@pytest.mark.parametrize("cls", [ROOT.TH1C, ROOT.TH1S, ROOT.TH1I, ROOT.TH1F, ROOT.TH1D])
def test_root_oned(cls):
    before = cls(rootname(), "title", 5, -2.0, 2.0)
    before.GetXaxis().SetTitle("title2")
    after = connect_root.toroot(connect_root.tostagg(before), rootname())
    check1d(before, after)

    before = cls(rootname(), "title", 5, -2.0, 2.0)
    before.GetXaxis().SetTitle("title2")
    for x in data: before.Fill(x)
    after = connect_root.toroot(connect_root.tostagg(before), rootname())
    check1d(before, after)

    before = cls(rootname(), "title", 5, -2.0, 2.0)
    before.GetXaxis().SetTitle("title2")
    for i, x in enumerate(["one", "two", "three", "four", "five"]):
        before.GetXaxis().SetBinLabel(i + 1, x)
    after = connect_root.toroot(connect_root.tostagg(before), rootname())
    check1d(before, after)

    before = cls(rootname(), "title", 5, -5.0, 5.0)
    before.GetXaxis().SetTitle("title2")
    for i, x in enumerate(["one", "two", "three", "four", "five"]):
        before.GetXaxis().SetBinLabel(i + 1, x)
    for x in data: before.Fill(x)
    after = connect_root.toroot(connect_root.tostagg(before), rootname())
    check1d(before, after)

    edges = numpy.array([-5.0, -3.0, 0.0, 5.0, 10.0, 100.0], dtype=numpy.float64)
    before = cls(rootname(), "title", 5, edges)
    before.GetXaxis().SetTitle("title2")
    after = connect_root.toroot(connect_root.tostagg(before), rootname())
    check1d(before, after)

    edges = numpy.array([-5.0, -3.0, 0.0, 5.0, 10.0, 100.0], dtype=numpy.float64)
    before = cls(rootname(), "title", 5, edges)
    before.GetXaxis().SetTitle("title2")
    for x in data: before.Fill(x)
    after = connect_root.toroot(connect_root.tostagg(before), rootname())
    check1d(before, after)
