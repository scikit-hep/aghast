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

import numpy
try:
    import ROOT
except ImportError:
    raise ImportError("\n\nInstall ROOT package with:\n\n    conda install -c conda-forge root")

from stagg import *

def toroot(obj):
    pass

def tostagg(obj, collection=False):
    if isinstance(obj, ROOT.TH1):
        if isinstance(obj, ROOT.TProfile):
            raise NotImplementedError

        elif isinstance(obj, ROOT.TProfile2D):
            raise NotImplementedError

        elif isinstance(obj, ROOT.TProfile3D):
            raise NotImplementedError

        if not isinstance(obj, (ROOT.TH2, ROOT.TH3)):
            fArray = obj.fArray
            if not isinstance(fArray, bytes):
                fArray = fArray.encode("latin-1")
            if isinstance(obj, ROOT.TH1C):
                sumwarray = numpy.frombuffer(fArray, count=obj.fN, dtype=numpy.int8)
            elif isinstance(obj, ROOT.TH1S):
                sumwarray = numpy.frombuffer(fArray, count=obj.fN, dtype=numpy.int16)
            elif isinstance(obj, ROOT.TH1I):
                sumwarray = numpy.frombuffer(fArray, count=obj.fN, dtype=numpy.int32)
            elif isinstance(obj, ROOT.TH1F):
                sumwarray = numpy.frombuffer(fArray, count=obj.fN, dtype=numpy.float32)
            elif isinstance(obj, ROOT.TH1D):
                sumwarray = numpy.frombuffer(fArray, count=obj.fN, dtype=numpy.float64)
            else:
                raise AssertionError("unrecognized type: {0}".format(type(obj)))

            xaxis = obj.GetXaxis()
            labels = xaxis.GetLabels()
            if labels:
                categories = list(labels)
            else:
                categories = None

            if obj.GetSumw2N() != 0:
                sumw2obj = h.GetSumw2()
                fArray = sumw2obj.fArray
                if not isinstance(fArray, bytes):
                    fArray = fArray.encode("latin-1")
                sumw2array = numpy.frombuffer(fArray, count=sumw2obj.fN, dtype=numpy.float64)
                if categories is not None:
                    sumwarray = sumwarray[1 : len(categories) + 1]
                    sumw2array = sumw2array[1 : len(categories) + 1]
                counts = WeightedCounts(
                    sumw=InterpretedInlineBuffer.fromarray(sumwarray),
                    sumw2=InterpretedInlineBuffer.fromarray(sumw2array))
            else:
                if categories is not None:
                    sumwarray = sumwarray[1 : len(categories) + 1]
                counts = UnweightedCounts(
                    counts=InterpretedInlineBuffer.fromarray(sumwarray))

            if categories is not None:
                binning = CategoryBinning(categories)

            elif xaxis.IsVariableBinSize():
                num = obj.GetNbinsX()
                edges = numpy.empty(num + 1, dtype=numpy.float64)
                xaxis.GetLowEdge(edges)
                edges[-1] = xaxis.GetBinUpEdge(num)
                binning = EdgesBinning(edges, overflow=RealOverflow(loc_underflow=BinLocation.below1, loc_overflow=BinLocation.above1))

            else:
                num = obj.GetNbinsX()
                low = xaxis.GetBinLowEdge(1)
                high = xaxis.GetBinUpEdge(num)
                binning = RegularBinning(num, RealInterval(low, high), overflow=RealOverflow(loc_underflow=BinLocation.below1, loc_overflow=BinLocation.above1))

            stats = numpy.zeros(4, numpy.float64)
            obj.GetStats(stats)
            entries = Moments(InterpretedInlineBuffer.fromarray(numpy.array([obj.GetEntries()], dtype=numpy.int64)), n=0, weightpower=0)
            sumw = Moments(InterpretedInlineBuffer.fromarray(stats[0:1]), n=0, weightpower=1)
            sumw2 = Moments(InterpretedInlineBuffer.fromarray(stats[1:2]), n=0, weightpower=2)
            sumwx = Moments(InterpretedInlineBuffer.fromarray(stats[2:3]), n=1, weightpower=1)
            sumwx2 = Moments(InterpretedInlineBuffer.fromarray(stats[3:4]), n=2, weightpower=1)
            statistics = Statistics(moments=[entries, sumw, sumw2, sumwx, sumwx2])

            title = xaxis.GetTitle()
            if title == "":
                title = None
            axis = Axis(binning, statistics=statistics, title=title)

            title = obj.GetTitle()
            if title == "":
                title = None
            return Histogram([axis], counts, title=title)

        elif isinstance(obj, ROOT.TH2):
            if isinstance(obj, ROOT.TH2C):
                raise NotImplementedError

            elif isinstance(obj, ROOT.TH2S):
                raise NotImplementedError

            elif isinstance(obj, ROOT.TH2I):
                raise NotImplementedError

            elif isinstance(obj, ROOT.TH2F):
                raise NotImplementedError

            elif isinstance(obj, ROOT.TH2D):
                raise NotImplementedError

        elif isinstance(obj, ROOT.TH3):
            if isinstance(obj, ROOT.TH3C):
                raise NotImplementedError

            elif isinstance(obj, ROOT.TH3S):
                raise NotImplementedError

            elif isinstance(obj, ROOT.TH3I):
                raise NotImplementedError

            elif isinstance(obj, ROOT.TH3F):
                raise NotImplementedError

            elif isinstance(obj, ROOT.TH3D):
                raise NotImplementedError

        else:
            raise TypeError("cannot convert {0}".format(type(obj).__name__))

    elif isinstance(obj, ROOT.TTree):
        raise NotImplementedError

    else:
        raise TypeError("cannot convert {0}".format(type(obj).__name__))
