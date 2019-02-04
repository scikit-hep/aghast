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

def toroot(obj, name):
    if isinstance(obj, Collection):
        raise NotImplementedError

    elif isinstance(obj, Histogram):
        axissummary = []
        slc = []
        for axis in obj.axis:
            if axis.binning is None:
                axissummary.append((RegularBinning, 1, 0.0, 1.0))
                slc.append(slice(-numpy.inf, numpy.inf))
            elif isinstance(axis.binning, IntegerBinning):
                axissummary.append((RegularBinning, 1 + axis.max - axis.min, axis.min - 0.5, axis.max + 0.5))
                slc.append(slice(-numpy.inf, numpy.inf))
            elif isinstance(axis.binning, RegularBinning):
                axissummary.append((RegularBinning, axis.num, axis.interval.low, axis.interval.high))
                slc.append(slice(-numpy.inf, numpy.inf))
            elif isinstance(axis.binning, HexagonalBinning):
                raise TypeError("no ROOT equivalent for HexagonalBinning")
            elif isinstance(axis.binning, EdgesBinning):
                axissummary.append((EdgesBinning, numpy.array(axis.edges, dtype=numpy.float64, copy=False)))
                slc.append(slice(-numpy.inf, numpy.inf))
            elif isinstance(axis.binning, IrregularBinning):
                axissummary.append((CategoryBinning, axis.toCategoryBinning().categories))
                slc.append(slice(None))
            elif isinstance(axis.binning, SparseRegularBinning):
                axissummary.append((CategoryBinning, axis.toCategoryBinning().categories))
                slc.append(slice(None))
            elif isinstance(axis.binning, FractionBinning):
                axissummary.append((CategoryBinning, axis.toCategoryBinning().categories))
                slc.append(slice(None))
            elif isinstance(axis.binning, PredicateBinning):
                axissummary.append((CategoryBinning, axis.toCategoryBinning().categories))
                slc.append(slice(None))
            elif isinstance(axis.binning, VariationBinning):
                axissummary.append((CategoryBinning, axis.toCategoryBinning().categories))
                slc.append(slice(None))
            else:
                raise AssertionError(type(axis.binning))
                
        if isinstance(obj.axis[0].binning, FractionBinning):
            raise NotImplementedError

        sumw = obj.counts[tuple(slc)]
        if isinstance(sumw, dict):
            sumw, sumw2 = sumw["sumw"], sumw["sumw2"]
            sumw2 = numpy.array(sumw2, dtype=numpy.float64, copy=False)
        else:
            sumw2 = None

        if len(self.profile) != 0:
            raise NotImplementedError

        if issubclass(sumw.dtype.type, numpy.int8):
            cls = ROOT.TH1C
            sumw = numpy.array(sumw, dtype=numpy.int8, copy=False)
        elif issubclass(sumw.dtype.type, (numpy.uint8, numpy.int16)):
            cls = ROOT.TH1S
            sumw = numpy.array(sumw, dtype=numpy.int16, copy=False)
        elif issubclass(sumw.dtype.type, (numpy.uint16, numpy.int32)):
            cls = ROOT.TH1I
            sumw = numpy.array(sumw, dtype=numpy.int32, copy=False)
        elif issubclass(sumw.dtype.type, numpy.float32):
            cls = ROOT.TH1F
            sumw = numpy.array(sumw, dtype=numpy.float32, copy=False)
        else:
            cls = ROOT.TH1D
            sumw = numpy.array(sumw, dtype=numpy.float64, copy=False)

        title = "" if obj.title is None else obj.title

        if len(axissummary) == 1:
            if axissummary[0] is RegularBinning:
                num, low, high = axissummary[1:]
                out = cls(name, title, num, low, high)
                xaxis = out.GetXaxis()
                
            elif axissummary[0] is EdgesBinning:
                edges = axissummary[1]
                out = cls(name, title, num, edges)
                xaxis = out.GetXaxis()

            elif axissummary[0] is CategoryBinning:
                categories = axissummary[1]
                out = cls(name, title, len(categories), 0.5, len(categories) + 0.5)
                xaxis = out.GetXaxis()
                for i, x in enumerate(categories):
                    xaxis.SetBinLabel(i + 1, x)

            else:
                raise AssertionError(axissummary[0])

            assert len(sumw.shape) == 1
            if sumw2 is not None:
                assert sumw.shape == sumw2.shape

            if len(sumw) == out.GetNbinsX():
                sumw = numpy.concatenate([numpy.array([0], dtype=sumw.dtype), sumw, numpy.array([0], dtype=sumw.dtype)])
                if sumw2 is not None:
                    sumw2 = numpy.concatenate([numpy.array([0], dtype=sumw2.dtype), sumw2, numpy.array([0], dtype=sumw2.dtype)])
            elif len(sumw) == out.GetNbinsX() + 1 and obj.axis[0].loc_underflow == BinLocation.nonexistent:
                sumw = numpy.concatenate([numpy.array([0], dtype=sumw.dtype), sumw])
                if sumw2 is not None:
                    sumw2 = numpy.concatenate([numpy.array([0], dtype=sumw2.dtype), sumw2])
            elif len(sumw) == out.GetNbinsX() + 1:
                sumw = numpy.concatenate([sumw, numpy.array([0], dtype=sumw.dtype)])
                if sumw2 is not None:
                    sumw2 = numpy.concatenate([sumw2, numpy.array([0], dtype=sumw2.dtype)])
            elif len(sumw) != out.GetNbinsX() + 2:
                raise AssertionError((len(sumw), out.GetNbinsX() + 2))

            sumwarray = numpy.frombuffer(obj.fArray, count=obj.fN, dtype=sumw.dtype)
            sumwarray[:] = sumw

            if sumw2 is not None:
                obj.Sumw2()
                sumw2obj = obj.GetSumw2()
                sumw2array = numpy.frombuffer(sumw2obj.fArray, count=sumw2obj.fN, dtype=numpy.float64)
                sumw2array[:] = sumw2
            
            xaxis.SetTitle("" if obj.axis[0].title is None else obj.axis[0].title)

            return out

        elif len(axis) == 2:
            raise NotImplementedError

        elif len(axis) == 3:
            raise NotImplementedError

        else:
            raise NotImplementedError

    elif isinstance(obj, Ntuple):
        raise NotImplementedError

    else:
        raise TypeError("cannot convert {0}".format(type(obj)))

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
            out = Histogram([axis], counts, title=title)

            if collection:
                raise NotImplementedError
            else:
                return out

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
