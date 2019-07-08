#!/usr/bin/env python

# BSD 3-Clause License; see https://github.com/scikit-hep/aghast/blob/master/LICENSE

import numpy
try:
    import ROOT
except ImportError:
    raise ImportError("\n\nInstall ROOT package with:\n\n    conda install -c conda-forge root")

from aghast import *

def getbincontents(obj):
    if isinstance(obj, (ROOT.TH1C, ROOT.TH2C, ROOT.TH3C)):
        out = numpy.empty(obj.GetNcells(), dtype=numpy.int8)
        arraytype = "char"
    elif isinstance(obj, (ROOT.TH1S, ROOT.TH2S, ROOT.TH3S)):
        out = numpy.empty(obj.GetNcells(), dtype=numpy.int16)
        arraytype = "short"
    elif isinstance(obj, (ROOT.TH1I, ROOT.TH2I, ROOT.TH3I)):
        out = numpy.empty(obj.GetNcells(), dtype=numpy.int32)
        arraytype = "int"
    elif isinstance(obj, (ROOT.TH1F, ROOT.TH2F, ROOT.TH3F)):
        out = numpy.empty(obj.GetNcells(), dtype=numpy.float32)
        arraytype = "float"
    elif isinstance(obj, (ROOT.TH1D, ROOT.TH2D, ROOT.TH3D)):
        out = numpy.empty(obj.GetNcells(), dtype=numpy.float64)
        arraytype = "double"
    else:
        raise AssertionError(type(obj))

    name = "_getbincontents_{0}".format(type(obj).__name__)
    if name not in getbincontents.run:
        ROOT.gInterpreter.Declare("""
        void %s(%s* hist, %s* array) {
            int n = hist->GetNcells();
            for (int i = 0;  i < n;  i++) {
                array[i] = hist->GetBinContent(i);
            }
        }""" % (name, type(obj).__name__, arraytype))
        getbincontents.run[name] = getattr(ROOT, name)

    getbincontents.run[name](obj, out)
    return out

getbincontents.run = {}

def setbincontents(obj, array):
    if isinstance(obj, (ROOT.TH1C, ROOT.TH2C, ROOT.TH3C)):
        array = numpy.array(array, dtype=numpy.int8, copy=False)
        arraytype = "char"
    elif isinstance(obj, (ROOT.TH1S, ROOT.TH2S, ROOT.TH3S)):
        array = numpy.array(array, dtype=numpy.int16, copy=False)
        arraytype = "short"
    elif isinstance(obj, (ROOT.TH1I, ROOT.TH2I, ROOT.TH3I)):
        array = numpy.array(array, dtype=numpy.int32, copy=False)
        arraytype = "int"
    elif isinstance(obj, (ROOT.TH1F, ROOT.TH2F, ROOT.TH3F)):
        array = numpy.array(array, dtype=numpy.float32, copy=False)
        arraytype = "float"
    elif isinstance(obj, (ROOT.TH1D, ROOT.TH2D, ROOT.TH3D)):
        array = numpy.array(array, dtype=numpy.float64, copy=False)
        arraytype = "double"
    else:
        raise AssertionError(type(obj))

    assert array.shape == (obj.GetNcells(),)

    name = "_setbincontents_{0}".format(type(obj).__name__)
    if name not in setbincontents.run:
        ROOT.gInterpreter.Declare("""
        void %s(%s* hist, %s* array) {
            int n = hist->GetNcells();
            for (int i = 0;  i < n;  i++) {
                hist->SetBinContent(i, array[i]);
            }
        }""" % (name, type(obj).__name__, arraytype))
        setbincontents.run[name] = getattr(ROOT, name)

    setbincontents.run[name](obj, array)

setbincontents.run = {}

def to_root(obj, name):
    if isinstance(obj, Collection):
        raise NotImplementedError

    elif isinstance(obj, Histogram):
        axissummary = []
        slc = []
        for axis in obj.axis:
            if axis.binning is None:
                axissummary.append((RegularBinning, 1, 0.0, 1.0))
                slc.append(slice(None if axis.binning.overflow is None or axis.binning.overflow.loc_underflow == BinLocation.nonexistent else -numpy.inf,
                                 None if axis.binning.overflow is None or axis.binning.overflow.loc_overflow == BinLocation.nonexistent else numpy.inf))
            elif isinstance(axis.binning, IntegerBinning):
                axissummary.append((RegularBinning, 1 + axis.binning.max - axis.binning.min, axis.binning.min - 0.5, axis.binning.max + 0.5))
                slc.append(slice(None if axis.binning.overflow is None or axis.binning.overflow.loc_underflow == BinLocation.nonexistent else -numpy.inf,
                                 None if axis.binning.overflow is None or axis.binning.overflow.loc_overflow == BinLocation.nonexistent else numpy.inf))
            elif isinstance(axis.binning, RegularBinning):
                axissummary.append((RegularBinning, axis.binning.num, axis.binning.interval.low, axis.binning.interval.high))
                slc.append(slice(None if axis.binning.overflow is None or axis.binning.overflow.loc_underflow == BinLocation.nonexistent else -numpy.inf,
                                 None if axis.binning.overflow is None or axis.binning.overflow.loc_overflow == BinLocation.nonexistent else numpy.inf))
            elif isinstance(axis.binning, HexagonalBinning):
                raise TypeError("no ROOT equivalent for HexagonalBinning")
            elif isinstance(axis.binning, EdgesBinning):
                axissummary.append((EdgesBinning, numpy.array(axis.binning.edges, dtype=numpy.float64, copy=False)))
                slc.append(slice(None if axis.binning.overflow is None or axis.binning.overflow.loc_underflow == BinLocation.nonexistent else -numpy.inf,
                                 None if axis.binning.overflow is None or axis.binning.overflow.loc_overflow == BinLocation.nonexistent else numpy.inf))
            elif isinstance(axis.binning, IrregularBinning):
                axissummary.append((CategoryBinning, axis.binning.toCategoryBinning().categories))
                slc.append(slice(None))
            elif isinstance(axis.binning, CategoryBinning):
                axissummary.append((CategoryBinning, axis.binning.categories))
                slc.append(slice(None))
            elif isinstance(axis.binning, SparseRegularBinning):
                axissummary.append((CategoryBinning, axis.binning.toCategoryBinning().categories))
                slc.append(slice(None))
            elif isinstance(axis.binning, FractionBinning):
                axissummary.append((CategoryBinning, axis.binning.toCategoryBinning().categories))
                slc.append(slice(None))
            elif isinstance(axis.binning, PredicateBinning):
                axissummary.append((CategoryBinning, axis.binning.toCategoryBinning().categories))
                slc.append(slice(None))
            elif isinstance(axis.binning, VariationBinning):
                axissummary.append((CategoryBinning, axis.binning.toCategoryBinning().categories))
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

        if len(obj.profile) != 0:
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
            if axissummary[0][0] is RegularBinning:
                num, low, high = axissummary[0][1:]
                out = cls(name, title, num, low, high)
                xaxis = out.GetXaxis()

            elif axissummary[0][0] is EdgesBinning:
                edges = axissummary[0][1]
                out = cls(name, title, len(edges) - 1, edges)
                xaxis = out.GetXaxis()

            elif axissummary[0][0] is CategoryBinning:
                categories = axissummary[0][1]
                out = cls(name, title, len(categories), 0.5, len(categories) + 0.5)

            else:
                raise AssertionError(axissummary[0][0])

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

            setbincontents(out, sumw)

            if sumw2 is not None:
                out.Sumw2()
                sumw2obj = out.GetSumw2()
                setbincontents(sumw2obj, sumw2)

            if axissummary[0][0] is CategoryBinning:
                xaxis = out.GetXaxis()
                for i, x in enumerate(categories):
                    xaxis.SetBinLabel(i + 1, x)

            numentries = None
            stats0 = None
            stats1 = None
            stats2 = None
            stats3 = None
            if len(obj.axis[0].statistics) != 0:
                for moment in obj.axis[0].statistics[0].moments:
                    sumwxn = moment.sumwxn.flatarray
                    if moment.n == 0 and moment.weightpower == 0 and len(sumwxn) == 1:
                        numentries, = sumwxn
                    if moment.n == 0 and moment.weightpower == 1 and len(sumwxn) == 1:
                        stats0, = sumwxn
                    if moment.n == 0 and moment.weightpower == 2 and len(sumwxn) == 1:
                        stats1, = sumwxn
                    if moment.n == 1 and moment.weightpower == 1 and len(sumwxn) == 1:
                        stats2, = sumwxn
                    if moment.n == 2 and moment.weightpower == 1 and len(sumwxn) == 1:
                        stats3, = sumwxn

            if numentries is None:
                if isinstance(obj.counts, UnweightedCounts):
                    numentries = obj.counts.flatarray.sum()
                elif obj.counts.unweighted is not None:
                    numentries = obj.counts.unweighted.flatarray.sum()
                else:
                    numentries = obj.counts.sumw.flatarray.sum()

            out.SetEntries(numentries)

            if stats0 is not None and stats1 is not None and stats2 is not None and stats3 is not None:
                stats = numpy.array([stats0, stats1, stats2, stats3], dtype=numpy.float64)
                out.PutStats(stats)

            xaxis.SetTitle("" if obj.axis[0].title is None else obj.axis[0].title)

            return out

        elif len(axissummary) == 2:
            raise NotImplementedError

        elif len(axissummary) == 3:
            raise NotImplementedError

        else:
            raise NotImplementedError

    elif isinstance(obj, Ntuple):
        raise NotImplementedError

    else:
        raise TypeError("cannot convert {0}".format(type(obj)))

def from_root(obj, collection=False):
    if isinstance(obj, ROOT.TH1):
        if isinstance(obj, ROOT.TProfile):
            raise NotImplementedError

        elif isinstance(obj, ROOT.TProfile2D):
            raise NotImplementedError

        elif isinstance(obj, ROOT.TProfile3D):
            raise NotImplementedError

        if not isinstance(obj, (ROOT.TH2, ROOT.TH3)):
            sumwarray = getbincontents(obj)

            xaxis = obj.GetXaxis()
            labels = xaxis.GetLabels()
            if labels:
                categories = [str(x) for x in labels]
            else:
                categories = None

            if obj.GetSumw2N() != 0:
                sumw2obj = obj.GetSumw2()
                sumw2array = getbincontents(sumw2obj)
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
            statistics = [Statistics(moments=[entries, sumw, sumw2, sumwx, sumwx2])]

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
            raise NotImplementedError

        elif isinstance(obj, ROOT.TH3):
            raise NotImplementedError

        else:
            raise TypeError("cannot convert {0}".format(type(obj).__name__))

    elif isinstance(obj, ROOT.TEfficiency):
        raise NotImplementedError

    elif isinstance(obj, ROOT.TTree):
        raise NotImplementedError

    else:
        raise TypeError("cannot convert {0}".format(type(obj).__name__))
