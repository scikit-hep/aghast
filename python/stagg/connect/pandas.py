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

import collections
import re

import numpy
try:
    import pandas
except ImportError:
    raise ImportError("Install pandas package with:\n    pip install pandas\nor\n    conda install pandas")

from stagg import *

def binning2index(binning):
    if binning is None:
        return pandas.CategoricalIndex(["all"])

    elif isinstance(binning, IntegerBinning):
        if binning.loc_underflow == BinLocation.nonexistent and binning.loc_overflow == BinLocation.nonexistent:
            return pandas.RangeIndex(binning.min, binning.max + 1)
        else:
            return binning2index(binning.toCategoryBinning())

    elif isinstance(binning, RegularBinning):
        if binning.overflow is None or (binning.overflow.loc_underflow == BinLocation.nonexistent and binning.overflow.loc_overflow == BinLocation.nonexistent and binning.overflow.loc_nanflow == BinLocation.nonexistent):
            return pandas.interval_range(binning.interval.low, binning.interval.high, binning.num, closed=("left" if binning.interval.low_inclusive else "right"))
        elif binning.overflow is None or binning.overflow.loc_nanflow == BinLocation.nonexistent:
            return binning2index(binning.toEdgesBinning())
        else:
            return binning2index(binning.toCategoryBinning())

    elif isinstance(binning, HexagonalBinning):
        raise NotImplementedError

    elif isinstance(binning, EdgesBinning):
        if binning.overflow is None or binning.overflow.loc_nanflow == BinLocation.nonexistent:
            if binning.overflow is None or (binning.overflow.loc_underflow == BinLocation.nonexistent and binning.overflow.loc_overflow == BinLocation.nonexistent):
                return pandas.IntervalIndex.from_breaks(binning.edges, closed=("left" if binning.low_inclusive else "right"))
            elif binning.overflow is not None and binning.overflow.loc_underflow.value <= BinLocation.nonexistent.value and binning.overflow.loc_overflow.value >= BinLocation.nonexistent.value:
                edges = numpy.empty(binning._binshape()[0] + 1, dtype=numpy.float64)
                shift = int(binning.overflow.loc_underflow != BinLocation.nonexistent)
                edges[shift : shift + len(binning.edges)] = binning.edges
                if binning.overflow.loc_underflow != BinLocation.nonexistent:
                    edges[0] = -numpy.inf
                if binning.overflow.loc_overflow != BinLocation.nonexistent:
                    edges[-1] = numpy.inf
                return pandas.IntervalIndex.from_breaks(edges, closed=("left" if binning.low_inclusive else "right"))
            else:
                return binning2index(binning.toIrregularBinning())
        else:
            return binning2index(binning.toCategoryBinning())

    elif isinstance(binning, IrregularBinning):
        if (binning.overflow is None or binning.overflow.loc_nanflow == BinLocation.nonexistent) and len(binning.intervals) != 0 and binning.intervals[0].low_inclusive != binning.intervals[0].high_inclusive and all(x.low_inclusive == binning.intervals[0].low_inclusive and x.high_inclusive == binning.intervals[0].high_inclusive for x in binning.intervals):
            left = numpy.empty(binning._binshape(), dtype=numpy.float64)
            right = numpy.empty(binning._binshape(), dtype=numpy.float64)
            flows = [] if binning.overflow is None else [(binning.overflow.loc_underflow, -numpy.inf), (binning.overflow.loc_overflow, numpy.inf)]
            low = numpy.inf
            high = -numpy.inf
            for interval in binning.intervals:
                if interval.low <= low:
                    low = interval.low
                if interval.high >= high:
                    high = interval.high
            i = 0
            for loc, val in BinLocation._belows(flows):
                if val == -numpy.inf:
                    left[i], right[i] = val, low
                if val == numpy.inf:
                    left[i], right[i] = high, val
                i += 1
            for interval in binning.intervals:
                left[i] = interval.low
                right[i] = interval.high
                i += 1
            for loc, val in BinLocation._aboves(flows):
                if val == -numpy.inf:
                    left[i], right[i] = val, low
                if val == numpy.inf:
                    left[i], right[i] = high, val
                i += 1
            return pandas.IntervalIndex.from_arrays(left, right, closed=("left" if binning.intervals[0].low_inclusive else "right"))
        else:
            return binning2index(binning.toCategoryBinning())

    elif isinstance(binning, CategoryBinning):
        categories = []
        flows = [(binning.loc_overflow,)]
        for loc, in BinLocation._belows(flows):
            categories.append("(other)")
        categories.extend(binning.categories)
        for loc, in BinLocation._aboves(flows):
            categories.append("(other)")
        return pandas.CategoricalIndex(categories)
        
    elif isinstance(binning, SparseRegularBinning):
        return binning2index(binning.toIrregularBinning())

    elif isinstance(binning, FractionBinning):
        return binning2index(binning.toCategoryBinning())

    elif isinstance(binning, PredicateBinning):
        return binning2index(binning.toCategoryBinning())

    elif isinstance(binning, VariationBinning):
        return binning2index(binning.toCategoryBinning())

    else:
        raise AssertionError(type(binning))

def topandas(obj):
    if isinstance(obj, Histogram):
        indexes = []
        names = []

        for axis in obj.allaxis:
            index = binning2index(axis.binning)

            target = None
            for i in range(len(indexes)):
                if target is None:
                    target = len(indexes[i])
                else:
                    assert target == len(indexes[i])
                indexes[i] = indexes[i].repeat(len(index))

            target = 1 if target is None else target

            # why doesn't pandas have an Index.tile function?
            x = index
            tiles = 1
            for j in range(int(numpy.floor(numpy.log2(target)))):
                x = x.append(x)
                tiles *= 2
            while tiles < target:
                x = x.append(index)
                tiles += 1

            assert len(x) == target * len(index)
            indexes.append(x)

            if axis.title is not None:
                names.append(axis.title)
            elif axis.expression is not None:
                names.append(axis.expression)
            else:
                names.append(None)

        assert len(indexes) == len(names)
        assert len(indexes) != 0
        if len(indexes) == 1:
            index = indexes[0]
        else:
            index = pandas.MultiIndex.from_arrays(indexes, names=names)

        if isinstance(obj.counts, UnweightedCounts):
            data = [obj.counts.flatarray]
            columns = ["unweighted"]
        elif isinstance(obj.counts, WeightedCounts):
            data = [obj.counts.sumw.flatarray]
            columns = ["sumw"]
            if obj.counts.sumw2 is not None:
                data.append(obj.counts.sumw2.flatarray)
                columns.append("sumw2")
            if obj.counts.unweighted is not None:
                data.append(obj.counts.unweighted.flatarray)
                columns.append("unweighted")

        if len(obj.profile) != 0:
            for i in range(len(columns)):
                columns[i] = ("counts", columns[i])

        for profile in obj.profile:
            for moment in profile.statistics.moments:
                data.append(moment.sumwxn.flatarray)
                columns.append((profile.expression, "sum" + ("w" + (repr(moment.weightpower) if moment.weightpower != 1 else "") if moment.weightpower != 0 else "") + ("x" + (repr(moment.n) if moment.n != 1 else "") if moment.n != 0 else "")))

            for quantile in profile.statistics.quantiles:
                data.append(quantile.values.flatarray)
                columns.append((profile.expression, ("p=" + ("%g" % quantile.p)) + (" (w" + (repr(quantile.weightpower) if quantile.weightpower != 1 else "") + ")" if quantile.weightpower != 0 else "")))

            if profile.statistics.mode is not None:
                data.append(profile.statistics.mode.values.flatarray)
                columns.append((profile.expression, "mode"))

            if profile.statistics.min is not None:
                data.append(profile.statistics.mode.values.flatarray)
                columns.append((profile.expression, "min"))

            if profile.statistics.max is not None:
                data.append(profile.statistics.mode.values.flatarray)
                columns.append((profile.expression, "max"))

        data = dict(zip(columns, data))

        if isinstance(columns[0], tuple):
            columns = pandas.MultiIndex.from_tuples(columns)

        return pandas.DataFrame(index=index, columns=columns, data=data)

    elif isinstance(obj, Ntuple):
        raise NotImplementedError

    elif isinstance(obj, Collection):
        raise NotImplementedError

    elif isinstance(obj, BinnedEvaluatedFunction):
        raise NotImplementedError

    else:
        raise TypeError("{0} has no pandas equivalent".format(type(obj).__name__))

def column2statistic(array, statexpr):
    m = column2statistic.moment.match(statexpr)
    if m is not None:
        if m.group(1) is None:
            weightpower = 0
        else:
            weightpower = m.group(2)
            if weightpower is None:
                weightpower = 1
            else:
                weightpower = int(weightpower)
        if m.group(3) is None:
            n = 0
        else:
            n = m.group(4)
            if n is None:
                n = 1
            else:
                n = int(n)
        return Moments(InterpretedInlineBuffer.fromarray(array), n=n, weightpower=weightpower)

    m = column2statistic.quantile.match(statexpr)
    if m is not None:
        p = float(m.group(1))
        weightpower = m.group(6)
        if weightpower is None:
            weightpower = 0
        elif weightpower == "":
            weightpower = 1
        else:
            weightpower = int(weightpower)
        return Quantiles(InterpretedInlineBuffer.fromarray(array), p=p, weightpower=weightpower)

    if statexpr == "mode":
        return Modes(InterpretedInlineBuffer.fromarray(array))

    if statexpr == "min":
        return Extremes(InterpretedInlineBuffer.fromarray(array))

    if statexpr == "max":
        return Extremes(InterpretedInlineBuffer.fromarray(array))

    return None

column2statistic.moment = re.compile(r"sum(w([+-]?\d+)?)?(x([+-]?\d+)?)?")
column2statistic.quantile = re.compile(r"p=([+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)( \(w(\d*)\))?")

def tostagg(obj):
    unweighted, sumw, sumw2 = None, None, None
    if isinstance(obj.columns, pandas.MultiIndex) and "counts" in obj.columns:
        if "unweighted" in obj["counts"].columns:
            unweighted = obj["counts"]["unweighted"].values
        if "sumw" in obj["counts"].columns:
            sumw = obj["counts"]["sumw"].values
        if "sumw2" in obj["counts"].columns:
            sumw2 = obj["counts"]["sumw2"].values
    else:
        if "unweighted" in obj.columns:
            unweighted = obj["unweighted"].values
        if "sumw" in obj.columns:
            sumw = obj["sumw"].values
        if "sumw2" in obj.columns:
            sumw2 = obj["sumw2"].values

    if unweighted is not None or sumw is not None or sumw2 is not None:
        handle_axis




        statistics = collections.OrderedDict()
        if isinstance(obj.columns, pandas.MultiIndex):
            for expression, statexpr in obj.columns:
                if expression != "counts":
                    array = obj[expression][statexpr].values

                    statistic = statistics.get(expression, None)
                    if statistic is None:
                        statistic = statistics[expression] = Statistics()

                    data = column2statistic(array, statexpr)
                    if isinstance(data, Moments):
                        statistic.moments = list(statistic.moments) + [data]
                    elif isinstance(data, Quantiles):
                        statistic.quantiles = list(statistic.quantiles) + [data]
                    elif isinstance(data, Modes):
                        statistic.mode = data
                    elif isinstance(data, Extremes) and statexpr == "min":
                        statistic.min = data
                    elif isinstance(data, Extremes) and statexpr == "max":
                        statistic.max = data

        else:
            for expression in obj.columns:
                if expression != "unweighted" and expression != "sumw" and expression != "sumw2":
                    array = obj[expression].values
                    statistics[expression] = Statistics(moments=[Moments(InterpretedInlineBuffer.fromarray(array), n=1, weightpower=(0 if sumw2 is None else 1))])

        profiles = []
        for expression, statistic in statistics.items():
            if len(statistic.moments) == 0 and len(statistic.quantiles) == 0 and statistic.mode is None and statistic.min is None and statistic.max is None:
                profiles.append(Profile(expression, statistic))
