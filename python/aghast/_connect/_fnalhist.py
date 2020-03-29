#!/usr/bin/env python

# BSD 3-Clause License; see https://github.com/scikit-hep/aghast/blob/master/LICENSE

import numpy

try:
    import fnal_column_analysis_tools.hist as hist
except ImportError:
    raise ImportError(
        "Install fnal_column_analysis_tools package with:\n    pip install fnal_column_analysis_tools"
    )

from aghast import *

_ovf_convention = lambda: RealOverflow(
    loc_underflow=BinLocation.below1,
    loc_overflow=BinLocation.above1,
    loc_nanflow=BinLocation.above2,
)


def from_fnalhist(obj):
    if not isinstance(obj, hist.Hist):
        raise TypeError(
            "cannot convert {0} to an aghast histogram".format(type(obj).__name__)
        )
    axes = []
    expanded_shape = []
    expanded_slices = []
    expansion_offsets = []
    for i, ax in enumerate(obj.axes()):
        if isinstance(ax, hist.Cat):
            cats = obj.identifiers(ax)
            expanded_shape.append(len(cats))
            expanded_slices.append(numpy.array(cats))
            expansion_offsets.append(i)
            axes.append(
                Axis(
                    CategoryBinning(categories=cats), expression=ax.name, title=ax.label
                )
            )
        elif isinstance(ax, hist.Bin) and ax._uniform:
            axes.append(
                Axis(
                    RegularBinning(
                        ax._bins,
                        RealInterval(ax._lo, ax._hi),
                        overflow=_ovf_convention(),
                    ),
                    expression=ax.name,
                    title=ax.label,
                )
            )
            expanded_shape.append(ax.size)
        elif isinstance(ax, hist.Bin) and not ax._uniform:
            axes.append(
                Axis(
                    EdgesBinning(ax._bins[:-1], overflow=_ovf_convention()),
                    expression=ax.name,
                    title=ax.label,
                )
            )
            expanded_shape.append(ax.size)
        else:
            raise NotImplementedError("unknown fnalhist Axis type")

    expanded_sumw = numpy.zeros(shape=expanded_shape, dtype=obj._dtype)
    if obj._sumw2 is not None:
        expanded_sumw2 = numpy.zeros(shape=expanded_shape, dtype=obj._dtype)

    walk_shape = [len(s) for s in expanded_slices]
    walk_indices = numpy.unravel_index(numpy.arange(numpy.prod(walk_shape)), walk_shape)
    extract_slices = list(zip(*(v[k] for v, k in zip(expanded_slices, walk_indices))))
    for iwalk, tup in enumerate(extract_slices):
        if tup in obj._sumw:
            insert_slice = [slice(None) for _ in expanded_shape]
            for walk_dim, ins_dim in enumerate(expansion_offsets):
                insert_slice[ins_dim] = walk_indices[walk_dim][iwalk]
            insert_slice = tuple(insert_slice)
            expanded_sumw[insert_slice] = obj._sumw[tup]
            if obj._sumw2 is not None:
                expanded_sumw2[insert_slice] = obj._sumw2[tup]

    if obj._sumw2 is not None:
        counts = WeightedCounts(
            sumw=InterpretedInlineBuffer.fromarray(expanded_sumw),
            sumw2=InterpretedInlineBuffer.fromarray(expanded_sumw2),
        )
    else:
        counts = UnweightedCounts(InterpretedInlineBuffer.fromarray(expanded_sumw))

    hout = Histogram(axis=axes, title=obj.label, counts=counts)
    return hout


def to_fnalhist(obj):
    if not isinstance(obj, Histogram):
        raise TypeError(
            "cannot convert {0} to a fnal_column_analysis_tools histogram".format(
                type(obj).__name__
            )
        )

    axes = []
    sparse_binning = {}
    dense_map = {}
    dense_shape = []
    for i, ax in enumerate(obj.axis):
        if isinstance(ax.binning, CategoryBinning):
            sparse_binning[i] = ax.binning.categories
            if ax.binning.loc_overflow.value < BinLocation.nonexistent.value:
                sparse_binning[i].insert(0, "")
            elif ax.binning.loc_overflow.value > BinLocation.nonexistent.value:
                sparse_binning[i].append("")
            new_ax = hist.Cat(ax.expression, ax.title)
            new_ax._categories = sparse_binning[i]
            axes.append(new_ax)
        elif isinstance(ax.binning, RegularBinning):
            dense_map[i] = [-numpy.inf, ..., numpy.inf, numpy.nan]
            dense_shape.append(ax.binning.num + 3)
            axes.append(
                hist.Bin(
                    ax.expression,
                    ax.title,
                    ax.binning.num,
                    ax.binning.interval.low,
                    ax.binning.interval.high,
                )
            )
        elif isinstance(ax.binning, EdgesBinning):
            dense_map[i] = [-numpy.inf, ..., numpy.inf, numpy.nan]
            dense_shape.append(ax.binning.edges.size + 2)
            axes.append(hist.Bin(ax.expression, ax.title, ax.binning.edges))
        else:
            raise TypeError(
                "unable to convert axes of type {0} to fnalhist axes".format(
                    type(ax).__name__
                )
            )

    hout = hist.Hist(obj.title, *axes)
    if isinstance(obj.counts, WeightedCounts):
        hout._init_sumw2()

    walk_shape = [len(s) for s in sparse_binning.values()]
    walk_indices = numpy.unravel_index(numpy.arange(numpy.prod(walk_shape)), walk_shape)
    for walk_index in zip(*walk_indices):
        insert_index = tuple(v[k] for k, v in zip(walk_index, sparse_binning.values()))
        extract_index = [None] * len(axes)
        for k, v in dense_map.items():
            extract_index[k] = v
        for k, v in zip(sparse_binning.keys(), walk_index):
            extract_index[k] = v
        extract_index = tuple(extract_index)
        if isinstance(obj.counts, UnweightedCounts):
            hout._sumw[insert_index] = obj.counts[extract_index]
        elif isinstance(obj.counts, WeightedCounts):
            counts = obj.counts[extract_index]
            hout._sumw[insert_index] = counts["sumw"]
            hout._sumw2[insert_index] = counts["sumw2"]

    return hout
