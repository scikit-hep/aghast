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

from __future__ import absolute_import

import sys
if sys.version_info[0] <= 2:
    # because absolute_import doesn't seem to work for PyPy 2.7
    import imp
    numpy_file, numpy_pathname, numpy_desc = imp.find_module("numpy")
    numpy = imp.load_module("numpy", numpy_file, numpy_pathname, numpy_desc)
else:
    import numpy

from aghast import *
import aghast.interface

def binning2array(binning):
    if not isinstance(binning, EdgesBinning):
        if not hasattr(binning, "toEdgesBinning"):
            raise TypeError("cannot convert {0} to a Numpy binning".format(type(binning).__name__))
        binning = binning.toEdgesBinning()
    if binning.overflow is not None and binning.overflow.loc_underflow != BinLocation.nonexistent and binning.overflow.loc_overflow != BinLocation.nonexistent:
        out = numpy.empty(len(binning.edges) + 2, dtype=binning.edges.dtype)
        out[0] = -numpy.inf
        out[-1] = numpy.inf
        out[1:-1] = binning.edges
    elif binning.overflow is not None and binning.overflow.loc_underflow != BinLocation.nonexistent:
        out = numpy.empty(len(binning.edges) + 1, dtype=binning.edges.dtype)
        out[0] = -numpy.inf
        out[1:] = binning.edges
    elif binning.overflow is not None and binning.overflow.loc_overflow != BinLocation.nonexistent:
        out = numpy.empty(len(binning.edges) + 1, dtype=binning.edges.dtype)
        out[-1] = numpy.inf
        out[:-1] = binning.edges
    else:
        out = binning.edges
    return out
        
def tonumpy(obj):
    if isinstance(obj, Histogram):
        edges = [binning2array(x.binning) for x in obj.axis]
        slices = ()
        for x in edges:
            start = -numpy.inf if x[0] == -numpy.inf else None
            stop = numpy.inf if x[-1] == numpy.inf else None
            slices = slices + (slice(start, stop),)
        counts = obj.counts[slices]
        if isinstance(counts, dict):
            counts = counts["sumw"]

        if len(edges) == 1:
            return counts, edges[0]
        elif len(edges) == 2:
            return counts, edges[0], edges[1]
        else:
            return counts, edges

    else:
        raise TypeError("cannot convert {0} to a Numpy histogram".format(type(obj).__name__))

def array2counts(array):
    if issubclass(array.dtype.type, numpy.integer) and (array >= 0).all():
        return UnweightedCounts(InterpretedInlineBuffer.fromarray(array))
    else:
        return WeightedCounts(InterpretedInlineBuffer.fromarray(array))

def array2binning(array):
    if len(array) <= 1:
        raise ValueError("binning array must have at least 2 elements: {0}".format(repr(array)))
    if not (array[1:] >= array[:-1]).all():
        raise ValueError("binning array must be monotonically increasing: {0}".format(repr(array)))
    bin_widths = array[1:] - array[:-1]
    bin_width = bin_widths.mean()
    if (numpy.absolute(bin_widths - bin_width) < 1e-10*(array[-1] - array[0])).all():
        return RegularBinning(len(array) - 1, RealInterval(array[0], array[-1]))
    else:
        return EdgesBinning(array)

def fromnumpy(obj):
    if isinstance(obj, tuple) and len(obj) == 2 and isinstance(obj[0], numpy.ndarray) and isinstance(obj[1], numpy.ndarray):
        counts, edges = obj
        return Histogram([Axis(array2binning(edges))], array2counts(counts))

    elif isinstance(obj, tuple) and len(obj) == 3 and isinstance(obj[0], numpy.ndarray) and isinstance(obj[1], numpy.ndarray) and isinstance(obj[2], numpy.ndarray):
        counts, xedges, yedges = obj
        return Histogram([Axis(array2binning(xedges)), Axis(array2binning(yedges))], array2counts(counts))

    elif isinstance(obj, tuple) and len(obj) == 2 and isinstance(obj[0], numpy.ndarray) and isinstance(obj[1], list) and all(isinstance(x, numpy.ndarray) for x in obj[1]):
        counts, edges = obj
        return Histogram([Axis(array2binning(x)) for x in edges], array2counts(counts))

    else:
        raise TypeError("not a recognized Numpy histogram type")
