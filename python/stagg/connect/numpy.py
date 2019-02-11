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

from stagg import *
import stagg.interface

def tonumpy(obj):
    raise NotImplementedError

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

def tostagg(obj):
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
