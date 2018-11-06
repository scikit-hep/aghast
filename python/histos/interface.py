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

import ctypes
import functools
import operator
import math
import struct
import sys

import numpy
import flatbuffers

import histos.histos_generated.Assignment
import histos.histos_generated.Axis
import histos.histos_generated.BinnedEvaluatedFunction
import histos.histos_generated.BinnedRegion
import histos.histos_generated.Binning
import histos.histos_generated.BinPosition
import histos.histos_generated.CategoryBinning
import histos.histos_generated.Chunk
import histos.histos_generated.Collection
import histos.histos_generated.ColumnChunk
import histos.histos_generated.Column
import histos.histos_generated.Correlation
import histos.histos_generated.Counts
import histos.histos_generated.DecorationLanguage
import histos.histos_generated.Decoration
import histos.histos_generated.DimensionOrder
import histos.histos_generated.Distribution
import histos.histos_generated.DistributionStats
import histos.histos_generated.DType
import histos.histos_generated.EdgesBinning
import histos.histos_generated.Endianness
import histos.histos_generated.EvaluatedFunction
import histos.histos_generated.ExternalType
import histos.histos_generated.Extremes
import histos.histos_generated.Filter
import histos.histos_generated.FractionalErrorMethod
import histos.histos_generated.FractionBinning
import histos.histos_generated.FunctionData
import histos.histos_generated.FunctionObjectData
import histos.histos_generated.FunctionObject
import histos.histos_generated.Function
import histos.histos_generated.GenericErrors
import histos.histos_generated.HexagonalBinning
import histos.histos_generated.HexagonalCoordinates
import histos.histos_generated.Histogram
import histos.histos_generated.IntegerBinning
import histos.histos_generated.InterpretedBuffer
import histos.histos_generated.InterpretedExternalBuffer
import histos.histos_generated.InterpretedInlineBuffer
import histos.histos_generated.IrregularBinning
import histos.histos_generated.MetadataLanguage
import histos.histos_generated.Metadata
import histos.histos_generated.Moments
import histos.histos_generated.NonRealMapping
import histos.histos_generated.Ntuple
import histos.histos_generated.ObjectData
import histos.histos_generated.Object
import histos.histos_generated.OverlappingFillStrategy
import histos.histos_generated.Page
import histos.histos_generated.ParameterizedFunction
import histos.histos_generated.Parameter
import histos.histos_generated.Profile
import histos.histos_generated.Quantiles
import histos.histos_generated.RawBuffer
import histos.histos_generated.RawExternalBuffer
import histos.histos_generated.RawInlineBuffer
import histos.histos_generated.RealInterval
import histos.histos_generated.RealOverflow
import histos.histos_generated.Region
import histos.histos_generated.RegularBinning
import histos.histos_generated.Slice
import histos.histos_generated.SparseRegularBinning
import histos.histos_generated.TicTacToeOverflowBinning
import histos.histos_generated.UnweightedCounts
import histos.histos_generated.Variation
import histos.histos_generated.WeightedCounts

import histos.checktype

def typedproperty(check):
    def setparent(self, value):
        if isinstance(value, Histos):
            if hasattr(value, "_parent"):
                raise ValueError("object is already attached to another hierarchy: {0}".format(repr(value)))
            else:
                value._parent = self

        elif ((sys.version_info[0] >= 3 and isinstance(value, (str, bytes))) or (sys.version_info[0] < 3 and isinstance(value, basestring))):
            pass

        else:
            try:
                value = list(value)
            except TypeError:
                pass
            else:
                for x in value:
                    setparent(self, x)
        
    @property
    def prop(self):
        private = "_" + check.paramname
        if not hasattr(self, private):
            setattr(self, private, check.fromflatbuffers(getattr(self._flatbuffers, check.paramname.capitalize())()))
        return getattr(self, private)

    @prop.setter
    def prop(self, value):
        setparent(self, value)
        private = "_" + check.paramname
        setattr(self, private, check(value))

    return prop

def _valid(obj, seen, shape):
    if obj is None:
        return shape
    else:
        if id(obj) in seen:
            raise ValueError("hierarchy is recursively nested")
        seen.add(id(obj))
        return obj._valid(seen, shape)

def _getbykey(self, field, where):
    lookup = "_lookup_" + field
    if not hasattr(self, lookup):
        setattr(self, lookup, {x.identifier: x for x in getattr(self, field)})
        if len(getattr(self, lookup)) != len(getattr(self, field)):
            raise ValueError("{0}.{1} keys must be unique".format(type(self).__name__, field))
    return getattr(self, lookup)[where]

class Histos(object):
    def __repr__(self):
        return "<{0} at 0x{1:012x}>".format(type(self).__name__, id(self))

    def _top(self):
        out = self
        seen = set([id(out)])
        while hasattr(out, "_parent"):
            out = out._parent
            if id(out) in seen:
                raise ValueError("hierarchy is recursively nested")
        if not isinstance(out, Collection):
            raise ValueError("{0} object is not nested in a hierarchy".format(type(self).__name__))
        return out

class Enum(object):
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def __repr__(self):
        return repr(self.name)

    def __str__(self):
        return str(self.name)

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self is other or (isinstance(other, Enum) and self.name == other.name)

    def __ne__(self, other):
        return not self.__eq__(other)
    
################################################# Metadata

class Metadata(Histos):
    unspecified = Enum("unspecified", histos.histos_generated.MetadataLanguage.MetadataLanguage.meta_unspecified)
    json = Enum("json", histos.histos_generated.MetadataLanguage.MetadataLanguage.meta_json)
    language = [unspecified, json]

    _params = {
        "data":     histos.checktype.CheckString("Metadata", "data", required=True),
        "language": histos.checktype.CheckEnum("Metadata", "language", required=True, choices=language),
        }

    data     = typedproperty(_params["data"])
    language = typedproperty(_params["language"])

    def __init__(self, data, language=unspecified):
        self.data = data
        self.language = language

    def _valid(self, seen, shape):
        return shape

################################################# Decoration

class Decoration(Histos):
    unspecified = Enum("unspecified", histos.histos_generated.DecorationLanguage.DecorationLanguage.deco_unspecified)
    css         = Enum("css", histos.histos_generated.DecorationLanguage.DecorationLanguage.deco_css)
    vega        = Enum("vega", histos.histos_generated.DecorationLanguage.DecorationLanguage.deco_vega)
    root_json   = Enum("root_json", histos.histos_generated.DecorationLanguage.DecorationLanguage.deco_root_json)
    language = [unspecified, css, vega, root_json]

    _params = {
        "data":     histos.checktype.CheckString("Metadata", "data", required=True),
        "language": histos.checktype.CheckEnum("Metadata", "language", required=True, choices=language),
        }

    data     = typedproperty(_params["data"])
    language = typedproperty(_params["language"])

    def __init__(self, data, language=unspecified):
        self.data = data
        self.language = language

    def _valid(self, seen, shape):
        return shape

################################################# Object

class Object(Histos):
    def __init__(self):
        raise TypeError("{0} is an abstract base class; do not construct".format(type(self).__name__))

################################################# Buffers

class Buffer(Histos):
    none = Enum("none", histos.histos_generated.Filter.Filter.filter_none)
    gzip = Enum("gzip", histos.histos_generated.Filter.Filter.filter_gzip)
    lzma = Enum("lzma", histos.histos_generated.Filter.Filter.filter_lzma)
    lz4  = Enum("lz4", histos.histos_generated.Filter.Filter.filter_lz4)
    filters = [none, gzip, lzma, lz4]

    def __init__(self):
        raise TypeError("{0} is an abstract base class; do not construct".format(type(self).__name__))

class InlineBuffer(object):
    def __init__(self):
        raise TypeError("{0} is an abstract base class; do not construct".format(type(self).__name__))

class ExternalBuffer(object):
    memory   = Enum("memory", histos.histos_generated.ExternalType.ExternalType.external_memory)
    samefile = Enum("samefile", histos.histos_generated.ExternalType.ExternalType.external_samefile)
    file     = Enum("file", histos.histos_generated.ExternalType.ExternalType.external_file)
    url      = Enum("url", histos.histos_generated.ExternalType.ExternalType.external_url)
    types = [memory, samefile, file, url]

    def __init__(self):
        raise TypeError("{0} is an abstract base class; do not construct".format(type(self).__name__))

class RawBuffer(object):
    def __init__(self):
        raise TypeError("{0} is an abstract base class; do not construct".format(type(self).__name__))

class DTypeEnum(Enum):
    def __init__(self, name, value, dtype):
        super(DTypeEnum, self).__init__(name, value)
        self.dtype = dtype

class EndiannessEnum(Enum):
    def __init__(self, name, value, endianness):
        super(EndiannessEnum, self).__init__(name, value)
        self.endianness = endianness

class DimensionOrderEnum(Enum):
    def __init__(self, name, value, dimension_order):
        super(DimensionOrderEnum, self).__init__(name, value)
        self.dimension_order = dimension_order

class Interpretation(object):
    none    = DTypeEnum("none", histos.histos_generated.DType.DType.dtype_none, numpy.dtype(numpy.uint8))
    int8    = DTypeEnum("int8", histos.histos_generated.DType.DType.dtype_int8, numpy.dtype(numpy.int8))
    uint8   = DTypeEnum("uint8", histos.histos_generated.DType.DType.dtype_uint8, numpy.dtype(numpy.uint8))
    int16   = DTypeEnum("int16", histos.histos_generated.DType.DType.dtype_int16, numpy.dtype(numpy.int16))
    uint16  = DTypeEnum("uint16", histos.histos_generated.DType.DType.dtype_uint16, numpy.dtype(numpy.uint16))
    int32   = DTypeEnum("int32", histos.histos_generated.DType.DType.dtype_int32, numpy.dtype(numpy.int32))
    uint32  = DTypeEnum("uint32", histos.histos_generated.DType.DType.dtype_uint32, numpy.dtype(numpy.uint32))
    int64   = DTypeEnum("int64", histos.histos_generated.DType.DType.dtype_int64, numpy.dtype(numpy.int64))
    uint64  = DTypeEnum("uint64", histos.histos_generated.DType.DType.dtype_uint64, numpy.dtype(numpy.uint64))
    float32 = DTypeEnum("float32", histos.histos_generated.DType.DType.dtype_float32, numpy.dtype(numpy.float32))
    float64 = DTypeEnum("float64", histos.histos_generated.DType.DType.dtype_float64, numpy.dtype(numpy.float64))
    dtypes = [none, int8, uint8, int16, uint16, int32, uint32, int64, uint64, float32, float64]

    little_endian = EndiannessEnum("little_endian", histos.histos_generated.Endianness.Endianness.little_endian, "<")
    big_endian    = EndiannessEnum("big_endian", histos.histos_generated.Endianness.Endianness.big_endian, ">")
    endiannesses = [little_endian, big_endian]

    @property
    def numpy_dtype(self):
        return self.dtype.dtype.newbyteorder(self._endianness.endianness)

class InterpretedBuffer(Interpretation):
    c_order       = DimensionOrderEnum("c_order", histos.histos_generated.DimensionOrder.DimensionOrder.c_order, "C")
    fortran_order = DimensionOrderEnum("fortran", histos.histos_generated.DimensionOrder.DimensionOrder.fortran_order, "F")
    orders = [c_order, fortran_order]

    def __init__(self):
        raise TypeError("{0} is an abstract base class; do not construct".format(type(self).__name__))

################################################# RawInlineBuffer

class RawInlineBuffer(Buffer, RawBuffer, InlineBuffer):
    _params = {
        "buffer": histos.checktype.CheckBuffer("RawInlineBuffer", "buffer", required=True),
        }

    buffer = typedproperty(_params["buffer"])

    def __init__(self, buffer):
        self.buffer = buffer

    @property
    def numpy_array(self):
        return numpy.frombuffer(self.buffer, dtype=InterpretedBuffer.none.dtype)

################################################# RawExternalBuffer

class RawExternalBuffer(Buffer, RawBuffer, ExternalBuffer):
    _params = {
        "pointer":          histos.checktype.CheckInteger("RawExternalBuffer", "pointer", required=True, min=0),
        "numbytes":         histos.checktype.CheckInteger("RawExternalBuffer", "numbytes", required=True, min=0),
        "external_type":    histos.checktype.CheckEnum("RawExternalBuffer", "external_type", required=True, choices=ExternalBuffer.types),
        }

    pointer       = typedproperty(_params["pointer"])
    numbytes      = typedproperty(_params["numbytes"])
    external_type = typedproperty(_params["external_type"])

    def __init__(self, pointer, numbytes, external_type=ExternalBuffer.memory):
        self.pointer = pointer
        self.numbytes = numbytes
        self.external_type = external_type

    @property
    def numpy_array(self):
        return numpy.ctypeslib.as_array(ctypes.cast(self.pointer, ctypes.POINTER(ctypes.c_uint8)), shape=(self.numbytes,))

################################################# InlineBuffer

class InterpretedInlineBuffer(Buffer, InterpretedBuffer, InlineBuffer):
    _params = {
        "buffer":           histos.checktype.CheckBuffer("InterpretedInlineBuffer", "buffer", required=True),
        "filters":          histos.checktype.CheckVector("InterpretedInlineBuffer", "filters", required=False, type=Buffer.filters),
        "postfilter_slice": histos.checktype.CheckSlice("InterpretedInlineBuffer", "postfilter_slice", required=False),
        "dtype":            histos.checktype.CheckEnum("InterpretedInlineBuffer", "dtype", required=False, choices=InterpretedBuffer.dtypes),
        "endianness":       histos.checktype.CheckEnum("InterpretedInlineBuffer", "endianness", required=False, choices=InterpretedBuffer.endiannesses),
        "dimension_order":  histos.checktype.CheckEnum("InterpretedInlineBuffer", "dimension_order", required=False, choices=InterpretedBuffer.orders),
        }

    buffer           = typedproperty(_params["buffer"])
    filters          = typedproperty(_params["filters"])
    postfilter_slice = typedproperty(_params["postfilter_slice"])
    dtype            = typedproperty(_params["dtype"])
    endianness       = typedproperty(_params["endianness"])
    dimension_order  = typedproperty(_params["dimension_order"])

    def __init__(self, buffer=None, filters=None, postfilter_slice=None, dtype=InterpretedBuffer.none, endianness=InterpretedBuffer.little_endian, dimension_order=InterpretedBuffer.c_order):
        if buffer is None:
            self._buffer = None     # placeholder for auto-generated buffer
        else:
            self.buffer = buffer
        self.filters = filters
        self.postfilter_slice = postfilter_slice
        self.dtype = dtype
        self.endianness = endianness
        self.dimension_order = dimension_order

    def _valid(self, seen, shape):
        if self.filters is None:
            if self._buffer is None:
                self._buffer = numpy.zeros(functools.reduce(operator.mul, shape, 1), dtype=self.numpy_dtype).view(InterpretedBuffer.none.dtype)
                buf = None
            else:
                buf = self.buffer

        else:
            raise NotImplementedError(self.filters)

        if buf is not None:
            if len(buf.shape) != 1:
                raise ValueError("InterpretedInlineBuffer.buffer shape is {0} but only one-dimensional arrays are allowed".format(buf.shape))
            elif buf.dtype != InterpretedBuffer.none.dtype:
                raise ValueError("InterpretedInlineBuffer.buffer underlying dtype is {0} but should be {1} (untyped)".format(buf.dtype, InterpretedBuffer.none.dtype))
            elif len(buf) != functools.reduce(operator.mul, shape, self.numpy_dtype.itemsize):
                raise ValueError("InterpretedInlineBuffer.buffer length is {0} but multiplicity at this position in the hierarchy is {1}".format(len(buf), functools.reduce(operator.mul, shape, self.numpy_dtype.itemsize)))

        self._shape = shape

        if self.postfilter_slice is not None:
            if self.postfilter_slice.step == 0:
                raise ValueError("slice step cannot be zero")
            raise NotImplementedError("handle postfilter_slice")

        return shape

    @property
    def numpy_array(self):
        self._top().checkvalid()
        return self._buffer.view(self.numpy_dtype).reshape(self._shape, order=self.dimension_order.dimension_order)

################################################# ExternalBuffer

class InterpretedExternalBuffer(Buffer, InterpretedBuffer, ExternalBuffer):
    _params = {
        "pointer":          histos.checktype.CheckInteger("ExternalBuffer", "pointer", required=True, min=0),
        "numbytes":         histos.checktype.CheckInteger("ExternalBuffer", "numbytes", required=True, min=0),
        "external_type":    histos.checktype.CheckEnum("ExternalBuffer", "external_type", required=False, choices=ExternalBuffer.types),
        "filters":          histos.checktype.CheckVector("ExternalBuffer", "filters", required=False, type=Buffer.filters),
        "postfilter_slice": histos.checktype.CheckSlice("ExternalBuffer", "postfilter_slice", required=False),
        "dtype":            histos.checktype.CheckEnum("ExternalBuffer", "dtype", required=False, choices=InterpretedBuffer.dtypes),
        "endianness":       histos.checktype.CheckEnum("ExternalBuffer", "endianness", required=False, choices=InterpretedBuffer.endiannesses),
        "dimension_order":  histos.checktype.CheckEnum("ExternalBuffer", "dimension_order", required=False, choices=InterpretedBuffer.orders),
        "location":         histos.checktype.CheckString("ExternalBuffer", "location", required=False),
        }

    pointer          = typedproperty(_params["pointer"])
    numbytes         = typedproperty(_params["numbytes"])
    external_type    = typedproperty(_params["external_type"])
    filters          = typedproperty(_params["filters"])
    postfilter_slice = typedproperty(_params["postfilter_slice"])
    dtype            = typedproperty(_params["dtype"])
    endianness       = typedproperty(_params["endianness"])
    dimension_order  = typedproperty(_params["dimension_order"])
    location         = typedproperty(_params["location"])

    def __init__(self, pointer=None, numbytes=None, external_type=ExternalBuffer.memory, filters=None, postfilter_slice=None, dtype=InterpretedBuffer.none, endianness=InterpretedBuffer.little_endian, dimension_order=InterpretedBuffer.c_order, location=""):
        if pointer is None and numbytes is None:
            self._pointer = None    # placeholder for auto-generated buffer
            self._numbytes = None
        else:
            self.pointer = pointer
            self.numbytes = numbytes
        self.external_type = external_type
        self.filters = filters
        self.postfilter_slice = postfilter_slice
        self.dtype = dtype
        self.endianness = endianness
        self.dimension_order = dimension_order
        self.location = location

    @property
    def numpy_array(self):
        self._top().checkvalid()
        out = numpy.ctypeslib.as_array(ctypes.cast(self.pointer, ctypes.POINTER(ctypes.c_uint8)), shape=(self.numbytes,))
        return out.view(self.numpy_dtype).reshape(self._shape, order=self.dimension_order.dimension_order)

################################################# Binning

class Binning(Histos):
    def __init__(self):
        raise TypeError("{0} is an abstract base class; do not construct".format(type(self).__name__))

################################################# FractionalBinning

class FractionalBinning(Binning):
    normal           = Enum("normal", histos.histos_generated.FractionalErrorMethod.FractionalErrorMethod.frac_normal)
    clopper_pearson  = Enum("clopper_pearson", histos.histos_generated.FractionalErrorMethod.FractionalErrorMethod.frac_clopper_pearson)
    wilson           = Enum("wilson", histos.histos_generated.FractionalErrorMethod.FractionalErrorMethod.frac_wilson)
    agresti_coull    = Enum("agresti_coull", histos.histos_generated.FractionalErrorMethod.FractionalErrorMethod.frac_agresti_coull)
    feldman_cousins  = Enum("feldman_cousins", histos.histos_generated.FractionalErrorMethod.FractionalErrorMethod.frac_feldman_cousins)
    jeffrey          = Enum("jeffrey", histos.histos_generated.FractionalErrorMethod.FractionalErrorMethod.frac_jeffrey)
    bayesian_uniform = Enum("bayesian_uniform", histos.histos_generated.FractionalErrorMethod.FractionalErrorMethod.frac_bayesian_uniform)
    error_methods = [normal, clopper_pearson, wilson, agresti_coull, feldman_cousins, jeffrey, bayesian_uniform]

    _params = {
        "error_method": histos.checktype.CheckEnum("FractionalBinning", "error_method", required=False, choices=error_methods),
        }

    error_method = typedproperty(_params["error_method"])

    def __init__(self, error_method=normal):
        self.error_method = error_method

    def _valid(self, seen, shape):
        return shape

################################################# BinPosition

class BinPosition(object):
    below3 = Enum("below3", histos.histos_generated.BinPosition.BinPosition.pos_below3)
    below2 = Enum("below2", histos.histos_generated.BinPosition.BinPosition.pos_below2)
    below1 = Enum("below1", histos.histos_generated.BinPosition.BinPosition.pos_below1)
    nonexistent = Enum("nonexistent", histos.histos_generated.BinPosition.BinPosition.pos_nonexistent)
    above1 = Enum("above1", histos.histos_generated.BinPosition.BinPosition.pos_above1)
    above2 = Enum("above2", histos.histos_generated.BinPosition.BinPosition.pos_above2)
    above3 = Enum("above3", histos.histos_generated.BinPosition.BinPosition.pos_above3)
    positions = [below3, below2, below1, nonexistent, above1, above2, above3]

################################################# IntegerBinning

class IntegerBinning(Binning, BinPosition):
    _params = {
        "min":           histos.checktype.CheckInteger("IntegerBinning", "min", required=True),
        "max":           histos.checktype.CheckInteger("IntegerBinning", "max", required=True),
        "pos_underflow": histos.checktype.CheckEnum("IntegerBinning", "pos_underflow", required=False, choices=BinPosition.positions),
        "pos_overflow":  histos.checktype.CheckEnum("IntegerBinning", "pos_overflow", required=False, choices=BinPosition.positions),
        }

    min           = typedproperty(_params["min"])
    max           = typedproperty(_params["max"])
    pos_underflow = typedproperty(_params["pos_underflow"])
    pos_overflow  = typedproperty(_params["pos_overflow"])

    def __init__(self, min, max, pos_underflow=BinPosition.nonexistent, pos_overflow=BinPosition.nonexistent):
        self.min = min
        self.max = max
        self.pos_underflow = pos_underflow
        self.pos_overflow = pos_overflow

    def _valid(self, seen, shape):
        if self.min >= self.max:
            raise ValueError("IntegerBinning.min ({0}) must be strictly less than IntegerBinning.max ({1})".format(self.min, self.max))

        if self.pos_underflow != BinPosition.nonexistent and self.pos_overflow != BinPosition.nonexistent and self.pos_underflow == self.pos_overflow:
            raise ValueError("IntegerBinning.pos_underflow and IntegerBinning.pos_overflow must not be equal unless they are both nonexistent")

        return (self.max - self.min + 1 + int(self.pos_underflow != BinPosition.nonexistent) + int(self.pos_overflow != BinPosition.nonexistent),)

################################################# RealInterval

class RealInterval(Histos):
    _params = {
        "low":            histos.checktype.CheckNumber("RealInterval", "low", required=True),
        "high":           histos.checktype.CheckNumber("RealInterval", "high", required=True),
        "low_inclusive":  histos.checktype.CheckBool("RealInterval", "low_inclusive", required=False),
        "high_inclusive": histos.checktype.CheckBool("RealInterval", "high_inclusive", required=False),
        }

    low            = typedproperty(_params["low"])
    high           = typedproperty(_params["high"])
    low_inclusive  = typedproperty(_params["low_inclusive"])
    high_inclusive = typedproperty(_params["high_inclusive"])

    def __init__(self, low, high, low_inclusive=True, high_inclusive=False):
        self.low = low
        self.high = high
        self.low_inclusive = low_inclusive
        self.high_inclusive = high_inclusive

    def _valid(self, seen, shape):
        if self.low > self.high:
            raise ValueError("RealInterval.low ({0}) must be less than or equal to RealInterval.high ({1})".format(self.low, self.high))
        if self.low == self.high and not self.low_inclusive and not self.high_inclusive:
            raise ValueError("RealInterval describes an empty set ({0} == {1} and both endpoints are exclusive)".format(self.low, self.high))
        return shape

################################################# RealOverflow

class RealOverflow(Histos, BinPosition):
    missing      = Enum("missing", histos.histos_generated.NonRealMapping.NonRealMapping.missing)
    in_underflow = Enum("in_underflow", histos.histos_generated.NonRealMapping.NonRealMapping.in_underflow)
    in_overflow  = Enum("in_overflow", histos.histos_generated.NonRealMapping.NonRealMapping.in_overflow)
    in_nanflow   = Enum("in_nanflow", histos.histos_generated.NonRealMapping.NonRealMapping.in_nanflow)
    mappings = [missing, in_underflow, in_overflow, in_nanflow]

    _params = {
        "pos_underflow": histos.checktype.CheckEnum("RealOverflow", "pos_underflow", required=False, choices=BinPosition.positions),
        "pos_overflow":  histos.checktype.CheckEnum("RealOverflow", "pos_overflow", required=False, choices=BinPosition.positions),
        "pos_nanflow":   histos.checktype.CheckEnum("RealOverflow", "pos_nanflow", required=False, choices=BinPosition.positions),
        "minf_mapping":  histos.checktype.CheckEnum("RealOverflow", "minf_mapping", required=False, choices=mappings),
        "pinf_mapping":  histos.checktype.CheckEnum("RealOverflow", "pinf_mapping", required=False, choices=mappings),
        "nan_mapping":   histos.checktype.CheckEnum("RealOverflow", "nan_mapping", required=False, choices=mappings),
        }

    pos_underflow = typedproperty(_params["pos_underflow"])
    pos_overflow  = typedproperty(_params["pos_overflow"])
    pos_nanflow   = typedproperty(_params["pos_nanflow"])
    minf_mapping  = typedproperty(_params["minf_mapping"])
    pinf_mapping  = typedproperty(_params["pinf_mapping"])
    nan_mapping   = typedproperty(_params["nan_mapping"])

    def __init__(self, pos_underflow=BinPosition.nonexistent, pos_overflow=BinPosition.nonexistent, pos_nanflow=BinPosition.nonexistent, minf_mapping=in_underflow, pinf_mapping=in_overflow, nan_mapping=in_nanflow):
        self.pos_underflow = pos_underflow
        self.pos_overflow = pos_overflow
        self.pos_nanflow = pos_nanflow
        self.minf_mapping = minf_mapping
        self.pinf_mapping = pinf_mapping
        self.nan_mapping = nan_mapping

    def _valid(self, seen, shape):
        if self.pos_underflow != BinPosition.nonexistent and self.pos_overflow != BinPosition.nonexistent and self.pos_underflow == self.pos_overflow:
            raise ValueError("RealOverflow.pos_underflow and RealOverflow.pos_overflow must not be equal unless they are both nonexistent")
        if self.pos_underflow != BinPosition.nonexistent and self.pos_nanflow != BinPosition.nonexistent and self.pos_underflow == self.pos_nanflow:
            raise ValueError("RealOverflow.pos_underflow and RealOverflow.pos_nanflow must not be equal unless they are both nonexistent")
        if self.pos_overflow != BinPosition.nonexistent and self.pos_nanflow != BinPosition.nonexistent and self.pos_overflow == self.pos_nanflow:
            raise ValueError("RealOverflow.pos_overflow and RealOverflow.pos_nanflow must not be equal unless they are both nonexistent")

        return (int(self.pos_underflow != BinPosition.nonexistent) + int(self.pos_overflow != BinPosition.nonexistent) + int(self.pos_nanflow != BinPosition.nonexistent),)

################################################# RegularBinning

class RegularBinning(Binning):
    _params = {
        "num":      histos.checktype.CheckInteger("RegularBinning", "num", required=True, min=1),
        "interval": histos.checktype.CheckClass("RegularBinning", "interval", required=True, type=RealInterval),
        "overflow": histos.checktype.CheckClass("RegularBinning", "overflow", required=False, type=RealOverflow),
        "circular": histos.checktype.CheckBool("RegularBinning", "circular", required=False),
        }

    num      = typedproperty(_params["num"])
    interval = typedproperty(_params["interval"])
    overflow = typedproperty(_params["overflow"])
    circular = typedproperty(_params["circular"])

    def __init__(self, num, interval, overflow=None, circular=False):
        self.num = num
        self.interval = interval
        self.overflow = overflow
        self.circular = circular

    def _valid(self, seen, shape):
        _valid(self.interval, seen, shape)
        if math.isinf(self.interval.low) or math.isnan(self.interval.low):
            raise ValueError("RegularBinning.interval.low must be finite")
        if math.isinf(self.interval.high) or math.isnan(self.interval.high):
            raise ValueError("RegularBinning.interval.high must be finite")
        if self.overflow is None:
            overflowdims, = RealOverflow()._valid(set(), shape)
        else:
            overflowdims, = _valid(self.overflow, seen, shape)
        return (self.num + overflowdims,)

################################################# TicTacToeOverflowBinning

class TicTacToeOverflowBinning(Binning):
    _params = {
        "xnum":      histos.checktype.CheckInteger("TicTacToeOverflowBinning", "xnum", required=True, min=1),
        "ynum":      histos.checktype.CheckInteger("TicTacToeOverflowBinning", "ynum", required=True, min=1),
        "xinterval": histos.checktype.CheckClass("TicTacToeOverflowBinning", "xinterval", required=True, type=RealInterval),
        "yinterval": histos.checktype.CheckClass("TicTacToeOverflowBinning", "yinterval", required=True, type=RealInterval),
        "xoverflow": histos.checktype.CheckClass("TicTacToeOverflowBinning", "xoverflow", required=False, type=RealOverflow),
        "yoverflow": histos.checktype.CheckClass("TicTacToeOverflowBinning", "yoverflow", required=False, type=RealOverflow),
        }

    xnum      = typedproperty(_params["xnum"])
    ynum      = typedproperty(_params["ynum"])
    xinterval = typedproperty(_params["xinterval"])
    yinterval = typedproperty(_params["yinterval"])
    xoverflow = typedproperty(_params["xoverflow"])
    yoverflow = typedproperty(_params["yoverflow"])

    def __init__(self, xnum, ynum, xinterval, yinterval, xoverflow=None, yoverflow=None):
        self.xnum = xnum
        self.ynum = ynum
        self.xinterval = xinterval
        self.yinterval = yinterval
        self.xoverflow = xoverflow
        self.yoverflow = yoverflow

    def _valid(self, seen, shape):
        _valid(self.xinterval, seen, shape)
        _valid(self.yinterval, seen, shape)
        if self.xoverflow is None:
            xoverflowdims, = RealOverflow()._valid(set(), shape)
        else:
            xoverflowdims, = _valid(self.xoverflow, seen, shape)
        if self.yoverflow is None:
            yoverflowdims, = RealOverflow()._valid(set(), shape)
        else:
            yoverflowdims, = _valid(self.yoverflow, seen, shape)

        return (self.xnum + xoverflowdims, self.ynum + yoverflowdims)
        
################################################# HexagonalBinning

class HexagonalBinning(Binning):
    offset         = Enum("offset", histos.histos_generated.HexagonalCoordinates.HexagonalCoordinates.hex_offset)
    doubled_offset = Enum("doubled_offset", histos.histos_generated.HexagonalCoordinates.HexagonalCoordinates.hex_doubled_offset)
    cube_xy        = Enum("cube_xy", histos.histos_generated.HexagonalCoordinates.HexagonalCoordinates.hex_cube_xy)
    cube_yz        = Enum("cube_yz", histos.histos_generated.HexagonalCoordinates.HexagonalCoordinates.hex_cube_yz)
    cube_xz        = Enum("cube_xz", histos.histos_generated.HexagonalCoordinates.HexagonalCoordinates.hex_cube_xz)
    coordinates = [offset, doubled_offset, cube_xy, cube_yz, cube_xz]

    _params = {
        "qmin":        histos.checktype.CheckInteger("HexagonalBinning", "qmin", required=True),
        "qmax":        histos.checktype.CheckInteger("HexagonalBinning", "qmax", required=True),
        "rmin":        histos.checktype.CheckInteger("HexagonalBinning", "rmin", required=True),
        "rmax":        histos.checktype.CheckInteger("HexagonalBinning", "rmax", required=True),
        "coordinates": histos.checktype.CheckEnum("HexagonalBinning", "coordinates", required=False, choices=coordinates),
        "xorigin":     histos.checktype.CheckNumber("HexagonalBinning", "xorigin", required=False, min_inclusive=False, max_inclusive=False),
        "yorigin":     histos.checktype.CheckNumber("HexagonalBinning", "yorigin", required=False, min_inclusive=False, max_inclusive=False),
        "qangle":      histos.checktype.CheckNumber("HexagonalBinning", "qangle", required=False, min=-0.5*math.pi, max=0.5*math.pi),
        "qoverflow":   histos.checktype.CheckClass("HexagonalBinning", "qoverflow", required=False, type=RealOverflow),
        "roverflow":   histos.checktype.CheckClass("HexagonalBinning", "roverflow", required=False, type=RealOverflow),
        }

    qmin        = typedproperty(_params["qmin"])
    qmax        = typedproperty(_params["qmax"])
    rmin        = typedproperty(_params["rmin"])
    rmax        = typedproperty(_params["rmax"])
    coordinates = typedproperty(_params["coordinates"])
    xorigin     = typedproperty(_params["xorigin"])
    yorigin     = typedproperty(_params["yorigin"])
    qangle      = typedproperty(_params["qangle"])
    qoverflow   = typedproperty(_params["qoverflow"])
    roverflow   = typedproperty(_params["roverflow"])

    def __init__(self, qmin, qmax, rmin, rmax, coordinates=offset, xorigin=0.0, yorigin=0.0, qangle=0.0, qoverflow=None, roverflow=None):
        self.qmin = qmin
        self.qmax = qmax
        self.rmin = rmin
        self.rmax = rmax
        self.coordinates = coordinates
        self.xorigin = xorigin
        self.yorigin = yorigin
        self.qoverflow = qoverflow
        self.roverflow = roverflow

    def _valid(self, seen, shape):
        qnum = self.qmax - self.qmin + 1
        rnum = self.rmax - self.rmin + 1
        if self.qoverflow is None:
            qoverflowdims, = RealOverflow()._valid(set(), shape)
        else:
            qoverflowdims, = _valid(self.qoverflow, seen, shape)
        if self.roverflow is None:
            roverflowdims, = RealOverflow()._valid(set(), shape)
        else:
            roverflowdims, = _valid(self.roverflow, seen, shape)
        return (qnum + qoverflowdims, rnum + roverflowdims)

################################################# EdgesBinning

class EdgesBinning(Binning):
    _params = {
        "edges":    histos.checktype.CheckVector("EdgesBinning", "edges", required=True, type=float, minlen=1),
        "overflow": histos.checktype.CheckClass("EdgesBinning", "overflow", required=False, type=RealOverflow),
        }

    edges    = typedproperty(_params["edges"])
    overflow = typedproperty(_params["overflow"])

    def __init__(self, edges, overflow=None):
        self.edges = edges
        self.overflow = overflow

    def _valid(self, seen, shape):
        if any(math.isinf(x) or math.isnan(x) for x in self.edges):
            raise ValueError("EdgesBinning.edges must all be finite")
        if not numpy.greater(self.edges[1:], self.edges[:-1]).all():
            raise ValueError("EdgesBinning.edges must be strictly increasing")
        if self.overflow is None:
            numoverflow, = RealOverflow()._valid(set(), shape)
        else:
            numoverflow, = _valid(self.overflow, seen, shape)
        return (len(self.edges) - 1 + numoverflow,)

################################################# EdgesBinning

class IrregularBinning(Binning):
    all = Enum("all", histos.histos_generated.OverlappingFillStrategy.OverlappingFillStrategy.overfill_all)
    first = Enum("first", histos.histos_generated.OverlappingFillStrategy.OverlappingFillStrategy.overfill_first)
    last = Enum("last", histos.histos_generated.OverlappingFillStrategy.OverlappingFillStrategy.overfill_last)
    overlapping_fill_strategies = [all, first, last]

    _params = {
        "intervals":        histos.checktype.CheckVector("IrregularBinning", "intervals", required=True, type=RealInterval),
        "overflow":         histos.checktype.CheckClass("IrregularBinning", "overflow", required=False, type=RealOverflow),
        "overlapping_fill": histos.checktype.CheckEnum("IrregularBinning", "overlapping_fill", required=False, choices=overlapping_fill_strategies),
        }

    intervals        = typedproperty(_params["intervals"])
    overflow         = typedproperty(_params["overflow"])
    overlapping_fill = typedproperty(_params["overlapping_fill"])

    def __init__(self, intervals, overflow=None, overlapping_fill=all):
        self.intervals = intervals
        self.overflow = overflow
        self.overlapping_fill = overlapping_fill

    def _valid(self, seen, shape):
        for x in self.intervals:
            _valid(x, seen, shape)
        if self.overflow is None:
            numoverflow, = RealOverflow()._valid(set(), shape)
        else:
            numoverflow, = _valid(self.overflow, seen, shape)
        return (len(self.intervals) + numoverflow,)

################################################# CategoryBinning

class CategoryBinning(Binning, BinPosition):
    _params = {
        "categories": histos.checktype.CheckVector("CategoryBinning", "categories", required=True, type=str),
        "pos_overflow":  histos.checktype.CheckEnum("CategoryBinning", "pos_overflow", required=False, choices=BinPosition.positions),
        }

    categories = typedproperty(_params["categories"])
    pos_overflow = typedproperty(_params["pos_overflow"])

    def __init__(self, categories, pos_overflow=BinPosition.nonexistent):
        self.categories = categories
        self.pos_overflow = pos_overflow

    def _valid(self, seen, shape):
        if len(self.categories) != len(set(self.categories)):
            raise ValueError("SparseRegularBinning.bins must be unique")
        return (len(self.categories) + (self.pos_overflow != BinPosition.nonexistent),)

################################################# SparseRegularBinning

class SparseRegularBinning(Binning, BinPosition):
    _params = {
        "bins":        histos.checktype.CheckVector("SparseRegularBinning", "bins", required=True, type=int),
        "bin_width":   histos.checktype.CheckNumber("SparseRegularBinning", "bin_width", required=True, min=0, min_inclusive=False),
        "origin":      histos.checktype.CheckNumber("SparseRegularBinning", "origin", required=False),
        "pos_nanflow": histos.checktype.CheckEnum("SparseRegularBinning", "pos_nanflow", required=False, choices=BinPosition.positions),
        }

    bins        = typedproperty(_params["bins"])
    bin_width   = typedproperty(_params["bin_width"])
    origin      = typedproperty(_params["origin"])
    pos_nanflow = typedproperty(_params["pos_nanflow"])

    def __init__(self, bins, bin_width, origin=0.0, pos_nanflow=BinPosition.nonexistent):
        self.bins = bins
        self.bin_width = bin_width
        self.origin = origin
        self.pos_nanflow = pos_nanflow

    def _valid(self, seen, shape):
        if len(self.bins) != len(numpy.unique(self.bins)):
            raise ValueError("SparseRegularBinning.bins must be unique")
        return (len(self.bins) + (self.pos_nanflow != BinPosition.nonexistent),)

################################################# Axis

class Axis(Histos):
    _params = {
        "binning":    histos.checktype.CheckClass("Axis", "binning", required=False, type=Binning),
        "expression": histos.checktype.CheckString("Axis", "expression", required=False),
        "title":      histos.checktype.CheckString("Axis", "title", required=False),
        "metadata":   histos.checktype.CheckClass("Axis", "metadata", required=False, type=Metadata),
        "decoration": histos.checktype.CheckClass("Axis", "decoration", required=False, type=Decoration),
        }

    binning    = typedproperty(_params["binning"])
    expression = typedproperty(_params["expression"])
    title      = typedproperty(_params["title"])
    metadata   = typedproperty(_params["metadata"])
    decoration = typedproperty(_params["decoration"])

    def __init__(self, binning=None, expression="", title="", metadata=None, decoration=None):
        self.binning = binning
        self.expression = expression
        self.title = title
        self.metadata = metadata
        self.decoration = decoration

    def _valid(self, seen, shape):
        if self.binning is None:
            binshape = (1,)
        else:
            binshape = _valid(self.binning, seen, shape)
        _valid(self.metadata, seen, shape)
        _valid(self.decoration, seen, shape)
        return binshape

################################################# Counts

class Counts(Histos):
    def __init__(self):
        raise TypeError("{0} is an abstract base class; do not construct".format(type(self).__name__))

################################################# UnweightedCounts

class UnweightedCounts(Counts):
    _params = {
        "counts":  histos.checktype.CheckClass("UnweightedCounts", "counts", required=True, type=InterpretedBuffer),
        }

    counts = typedproperty(_params["counts"])

    def __init__(self, counts):
        self.counts = counts

################################################# WeightedCounts

class WeightedCounts(Counts):
    _params = {
        "sumw":   histos.checktype.CheckClass("WeightedCounts", "sumw", required=True, type=InterpretedBuffer),
        "sumw2":  histos.checktype.CheckClass("WeightedCounts", "sumw2", required=True, type=InterpretedBuffer),
        "counts": histos.checktype.CheckClass("WeightedCounts", "counts", required=False, type=UnweightedCounts),
        }

    sumw   = typedproperty(_params["sumw"])
    sumw2  = typedproperty(_params["sumw2"])
    counts = typedproperty(_params["counts"])

    def __init__(self, sumw, sumw2, counts=None):
        self.sumw = sumw
        self.sumw2 = sumw2
        self.counts = counts

################################################# Correlation

class Correlation(Histos):
    _params = {
        "sumwx":   histos.checktype.CheckClass("Correlation", "sumwx", required=True, type=InterpretedBuffer),
        "sumwxy":  histos.checktype.CheckClass("Correlation", "sumwxy", required=True, type=InterpretedBuffer),
        }

    sumwx  = typedproperty(_params["sumwx"])
    sumwxy = typedproperty(_params["sumwxy"])

    def __init__(self, sumwx, sumwxy):
        self.sumwx = sumwx
        self.sumwxy  = sumwxy 

################################################# Extremes

class Extremes(Histos):
    _params = {
        "min":           histos.checktype.CheckClass("Extremes", "min", required=True, type=InterpretedBuffer),
        "max":           histos.checktype.CheckClass("Extremes", "max", required=True, type=InterpretedBuffer),
        "excludes_minf": histos.checktype.CheckBool("Extremes", "excludes_minf", required=False),
        "excludes_pinf": histos.checktype.CheckBool("Extremes", "excludes_pinf", required=False),
        "excludes_nan":  histos.checktype.CheckBool("Extremes", "excludes_nan", required=False),
        }

    min           = typedproperty(_params["min"])
    max           = typedproperty(_params["max"])
    excludes_minf = typedproperty(_params["excludes_minf"])
    excludes_pinf = typedproperty(_params["excludes_pinf"])
    excludes_nan  = typedproperty(_params["excludes_nan"])

    def __init__(self, min, max, excludes_minf=False, excludes_pinf=False, excludes_nan=True):
        self.min = min
        self.max = max
        self.excludes_minf = excludes_minf
        self.excludes_pinf = excludes_pinf
        self.excludes_nan = excludes_nan

################################################# Moments

class Moments(Histos):
    _params = {
        "sumwn": histos.checktype.CheckClass("Moments", "sumwn", required=True, type=InterpretedBuffer),
        "n":     histos.checktype.CheckInteger("Moments", "n", required=True, min=1),
        }

    sumwn = typedproperty(_params["sumwn"])
    n     = typedproperty(_params["n"])

    def __init__(self, sumwn, n):
        self.sumwn = sumwn
        self.n = n

################################################# Quantiles

class Quantiles(Histos):
    _params = {
        "values": histos.checktype.CheckClass("Quantiles", "values", required=True, type=InterpretedBuffer),
        "p":      histos.checktype.CheckNumber("Quantiles", "p", required=True, min=0.0, max=1.0),
        }

    values = typedproperty(_params["values"])
    p      = typedproperty(_params["p"])

    def __init__(self, values, p=0.5):
        self.values = values
        self.p = p

################################################# GenericErrors

class GenericErrors(Histos):
    _params = {
        "errors": histos.checktype.CheckClass("GenericErrors", "errors", required=True, type=InterpretedBuffer),
        "p":      histos.checktype.CheckNumber("GenericErrors", "p", required=False, min=0.0, max=1.0),
        }

    errors = typedproperty(_params["errors"])
    p      = typedproperty(_params["p"])

    def __init__(self, errors, p=math.erf(math.sqrt(0.5))):
        self.errors = errors
        self.p = p

################################################# DistributionStats

class DistributionStats(Histos):
    _params = {
        "correlation":    histos.checktype.CheckClass("DistributionStats", "correlation", required=False, type=Correlation),
        "extremes":       histos.checktype.CheckClass("DistributionStats", "extremes", required=False, type=Extremes),
        "moments":        histos.checktype.CheckVector("DistributionStats", "moments", required=False, type=Moments),
        "quantiles":      histos.checktype.CheckVector("DistributionStats", "quantiles", required=False, type=Quantiles),
        "generic_errors": histos.checktype.CheckVector("DistributionStats", "generic_errors", required=False, type=GenericErrors),
        }

    correlation    = typedproperty(_params["correlation"])
    extremes       = typedproperty(_params["extremes"])
    moments        = typedproperty(_params["moments"])
    quantiles      = typedproperty(_params["quantiles"])
    generic_errors = typedproperty(_params["generic_errors"])

    def __init__(self, correlation=None, extremes=None, moments=None, quantiles=None, generic_errors=None):
        self.correlation = correlation
        self.extremes = extremes
        self.moments = moments
        self.quantiles = quantiles
        self.generic_errors = generic_errors

################################################# Distribution

class Distribution(Histos):
    _params = {
        "counts": histos.checktype.CheckClass("Distribution", "counts", required=True, type=Counts),
        "stats":  histos.checktype.CheckClass("Distribution", "stats", required=False, type=DistributionStats),
        }

    counts = typedproperty(_params["counts"])
    stats  = typedproperty(_params["stats"])

    def __init__(self, counts, stats=None):
        self.counts = counts
        self.stats = stats

################################################# Profile

class Profile(Histos):
    _params = {
        "expression": histos.checktype.CheckString("Profile", "expression", required=True),
        "title":      histos.checktype.CheckString("Profile", "title", required=False),
        "metadata":   histos.checktype.CheckClass("Profile", "metadata", required=False, type=Metadata),
        "decoration": histos.checktype.CheckClass("Profile", "decoration", required=False, type=Decoration),
        }

    expression = typedproperty(_params["expression"])
    title      = typedproperty(_params["title"])
    metadata   = typedproperty(_params["metadata"])
    decoration = typedproperty(_params["decoration"])

    def __init__(self, expression, title="", metadata=None, decoration=None):
        self.expression = expression
        self.title = title
        self.metadata = metadata
        self.decoration = decoration

################################################# Parameter

class Parameter(Histos):
    _params = {
        "identifier": histos.checktype.CheckKey("Parameter", "identifier", required=True, type=str),
        "value":      histos.checktype.CheckNumber("Parameter", "value", required=True),
        }

    identifier = typedproperty(_params["identifier"])
    value      = typedproperty(_params["value"])

    def __init__(self, identifier, value):
        self.identifier = identifier
        self.value = value

    def _valid(self, seen, shape):
        return shape

################################################# Function

class Function(Histos):
    def __init__(self):
        raise TypeError("{0} is an abstract base class; do not construct".format(type(self).__name__))

################################################# FunctionObject

class FunctionObject(Object):
    def __init__(self):
        raise TypeError("{0} is an abstract base class; do not construct".format(type(self).__name__))

################################################# ParameterizedFunction

class ParameterizedFunction(Function, FunctionObject):
    _params = {
        "identifier": histos.checktype.CheckKey("ParameterizedFunction", "identifier", required=True, type=str),
        "expression": histos.checktype.CheckString("ParameterizedFunction", "expression", required=True),
        "parameters": histos.checktype.CheckVector("ParameterizedFunction", "parameters", required=True, type=Parameter),
        "contours":   histos.checktype.CheckVector("ParameterizedFunction", "contours", required=False, type=float),
        "title":      histos.checktype.CheckString("ParameterizedFunction", "title", required=False),
        "metadata":   histos.checktype.CheckClass("ParameterizedFunction", "metadata", required=False, type=Metadata),
        "decoration": histos.checktype.CheckClass("ParameterizedFunction", "decoration", required=False, type=Decoration),
        }

    identifier = typedproperty(_params["identifier"])
    expression = typedproperty(_params["expression"])
    parameters = typedproperty(_params["parameters"])
    contours   = typedproperty(_params["contours"])
    title      = typedproperty(_params["title"])
    metadata   = typedproperty(_params["metadata"])
    decoration = typedproperty(_params["decoration"])

    def __init__(self, identifier, expression, parameters, contours=None, title="", metadata=None, decoration=None):
        self.identifier = identifier
        self.expression = expression
        self.parameters = parameters
        self.contours = contours
        self.title = title
        self.metadata = metadata
        self.decoration = decoration

    def _valid(self, seen, shape):
        if len(set(x.identifier for x in self.parameters)) != len(self.parameters):
            raise ValueError("ParameterizedFunction.parameters keys must be unique")

        for x in self.parameters:
            _valid(x, seen, shape)

        if self.contours is not None:
            if len(self.contours) != len(numpy.unique(self.contours)):
                raise ValueError("ParameterizedFunction.contours must be unique")

        return shape

################################################# EvaluatedFunction

class EvaluatedFunction(Function):
    _params = {
        "identifier":     histos.checktype.CheckKey("EvaluatedFunction", "identifier", required=True, type=str),
        "values":         histos.checktype.CheckVector("EvaluatedFunction", "values", required=True, type=float),
        "derivatives":    histos.checktype.CheckVector("EvaluatedFunction", "derivatives", required=False, type=float),
        "generic_errors": histos.checktype.CheckVector("EvaluatedFunction", "generic_errors", required=False, type=GenericErrors),
        "title":          histos.checktype.CheckString("EvaluatedFunction", "title", required=False),
        "metadata":       histos.checktype.CheckClass("EvaluatedFunction", "metadata", required=False, type=Metadata),
        "decoration":     histos.checktype.CheckClass("EvaluatedFunction", "decoration", required=False, type=Decoration),
        }

    identifier     = typedproperty(_params["identifier"])
    values         = typedproperty(_params["values"])
    derivatives    = typedproperty(_params["derivatives"])
    generic_errors = typedproperty(_params["generic_errors"])
    title          = typedproperty(_params["title"])
    metadata       = typedproperty(_params["metadata"])
    decoration     = typedproperty(_params["decoration"])

    def __init__(self, identifier, values, derivatives=None, generic_errors=None, title="", metadata=None, decoration=None):
        self.identifier = identifier
        self.values = values
        self.derivatives = derivatives
        self.generic_errors = generic_errors
        self.title = title
        self.metadata = metadata
        self.decoration = decoration

################################################# BinnedEvaluatedFunction

class BinnedEvaluatedFunction(FunctionObject):
    _params = {
        "identifier":     histos.checktype.CheckKey("BinnedEvaluatedFunction", "identifier", required=True, type=str),
        "axis":           histos.checktype.CheckVector("BinnedEvaluatedFunction", "axis", required=True, type=Axis, minlen=1),
        "values":         histos.checktype.CheckClass("BinnedEvaluatedFunction", "values", required=True, type=InterpretedBuffer),
        "derivatives":    histos.checktype.CheckClass("BinnedEvaluatedFunction", "derivatives", required=False, type=InterpretedBuffer),
        "generic_errors": histos.checktype.CheckVector("BinnedEvaluatedFunction", "generic_errors", required=False, type=GenericErrors),
        "title":          histos.checktype.CheckString("BinnedEvaluatedFunction", "title", required=False),
        "metadata":       histos.checktype.CheckClass("BinnedEvaluatedFunction", "metadata", required=False, type=Metadata),
        "decoration":     histos.checktype.CheckClass("BinnedEvaluatedFunction", "decoration", required=False, type=Decoration),
        }

    identifier     = typedproperty(_params["identifier"])
    axis           = typedproperty(_params["axis"])
    values         = typedproperty(_params["values"])
    derivatives    = typedproperty(_params["derivatives"])
    generic_errors = typedproperty(_params["generic_errors"])
    title          = typedproperty(_params["title"])
    metadata       = typedproperty(_params["metadata"])
    decoration     = typedproperty(_params["decoration"])

    def __init__(self, identifier, axis, values, derivatives=None, generic_errors=None, title="", metadata=None, decoration=None):
        self.identifier = identifier
        self.axis = axis
        self.values = values
        self.derivatives = derivatives
        self.generic_errors = generic_errors
        self.title = title
        self.metadata = metadata
        self.decoration = decoration

    def _valid(self, seen, shape):
        binshape = ()
        for x in self.axis:
            binshape = binshape + _valid(x, seen, shape)
        _valid(self.values, seen, shape + binshape)
        _valid(self.derivatives, seen, shape)
        _valid(self.generic_errors, seen, shape)
        _valid(self.metadata, seen, shape)
        _valid(self.decoration, seen, shape)
        return shape

################################################# Histogram

class Histogram(Object):
    _params = {
        "identifier":     histos.checktype.CheckKey("Histogram", "identifier", required=True, type=str),
        "axis":           histos.checktype.CheckVector("Histogram", "axis", required=True, type=Axis, minlen=1),
        "distribution":   histos.checktype.CheckClass("Histogram", "distribution", required=True, type=Distribution),
        "profiles":       histos.checktype.CheckVector("Histogram", "profiles", required=False, type=Profile),
        "unbinned_stats": histos.checktype.CheckVector("Histogram", "unbinned_stats", required=False, type=DistributionStats),
        "profile_stats":  histos.checktype.CheckVector("Histogram", "profile_stats", required=False, type=DistributionStats),
        "functions":      histos.checktype.CheckVector("Histogram", "functions", required=False, type=Function),
        "title":          histos.checktype.CheckString("Histogram", "title", required=False),
        "metadata":       histos.checktype.CheckClass("Histogram", "metadata", required=False, type=Metadata),
        "decoration":     histos.checktype.CheckClass("Histogram", "decoration", required=False, type=Decoration),
        }

    identifier     = typedproperty(_params["identifier"])
    axis           = typedproperty(_params["axis"])
    distribution   = typedproperty(_params["distribution"])
    profiles       = typedproperty(_params["profiles"])
    unbinned_stats = typedproperty(_params["unbinned_stats"])
    profile_stats  = typedproperty(_params["profile_stats"])
    functions      = typedproperty(_params["functions"])
    title          = typedproperty(_params["title"])
    metadata       = typedproperty(_params["metadata"])
    decoration     = typedproperty(_params["decoration"])

    def __init__(self, identifier, axis, distribution, profiles=None, unbinned_stats=None, profile_stats=None, functions=None, title="", metadata=None, decoration=None):
        self.identifier = identifier
        self.axis = axis
        self.distribution = distribution
        self.profiles = profiles
        self.unbinned_stats = unbinned_stats
        self.profile_stats = profile_stats
        self.functions = functions
        self.title = title
        self.metadata = metadata
        self.decoration = decoration

################################################# Page

class Page(Histos):
    _params = {
        "buffer": histos.checktype.CheckClass("Page", "buffer", required=True, type=RawBuffer),
        }

    buffer = typedproperty(_params["buffer"])

    def __init__(self, buffer):
        self.buffer = buffer

    @property
    def numpy_array(self):
        self._top().checkvalid()
        return self._array

    def _valid(self, seen, shape, column, numentries):
        buf = self.buffer.numpy_array
        if column.filters is not None:
            raise NotImplementedError("handle column.filters")

        if column.postfilter_slice is not None:
            start = self.postfilter_slice.start if self.postfilter_slice.has_start else None
            stop = self.postfilter_slice.stop if self.postfilter_slice.has_stop else None
            step = self.postfilter_slice.step if self.postfilter_slice.has_step else None
            buf = buf[start:stop:step]

        if len(buf) != column.numpy_dtype.itemsize * numentries:
            raise ValueError("Page.buffer length is {0} bytes but ColumnChunk.page_offsets claims {1} entries of {2} bytes each".format(len(buf), numentries, column.numpy_dtype.itemsize))

        self._array = buf.view(column.numpy_dtype).reshape((numentries,))

        return numentries

################################################# ColumnChunk

class ColumnChunk(Histos):
    _params = {
        "pages":         histos.checktype.CheckVector("ColumnChunk", "pages", required=True, type=Page),
        "page_offsets":  histos.checktype.CheckVector("ColumnChunk", "page_offsets", required=True, type=int, minlen=1),
        "page_extremes": histos.checktype.CheckVector("ColumnChunk", "page_extremes", required=False, type=Extremes),
        }

    pages         = typedproperty(_params["pages"])
    page_offsets  = typedproperty(_params["page_offsets"])
    page_extremes = typedproperty(_params["page_extremes"])

    def __init__(self, pages, page_offsets, page_extremes=None):
        self.pages = pages
        self.page_offsets = page_offsets
        self.page_extremes = page_extremes

    def _valid(self, seen, shape, column):
        # have to do recursive check because ColumnChunk._valid is called directly by Chunk._valid
        if id(self) in seen:
            raise ValueError("hierarchy is recursively nested")
        seen.add(id(self))

        if self.page_offsets[0] != 0:
            raise ValueError("ColumnChunk.page_offsets must start with 0")
        if not numpy.greater_equal(self.page_offsets[1:], self.page_offsets[:-1]).all():
            raise ValueError("ColumnChunk.page_offsets must be monotonically increasing")

        if len(self.page_offsets) != len(self.pages) + 1:
            raise ValueError("ColumnChunk.page_offsets length is {0}, but it must be one longer than ColumnChunk.pages, which is {1}".format(len(self.page_offsets), len(self.pages)))

        for i, x in enumerate(self.pages):
            x._valid(seen, shape, column, self.page_offsets[i + 1] - self.page_offsets[i])

        if self.page_extremes is not None:
            if len(self.page_extremes) != len(self.pages):
                raise ValueError("ColumnChunk.page_extremes length {0} must be equal to ColumnChunk.pages length {1}".format(len(self.page_extremes), len(self.pages)))

            for x in self.page_extremes:
                _valid(x, seen, shape)

            raise NotImplementedError("check extremes")

        return self.page_offsets[-1]

################################################# Chunk

class Chunk(Histos):
    _params = {
        "columns":  histos.checktype.CheckVector("Chunk", "columns", required=True, type=ColumnChunk),
        "metadata": histos.checktype.CheckClass("Chunk", "metadata", required=False, type=Metadata),
        }

    columns  = typedproperty(_params["columns"])
    metadata = typedproperty(_params["metadata"])

    def __init__(self, columns, metadata=None):
        self.columns = columns
        self.metadata = metadata

    def _valid(self, seen, shape, columns, numentries):
        # have to do recursive check because Chunk._valid is called directly by Ntuple._valid
        if id(self) in seen:
            raise ValueError("hierarchy is recursively nested")
        seen.add(id(self))

        if len(self.columns) != len(columns):
            raise ValueError("Chunk.columns has length {0}, but Ntuple.columns has length {1}".format(len(self.columns), len(columns)))

        for x, y in zip(self.columns, columns):
            num = x._valid(seen, shape, y)
            if numentries is not None and num != numentries:
                raise ValueError("Chunk.column {0} has {1} entries but Chunk has {2} entries".format(repr(y.identifier), num, numentries))

        _valid(self.metadata, seen, shape)
        return numentries

################################################# Column

class Column(Histos, Interpretation):
    _params = {
        "identifier":       histos.checktype.CheckKey("Column", "identifier", required=True, type=str),
        "dtype":            histos.checktype.CheckEnum("Column", "dtype", required=True, choices=Interpretation.dtypes),
        "endianness":       histos.checktype.CheckEnum("Column", "endianness", required=False, choices=Interpretation.endiannesses),
        "filters":          histos.checktype.CheckVector("Column", "filters", required=False, type=Buffer.filters),
        "postfilter_slice": histos.checktype.CheckSlice("Column", "postfilter_slice", required=False),
        "title":            histos.checktype.CheckString("Column", "title", required=False),
        "metadata":         histos.checktype.CheckClass("Column", "metadata", required=False, type=Metadata),
        "decoration":       histos.checktype.CheckClass("Column", "decoration", required=False, type=Decoration),
        }

    identifier       = typedproperty(_params["identifier"])
    dtype            = typedproperty(_params["dtype"])
    endianness       = typedproperty(_params["endianness"])
    filters          = typedproperty(_params["filters"])
    postfilter_slice = typedproperty(_params["filters"])
    title            = typedproperty(_params["title"])
    metadata         = typedproperty(_params["metadata"])
    decoration       = typedproperty(_params["decoration"])

    def __init__(self, identifier, dtype, endianness=InterpretedBuffer.little_endian, dimension_order=InterpretedBuffer.c_order, filters=None, postfilter_slice=None, title="", metadata=None, decoration=None):
        self.identifier = identifier
        self.dtype = dtype
        self.endianness = endianness
        self.filters = filters
        self.postfilter_slice = postfilter_slice
        self.title = title
        self.metadata = metadata
        self.decoration = decoration

    def _valid(self, seen, shape):
        if self.postfilter_slice is not None:
            if self.postfilter_slice.step == 0:
                raise ValueError("slice step cannot be zero")
        return shape

################################################# Ntuple

class Ntuple(Object):
    _params = {
        "identifier":     histos.checktype.CheckKey("Ntuple", "identifier", required=True, type=str),
        "columns":        histos.checktype.CheckVector("Ntuple", "columns", required=True, type=Column, minlen=1),
        "chunks":         histos.checktype.CheckVector("Ntuple", "chunks", required=True, type=Chunk),
        "chunk_offsets":  histos.checktype.CheckVector("Ntuple", "chunk_offsets", required=False, type=int, minlen=1),
        "unbinned_stats": histos.checktype.CheckVector("Ntuple", "unbinned_stats", required=False, type=DistributionStats),
        "functions":      histos.checktype.CheckVector("Ntuple", "functions", required=False, type=FunctionObject),
        "title":          histos.checktype.CheckString("Ntuple", "title", required=False),
        "metadata":       histos.checktype.CheckClass("Ntuple", "metadata", required=False, type=Metadata),
        "decoration":     histos.checktype.CheckClass("Ntuple", "decoration", required=False, type=Decoration),
        }

    identifier     = typedproperty(_params["identifier"])
    columns        = typedproperty(_params["columns"])
    chunks         = typedproperty(_params["chunks"])
    chunk_offsets  = typedproperty(_params["chunk_offsets"])
    unbinned_stats = typedproperty(_params["unbinned_stats"])
    functions      = typedproperty(_params["functions"])
    title          = typedproperty(_params["title"])
    metadata       = typedproperty(_params["metadata"])
    decoration     = typedproperty(_params["decoration"])

    def __init__(self, identifier, columns, chunks, chunk_offsets=None, unbinned_stats=None, functions=None, title="", metadata=None, decoration=None):
        self.identifier = identifier
        self.columns = columns
        self.chunks = chunks
        self.chunk_offsets = chunk_offsets
        self.unbinned_stats = unbinned_stats
        self.functions = functions
        self.title = title
        self.metadata = metadata
        self.decoration = decoration

    def _valid(self, seen, shape):
        if len(set(x.identifier for x in self.columns)) != len(self.columns):
            raise ValueError("Ntuple.columns keys must be unique")

        for x in self.columns:
            _valid(x, seen, shape)

        if self.chunk_offsets is None:
            for x in self.chunks:
                x._valid(seen, shape, self.columns, None)

        else:
            if self.chunk_offsets[0] != 0:
                raise ValueError("Ntuple.chunk_offsets must start with 0")
            if not numpy.greater_equal(self.chunk_offsets[1:], self.chunk_offsets[:-1]).all():
                raise ValueError("Ntuple.chunk_offsets must be monotonically increasing")

            if len(self.chunk_offsets) != len(self.chunks) + 1:
                raise ValueError("Ntuple.chunk_offsets length is {0}, but it must be one longer than Ntuple.chunks, which is {1}".format(len(self.chunk_offsets), len(self.chunks)))

            for i, x in enumerate(self.chunks):
                x._valid(seen, shape, self.columns, self.chunk_offsets[i + 1] - self.chunk_offsets[i])

        if self.unbinned_stats is not None:
            for x in self.unbinned_stats:
                _valid(x, seen, shape)

        if self.functions is not None:
            for x in self.functions:
                _valid(x, seen, shape)

        _valid(self.metadata, seen, shape)
        _valid(self.decoration, seen, shape)

        return shape

################################################# Region

class Region(Histos):
    _params = {
        "expressions":  histos.checktype.CheckVector("Region", "expressions", required=True, type=str),
        }

    expressions = typedproperty(_params["expressions"])

    def __init__(self, expressions):
        self.expressions  = expressions 

################################################# BinnedRegion

class BinnedRegion(Histos):
    _params = {
        "expression": histos.checktype.CheckString("BinnedRegion", "expression", required=True),
        "binning":    histos.checktype.CheckClass("BinnedRegion", "binning", required=True, type=Binning),
        }

    expression = typedproperty(_params["expression"])
    binning    = typedproperty(_params["binning"])

    def __init__(self, expression, binning):
        self.expression = expression
        self.binning  = binning 

################################################# Assignment

class Assignment(Histos):
    _params = {
        "identifier": histos.checktype.CheckKey("Assignment", "identifier", required=True, type=str),
        "expression": histos.checktype.CheckString("Assignment", "expression", required=True),
        }

    identifier = typedproperty(_params["identifier"])
    expression = typedproperty(_params["expression"])

    def __init__(self, identifier, expression):
        self.identifier = identifier
        self.expression  = expression 

################################################# Variation

class Variation(Histos):
    _params = {
        "assignments":         histos.checktype.CheckVector("Variation", "assignments", required=True, type=Assignment),
        "systematic":          histos.checktype.CheckVector("Variation", "systematic", required=False, type=float),
        "category_systematic": histos.checktype.CheckVector("Variation", "category_systematic", required=False, type=str),
        }

    assignments         = typedproperty(_params["assignments"])
    systematic          = typedproperty(_params["systematic"])
    category_systematic = typedproperty(_params["category_systematic"])

    def __init__(self, assignments, systematic=None, category_systematic=None):
        self.assignments = assignments
        self.systematic = systematic
        self.category_systematic = category_systematic

################################################# Collection

class Collection(Histos):
    def tobuffer(self):
        self._valid(set(), ())
        builder = flatbuffers.Builder(1024)
        builder.Finish(self._toflatbuffers(builder, None))
        return builder.Output()

    @classmethod
    def frombuffer(cls, buffer, offset=0):
        out = cls.__new__(cls)
        out._flatbuffers = histos.histos_generated.Collection.Collection.GetRootAsCollection(buffer, offset)
        return out

    def toarray(self):
        return numpy.frombuffer(self.tobuffer(), dtype=numpy.uint8)

    @classmethod
    def fromarray(cls, array):
        return cls.frombuffer(array)

    def tofile(self, file):
        self._valid(set(), ())

        opened = False
        if not hasattr(file, "write"):
            file = open(file, "wb")
            opened = True

        if not hasattr(file, "tell"):
            class FileLike(object):
                def __init__(self, file):
                    self.file = file
                    self.offset = 0
                def write(self, data):
                    self.file.write(data)
                    self.offset += len(data)
                def close(self):
                    try:
                        self.file.close()
                    except:
                        pass
                def tell(self):
                    return self.offset
            file = FileLike(file)

        try:
            file.write(b"hist")
            builder = flatbuffers.Builder(1024)
            builder.Finish(self._toflatbuffers(builder, False, file))
            offset = file.tell()
            file.write(builder.Output())
            file.write(struct.pack("<Q", offset))
            file.write(b"hist")

        finally:
            if opened:
                file.close()

    @classmethod
    def fromfile(cls, file, mode="r+"):
        if isinstance(file, str):
            file = numpy.memmap(file, dtype=numpy.uint8, mode=mode)
        if file[:4].tostring() != b"hist":
            raise OSError("file does not begin with magic 'hist'")
        if file[-4:].tostring() != b"hist":
            raise OSError("file does not end with magic 'hist'")
        offset, = struct.unpack("<Q", file[-12:-4])
        return cls.frombuffer(file[offset:-12])

    _params = {
        "identifier":     histos.checktype.CheckString("Collection", "identifier", required=True),
        "objects":        histos.checktype.CheckVector("Collection", "objects", required=True, type=Object),
        "collections":    histos.checktype.CheckVector("Collection", "collections", required=False, type=None),
        "regions":        histos.checktype.CheckVector("Collection", "regions", required=False, type=Region),
        "binned_regions": histos.checktype.CheckVector("Collection", "binned_regions", required=False, type=BinnedRegion),
        "variations":     histos.checktype.CheckVector("Collection", "variations", required=False, type=Variation),
        "title":          histos.checktype.CheckString("Collection", "title", required=False),
        "metadata":       histos.checktype.CheckClass("Collection", "metadata", required=False, type=Metadata),
        "decoration":     histos.checktype.CheckClass("Collection", "decoration", required=False, type=Decoration),
        }

    identifier     = typedproperty(_params["identifier"])
    objects        = typedproperty(_params["objects"])
    collections    = typedproperty(_params["collections"])
    regions        = typedproperty(_params["regions"])
    binned_regions = typedproperty(_params["binned_regions"])
    variations     = typedproperty(_params["variations"])
    title          = typedproperty(_params["title"])
    metadata       = typedproperty(_params["metadata"])
    decoration     = typedproperty(_params["decoration"])

    def __init__(self, identifier, objects, collections=None, regions=None, binned_regions=None, variations=None, title="", metadata=None, decoration=None):
        self.identifier = identifier
        self.objects = objects
        self.collections = collections
        self.regions = regions
        self.binned_regions = binned_regions
        self.variations = variations
        self.title = title
        self.metadata = metadata
        self.decoration = decoration

    def _valid(self, seen, shape):
        if len(set(x.identifier for x in self.objects)) != len(self.objects):
            raise ValueError("Collection.objects keys must be unique")

        for x in self.objects:
            _valid(x, seen, shape)

        _valid(self.metadata, seen, shape)
        _valid(self.decoration, seen, shape)
        return shape

    def __getitem__(self, where):
        return _getbykey(self, "objects", where)

    def __repr__(self):
        return "<{0} {1} at 0x{2:012x}>".format(type(self).__name__, repr(self.identifier), id(self))

    @property
    def isvalid(self):
        try:
            self._valid(set(), ())
        except ValueError:
            return False
        else:
            return True

    def checkvalid(self):
        self._valid(set(), ())

    def _toflatbuffers(self, builder, file):
        identifier = builder.CreateString(self._identifier)
        if len(self._title) > 0:
            title = builder.CreateString(self._title)
        histos.histos_generated.Collection.CollectionStart(builder)
        histos.histos_generated.Collection.CollectionAddIdentifier(builder, identifier)
        if len(self._title) > 0:
            histos.histos_generated.Collection.CollectionAddTitle(builder, title)
        return histos.histos_generated.Collection.CollectionEnd(builder)

Collection._params["collections"].type = Collection
