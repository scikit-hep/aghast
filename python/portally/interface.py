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

import portally.portally_generated.MetadataLanguage
import portally.portally_generated.Metadata
import portally.portally_generated.DecorationLanguage
import portally.portally_generated.Decoration
import portally.portally_generated.DType
import portally.portally_generated.Endianness
import portally.portally_generated.DimensionOrder
import portally.portally_generated.Filter
import portally.portally_generated.Slice
import portally.portally_generated.ExternalType
import portally.portally_generated.RawInlineBuffer
import portally.portally_generated.RawExternalBuffer
import portally.portally_generated.InterpretedInlineBuffer
import portally.portally_generated.InterpretedExternalBuffer
import portally.portally_generated.RawBuffer
import portally.portally_generated.InterpretedBuffer
import portally.portally_generated.StatisticFilter
import portally.portally_generated.Moments
import portally.portally_generated.Extremes
import portally.portally_generated.Quantiles
import portally.portally_generated.Modes
import portally.portally_generated.Statistics
import portally.portally_generated.Correlations
import portally.portally_generated.BinPosition
import portally.portally_generated.IntegerBinning
import portally.portally_generated.RealInterval
import portally.portally_generated.NonRealMapping
import portally.portally_generated.RealOverflow
import portally.portally_generated.RegularBinning
import portally.portally_generated.TicTacToeOverflowBinning
import portally.portally_generated.HexagonalCoordinates
import portally.portally_generated.HexagonalBinning
import portally.portally_generated.EdgesBinning
import portally.portally_generated.OverlappingFillStrategy
import portally.portally_generated.IrregularBinning
import portally.portally_generated.CategoryBinning
import portally.portally_generated.SparseRegularBinning
import portally.portally_generated.FractionLayout
import portally.portally_generated.FractionErrorMethod
import portally.portally_generated.FractionBinning
import portally.portally_generated.Binning
import portally.portally_generated.Axis
import portally.portally_generated.Profile
import portally.portally_generated.UnweightedCounts
import portally.portally_generated.WeightedCounts
import portally.portally_generated.Counts
import portally.portally_generated.Parameter
import portally.portally_generated.ParameterizedFunction
import portally.portally_generated.EvaluatedFunction
import portally.portally_generated.FunctionData
import portally.portally_generated.Function
import portally.portally_generated.BinnedEvaluatedFunction
import portally.portally_generated.FunctionObjectData
import portally.portally_generated.FunctionObject
import portally.portally_generated.Histogram
import portally.portally_generated.Page
import portally.portally_generated.ColumnChunk
import portally.portally_generated.Chunk
import portally.portally_generated.Column
import portally.portally_generated.NtupleInstance
import portally.portally_generated.Ntuple
import portally.portally_generated.ObjectData
import portally.portally_generated.Object
import portally.portally_generated.Region
import portally.portally_generated.BinnedRegion
import portally.portally_generated.Assignment
import portally.portally_generated.Variation
import portally.portally_generated.Collection

import portally.checktype

def typedproperty(check):
    def setparent(self, value):
        if isinstance(value, Portally):
            if hasattr(value, "_parent"):
                raise ValueError("already attached to another hierarchy: {0}".format(repr(value)))
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
            assert hasattr(self, "_flatbuffers"), "not derived from a flatbuffer or not properly initialized"
            setattr(self, private, check.fromflatbuffers(getattr(self._flatbuffers, check.paramname.capitalize())()))
        return getattr(self, private)

    @prop.setter
    def prop(self, value):
        setparent(self, value)
        private = "_" + check.paramname
        setattr(self, private, check(value))

    return prop

def _valid(obj, seen, only, *args):
    if obj is None:
        return args[0]
    else:
        if only is None or id(obj) in only:
            if id(obj) in seen:
                raise ValueError("hierarchy is recursively nested")
            seen.add(id(obj))
            obj._validtypes()
            return obj._valid(seen, only, *args)
        else:
            return args[0]

def _getbykey(self, field, where):
    lookup = "_lookup_" + field
    if not hasattr(self, lookup):
        setattr(self, lookup, {x.identifier: x for x in getattr(self, field)})
        if len(getattr(self, lookup)) != len(getattr(self, field)):
            raise ValueError("{0}.{1} keys must be unique".format(type(self).__name__, field))
    return getattr(self, lookup)[where]

class Portally(object):
    def __repr__(self):
        return "<{0} at 0x{1:012x}>".format(type(self).__name__, id(self))

    def _shape(self, path, shape):
        for x in path:
            if self is x:
                raise ValueError("hierarchy is recursively nested")
        path = (self,) + path
        if hasattr(self, "_parent"):
            return self._parent._shape(path, shape)
        elif shape == ():
            return (1,)
        else:
            return shape

    def _top(self):
        out = self
        seen = set([id(out)])
        while hasattr(out, "_parent"):
            out = out._parent
            if id(out) in seen:
                raise ValueError("hierarchy is recursively nested")
            seen.add(id(out))
        if not isinstance(out, Collection):
            raise ValueError("{0} object is not nested in a hierarchy".format(type(self).__name__))
        return out, seen

    def _topvalid(self):
        top, only = self._top()
        top._valid(set(), only, ())

    _validtypesskip = ()
    def _validtypes(self):
        for n, x in self._params.items():
            if not (n in self._validtypesskip and getattr(self, n) is None):
                x(getattr(self, n))

    def _valid(self, seen, only, shape):
        raise NotImplementedError("missing _valid implementation")

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

class MetadataLanguageEnum(Enum): pass

class Metadata(Portally):
    unspecified = MetadataLanguageEnum("unspecified", portally.portally_generated.MetadataLanguage.MetadataLanguage.meta_unspecified)
    json = MetadataLanguageEnum("json", portally.portally_generated.MetadataLanguage.MetadataLanguage.meta_json)
    language = [unspecified, json]

    _params = {
        "data":     portally.checktype.CheckString("Metadata", "data", required=True),
        "language": portally.checktype.CheckEnum("Metadata", "language", required=True, choices=language),
        }

    data     = typedproperty(_params["data"])
    language = typedproperty(_params["language"])

    def __init__(self, data, language=unspecified):
        self.data = data
        self.language = language

    def _valid(self, seen, only, shape):
        return shape

################################################# Decoration

class DecorationLanguageEnum(Enum): pass

class Decoration(Portally):
    unspecified = DecorationLanguageEnum("unspecified", portally.portally_generated.DecorationLanguage.DecorationLanguage.deco_unspecified)
    css         = DecorationLanguageEnum("css", portally.portally_generated.DecorationLanguage.DecorationLanguage.deco_css)
    vega        = DecorationLanguageEnum("vega", portally.portally_generated.DecorationLanguage.DecorationLanguage.deco_vega)
    root_json   = DecorationLanguageEnum("root_json", portally.portally_generated.DecorationLanguage.DecorationLanguage.deco_root_json)
    language = [unspecified, css, vega, root_json]

    _params = {
        "data":     portally.checktype.CheckString("Decoration", "data", required=True),
        "language": portally.checktype.CheckEnum("Decoration", "language", required=True, choices=language),
        }

    data     = typedproperty(_params["data"])
    language = typedproperty(_params["language"])

    def __init__(self, data, language=unspecified):
        self.data = data
        self.language = language

    def _valid(self, seen, only, shape):
        return shape

################################################# Buffers

class BufferFilterEnum(Enum): pass

class Buffer(Portally):
    none = BufferFilterEnum("none", portally.portally_generated.Filter.Filter.filter_none)
    gzip = BufferFilterEnum("gzip", portally.portally_generated.Filter.Filter.filter_gzip)
    lzma = BufferFilterEnum("lzma", portally.portally_generated.Filter.Filter.filter_lzma)
    lz4  = BufferFilterEnum("lz4", portally.portally_generated.Filter.Filter.filter_lz4)
    filters = [none, gzip, lzma, lz4]

    def __init__(self):
        raise TypeError("{0} is an abstract base class; do not construct".format(type(self).__name__))

class InlineBuffer(object):
    def __init__(self):
        raise TypeError("{0} is an abstract base class; do not construct".format(type(self).__name__))

class ExternalTypeEnum(Enum): pass

class ExternalBuffer(object):
    memory   = ExternalTypeEnum("memory", portally.portally_generated.ExternalType.ExternalType.external_memory)
    samefile = ExternalTypeEnum("samefile", portally.portally_generated.ExternalType.ExternalType.external_samefile)
    file     = ExternalTypeEnum("file", portally.portally_generated.ExternalType.ExternalType.external_file)
    url      = ExternalTypeEnum("url", portally.portally_generated.ExternalType.ExternalType.external_url)
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

class Interpretation(object):
    none    = DTypeEnum("none", portally.portally_generated.DType.DType.dtype_none, numpy.dtype(numpy.uint8))
    int8    = DTypeEnum("int8", portally.portally_generated.DType.DType.dtype_int8, numpy.dtype(numpy.int8))
    uint8   = DTypeEnum("uint8", portally.portally_generated.DType.DType.dtype_uint8, numpy.dtype(numpy.uint8))
    int16   = DTypeEnum("int16", portally.portally_generated.DType.DType.dtype_int16, numpy.dtype(numpy.int16))
    uint16  = DTypeEnum("uint16", portally.portally_generated.DType.DType.dtype_uint16, numpy.dtype(numpy.uint16))
    int32   = DTypeEnum("int32", portally.portally_generated.DType.DType.dtype_int32, numpy.dtype(numpy.int32))
    uint32  = DTypeEnum("uint32", portally.portally_generated.DType.DType.dtype_uint32, numpy.dtype(numpy.uint32))
    int64   = DTypeEnum("int64", portally.portally_generated.DType.DType.dtype_int64, numpy.dtype(numpy.int64))
    uint64  = DTypeEnum("uint64", portally.portally_generated.DType.DType.dtype_uint64, numpy.dtype(numpy.uint64))
    float32 = DTypeEnum("float32", portally.portally_generated.DType.DType.dtype_float32, numpy.dtype(numpy.float32))
    float64 = DTypeEnum("float64", portally.portally_generated.DType.DType.dtype_float64, numpy.dtype(numpy.float64))
    dtypes = [none, int8, uint8, int16, uint16, int32, uint32, int64, uint64, float32, float64]

    little_endian = EndiannessEnum("little_endian", portally.portally_generated.Endianness.Endianness.little_endian, "<")
    big_endian    = EndiannessEnum("big_endian", portally.portally_generated.Endianness.Endianness.big_endian, ">")
    endiannesses = [little_endian, big_endian]

    def __init__(self):
        raise TypeError("{0} is an abstract base class; do not construct".format(type(self).__name__))

    @property
    def numpy_dtype(self):
        return self.dtype.dtype.newbyteorder(self._endianness.endianness)

    @classmethod
    def from_numpy_dtype(cls, dtype):
        dtype = numpy.dtype(dtype)

        if dtype.byteorder == "=":
            endianness = cls.little_endian if sys.byteorder == "little" else cls.big_endian
        elif dtype.byteorder == ">":
            endianness = cls.big_endian
        elif dtype.byteorder == "<":
            endianness = cls.little_endian

        if dtype.kind == "i":
            if dtype.itemsize == 1:
                return cls.int8, endianness
            elif dtype.itemsize == 2:
                return cls.int16, endianness
            elif dtype.itemsize == 4:
                return cls.int32, endianness
            elif dtype.itemsize == 8:
                return cls.int64, endianness

        elif dtype.kind == "u":
            if dtype.itemsize == 1:
                return cls.uint8, endianness
            elif dtype.itemsize == 2:
                return cls.uint16, endianness
            elif dtype.itemsize == 4:
                return cls.uint32, endianness
            elif dtype.itemsize == 8:
                return cls.uint64, endianness

        elif dtype.kind == "f":
            if dtype.itemsize == 4:
                return cls.float32, endianness
            elif dtype.itemsize == 8:
                return cls.float64, endianness

        raise ValueError("numpy dtype {0} does not correspond to any Interpretation dtype, endianness pair".format(str(dtype)))

class DimensionOrderEnum(Enum):
    def __init__(self, name, value, dimension_order):
        super(DimensionOrderEnum, self).__init__(name, value)
        self.dimension_order = dimension_order

class InterpretedBuffer(Interpretation):
    c_order       = DimensionOrderEnum("c_order", portally.portally_generated.DimensionOrder.DimensionOrder.c_order, "C")
    fortran_order = DimensionOrderEnum("fortran", portally.portally_generated.DimensionOrder.DimensionOrder.fortran_order, "F")
    orders = [c_order, fortran_order]

    def __init__(self):
        raise TypeError("{0} is an abstract base class; do not construct".format(type(self).__name__))

################################################# RawInlineBuffer

class RawInlineBuffer(Buffer, RawBuffer, InlineBuffer):
    _params = {
        "buffer": portally.checktype.CheckBuffer("RawInlineBuffer", "buffer", required=True),
        }

    buffer = typedproperty(_params["buffer"])

    _validtypesskip = ("buffer",)

    def __init__(self, buffer=None):
        if buffer is None:
            self._buffer = None     # placeholder for auto-generated buffer
        else:
            self.buffer = buffer

    def _valid(self, seen, only, numbytes):
        if self._buffer is None:
            self._buffer = numpy.empty(numbytes, dtype=InterpretedBuffer.none.dtype)
        self._array = numpy.frombuffer(self._buffer, dtype=InterpretedBuffer.none.dtype)

    @property
    def numpy_array(self):
        self._topvalid()
        return self._array

################################################# RawExternalBuffer

class RawExternalBuffer(Buffer, RawBuffer, ExternalBuffer):
    _params = {
        "pointer":          portally.checktype.CheckInteger("RawExternalBuffer", "pointer", required=True, min=0),
        "numbytes":         portally.checktype.CheckInteger("RawExternalBuffer", "numbytes", required=True, min=0),
        "external_type":    portally.checktype.CheckEnum("RawExternalBuffer", "external_type", required=True, choices=ExternalBuffer.types),
        }

    pointer       = typedproperty(_params["pointer"])
    numbytes      = typedproperty(_params["numbytes"])
    external_type = typedproperty(_params["external_type"])

    _validtypesskip = ("pointer", "numbytes")

    def __init__(self, pointer=None, numbytes=None, external_type=ExternalBuffer.memory):
        if pointer is None and numbytes is None:
            self._pointer = None    # placeholder for auto-generated buffer
            self._numbytes = None
        else:
            self.pointer = pointer
            self.numbytes = numbytes
        self.external_type = external_type

    def _valid(self, seen, only, numbytes):
        if self._pointer is None or self._numbytes is None:
            self._buffer = numpy.empty(numbytes, dtype=InterpretedBuffer.none.dtype)
            self._pointer = self._buffer.ctypes.data
            self._numbytes = self._buffer.nbytes
        else:
            self._buffer = numpy.ctypeslib.as_array(ctypes.cast(self.pointer, ctypes.POINTER(ctypes.c_uint8)), shape=(self.numbytes,))

        if len(self._buffer) != numbytes:
            raise ValueError("RawExternalBuffer.buffer length is {0} but it should be {1} bytes".format(len(self._buffer), numbytes))

        self._array = self._buffer

    @property
    def numpy_array(self):
        self._topvalid()
        return self._buffer

################################################# InlineBuffer

class InterpretedInlineBuffer(Buffer, InterpretedBuffer, InlineBuffer):
    _params = {
        "buffer":           portally.checktype.CheckBuffer("InterpretedInlineBuffer", "buffer", required=True),
        "filters":          portally.checktype.CheckVector("InterpretedInlineBuffer", "filters", required=False, type=Buffer.filters),
        "postfilter_slice": portally.checktype.CheckSlice("InterpretedInlineBuffer", "postfilter_slice", required=False),
        "dtype":            portally.checktype.CheckEnum("InterpretedInlineBuffer", "dtype", required=False, choices=InterpretedBuffer.dtypes),
        "endianness":       portally.checktype.CheckEnum("InterpretedInlineBuffer", "endianness", required=False, choices=InterpretedBuffer.endiannesses),
        "dimension_order":  portally.checktype.CheckEnum("InterpretedInlineBuffer", "dimension_order", required=False, choices=InterpretedBuffer.orders),
        }

    buffer           = typedproperty(_params["buffer"])
    filters          = typedproperty(_params["filters"])
    postfilter_slice = typedproperty(_params["postfilter_slice"])
    dtype            = typedproperty(_params["dtype"])
    endianness       = typedproperty(_params["endianness"])
    dimension_order  = typedproperty(_params["dimension_order"])

    _validtypesskip = ("buffer",)

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

    def _valid(self, seen, only, shape):
        if self._buffer is None:
            self._buffer = numpy.zeros(functools.reduce(operator.mul, shape, self.numpy_dtype.itemsize), dtype=InterpretedBuffer.none.dtype)

        if self.filters is None:
            self._array = self._buffer
        else:
            raise NotImplementedError(self.filters)

        if self._array.dtype.itemsize != 1:
            self._array = self._array.view(InterpretedBuffer.none.dtype)
        if len(self._array.shape) != 1:
            self._array = self._array.reshape(-1)

        if self.postfilter_slice is not None:
            start = self.postfilter_slice.start if self.postfilter_slice.has_start else None
            stop = self.postfilter_slice.stop if self.postfilter_slice.has_stop else None
            step = self.postfilter_slice.step if self.postfilter_slice.has_step else None
            self._array = self._array[start:stop:step]
        
        try:
            self._array = self._array.view(self.numpy_dtype)
        except ValueError:
            raise ValueError("InterpretedInlineBuffer.buffer raw length is {0} bytes but this does not fit an itemsize of {1} bytes".format(len(self._array), self.numpy._dtype.itemsize))

        tmp = self._shape((), ())
        assert shape == tmp, "{} != {}".format(shape, tmp)

        if len(self._array) != functools.reduce(operator.mul, shape, 1):
            raise ValueError("InterpretedInlineBuffer.buffer length as {0} is {1} but multiplicity at this position in the hierarchy is {2}".format(self.numpy_dtype, len(self._array), functools.reduce(operator.mul, shape, 1)))

        self._array = self._array.reshape(shape, order=self.dimension_order.dimension_order)
        return shape

    @classmethod
    def fromarray(cls, array):
        dtype, endianness = Interpretation.from_numpy_dtype(array.dtype)
        order = InterpretedBuffer.fortran_order if numpy.isfortran(array) else InterpretedBuffer.c_order
        return cls(array, dtype=dtype, endianness=endianness, dimension_order=order)

    @property
    def numpy_array(self):
        self._topvalid()
        return self._array

################################################# ExternalBuffer

class InterpretedExternalBuffer(Buffer, InterpretedBuffer, ExternalBuffer):
    _params = {
        "pointer":          portally.checktype.CheckInteger("InterpretedExternalBuffer", "pointer", required=True, min=0),
        "numbytes":         portally.checktype.CheckInteger("InterpretedExternalBuffer", "numbytes", required=True, min=0),
        "external_type":    portally.checktype.CheckEnum("InterpretedExternalBuffer", "external_type", required=False, choices=ExternalBuffer.types),
        "filters":          portally.checktype.CheckVector("InterpretedExternalBuffer", "filters", required=False, type=Buffer.filters),
        "postfilter_slice": portally.checktype.CheckSlice("InterpretedExternalBuffer", "postfilter_slice", required=False),
        "dtype":            portally.checktype.CheckEnum("InterpretedExternalBuffer", "dtype", required=False, choices=InterpretedBuffer.dtypes),
        "endianness":       portally.checktype.CheckEnum("InterpretedExternalBuffer", "endianness", required=False, choices=InterpretedBuffer.endiannesses),
        "dimension_order":  portally.checktype.CheckEnum("InterpretedExternalBuffer", "dimension_order", required=False, choices=InterpretedBuffer.orders),
        "location":         portally.checktype.CheckString("InterpretedExternalBuffer", "location", required=False),
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

    _validtypesskip = ("pointer", "numbytes")

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

    def _valid(self, seen, only, shape):
        if self._pointer is None or self._numbytes is None:
            self._buffer = numpy.zeros(functools.reduce(operator.mul, shape, self.numpy_dtype.itemsize), dtype=InterpretedBuffer.none.dtype)
            self._pointer = self._buffer.ctypes.data
            self._numbytes = self._buffer.nbytes
        else:
            self._buffer = numpy.ctypeslib.as_array(ctypes.cast(self.pointer, ctypes.POINTER(ctypes.c_uint8)), shape=(self.numbytes,))

        if self.filters is None:
            self._array = self._buffer
        else:
            raise NotImplementedError(self.filters)

        if self.postfilter_slice is not None:
            start = self.postfilter_slice.start if self.postfilter_slice.has_start else None
            stop = self.postfilter_slice.stop if self.postfilter_slice.has_stop else None
            step = self.postfilter_slice.step if self.postfilter_slice.has_step else None
            self._array = self._array[start:stop:step]

        try:
            self._array = self._array.view(self.numpy_dtype)
        except ValueError:
            raise ValueError("InterpretedExternalBuffer.buffer raw length is {0} bytes but this does not fit an itemsize of {1} bytes".format(len(self._array), self.numpy._dtype.itemsize))

        if len(self._array) != functools.reduce(operator.mul, shape, 1):
            raise ValueError("InterpretedExternalBuffer.buffer length is {0} but multiplicity at this position in the hierarchy is {1}".format(len(self._array), functools.reduce(operator.mul, shape, 1)))

        self._array = self._array.reshape(shape, order=self.dimension_order.dimension_order)
        return shape

    @property
    def numpy_array(self):
        self._topvalid()
        return self._array

################################################# StatisticFilter

class StatisticFilter(Portally):
    _params = {
        "minimum": portally.checktype.CheckNumber("StatisticFilter", "minimum", required=False),
        "maximum": portally.checktype.CheckNumber("StatisticFilter", "maximum", required=False),
        "excludes_minf": portally.checktype.CheckBool("StatisticFilter", "excludes_minf", required=False),
        "excludes_pinf": portally.checktype.CheckBool("StatisticFilter", "excludes_pinf", required=False),
        "excludes_nan":  portally.checktype.CheckBool("StatisticFilter", "excludes_nan", required=False),
        }

    minimum       = typedproperty(_params["minimum"])
    maximum       = typedproperty(_params["maximum"])
    excludes_minf = typedproperty(_params["excludes_minf"])
    excludes_pinf = typedproperty(_params["excludes_pinf"])
    excludes_nan  = typedproperty(_params["excludes_nan"])

    def __init__(self, minimum=None, maximum=None, excludes_minf=None, excludes_pinf=None, excludes_nan=None):
        self.minimum = minimum
        self.maximum = maximum
        self.excludes_minf = excludes_minf
        self.excludes_pinf = excludes_pinf
        self.excludes_nan = excludes_nan

################################################# Moments

class Moments(Portally):
    _params = {
        "sumwxn":   portally.checktype.CheckClass("Moments", "sumwxn", required=True, type=InterpretedBuffer),
        "n":        portally.checktype.CheckInteger("Moments", "n", required=True, min=0),
        "weighted": portally.checktype.CheckBool("Moments", "weighted", required=False),
        "filter": portally.checktype.CheckClass("Moments", "filter", required=False, type=StatisticFilter),
        }

    sumwxn   = typedproperty(_params["sumwxn"])
    n        = typedproperty(_params["n"])
    weighted = typedproperty(_params["weighted"])
    filter   = typedproperty(_params["filter"])

    def __init__(self, sumwxn, n, weighted=True, filter=None):
        self.sumwxn = sumwxn
        self.n = n
        self.weighted = weighted
        self.filter = filter

    def _valid(self, seen, only, shape):
        return _valid(self.sumwxn, seen, only, shape)

################################################# Extremes

class Extremes(Portally):
    _params = {
        "values": portally.checktype.CheckClass("Extremes", "values", required=True, type=InterpretedBuffer),
        "filter": portally.checktype.CheckClass("Extremes", "filter", required=False, type=StatisticFilter),
        }

    values = typedproperty(_params["values"])
    filter = typedproperty(_params["filter"])

    def __init__(self, values, filter=None):
        self.values = values
        self.filter = filter

    def _valid(self, seen, only, shape):
        return _valid(self.values, seen, only, shape)

################################################# Quantiles

class Quantiles(Portally):
    _params = {
        "values": portally.checktype.CheckClass("Quantiles", "values", required=True, type=InterpretedBuffer),
        "p":      portally.checktype.CheckNumber("Quantiles", "p", required=True, min=0.0, max=1.0),
        "weighted": portally.checktype.CheckBool("Quantiles", "n", required=False),
        "filter": portally.checktype.CheckClass("Quantiles", "filter", required=False, type=StatisticFilter),
        }

    values   = typedproperty(_params["values"])
    p        = typedproperty(_params["p"])
    weighted = typedproperty(_params["weighted"])
    filter   = typedproperty(_params["filter"])

    def __init__(self, values, p=0.5, weighted=True, filter=None):
        self.values = values
        self.p = p
        self.weighted = weighted
        self.filter = filter

    def _valid(self, seen, only, shape):
        return _valid(self.values, seen, only, shape)

################################################# Modes

class Modes(Portally):
    _params = {
        "values": portally.checktype.CheckClass("Modes", "values", required=True, type=InterpretedBuffer),
        "filter": portally.checktype.CheckClass("Modes", "filter", required=False, type=StatisticFilter),
        }

    values   = typedproperty(_params["values"])
    filter   = typedproperty(_params["filter"])

    def __init__(self, values, filter=None):
        self.values = values
        self.filter = filter

    def _valid(self, seen, only, shape):
        return _valid(self.values, seen, only, shape)

################################################# Statistics

class Statistics(Portally):
    _params = {
        "moments":   portally.checktype.CheckVector("Statistics", "moments", required=False, type=Moments),
        "quantiles": portally.checktype.CheckVector("Statistics", "quantiles", required=False, type=Quantiles),
        "modes":     portally.checktype.CheckClass("Statistics", "modes", required=False, type=Modes),
        "minima":    portally.checktype.CheckClass("Statistics", "minima", required=False, type=Extremes),
        "maxima":    portally.checktype.CheckClass("Statistics", "maxima", required=False, type=Extremes),
        }

    moments   = typedproperty(_params["moments"])
    quantiles = typedproperty(_params["quantiles"])
    modes     = typedproperty(_params["modes"])
    minima    = typedproperty(_params["minima"])
    maxima    = typedproperty(_params["maxima"])

    def __init__(self, moments=None, quantiles=None, modes=None, minima=None, maxima=None):
        self.moments = moments
        self.quantiles = quantiles
        self.modes = modes
        self.minima = minima
        self.maxima = maxima

    def _valid(self, seen, only, shape):
        if shape == ():
            statshape = (1,)
        else:
            statshape = shape

        if self.moments is not None:
            if len(set((x.n, x.weighted) for x in self.moments)) != len(self.moments):
                raise ValueError("Statistics.moments must have unique (n, weighted)")
            for x in self.moments:
                _valid(x, seen, only, statshape)

        if self.quantiles is not None:
            if len(set((x.p, x.weighted) for x in self.quantiles)) != len(self.quantiles):
                raise ValueError("Statistics.quantiles must have unique (p, weighted)")
            for x in self.quantiles:
                _valid(x, seen, only, statshape)

        _valid(self.modes, seen, only, statshape)
        _valid(self.minima, seen, only, statshape)
        _valid(self.maxima, seen, only, statshape)

        return shape

################################################# Correlations

class Correlations(Portally):
    _params = {
        "xindex": portally.checktype.CheckInteger("Correlations", "xindex", required=True, min=0),
        "yindex": portally.checktype.CheckInteger("Correlations", "yindex", required=True, min=0),
        "sumwxy": portally.checktype.CheckClass("Correlations", "sumwxy", required=True, type=InterpretedBuffer),
        "weighted": portally.checktype.CheckBool("Correlations", "n", required=False),
        "filter": portally.checktype.CheckClass("Correlations", "filter", required=False, type=StatisticFilter),
        }

    xindex   = typedproperty(_params["xindex"])
    yindex   = typedproperty(_params["yindex"])
    sumwxy   = typedproperty(_params["sumwxy"])
    weighted = typedproperty(_params["weighted"])
    filter   = typedproperty(_params["filter"])

    def __init__(self, xindex, yindex, sumwxy, weighted=True, filter=None):
        self.xindex = xindex
        self.yindex = yindex
        self.sumwxy = sumwxy
        self.weighted = weighted
        self.filter = filter

    def _valid(self, seen, only, shape):
        if shape == ():
            corrshape = (1,)
        else:
            corrshape = shape
        _valid(self.sumwxy, seen, only, corrshape)
        return shape

    @staticmethod
    def _validindexes(correlations, numvars):
        pairs = [(x.xindex, x.yindex, x.weighted) for x in correlations]
        if len(set(pairs)) != len(pairs):
            raise ValueError("Correlations.xindex, yindex pairs must be unique")
        if any(x.xindex >= numvars for x in correlations):
            raise ValueError("Correlations.xindex must all be less than the number of axis or column variables {}".format(numvars))
        if any(x.yindex >= numvars for x in correlations):
            raise ValueError("Correlations.yindex must all be less than the number of axis or column variables {}".format(numvars))

################################################# Binning

class Binning(Portally):
    def __init__(self):
        raise TypeError("{0} is an abstract base class; do not construct".format(type(self).__name__))

    @property
    def isnumerical(self):
        return True

    @property
    def dimensions(self):
        return len(self._binshape())

################################################# BinPosition

class BinPositionEnum(Enum): pass

class BinPosition(object):
    below3      = BinPositionEnum("below3", portally.portally_generated.BinPosition.BinPosition.pos_below3)
    below2      = BinPositionEnum("below2", portally.portally_generated.BinPosition.BinPosition.pos_below2)
    below1      = BinPositionEnum("below1", portally.portally_generated.BinPosition.BinPosition.pos_below1)
    nonexistent = BinPositionEnum("nonexistent", portally.portally_generated.BinPosition.BinPosition.pos_nonexistent)
    above1      = BinPositionEnum("above1", portally.portally_generated.BinPosition.BinPosition.pos_above1)
    above2      = BinPositionEnum("above2", portally.portally_generated.BinPosition.BinPosition.pos_above2)
    above3      = BinPositionEnum("above3", portally.portally_generated.BinPosition.BinPosition.pos_above3)
    positions = [below3, below2, below1, nonexistent, above1, above2, above3]

    def __init__(self):
        raise TypeError("{0} is an abstract base class; do not construct".format(type(self).__name__))

################################################# IntegerBinning

class IntegerBinning(Binning, BinPosition):
    _params = {
        "minimum":       portally.checktype.CheckInteger("IntegerBinning", "minimum", required=True),
        "maximum":       portally.checktype.CheckInteger("IntegerBinning", "maximum", required=True),
        "pos_underflow": portally.checktype.CheckEnum("IntegerBinning", "pos_underflow", required=False, choices=BinPosition.positions),
        "pos_overflow":  portally.checktype.CheckEnum("IntegerBinning", "pos_overflow", required=False, choices=BinPosition.positions),
        }

    minimum       = typedproperty(_params["minimum"])
    maximum       = typedproperty(_params["maximum"])
    pos_underflow = typedproperty(_params["pos_underflow"])
    pos_overflow  = typedproperty(_params["pos_overflow"])

    def __init__(self, minimum, maximum, pos_underflow=BinPosition.nonexistent, pos_overflow=BinPosition.nonexistent):
        self.minimum = minimum
        self.maximum = maximum
        self.pos_underflow = pos_underflow
        self.pos_overflow = pos_overflow

    def _valid(self, seen, only, shape):
        if self.minimum >= self.maximum:
            raise ValueError("IntegerBinning.minimum ({0}) must be strictly less than IntegerBinning.maximum ({1})".format(self.minimum, self.maximum))

        if self.pos_underflow != BinPosition.nonexistent and self.pos_overflow != BinPosition.nonexistent and self.pos_underflow == self.pos_overflow:
            raise ValueError("IntegerBinning.pos_underflow and IntegerBinning.pos_overflow must not be equal unless they are both nonexistent")

        return (self.maximum - self.minimum + 1 + int(self.pos_underflow != BinPosition.nonexistent) + int(self.pos_overflow != BinPosition.nonexistent),)

    def _binshape(self):
        return (self.maximum - self.minimum + 1 + int(self.pos_underflow != BinPosition.nonexistent) + int(self.pos_overflow != BinPosition.nonexistent),)

################################################# RealInterval

class RealInterval(Portally):
    _params = {
        "low":            portally.checktype.CheckNumber("RealInterval", "low", required=True),
        "high":           portally.checktype.CheckNumber("RealInterval", "high", required=True),
        "low_inclusive":  portally.checktype.CheckBool("RealInterval", "low_inclusive", required=False),
        "high_inclusive": portally.checktype.CheckBool("RealInterval", "high_inclusive", required=False),
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

    def _valid(self, seen, only, shape):
        if self.low > self.high:
            raise ValueError("RealInterval.low ({0}) must be less than or equal to RealInterval.high ({1})".format(self.low, self.high))

        if self.low == self.high and not self.low_inclusive and not self.high_inclusive:
            raise ValueError("RealInterval describes an empty set ({0} == {1} and both endpoints are exclusive)".format(self.low, self.high))

        return shape

################################################# RealOverflow

class NonRealMappingEnum(Enum): pass

class RealOverflow(Portally, BinPosition):
    missing      = NonRealMappingEnum("missing", portally.portally_generated.NonRealMapping.NonRealMapping.missing)
    in_underflow = NonRealMappingEnum("in_underflow", portally.portally_generated.NonRealMapping.NonRealMapping.in_underflow)
    in_overflow  = NonRealMappingEnum("in_overflow", portally.portally_generated.NonRealMapping.NonRealMapping.in_overflow)
    in_nanflow   = NonRealMappingEnum("in_nanflow", portally.portally_generated.NonRealMapping.NonRealMapping.in_nanflow)
    mappings = [missing, in_underflow, in_overflow, in_nanflow]

    _params = {
        "pos_underflow": portally.checktype.CheckEnum("RealOverflow", "pos_underflow", required=False, choices=BinPosition.positions),
        "pos_overflow":  portally.checktype.CheckEnum("RealOverflow", "pos_overflow", required=False, choices=BinPosition.positions),
        "pos_nanflow":   portally.checktype.CheckEnum("RealOverflow", "pos_nanflow", required=False, choices=BinPosition.positions),
        "minf_mapping":  portally.checktype.CheckEnum("RealOverflow", "minf_mapping", required=False, choices=mappings),
        "pinf_mapping":  portally.checktype.CheckEnum("RealOverflow", "pinf_mapping", required=False, choices=mappings),
        "nan_mapping":   portally.checktype.CheckEnum("RealOverflow", "nan_mapping", required=False, choices=mappings),
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

    def _valid(self, seen, only, shape):
        if self.pos_underflow != BinPosition.nonexistent and self.pos_overflow != BinPosition.nonexistent and self.pos_underflow == self.pos_overflow:
            raise ValueError("RealOverflow.pos_underflow and RealOverflow.pos_overflow must not be equal unless they are both nonexistent")

        if self.pos_underflow != BinPosition.nonexistent and self.pos_nanflow != BinPosition.nonexistent and self.pos_underflow == self.pos_nanflow:
            raise ValueError("RealOverflow.pos_underflow and RealOverflow.pos_nanflow must not be equal unless they are both nonexistent")

        if self.pos_overflow != BinPosition.nonexistent and self.pos_nanflow != BinPosition.nonexistent and self.pos_overflow == self.pos_nanflow:
            raise ValueError("RealOverflow.pos_overflow and RealOverflow.pos_nanflow must not be equal unless they are both nonexistent")

        return (int(self.pos_underflow != BinPosition.nonexistent) + int(self.pos_overflow != BinPosition.nonexistent) + int(self.pos_nanflow != BinPosition.nonexistent),)

    def _numbins(self):
        return int(self.pos_underflow != BinPosition.nonexistent) + int(self.pos_overflow != BinPosition.nonexistent) + int(self.pos_nanflow != BinPosition.nonexistent)

################################################# RegularBinning

class RegularBinning(Binning):
    _params = {
        "num":      portally.checktype.CheckInteger("RegularBinning", "num", required=True, min=1),
        "interval": portally.checktype.CheckClass("RegularBinning", "interval", required=True, type=RealInterval),
        "overflow": portally.checktype.CheckClass("RegularBinning", "overflow", required=False, type=RealOverflow),
        "circular": portally.checktype.CheckBool("RegularBinning", "circular", required=False),
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

    def _valid(self, seen, only, shape):
        _valid(self.interval, seen, only, shape)

        if math.isinf(self.interval.low):
            raise ValueError("RegularBinning.interval.low must be finite")

        if math.isinf(self.interval.high):
            raise ValueError("RegularBinning.interval.high must be finite")

        if self.overflow is None:
            overflowdims, = RealOverflow()._valid(set(), None, shape)
        else:
            overflowdims, = _valid(self.overflow, seen, None, shape)

        return (self.num + overflowdims,)

    def _binshape(self):
        if self.overflow is None:
            numoverflowbins = 0
        else:
            numoverflowbins = self.overflow._numbins()
        return (self.num + numoverflowbins,)

################################################# TicTacToeOverflowBinning

class TicTacToeOverflowBinning(Binning):
    _params = {
        "xnum":      portally.checktype.CheckInteger("TicTacToeOverflowBinning", "xnum", required=True, min=1),
        "ynum":      portally.checktype.CheckInteger("TicTacToeOverflowBinning", "ynum", required=True, min=1),
        "xinterval": portally.checktype.CheckClass("TicTacToeOverflowBinning", "xinterval", required=True, type=RealInterval),
        "yinterval": portally.checktype.CheckClass("TicTacToeOverflowBinning", "yinterval", required=True, type=RealInterval),
        "xoverflow": portally.checktype.CheckClass("TicTacToeOverflowBinning", "xoverflow", required=False, type=RealOverflow),
        "yoverflow": portally.checktype.CheckClass("TicTacToeOverflowBinning", "yoverflow", required=False, type=RealOverflow),
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

    def _valid(self, seen, only, shape):
        _valid(self.xinterval, seen, None, shape)
        _valid(self.yinterval, seen, None, shape)
        if self.xoverflow is None:
            xoverflowdims, = RealOverflow()._valid(set(), None, shape)
        else:
            xoverflowdims, = _valid(self.xoverflow, seen, None, shape)
        if self.yoverflow is None:
            yoverflowdims, = RealOverflow()._valid(set(), None, shape)
        else:
            yoverflowdims, = _valid(self.yoverflow, seen, None, shape)

        return (self.xnum + xoverflowdims, self.ynum + yoverflowdims)

    def _binshape(self):
        if self.xoverflow is None:
            numxoverflowbins = 0
        else:
            numxoverflowbins = self.xoverflow._numbins()
        if self.yoverflow is None:
            numyoverflowbins = 0
        else:
            numyoverflowbins = self.yoverflow._numbins()
        return (self.xnum + numxoverflowbins, self.ynum + numyoverflowbins)
        
################################################# HexagonalBinning

class HexagonalCoordinatesEnum(Enum): pass

class HexagonalBinning(Binning):
    offset         = HexagonalCoordinatesEnum("offset", portally.portally_generated.HexagonalCoordinates.HexagonalCoordinates.hex_offset)
    doubled_offset = HexagonalCoordinatesEnum("doubled_offset", portally.portally_generated.HexagonalCoordinates.HexagonalCoordinates.hex_doubled_offset)
    cube_xy        = HexagonalCoordinatesEnum("cube_xy", portally.portally_generated.HexagonalCoordinates.HexagonalCoordinates.hex_cube_xy)
    cube_yz        = HexagonalCoordinatesEnum("cube_yz", portally.portally_generated.HexagonalCoordinates.HexagonalCoordinates.hex_cube_yz)
    cube_xz        = HexagonalCoordinatesEnum("cube_xz", portally.portally_generated.HexagonalCoordinates.HexagonalCoordinates.hex_cube_xz)
    coordinates = [offset, doubled_offset, cube_xy, cube_yz, cube_xz]

    _params = {
        "qmin":        portally.checktype.CheckInteger("HexagonalBinning", "qmin", required=True),
        "qmax":        portally.checktype.CheckInteger("HexagonalBinning", "qmax", required=True),
        "rmin":        portally.checktype.CheckInteger("HexagonalBinning", "rmin", required=True),
        "rmax":        portally.checktype.CheckInteger("HexagonalBinning", "rmax", required=True),
        "coordinates": portally.checktype.CheckEnum("HexagonalBinning", "coordinates", required=False, choices=coordinates),
        "xorigin":     portally.checktype.CheckNumber("HexagonalBinning", "xorigin", required=False, min_inclusive=False, max_inclusive=False),
        "yorigin":     portally.checktype.CheckNumber("HexagonalBinning", "yorigin", required=False, min_inclusive=False, max_inclusive=False),
        "qangle":      portally.checktype.CheckNumber("HexagonalBinning", "qangle", required=False, min=-0.5*math.pi, max=0.5*math.pi),
        "qoverflow":   portally.checktype.CheckClass("HexagonalBinning", "qoverflow", required=False, type=RealOverflow),
        "roverflow":   portally.checktype.CheckClass("HexagonalBinning", "roverflow", required=False, type=RealOverflow),
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
        self.qangle = qangle
        self.qoverflow = qoverflow
        self.roverflow = roverflow

    def _valid(self, seen, only, shape):
        if self.qmin >= self.qmax:
            raise ValueError("HexagonalBinning.qmin ({0}) must be strictly less than HexagonalBinning.qmax ({1})".format(self.qmin, self.qmax))
        if self.rmin >= self.rmax:
            raise ValueError("HexagonalBinning.rmin ({0}) must be strictly less than HexagonalBinning.rmax ({1})".format(self.rmin, self.rmax))
        qnum = self.qmax - self.qmin + 1
        rnum = self.rmax - self.rmin + 1

        if self.qoverflow is None:
            qoverflowdims, = RealOverflow()._valid(set(), None, shape)
        else:
            qoverflowdims, = _valid(self.qoverflow, seen, None, shape)
        if self.roverflow is None:
            roverflowdims, = RealOverflow()._valid(set(), None, shape)
        else:
            roverflowdims, = _valid(self.roverflow, seen, None, shape)
        return (qnum + qoverflowdims, rnum + roverflowdims)

    def _binshape(self):
        qnum = self.qmax - self.qmin + 1
        rnum = self.rmax - self.rmin + 1
        if self.qoverflow is None:
            numqoverflowbins = 0
        else:
            numqoverflowbins = self.qoverflow._numbins()
        if self.roverflow is None:
            numroverflowbins = 0
        else:
            numroverflowbins = self.roverflow._numbins()
        return (qnum + numqoverflowbins, rnum + numroverflowbins)

################################################# EdgesBinning

class EdgesBinning(Binning):
    _params = {
        "edges":    portally.checktype.CheckVector("EdgesBinning", "edges", required=True, type=float, minlen=1),
        "overflow": portally.checktype.CheckClass("EdgesBinning", "overflow", required=False, type=RealOverflow),
        }

    edges    = typedproperty(_params["edges"])
    overflow = typedproperty(_params["overflow"])

    def __init__(self, edges, overflow=None):
        self.edges = edges
        self.overflow = overflow

    def _valid(self, seen, only, shape):
        if numpy.isinf(self.edges).any():
            raise ValueError("EdgesBinning.edges must all be finite")
        if not numpy.greater(self.edges[1:], self.edges[:-1]).all():
            raise ValueError("EdgesBinning.edges must be strictly increasing")
        if self.overflow is None:
            numoverflow, = RealOverflow()._valid(set(), None, shape)
        else:
            numoverflow, = _valid(self.overflow, seen, None, shape)
        return (len(self.edges) - 1 + numoverflow,)

    def _binshape(self):
        if self.overflow is None:
            numoverflowbins = 0
        else:
            numoverflowbins = self.overflow._numbins()
        return (len(self.edges) - 1 + numoverflowbins,)

################################################# EdgesBinning

class OverlappingFillStrategyEnum(Enum): pass

class IrregularBinning(Binning):
    all   = OverlappingFillStrategyEnum("all", portally.portally_generated.OverlappingFillStrategy.OverlappingFillStrategy.overfill_all)
    first = OverlappingFillStrategyEnum("first", portally.portally_generated.OverlappingFillStrategy.OverlappingFillStrategy.overfill_first)
    last  = OverlappingFillStrategyEnum("last", portally.portally_generated.OverlappingFillStrategy.OverlappingFillStrategy.overfill_last)
    overlapping_fill_strategies = [all, first, last]

    _params = {
        "intervals":        portally.checktype.CheckVector("IrregularBinning", "intervals", required=True, type=RealInterval),
        "overflow":         portally.checktype.CheckClass("IrregularBinning", "overflow", required=False, type=RealOverflow),
        "overlapping_fill": portally.checktype.CheckEnum("IrregularBinning", "overlapping_fill", required=False, choices=overlapping_fill_strategies),
        }

    intervals        = typedproperty(_params["intervals"])
    overflow         = typedproperty(_params["overflow"])
    overlapping_fill = typedproperty(_params["overlapping_fill"])

    def __init__(self, intervals, overflow=None, overlapping_fill=all):
        self.intervals = intervals
        self.overflow = overflow
        self.overlapping_fill = overlapping_fill

    def _valid(self, seen, only, shape):
        for x in self.intervals:
            _valid(x, seen, only, shape)
        if self.overflow is None:
            numoverflow, = RealOverflow()._valid(set(), None, shape)
        else:
            numoverflow, = _valid(self.overflow, seen, None, shape)
        return (len(self.intervals) + numoverflow,)

    def _binshape(self):
        if self.overflow is None:
            numoverflowbins = 0
        else:
            numoverflowbins = self.overflow._numbins()
        return (len(self.intervals) + numoverflowbins,)

################################################# CategoryBinning

class CategoryBinning(Binning, BinPosition):
    _params = {
        "categories": portally.checktype.CheckVector("CategoryBinning", "categories", required=True, type=str),
        "pos_overflow":  portally.checktype.CheckEnum("CategoryBinning", "pos_overflow", required=False, choices=BinPosition.positions),
        }

    categories = typedproperty(_params["categories"])
    pos_overflow = typedproperty(_params["pos_overflow"])

    def __init__(self, categories, pos_overflow=BinPosition.nonexistent):
        self.categories = categories
        self.pos_overflow = pos_overflow

    def _valid(self, seen, only, shape):
        if len(self.categories) != len(set(self.categories)):
            raise ValueError("SparseRegularBinning.bins must be unique")

        return (len(self.categories) + (self.pos_overflow != BinPosition.nonexistent),)

    @property
    def isnumerical(self):
        return False

    def _binshape(self):
        return (len(self.categories) + (self.pos_overflow != BinPosition.nonexistent),)

################################################# SparseRegularBinning

class SparseRegularBinning(Binning, BinPosition):
    _params = {
        "bins":        portally.checktype.CheckVector("SparseRegularBinning", "bins", required=True, type=int),
        "bin_width":   portally.checktype.CheckNumber("SparseRegularBinning", "bin_width", required=True, min=0, min_inclusive=False),
        "origin":      portally.checktype.CheckNumber("SparseRegularBinning", "origin", required=False),
        "pos_nanflow": portally.checktype.CheckEnum("SparseRegularBinning", "pos_nanflow", required=False, choices=BinPosition.positions),
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

    def _valid(self, seen, only, shape):
        if len(self.bins) != len(numpy.unique(self.bins)):
            raise ValueError("SparseRegularBinning.bins must be unique")

        return (len(self.bins) + (self.pos_nanflow != BinPosition.nonexistent),)

    def _binshape(self):
        return (len(self.bins) + (self.pos_nanflow != BinPosition.nonexistent),)

################################################# FractionBinning

class FractionLayoutEnum(Enum): pass

class FractionErrorMethodEnum(Enum): pass

class FractionBinning(Binning):
    passall  = FractionLayoutEnum("passall", portally.portally_generated.FractionLayout.FractionLayout.frac_passall)
    failall  = FractionLayoutEnum("failall", portally.portally_generated.FractionLayout.FractionLayout.frac_failall)
    passfail = FractionLayoutEnum("passfail", portally.portally_generated.FractionLayout.FractionLayout.frac_passfail)
    layouts = [passall, failall, passfail]

    normal           = FractionErrorMethodEnum("normal", portally.portally_generated.FractionErrorMethod.FractionErrorMethod.frac_normal)
    clopper_pearson  = FractionErrorMethodEnum("clopper_pearson", portally.portally_generated.FractionErrorMethod.FractionErrorMethod.frac_clopper_pearson)
    wilson           = FractionErrorMethodEnum("wilson", portally.portally_generated.FractionErrorMethod.FractionErrorMethod.frac_wilson)
    agresti_coull    = FractionErrorMethodEnum("agresti_coull", portally.portally_generated.FractionErrorMethod.FractionErrorMethod.frac_agresti_coull)
    feldman_cousins  = FractionErrorMethodEnum("feldman_cousins", portally.portally_generated.FractionErrorMethod.FractionErrorMethod.frac_feldman_cousins)
    jeffrey          = FractionErrorMethodEnum("jeffrey", portally.portally_generated.FractionErrorMethod.FractionErrorMethod.frac_jeffrey)
    bayesian_uniform = FractionErrorMethodEnum("bayesian_uniform", portally.portally_generated.FractionErrorMethod.FractionErrorMethod.frac_bayesian_uniform)
    error_methods = [normal, clopper_pearson, wilson, agresti_coull, feldman_cousins, jeffrey, bayesian_uniform]

    _params = {
        "layout": portally.checktype.CheckEnum("FractionBinning", "layout", required=False, choices=layouts),
        "layout_reversed": portally.checktype.CheckBool("FractionBinning", "layout_reversed", required=False),
        "error_method": portally.checktype.CheckEnum("FractionBinning", "error_method", required=False, choices=error_methods),
        }

    layout          = typedproperty(_params["layout"])
    layout_reversed = typedproperty(_params["layout_reversed"])
    error_method    = typedproperty(_params["error_method"])

    def __init__(self, layout=passall, layout_reversed=False, error_method=normal):
        self.layout = layout
        self.layout_reversed = layout_reversed
        self.error_method = error_method

    def _valid(self, seen, only, shape):
        return (2,)

    @property
    def isnumerical(self):
        return False

    def _binshape(self):
        return (2,)

################################################# Axis

class Axis(Portally):
    _params = {
        "binning":    portally.checktype.CheckClass("Axis", "binning", required=False, type=Binning),
        "expression": portally.checktype.CheckString("Axis", "expression", required=False),
        "statistics": portally.checktype.CheckClass("Axis", "statistics", required=False, type=Statistics),
        "title":      portally.checktype.CheckString("Axis", "title", required=False),
        "metadata":   portally.checktype.CheckClass("Axis", "metadata", required=False, type=Metadata),
        "decoration": portally.checktype.CheckClass("Axis", "decoration", required=False, type=Decoration),
        }

    binning    = typedproperty(_params["binning"])
    expression = typedproperty(_params["expression"])
    statistics = typedproperty(_params["statistics"])
    title      = typedproperty(_params["title"])
    metadata   = typedproperty(_params["metadata"])
    decoration = typedproperty(_params["decoration"])

    def __init__(self, binning=None, expression="", statistics=None, title="", metadata=None, decoration=None):
        self.binning = binning
        self.expression = expression
        self.statistics = statistics
        self.title = title
        self.metadata = metadata
        self.decoration = decoration

    def _valid(self, seen, only, shape):
        if self.binning is None:
            binshape = (1,)
        else:
            binshape = _valid(self.binning, seen, None, shape)

        _valid(self.statistics, seen, only, shape)

        return binshape

    def _binshape(self):
        if self.binning is None:
            return (1,)
        else:
            return self.binning._binshape()

################################################# Profile

class Profile(Portally):
    _params = {
        "expression": portally.checktype.CheckString("Profile", "expression", required=True),
        "statistics": portally.checktype.CheckClass("Profile", "statistics", required=True, type=Statistics),
        "title":      portally.checktype.CheckString("Profile", "title", required=False),
        "metadata":   portally.checktype.CheckClass("Profile", "metadata", required=False, type=Metadata),
        "decoration": portally.checktype.CheckClass("Profile", "decoration", required=False, type=Decoration),
        }

    expression = typedproperty(_params["expression"])
    statistics = typedproperty(_params["statistics"])
    title      = typedproperty(_params["title"])
    metadata   = typedproperty(_params["metadata"])
    decoration = typedproperty(_params["decoration"])

    def __init__(self, expression, statistics, title="", metadata=None, decoration=None):
        self.expression = expression
        self.statistics = statistics
        self.title = title
        self.metadata = metadata
        self.decoration = decoration

    def _valid(self, seen, only, shape):
        _valid(self.statistics, seen, only, shape)
        return shape

################################################# Counts

class Counts(Portally):
    def __init__(self):
        raise TypeError("{0} is an abstract base class; do not construct".format(type(self).__name__))

################################################# UnweightedCounts

class UnweightedCounts(Counts):
    _params = {
        "counts": portally.checktype.CheckClass("UnweightedCounts", "counts", required=True, type=InterpretedBuffer),
        }

    counts = typedproperty(_params["counts"])

    def __init__(self, counts):
        self.counts = counts

    def _valid(self, seen, only, shape):
        _valid(self.counts, seen, only, shape)

################################################# WeightedCounts

class WeightedCounts(Counts):
    _params = {
        "sumw":       portally.checktype.CheckClass("WeightedCounts", "sumw", required=True, type=InterpretedBuffer),
        "sumw2":      portally.checktype.CheckClass("WeightedCounts", "sumw2", required=False, type=InterpretedBuffer),
        "unweighted": portally.checktype.CheckClass("WeightedCounts", "unweighted", required=False, type=UnweightedCounts),
        }

    sumw       = typedproperty(_params["sumw"])
    sumw2      = typedproperty(_params["sumw2"])
    unweighted = typedproperty(_params["unweighted"])

    def __init__(self, sumw, sumw2=None, unweighted=None):
        self.sumw = sumw
        self.sumw2 = sumw2
        self.unweighted = unweighted

    def _valid(self, seen, only, shape):
        _valid(self.sumw, seen, only, shape)
        _valid(self.sumw2, seen, only, shape)
        _valid(self.unweighted, seen, only, shape)

################################################# Parameter

class Parameter(Portally):
    _params = {
        "identifier": portally.checktype.CheckKey("Parameter", "identifier", required=True, type=str),
        "values":     portally.checktype.CheckClass("Parameter", "values", required=True, type=InterpretedBuffer),
        }

    identifier = typedproperty(_params["identifier"])
    values     = typedproperty(_params["values"])

    def __init__(self, identifier, values):
        self.identifier = identifier
        self.values = values

    def _valid(self, seen, only, shape):
        if shape == ():
            parshape = (1,)
        else:
            parshape = shape
        _valid(self.values, seen, only, parshape)
        return shape

################################################# Function

class Function(Portally):
    def __init__(self):
        raise TypeError("{0} is an abstract base class; do not construct".format(type(self).__name__))

################################################# Object

class Object(Portally):
    def __init__(self):
        raise TypeError("{0} is an abstract base class; do not construct".format(type(self).__name__))

################################################# FunctionObject

class FunctionObject(Object):
    def __init__(self):
        raise TypeError("{0} is an abstract base class; do not construct".format(type(self).__name__))

################################################# ParameterizedFunction

class ParameterizedFunction(Function, FunctionObject):
    _params = {
        "identifier": portally.checktype.CheckKey("ParameterizedFunction", "identifier", required=True, type=str),
        "expression": portally.checktype.CheckString("ParameterizedFunction", "expression", required=True),
        "parameters": portally.checktype.CheckVector("ParameterizedFunction", "parameters", required=False, type=Parameter),
        "title":      portally.checktype.CheckString("ParameterizedFunction", "title", required=False),
        "metadata":   portally.checktype.CheckClass("ParameterizedFunction", "metadata", required=False, type=Metadata),
        "decoration": portally.checktype.CheckClass("ParameterizedFunction", "decoration", required=False, type=Decoration),
        "script":     portally.checktype.CheckString("ParameterizedFunction", "script", required=False),
        }

    identifier = typedproperty(_params["identifier"])
    expression = typedproperty(_params["expression"])
    parameters = typedproperty(_params["parameters"])
    title      = typedproperty(_params["title"])
    metadata   = typedproperty(_params["metadata"])
    decoration = typedproperty(_params["decoration"])
    script     = typedproperty(_params["script"])

    def __init__(self, identifier, expression, parameters=None, title="", metadata=None, decoration=None, script=""):
        self.identifier = identifier
        self.expression = expression
        self.parameters = parameters
        self.title = title
        self.metadata = metadata
        self.decoration = decoration
        self.script = script

    def _valid(self, seen, only, shape):
        if self.parameters is not None:
            if len(set(x.identifier for x in self.parameters)) != len(self.parameters):
                raise ValueError("ParameterizedFunction.parameters keys must be unique")
            for x in self.parameters:
                _valid(x, seen, only, shape)

        return shape

################################################# EvaluatedFunction

class EvaluatedFunction(Function):
    _params = {
        "identifier":  portally.checktype.CheckKey("EvaluatedFunction", "identifier", required=True, type=str),
        "values":      portally.checktype.CheckClass("EvaluatedFunction", "values", required=True, type=InterpretedBuffer),
        "derivatives": portally.checktype.CheckClass("EvaluatedFunction", "derivatives", required=False, type=InterpretedBuffer),
        "errors":      portally.checktype.CheckVector("EvaluatedFunction", "errors", required=False, type=Quantiles),
        "title":       portally.checktype.CheckString("EvaluatedFunction", "title", required=False),
        "metadata":    portally.checktype.CheckClass("EvaluatedFunction", "metadata", required=False, type=Metadata),
        "decoration":  portally.checktype.CheckClass("EvaluatedFunction", "decoration", required=False, type=Decoration),
        "script":      portally.checktype.CheckString("EvaluatedFunction", "script", required=False),
        }

    identifier  = typedproperty(_params["identifier"])
    values      = typedproperty(_params["values"])
    derivatives = typedproperty(_params["derivatives"])
    errors      = typedproperty(_params["errors"])
    title       = typedproperty(_params["title"])
    metadata    = typedproperty(_params["metadata"])
    decoration  = typedproperty(_params["decoration"])
    script      = typedproperty(_params["script"])

    def __init__(self, identifier, values, derivatives=None, errors=None, title="", metadata=None, decoration=None, script=""):
        self.identifier = identifier
        self.values = values
        self.derivatives = derivatives
        self.errors = errors
        self.title = title
        self.metadata = metadata
        self.decoration = decoration
        self.script = script

    def _valid(self, seen, only, shape):
        _valid(self.values, seen, only, shape)
        _valid(self.derivatives, seen, only, shape)
        if self.errors is not None:
            for x in self.errors:
                _valid(x, seen, only, shape)
        return shape

################################################# BinnedEvaluatedFunction

class BinnedEvaluatedFunction(FunctionObject):
    _params = {
        "identifier":  portally.checktype.CheckKey("BinnedEvaluatedFunction", "identifier", required=True, type=str),
        "axis":        portally.checktype.CheckVector("BinnedEvaluatedFunction", "axis", required=True, type=Axis, minlen=1),
        "values":      portally.checktype.CheckClass("BinnedEvaluatedFunction", "values", required=True, type=InterpretedBuffer),
        "derivatives": portally.checktype.CheckClass("BinnedEvaluatedFunction", "derivatives", required=False, type=InterpretedBuffer),
        "errors":      portally.checktype.CheckVector("BinnedEvaluatedFunction", "errors", required=False, type=Quantiles),
        "title":       portally.checktype.CheckString("BinnedEvaluatedFunction", "title", required=False),
        "metadata":    portally.checktype.CheckClass("BinnedEvaluatedFunction", "metadata", required=False, type=Metadata),
        "decoration":  portally.checktype.CheckClass("BinnedEvaluatedFunction", "decoration", required=False, type=Decoration),
        "script":      portally.checktype.CheckString("BinnedEvaluatedFunction", "script", required=False),
        }

    identifier  = typedproperty(_params["identifier"])
    axis        = typedproperty(_params["axis"])
    values      = typedproperty(_params["values"])
    derivatives = typedproperty(_params["derivatives"])
    errors      = typedproperty(_params["errors"])
    title       = typedproperty(_params["title"])
    metadata    = typedproperty(_params["metadata"])
    decoration  = typedproperty(_params["decoration"])
    script      = typedproperty(_params["script"])

    def __init__(self, identifier, axis, values, derivatives=None, errors=None, title="", metadata=None, decoration=None, script=""):
        self.identifier = identifier
        self.axis = axis
        self.values = values
        self.derivatives = derivatives
        self.errors = errors
        self.title = title
        self.metadata = metadata
        self.decoration = decoration
        self.script = script

    def _valid(self, seen, only, shape):
        binshape = shape
        for x in self.axis:
            binshape = binshape + _valid(x, seen, None, shape)

        _valid(self.values, seen, only, binshape)
        _valid(self.derivatives, seen, only, binshape)
        if self.errors is not None:
            for x in self.errors:
                _valid(x, seen, only, binshape)

        return shape

    def _shape(self, path, shape):
        shape = ()
        if len(path) > 0 and isinstance(path[0], (InterpretedBuffer, Quantiles)):
            for x in self.axis:
                shape = shape + x._binshape()

        path = (self,) + path
        if hasattr(self, "_parent"):
            return self._parent._shape(path, shape)
        elif shape == ():
            return (1,)
        else:
            return shape

################################################# Histogram

class Histogram(Object):
    _params = {
        "identifier":           portally.checktype.CheckKey("Histogram", "identifier", required=True, type=str),
        "axis":                 portally.checktype.CheckVector("Histogram", "axis", required=True, type=Axis, minlen=1),
        "counts":               portally.checktype.CheckClass("Histogram", "counts", required=True, type=Counts),
        "profile":              portally.checktype.CheckVector("Histogram", "profile", required=False, type=Profile),
        "axis_correlations":    portally.checktype.CheckVector("Histogram", "axis_correlations", required=False, type=Correlations),
        "profile_correlations": portally.checktype.CheckVector("Histogram", "profile_correlations", required=False, type=Correlations),
        "functions":            portally.checktype.CheckVector("Histogram", "functions", required=False, type=Function),
        "title":                portally.checktype.CheckString("Histogram", "title", required=False),
        "metadata":             portally.checktype.CheckClass("Histogram", "metadata", required=False, type=Metadata),
        "decoration":           portally.checktype.CheckClass("Histogram", "decoration", required=False, type=Decoration),
        "script":               portally.checktype.CheckString("Histogram", "script", required=False),
        }

    identifier           = typedproperty(_params["identifier"])
    axis                 = typedproperty(_params["axis"])
    counts               = typedproperty(_params["counts"])
    profile              = typedproperty(_params["profile"])
    axis_correlations    = typedproperty(_params["axis_correlations"])
    profile_correlations = typedproperty(_params["profile_correlations"])
    functions            = typedproperty(_params["functions"])
    title                = typedproperty(_params["title"])
    metadata             = typedproperty(_params["metadata"])
    decoration           = typedproperty(_params["decoration"])
    script               = typedproperty(_params["script"])

    def __init__(self, identifier, axis, counts, profile=None, axis_correlations=None, profile_correlations=None, functions=None, title="", metadata=None, decoration=None, script=""):
        self.identifier = identifier
        self.axis = axis
        self.counts = counts
        self.profile = profile
        self.axis_correlations = axis_correlations
        self.profile_correlations = profile_correlations
        self.functions = functions
        self.title = title
        self.metadata = metadata
        self.decoration = decoration
        self.script = script

    def _valid(self, seen, only, shape):
        binshape = shape
        for x in self.axis:
            binshape = binshape + _valid(x, seen, None, shape)

        _valid(self.counts, seen, only, binshape)

        if self.axis_correlations is not None:
            Correlations._validindexes(self.axis_correlations, len(binshape) - len(shape))
            for x in self.axis_correlations:
                _valid(x, seen, only, shape)

        if self.profile is not None:
            numprofile = len(self.profile)
            for x in self.profile:
                _valid(x, seen, only, binshape)
        else:
            numprofile = 0

        if self.profile_correlations is not None:
            Correlations._validindexes(self.profile_correlations, numprofile)
            for x in self.profile_correlations:
                _valid(x, seen, only, shape)

        if self.functions is not None:
            if len(set(x.identifier for x in self.functions)) != len(self.functions):
                raise ValueError("Histogram.functions keys must be unique")
            for x in self.functions:
                _valid(x, seen, only, binshape)

        return shape

    def _shape(self, path, shape):
        shape = ()
        if len(path) > 0 and isinstance(path[0], (Counts, Profile, Function)):
            for x in self.axis:
                shape = shape + x._binshape()

        path = (self,) + path
        if hasattr(self, "_parent"):
            return self._parent._shape(path, shape)
        elif shape == ():
            return (1,)
        else:
            return shape

################################################# Page

class Page(Portally):
    _params = {
        "buffer": portally.checktype.CheckClass("Page", "buffer", required=True, type=RawBuffer),
        }

    buffer = typedproperty(_params["buffer"])

    def __init__(self, buffer):
        self.buffer = buffer

    def _valid(self, seen, only, column, numentries):
        _valid(self.buffer, seen, None, column.numpy_dtype.itemsize * numentries)
        buf = self.buffer._array

        if column.filters is not None:
            raise NotImplementedError("handle column.filters")

        if column.postfilter_slice is not None:
            start = column.postfilter_slice.start if column.postfilter_slice.has_start else None
            stop = column.postfilter_slice.stop if column.postfilter_slice.has_stop else None
            step = column.postfilter_slice.step if column.postfilter_slice.has_step else None
            buf = buf[start:stop:step]

        if len(buf) != column.numpy_dtype.itemsize * numentries:
            raise ValueError("Page.buffer length is {0} bytes but ColumnChunk.page_offsets claims {1} entries of {2} bytes each".format(len(buf), numentries, column.numpy_dtype.itemsize))

        self._array = buf.view(column.numpy_dtype).reshape((numentries,))

        return numentries

    @property
    def numpy_array(self):
        self._topvalid()
        return self._array

################################################# ColumnChunk

class ColumnChunk(Portally):
    _params = {
        "pages":         portally.checktype.CheckVector("ColumnChunk", "pages", required=True, type=Page),
        "page_offsets":  portally.checktype.CheckVector("ColumnChunk", "page_offsets", required=True, type=int, minlen=1),
        "page_extremes": portally.checktype.CheckVector("ColumnChunk", "page_extremes", required=False, type=Extremes),
        }

    pages         = typedproperty(_params["pages"])
    page_offsets  = typedproperty(_params["page_offsets"])
    page_extremes = typedproperty(_params["page_extremes"])

    def __init__(self, pages, page_offsets, page_extremes=None):
        self.pages = pages
        self.page_offsets = page_offsets
        self.page_extremes = page_extremes

    def _valid(self, seen, only, column):
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
            _valid(x, seen, only, column, self.page_offsets[i + 1] - self.page_offsets[i])

        if self.page_extremes is not None:
            if len(self.page_extremes) != len(self.pages):
                raise ValueError("ColumnChunk.page_extremes length {0} must be equal to ColumnChunk.pages length {1}".format(len(self.page_extremes), len(self.pages)))

            for x in self.page_extremes:
                _valid(x, seen, only, ())

            raise NotImplementedError("check extremes")

        return self.page_offsets[-1]

    @property
    def numpy_array(self):
        out = [x.numpy_array for x in self.pages]
        if len(out) == 0:
            self._topvalid()
            column = self._parent._parent._parent.columns[self._parent.columns.index(self)]
            return numpy.empty(0, column.numpy_dtype)

        elif len(out) == 1:
            return out[0]

        else:
            return numpy.concatenate(out)

################################################# Chunk

class Chunk(Portally):
    _params = {
        "columns":  portally.checktype.CheckVector("Chunk", "columns", required=True, type=ColumnChunk),
        "metadata": portally.checktype.CheckClass("Chunk", "metadata", required=False, type=Metadata),
        }

    columns  = typedproperty(_params["columns"])
    metadata = typedproperty(_params["metadata"])

    def __init__(self, columns, metadata=None):
        self.columns = columns
        self.metadata = metadata

    def _valid(self, seen, only, columns, numentries):
        # have to do recursive check because Chunk._valid is called directly by Ntuple._valid
        if id(self) in seen:
            raise ValueError("hierarchy is recursively nested")
        seen.add(id(self))

        if len(self.columns) != len(columns):
            raise ValueError("Chunk.columns has length {0}, but Ntuple.columns has length {1}".format(len(self.columns), len(columns)))

        for x, y in zip(self.columns, columns):
            if only is None or id(x) in only:
                x._validtypes()
                num = x._valid(seen, only, y)
                if numentries is not None and num != numentries:
                    raise ValueError("Chunk.column {0} has {1} entries but Chunk has {2} entries".format(repr(y.identifier), num, numentries))

        return numentries

    @property
    def numpy_arrays(self):
        if not isinstance(getattr(self, "_parent", None), NtupleInstance) or not isinstance(getattr(self._parent, "_parent", None), Ntuple):
            raise ValueError("{0} object is not nested in a hierarchy".format(type(self).__name__))

        if len(self.columns) != len(self._parent._parent.columns):
            raise ValueError("Chunk.columns has length {0}, but Ntuple.columns has length {1}".format(len(self.columns), len(self._parent._parent.columns)))

        return {y.identifier: x.numpy_array for x, y in zip(self.columns, self._parent._parent.columns)}

################################################# Column

class Column(Portally, Interpretation):
    _params = {
        "identifier":       portally.checktype.CheckKey("Column", "identifier", required=True, type=str),
        "dtype":            portally.checktype.CheckEnum("Column", "dtype", required=True, choices=Interpretation.dtypes),
        "endianness":       portally.checktype.CheckEnum("Column", "endianness", required=False, choices=Interpretation.endiannesses),
        "filters":          portally.checktype.CheckVector("Column", "filters", required=False, type=Buffer.filters),
        "postfilter_slice": portally.checktype.CheckSlice("Column", "postfilter_slice", required=False),
        "title":            portally.checktype.CheckString("Column", "title", required=False),
        "metadata":         portally.checktype.CheckClass("Column", "metadata", required=False, type=Metadata),
        "decoration":       portally.checktype.CheckClass("Column", "decoration", required=False, type=Decoration),
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

    def _valid(self, seen, only):
        if self.postfilter_slice is not None:
            if self.postfilter_slice.step == 0:
                raise ValueError("slice step cannot be zero")

################################################# NtupleInstance

class NtupleInstance(Portally):
    _params = {
        "chunks":        portally.checktype.CheckVector("NtupleInstance", "chunks", required=True, type=Chunk),
        "chunk_offsets": portally.checktype.CheckVector("NtupleInstance", "chunk_offsets", required=False, type=int, minlen=1),
        }

    chunks              = typedproperty(_params["chunks"])
    chunk_offsets       = typedproperty(_params["chunk_offsets"])

    def __init__(self, chunks, chunk_offsets=None):
        self.chunks = chunks
        self.chunk_offsets = chunk_offsets

    def _valid(self, seen, only):
        if not isinstance(getattr(self, "_parent", None), Ntuple):
            raise ValueError("{0} object is not nested in a hierarchy".format(type(self).__name__))

        if self.chunk_offsets is None:
            for x in self.chunks:
                if only is None or id(x) in only:
                    x._validtypes()
                    x._valid(seen, only, self._parent.columns, None)

        else:
            if self.chunk_offsets[0] != 0:
                raise ValueError("Ntuple.chunk_offsets must start with 0")
            if not numpy.greater_equal(self.chunk_offsets[1:], self.chunk_offsets[:-1]).all():
                raise ValueError("Ntuple.chunk_offsets must be monotonically increasing")

            if len(self.chunk_offsets) != len(self.chunks) + 1:
                raise ValueError("Ntuple.chunk_offsets length is {0}, but it must be one longer than Ntuple.chunks, which is {1}".format(len(self.chunk_offsets), len(self.chunks)))

            for i, x in enumerate(self.chunks):
                if only is None or id(x) in only:
                    x._validtypes()
                    x._valid(seen, only, self._parent.columns, self.chunk_offsets[i + 1] - self.chunk_offsets[i])

    @property
    def numpy_arrays(self):
        self._valid(set(), None)
        for x in self.chunks:
            yield x.numpy_arrays

################################################# Ntuple

class Ntuple(Object):
    _params = {
        "identifier":          portally.checktype.CheckKey("Ntuple", "identifier", required=True, type=str),
        "columns":             portally.checktype.CheckVector("Ntuple", "columns", required=True, type=Column, minlen=1),
        "instances":           portally.checktype.CheckVector("Ntuple", "instances", required=True, type=NtupleInstance, minlen=1),
        "column_statistics":   portally.checktype.CheckVector("Ntuple", "column_statistics", required=False, type=Statistics),
        "column_correlations": portally.checktype.CheckVector("Ntuple", "column_correlations", required=False, type=Correlations),
        "functions":           portally.checktype.CheckVector("Ntuple", "functions", required=False, type=FunctionObject),
        "title":               portally.checktype.CheckString("Ntuple", "title", required=False),
        "metadata":            portally.checktype.CheckClass("Ntuple", "metadata", required=False, type=Metadata),
        "decoration":          portally.checktype.CheckClass("Ntuple", "decoration", required=False, type=Decoration),
        "script":              portally.checktype.CheckString("Ntuple", "script", required=False),
        }

    identifier          = typedproperty(_params["identifier"])
    columns             = typedproperty(_params["columns"])
    instances           = typedproperty(_params["instances"])
    column_statistics   = typedproperty(_params["column_statistics"])
    column_correlations = typedproperty(_params["column_correlations"])
    functions           = typedproperty(_params["functions"])
    title               = typedproperty(_params["title"])
    metadata            = typedproperty(_params["metadata"])
    decoration          = typedproperty(_params["decoration"])
    script              = typedproperty(_params["script"])

    def __init__(self, identifier, columns, instances, column_statistics=None, column_correlations=None, functions=None, title="", metadata=None, decoration=None, script=""):
        self.identifier = identifier
        self.columns = columns
        self.instances = instances
        self.column_statistics = column_statistics
        self.column_correlations = column_correlations
        self.functions = functions
        self.title = title
        self.metadata = metadata
        self.decoration = decoration
        self.script = script

    def _valid(self, seen, only, shape):
        if len(set(x.identifier for x in self.columns)) != len(self.columns):
            raise ValueError("Ntuple.columns keys must be unique")

        for x in self.columns:
            if only is None or id(x) in only:
                x._validtypes()
                x._valid(seen, only)

        if len(self.instances) != functools.reduce(operator.mul, shape, 1):
            raise ValueError("Ntuple.instances length is {0} but multiplicity at this position in the hierarchy is {1}".format(len(self.instances), functools.reduce(operator.mul, shape, 1)))

        for x in self.instances:
            if only is None or id(x) in only:
                x._validtypes()
                x._valid(seen, only)

        if self.column_statistics is not None:
            for x in self.column_statistics:
                _valid(x, seen, only, shape)

        if self.column_correlations is not None:
            Correlations._validindexes(self.column_correlations, len(self.columns))
            for x in self.column_correlations:
                _valid(x, seen, only, shape)

        if self.functions is not None:
            for x in self.functions:
                _valid(x, seen, only, shape)

        return shape

################################################# Region

class Region(Portally):
    _params = {
        "expressions":  portally.checktype.CheckVector("Region", "expressions", required=True, type=str),
        }

    expressions = typedproperty(_params["expressions"])

    def __init__(self, expressions):
        self.expressions  = expressions 

################################################# BinnedRegion

class BinnedRegion(Portally):
    _params = {
        "expression": portally.checktype.CheckString("BinnedRegion", "expression", required=True),
        "binning":    portally.checktype.CheckClass("BinnedRegion", "binning", required=True, type=Binning),
        }

    expression = typedproperty(_params["expression"])
    binning    = typedproperty(_params["binning"])

    def __init__(self, expression, binning):
        self.expression = expression
        self.binning  = binning 

################################################# Assignment

class Assignment(Portally):
    _params = {
        "identifier": portally.checktype.CheckKey("Assignment", "identifier", required=True, type=str),
        "expression": portally.checktype.CheckString("Assignment", "expression", required=True),
        }

    identifier = typedproperty(_params["identifier"])
    expression = typedproperty(_params["expression"])

    def __init__(self, identifier, expression):
        self.identifier = identifier
        self.expression  = expression 

################################################# Variation

class Variation(Portally):
    _params = {
        "assignments":         portally.checktype.CheckVector("Variation", "assignments", required=True, type=Assignment),
        "systematic":          portally.checktype.CheckVector("Variation", "systematic", required=False, type=float),
        "category_systematic": portally.checktype.CheckVector("Variation", "category_systematic", required=False, type=str),
        }

    assignments         = typedproperty(_params["assignments"])
    systematic          = typedproperty(_params["systematic"])
    category_systematic = typedproperty(_params["category_systematic"])

    def __init__(self, assignments, systematic=None, category_systematic=None):
        self.assignments = assignments
        self.systematic = systematic
        self.category_systematic = category_systematic

################################################# Collection

class Collection(Portally):
    _params = {
        "identifier":     portally.checktype.CheckString("Collection", "identifier", required=True),
        "objects":        portally.checktype.CheckVector("Collection", "objects", required=True, type=Object),
        "collections":    portally.checktype.CheckVector("Collection", "collections", required=False, type=None),
        "regions":        portally.checktype.CheckVector("Collection", "regions", required=False, type=Region),
        "binned_regions": portally.checktype.CheckVector("Collection", "binned_regions", required=False, type=BinnedRegion),
        "variations":     portally.checktype.CheckVector("Collection", "variations", required=False, type=Variation),
        "title":          portally.checktype.CheckString("Collection", "title", required=False),
        "metadata":       portally.checktype.CheckClass("Collection", "metadata", required=False, type=Metadata),
        "decoration":     portally.checktype.CheckClass("Collection", "decoration", required=False, type=Decoration),
        "script":         portally.checktype.CheckString("Collection", "script", required=False),
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
    script         = typedproperty(_params["script"])

    def __init__(self, identifier, objects, collections=None, regions=None, binned_regions=None, variations=None, title="", metadata=None, decoration=None, script=""):
        self.identifier = identifier
        self.objects = objects
        self.collections = collections
        self.regions = regions
        self.binned_regions = binned_regions
        self.variations = variations
        self.title = title
        self.metadata = metadata
        self.decoration = decoration
        self.script = script

    def _valid(self, seen, only, shape):
        if len(set(x.identifier for x in self.objects)) != len(self.objects):
            raise ValueError("Collection.objects keys must be unique")

        for x in self.objects:
            _valid(x, seen, only, shape)

        return shape

    def tobuffer(self):
        self.checkvalid()
        builder = flatbuffers.Builder(1024)
        builder.Finish(self._toflatbuffers(builder, None))
        return builder.Output()

    @classmethod
    def frombuffer(cls, buffer, offset=0):
        out = cls.__new__(cls)
        out._flatbuffers = portally.portally_generated.Collection.Collection.GetRootAsCollection(buffer, offset)
        return out

    def toarray(self):
        return numpy.frombuffer(self.tobuffer(), dtype=numpy.uint8)

    @classmethod
    def fromarray(cls, array):
        return cls.frombuffer(array)

    def tofile(self, file):
        self.checkvalid()

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

    def __getitem__(self, where):
        return _getbykey(self, "objects", where)

    def __repr__(self):
        return "<{0} {1} at 0x{2:012x}>".format(type(self).__name__, repr(self.identifier), id(self))

    @property
    def isvalid(self):
        try:
            self.checkvalid()
        except ValueError:
            return False
        else:
            return True

    def checkvalid(self):
        self._valid(set(), None, ())

    def _toflatbuffers(self, builder, file):
        identifier = builder.CreateString(self._identifier)
        if len(self._title) > 0:
            title = builder.CreateString(self._title)
        portally.portally_generated.Collection.CollectionStart(builder)
        portally.portally_generated.Collection.CollectionAddIdentifier(builder, identifier)
        if len(self._title) > 0:
            portally.portally_generated.Collection.CollectionAddTitle(builder, title)
        return portally.portally_generated.Collection.CollectionEnd(builder)

Collection._params["collections"].type = Collection
