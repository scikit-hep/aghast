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

import portally.portally_generated.Assignment
import portally.portally_generated.Axis
import portally.portally_generated.BinnedEvaluatedFunction
import portally.portally_generated.BinnedRegion
import portally.portally_generated.Binning
import portally.portally_generated.BinPosition
import portally.portally_generated.CategoryBinning
import portally.portally_generated.Chunk
import portally.portally_generated.Collection
import portally.portally_generated.Column
import portally.portally_generated.ColumnChunk
import portally.portally_generated.Correlation
import portally.portally_generated.Counts
import portally.portally_generated.Decoration
import portally.portally_generated.DecorationLanguage
import portally.portally_generated.Descriptive
import portally.portally_generated.DescriptiveFilter
import portally.portally_generated.DimensionOrder
import portally.portally_generated.DType
import portally.portally_generated.EdgesBinning
import portally.portally_generated.Endianness
import portally.portally_generated.EvaluatedFunction
import portally.portally_generated.ExternalType
import portally.portally_generated.Extremes
import portally.portally_generated.Filter
import portally.portally_generated.FractionalErrorMethod
import portally.portally_generated.FractionBinning
import portally.portally_generated.Function
import portally.portally_generated.FunctionData
import portally.portally_generated.FunctionObject
import portally.portally_generated.FunctionObjectData
import portally.portally_generated.HexagonalBinning
import portally.portally_generated.HexagonalCoordinates
import portally.portally_generated.Histogram
import portally.portally_generated.IntegerBinning
import portally.portally_generated.InterpretedBuffer
import portally.portally_generated.InterpretedExternalBuffer
import portally.portally_generated.InterpretedInlineBuffer
import portally.portally_generated.IrregularBinning
import portally.portally_generated.Metadata
import portally.portally_generated.MetadataLanguage
import portally.portally_generated.Modes
import portally.portally_generated.Moments
import portally.portally_generated.NonRealMapping
import portally.portally_generated.Ntuple
import portally.portally_generated.NtupleInstance
import portally.portally_generated.Object
import portally.portally_generated.ObjectData
import portally.portally_generated.OverlappingFillStrategy
import portally.portally_generated.Page
import portally.portally_generated.Parameter
import portally.portally_generated.ParameterizedFunction
import portally.portally_generated.Profile
import portally.portally_generated.Quantiles
import portally.portally_generated.RawBuffer
import portally.portally_generated.RawExternalBuffer
import portally.portally_generated.RawInlineBuffer
import portally.portally_generated.RealInterval
import portally.portally_generated.RealOverflow
import portally.portally_generated.Region
import portally.portally_generated.RegularBinning
import portally.portally_generated.Slice
import portally.portally_generated.SparseRegularBinning
import portally.portally_generated.TicTacToeOverflowBinning
import portally.portally_generated.UnweightedCounts
import portally.portally_generated.Variation
import portally.portally_generated.WeightedCounts

import portally.checktype

def typedproperty(check):
    def setparent(self, value):
        if isinstance(value, Portally):
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

def _valid(obj, seen, only, shape):
    if obj is None:
        return shape
    else:
        if id(obj) in seen:
            raise ValueError("hierarchy is recursively nested")
        seen.add(id(obj))
        obj._validtypes()
        return obj._valid(seen, only, shape)

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

class Metadata(Portally):
    unspecified = Enum("unspecified", portally.portally_generated.MetadataLanguage.MetadataLanguage.meta_unspecified)
    json = Enum("json", portally.portally_generated.MetadataLanguage.MetadataLanguage.meta_json)
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

class Decoration(Portally):
    unspecified = Enum("unspecified", portally.portally_generated.DecorationLanguage.DecorationLanguage.deco_unspecified)
    css         = Enum("css", portally.portally_generated.DecorationLanguage.DecorationLanguage.deco_css)
    vega        = Enum("vega", portally.portally_generated.DecorationLanguage.DecorationLanguage.deco_vega)
    root_json   = Enum("root_json", portally.portally_generated.DecorationLanguage.DecorationLanguage.deco_root_json)
    language = [unspecified, css, vega, root_json]

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

################################################# Object

class Object(Portally):
    def __init__(self):
        raise TypeError("{0} is an abstract base class; do not construct".format(type(self).__name__))

################################################# Buffers

class Buffer(Portally):
    none = Enum("none", portally.portally_generated.Filter.Filter.filter_none)
    gzip = Enum("gzip", portally.portally_generated.Filter.Filter.filter_gzip)
    lzma = Enum("lzma", portally.portally_generated.Filter.Filter.filter_lzma)
    lz4  = Enum("lz4", portally.portally_generated.Filter.Filter.filter_lz4)
    filters = [none, gzip, lzma, lz4]

    def __init__(self):
        raise TypeError("{0} is an abstract base class; do not construct".format(type(self).__name__))

class InlineBuffer(object):
    def __init__(self):
        raise TypeError("{0} is an abstract base class; do not construct".format(type(self).__name__))

class ExternalBuffer(object):
    memory   = Enum("memory", portally.portally_generated.ExternalType.ExternalType.external_memory)
    samefile = Enum("samefile", portally.portally_generated.ExternalType.ExternalType.external_samefile)
    file     = Enum("file", portally.portally_generated.ExternalType.ExternalType.external_file)
    url      = Enum("url", portally.portally_generated.ExternalType.ExternalType.external_url)
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

    @classmethod
    def fromarray(cls, array):
        dtype, endianness = Interpretation.from_numpy_dtype(array.dtype)
        order = InterpretedBuffer.fortran_order if numpy.isfortran(array) else InterpretedBuffer.c_order
        return cls(array, dtype=dtype, endianness=endianness, dimension_order=order)

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

        if len(self._array) != functools.reduce(operator.mul, shape, self.numpy_dtype.itemsize):
            raise ValueError("InterpretedInlineBuffer.buffer length is {0} but multiplicity at this position in the hierarchy is {1}".format(len(self._array), functools.reduce(operator.mul, shape, self.numpy_dtype.itemsize)))

        self._array = self._array.view(self.numpy_dtype).reshape(shape, order=self.dimension_order.dimension_order)
        return shape

    @property
    def numpy_array(self):
        self._topvalid()
        return self._array

################################################# ExternalBuffer

class InterpretedExternalBuffer(Buffer, InterpretedBuffer, ExternalBuffer):
    _params = {
        "pointer":          portally.checktype.CheckInteger("ExternalBuffer", "pointer", required=True, min=0),
        "numbytes":         portally.checktype.CheckInteger("ExternalBuffer", "numbytes", required=True, min=0),
        "external_type":    portally.checktype.CheckEnum("ExternalBuffer", "external_type", required=False, choices=ExternalBuffer.types),
        "filters":          portally.checktype.CheckVector("ExternalBuffer", "filters", required=False, type=Buffer.filters),
        "postfilter_slice": portally.checktype.CheckSlice("ExternalBuffer", "postfilter_slice", required=False),
        "dtype":            portally.checktype.CheckEnum("ExternalBuffer", "dtype", required=False, choices=InterpretedBuffer.dtypes),
        "endianness":       portally.checktype.CheckEnum("ExternalBuffer", "endianness", required=False, choices=InterpretedBuffer.endiannesses),
        "dimension_order":  portally.checktype.CheckEnum("ExternalBuffer", "dimension_order", required=False, choices=InterpretedBuffer.orders),
        "location":         portally.checktype.CheckString("ExternalBuffer", "location", required=False),
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

        if len(self._array) != functools.reduce(operator.mul, shape, self.numpy_dtype.itemsize):
            raise ValueError("InterpretedExternalBuffer.buffer length is {0} but multiplicity at this position in the hierarchy is {1}".format(len(self._array), functools.reduce(operator.mul, shape, self.numpy_dtype.itemsize)))

        self._array = self._array.view(self.numpy_dtype).reshape(shape, order=self.dimension_order.dimension_order)
        return shape

    @property
    def numpy_array(self):
        self._topvalid()
        return self._array

################################################# Binning

class Binning(Portally):
    def __init__(self):
        raise TypeError("{0} is an abstract base class; do not construct".format(type(self).__name__))

################################################# FractionalBinning

class FractionalBinning(Binning):
    normal           = Enum("normal", portally.portally_generated.FractionalErrorMethod.FractionalErrorMethod.frac_normal)
    clopper_pearson  = Enum("clopper_pearson", portally.portally_generated.FractionalErrorMethod.FractionalErrorMethod.frac_clopper_pearson)
    wilson           = Enum("wilson", portally.portally_generated.FractionalErrorMethod.FractionalErrorMethod.frac_wilson)
    agresti_coull    = Enum("agresti_coull", portally.portally_generated.FractionalErrorMethod.FractionalErrorMethod.frac_agresti_coull)
    feldman_cousins  = Enum("feldman_cousins", portally.portally_generated.FractionalErrorMethod.FractionalErrorMethod.frac_feldman_cousins)
    jeffrey          = Enum("jeffrey", portally.portally_generated.FractionalErrorMethod.FractionalErrorMethod.frac_jeffrey)
    bayesian_uniform = Enum("bayesian_uniform", portally.portally_generated.FractionalErrorMethod.FractionalErrorMethod.frac_bayesian_uniform)
    error_methods = [normal, clopper_pearson, wilson, agresti_coull, feldman_cousins, jeffrey, bayesian_uniform]

    _params = {
        "error_method": portally.checktype.CheckEnum("FractionalBinning", "error_method", required=False, choices=error_methods),
        }

    error_method = typedproperty(_params["error_method"])

    def __init__(self, error_method=normal):
        self.error_method = error_method

    def _valid(self, seen, only, shape):
        return shape

################################################# BinPosition

class BinPosition(object):
    below3 = Enum("below3", portally.portally_generated.BinPosition.BinPosition.pos_below3)
    below2 = Enum("below2", portally.portally_generated.BinPosition.BinPosition.pos_below2)
    below1 = Enum("below1", portally.portally_generated.BinPosition.BinPosition.pos_below1)
    nonexistent = Enum("nonexistent", portally.portally_generated.BinPosition.BinPosition.pos_nonexistent)
    above1 = Enum("above1", portally.portally_generated.BinPosition.BinPosition.pos_above1)
    above2 = Enum("above2", portally.portally_generated.BinPosition.BinPosition.pos_above2)
    above3 = Enum("above3", portally.portally_generated.BinPosition.BinPosition.pos_above3)
    positions = [below3, below2, below1, nonexistent, above1, above2, above3]

################################################# IntegerBinning

class IntegerBinning(Binning, BinPosition):
    _params = {
        "min":           portally.checktype.CheckInteger("IntegerBinning", "min", required=True),
        "max":           portally.checktype.CheckInteger("IntegerBinning", "max", required=True),
        "pos_underflow": portally.checktype.CheckEnum("IntegerBinning", "pos_underflow", required=False, choices=BinPosition.positions),
        "pos_overflow":  portally.checktype.CheckEnum("IntegerBinning", "pos_overflow", required=False, choices=BinPosition.positions),
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

    def _valid(self, seen, only, shape):
        if self.min >= self.max:
            raise ValueError("IntegerBinning.min ({0}) must be strictly less than IntegerBinning.max ({1})".format(self.min, self.max))

        if self.pos_underflow != BinPosition.nonexistent and self.pos_overflow != BinPosition.nonexistent and self.pos_underflow == self.pos_overflow:
            raise ValueError("IntegerBinning.pos_underflow and IntegerBinning.pos_overflow must not be equal unless they are both nonexistent")

        return (self.max - self.min + 1 + int(self.pos_underflow != BinPosition.nonexistent) + int(self.pos_overflow != BinPosition.nonexistent),)

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

class RealOverflow(Portally, BinPosition):
    missing      = Enum("missing", portally.portally_generated.NonRealMapping.NonRealMapping.missing)
    in_underflow = Enum("in_underflow", portally.portally_generated.NonRealMapping.NonRealMapping.in_underflow)
    in_overflow  = Enum("in_overflow", portally.portally_generated.NonRealMapping.NonRealMapping.in_overflow)
    in_nanflow   = Enum("in_nanflow", portally.portally_generated.NonRealMapping.NonRealMapping.in_nanflow)
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
        if only is None or id(self.interval) in only:
            _valid(self.interval, seen, only, shape)
        if math.isinf(self.interval.low) or math.isnan(self.interval.low):
            raise ValueError("RegularBinning.interval.low must be finite")
        if math.isinf(self.interval.high) or math.isnan(self.interval.high):
            raise ValueError("RegularBinning.interval.high must be finite")
        if self.overflow is None:
            overflowdims, = RealOverflow()._valid(set(), None, shape)
        else:
            overflowdims, = _valid(self.overflow, seen, only, shape)
        return (self.num + overflowdims,)

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
        if only is None or id(self.xinterval) in only:
            _valid(self.xinterval, seen, only, shape)
        if only is None or id(self.yinterval) in only:
            _valid(self.yinterval, seen, only, shape)
        if self.xoverflow is None:
            xoverflowdims, = RealOverflow()._valid(set(), None, shape)
        else:
            xoverflowdims, = _valid(self.xoverflow, seen, only, shape)
        if self.yoverflow is None:
            yoverflowdims, = RealOverflow()._valid(set(), None, shape)
        else:
            yoverflowdims, = _valid(self.yoverflow, seen, only, shape)

        return (self.xnum + xoverflowdims, self.ynum + yoverflowdims)
        
################################################# HexagonalBinning

class HexagonalBinning(Binning):
    offset         = Enum("offset", portally.portally_generated.HexagonalCoordinates.HexagonalCoordinates.hex_offset)
    doubled_offset = Enum("doubled_offset", portally.portally_generated.HexagonalCoordinates.HexagonalCoordinates.hex_doubled_offset)
    cube_xy        = Enum("cube_xy", portally.portally_generated.HexagonalCoordinates.HexagonalCoordinates.hex_cube_xy)
    cube_yz        = Enum("cube_yz", portally.portally_generated.HexagonalCoordinates.HexagonalCoordinates.hex_cube_yz)
    cube_xz        = Enum("cube_xz", portally.portally_generated.HexagonalCoordinates.HexagonalCoordinates.hex_cube_xz)
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
        qnum = self.qmax - self.qmin + 1
        rnum = self.rmax - self.rmin + 1
        if self.qoverflow is None:
            qoverflowdims, = RealOverflow()._valid(set(), None, shape)
        else:
            qoverflowdims, = _valid(self.qoverflow, seen, only, shape)
        if self.roverflow is None:
            roverflowdims, = RealOverflow()._valid(set(), None, shape)
        else:
            roverflowdims, = _valid(self.roverflow, seen, only, shape)
        return (qnum + qoverflowdims, rnum + roverflowdims)

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
        if any(math.isinf(x) or math.isnan(x) for x in self.edges):
            raise ValueError("EdgesBinning.edges must all be finite")
        if not numpy.greater(self.edges[1:], self.edges[:-1]).all():
            raise ValueError("EdgesBinning.edges must be strictly increasing")
        if self.overflow is None:
            numoverflow, = RealOverflow()._valid(set(), None, shape)
        else:
            numoverflow, = _valid(self.overflow, seen, only, shape)
        return (len(self.edges) - 1 + numoverflow,)

################################################# EdgesBinning

class IrregularBinning(Binning):
    all = Enum("all", portally.portally_generated.OverlappingFillStrategy.OverlappingFillStrategy.overfill_all)
    first = Enum("first", portally.portally_generated.OverlappingFillStrategy.OverlappingFillStrategy.overfill_first)
    last = Enum("last", portally.portally_generated.OverlappingFillStrategy.OverlappingFillStrategy.overfill_last)
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
            if only is None or id(x) in only:
                _valid(x, seen, only, shape)
        if self.overflow is None:
            numoverflow, = RealOverflow()._valid(set(), None, shape)
        else:
            numoverflow, = _valid(self.overflow, seen, only, shape)
        return (len(self.intervals) + numoverflow,)

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

################################################# Axis

class Axis(Portally):
    _params = {
        "binning":    portally.checktype.CheckClass("Axis", "binning", required=False, type=Binning),
        "expression": portally.checktype.CheckString("Axis", "expression", required=False),
        "title":      portally.checktype.CheckString("Axis", "title", required=False),
        "metadata":   portally.checktype.CheckClass("Axis", "metadata", required=False, type=Metadata),
        "decoration": portally.checktype.CheckClass("Axis", "decoration", required=False, type=Decoration),
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

    def _valid(self, seen, only, shape):
        if self.binning is None:
            binshape = (1,)
        else:
            binshape = _valid(self.binning, seen, only, shape)
        if only is None or id(self.metadata) in only:
            _valid(self.metadata, seen, only, shape)
        if only is None or id(self.decoration) in only:
            _valid(self.decoration, seen, only, shape)
        return binshape

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
        if only is None or id(self.counts) in only:
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
        if only is None or id(self.sumw) in only:
            _valid(self.sumw, seen, only, shape)
        if only is None or id(self.sumw2) in only:
            _valid(self.sumw2, seen, only, shape)
        if only is None or id(self.unweighted) in only:
            _valid(self.unweighted, seen, only, shape)

################################################# DescriptiveFilter

class DescriptiveFilter(Portally):
    _params = {
        "minimum": portally.checktype.CheckNumber("DescriptiveFilter", "minimum", required=False),
        "maximum": portally.checktype.CheckNumber("DescriptiveFilter", "maximum", required=False),
        "excludes_minf": portally.checktype.CheckBool("DescriptiveFilter", "excludes_minf", required=False),
        "excludes_pinf": portally.checktype.CheckBool("DescriptiveFilter", "excludes_pinf", required=False),
        "excludes_nan":  portally.checktype.CheckBool("DescriptiveFilter", "excludes_nan", required=False),
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
        "sumwxn": portally.checktype.CheckClass("Moments", "sumwxn", required=True, type=InterpretedBuffer),
        "n":      portally.checktype.CheckInteger("Moments", "n", required=True, min=1),
        "filter": portally.checktype.CheckClass("Moments", "filter", required=False, type=DescriptiveFilter),
        }

    sumwxn = typedproperty(_params["sumwxn"])
    n      = typedproperty(_params["n"])
    filter = typedproperty(_params["filter"])

    def __init__(self, sumwxn, n, filter=None):
        self.sumwxn = sumwxn
        self.n = n
        self.filter = filter

################################################# Extremes

class Extremes(Portally):
    _params = {
        "values": portally.checktype.CheckClass("Extremes", "values", required=True, type=InterpretedBuffer),
        "filter": portally.checktype.CheckClass("Extremes", "filter", required=False, type=DescriptiveFilter),
        }

    values = typedproperty(_params["values"])
    filter = typedproperty(_params["filter"])

    def __init__(self, values, filter=None):
        self.values = values
        self.filter = filter

################################################# Quantiles

class Quantiles(Portally):
    _params = {
        "values": portally.checktype.CheckClass("Quantiles", "values", required=True, type=InterpretedBuffer),
        "p":      portally.checktype.CheckNumber("Quantiles", "p", required=True, min=0.0, max=1.0),
        "filter": portally.checktype.CheckClass("Quantiles", "filter", required=False, type=DescriptiveFilter),
        }

    values = typedproperty(_params["values"])
    p      = typedproperty(_params["p"])
    filter = typedproperty(_params["filter"])

    def __init__(self, values, p=0.5, filter=None):
        self.values = values
        self.p = p
        self.filter = filter

################################################# Modes

class Modes(Portally):
    _params = {
        "values": portally.checktype.CheckClass("Modes", "values", required=True, type=InterpretedBuffer),
        "filter": portally.checktype.CheckClass("Modes", "filter", required=False, type=DescriptiveFilter),
        }

    values = typedproperty(_params["values"])
    filter = typedproperty(_params["filter"])

    def __init__(self, values, filter=None):
        self.values = values
        self.filter = filter

################################################# Descriptive

class Descriptive(Portally):
    _params = {
        "moments":   portally.checktype.CheckVector("Descriptive", "moments", required=False, type=Moments),
        "quantiles": portally.checktype.CheckVector("Descriptive", "quantiles", required=False, type=Quantiles),
        "modes":     portally.checktype.CheckClass("Descriptive", "modes", required=False, type=Modes),
        "minima":    portally.checktype.CheckClass("Descriptive", "minima", required=False, type=Extremes),
        "maxima":    portally.checktype.CheckClass("Descriptive", "maxima", required=False, type=Extremes),
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
        HERE

################################################# Correlation

class Correlation(Portally):
    _params = {
        "sumwxy": portally.checktype.CheckClass("Correlation", "sumwxy", required=True, type=InterpretedBuffer),
        "filter": portally.checktype.CheckClass("Modes", "filter", required=False, type=DescriptiveFilter),
        }

    sumwxy = typedproperty(_params["sumwxy"])
    filter = typedproperty(_params["filter"])

    def __init__(self, sumwxy, filter=None):
        self.sumwxy = sumwxy
        self.filter = filter

################################################# Profile

class Profile(Portally):
    _params = {
        "expression":  portally.checktype.CheckString("Profile", "expression", required=True),
        "descriptive": portally.checktype.CheckString("Profile", "descriptive", required=True),
        "title":       portally.checktype.CheckString("Profile", "title", required=False),
        "metadata":    portally.checktype.CheckClass("Profile", "metadata", required=False, type=Metadata),
        "decoration":  portally.checktype.CheckClass("Profile", "decoration", required=False, type=Decoration),
        }

    expression  = typedproperty(_params["expression"])
    descriptive = typedproperty(_params["descriptive"])
    title       = typedproperty(_params["title"])
    metadata    = typedproperty(_params["metadata"])
    decoration  = typedproperty(_params["decoration"])

    def __init__(self, expression, descriptive, title="", metadata=None, decoration=None):
        self.expression = expression
        self.descriptive = descriptive
        self.title = title
        self.metadata = metadata
        self.decoration = decoration

################################################# Parameter

class Parameter(Portally):
    _params = {
        "identifier": portally.checktype.CheckKey("Parameter", "identifier", required=True, type=str),
        "value":      portally.checktype.CheckNumber("Parameter", "value", required=True),
        }

    identifier = typedproperty(_params["identifier"])
    value      = typedproperty(_params["value"])

    def __init__(self, identifier, value):
        self.identifier = identifier
        self.value = value

    def _valid(self, seen, only, shape):
        return shape

################################################# Function

class Function(Portally):
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
        "parameters": portally.checktype.CheckVector("ParameterizedFunction", "parameters", required=True, type=Parameter),
        "contours":   portally.checktype.CheckVector("ParameterizedFunction", "contours", required=False, type=float),
        "title":      portally.checktype.CheckString("ParameterizedFunction", "title", required=False),
        "metadata":   portally.checktype.CheckClass("ParameterizedFunction", "metadata", required=False, type=Metadata),
        "decoration": portally.checktype.CheckClass("ParameterizedFunction", "decoration", required=False, type=Decoration),
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

    def _valid(self, seen, only, shape):
        if len(set(x.identifier for x in self.parameters)) != len(self.parameters):
            raise ValueError("ParameterizedFunction.parameters keys must be unique")

        for x in self.parameters:
            if only is None or id(x) in only:
                _valid(x, seen, only, shape)

        if self.contours is not None:
            if len(self.contours) != len(numpy.unique(self.contours)):
                raise ValueError("ParameterizedFunction.contours must be unique")

        return shape

################################################# EvaluatedFunction

class EvaluatedFunction(Function):
    _params = {
        "identifier":  portally.checktype.CheckKey("EvaluatedFunction", "identifier", required=True, type=str),
        "values":      portally.checktype.CheckVector("EvaluatedFunction", "values", required=True, type=float),
        "derivatives": portally.checktype.CheckVector("EvaluatedFunction", "derivatives", required=False, type=float),
        "errors":      portally.checktype.CheckVector("EvaluatedFunction", "errors", required=False, type=Quantiles),
        "title":       portally.checktype.CheckString("EvaluatedFunction", "title", required=False),
        "metadata":    portally.checktype.CheckClass("EvaluatedFunction", "metadata", required=False, type=Metadata),
        "decoration":  portally.checktype.CheckClass("EvaluatedFunction", "decoration", required=False, type=Decoration),
        }

    identifier  = typedproperty(_params["identifier"])
    values      = typedproperty(_params["values"])
    derivatives = typedproperty(_params["derivatives"])
    errors      = typedproperty(_params["errors"])
    title       = typedproperty(_params["title"])
    metadata    = typedproperty(_params["metadata"])
    decoration  = typedproperty(_params["decoration"])

    def __init__(self, identifier, values, derivatives=None, errors=None, title="", metadata=None, decoration=None):
        self.identifier = identifier
        self.values = values
        self.derivatives = derivatives
        self.errors = errors
        self.title = title
        self.metadata = metadata
        self.decoration = decoration

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
        }

    identifier  = typedproperty(_params["identifier"])
    axis        = typedproperty(_params["axis"])
    values      = typedproperty(_params["values"])
    derivatives = typedproperty(_params["derivatives"])
    errors      = typedproperty(_params["errors"])
    title       = typedproperty(_params["title"])
    metadata    = typedproperty(_params["metadata"])
    decoration  = typedproperty(_params["decoration"])

    def __init__(self, identifier, axis, values, derivatives=None, errors=None, title="", metadata=None, decoration=None):
        self.identifier = identifier
        self.axis = axis
        self.values = values
        self.derivatives = derivatives
        self.errors = errors
        self.title = title
        self.metadata = metadata
        self.decoration = decoration

    def _valid(self, seen, only, shape):
        binshape = shape
        for x in self.axis:
            binshape = binshape + _valid(x, seen, only, shape)

        if only is None or id(self.values) in only:
            _valid(self.values, seen, only, binshape)
        if only is None or id(self.derivatives) in only:
            _valid(self.derivatives, seen, only, binshape)
        if only is None or id(self.errors) in only:
            _valid(self.errors, seen, only, binshape)

        if only is None or id(self.metadata) in only:
            _valid(self.metadata, seen, only, shape)
        if only is None or id(self.decoration) in only:
            _valid(self.decoration, seen, only, shape)

        return shape

################################################# Histogram

class Histogram(Object):
    _params = {
        "identifier":          portally.checktype.CheckKey("Histogram", "identifier", required=True, type=str),
        "axis":                portally.checktype.CheckVector("Histogram", "axis", required=True, type=Axis, minlen=1),
        "counts":              portally.checktype.CheckClass("Histogram", "counts", required=True, type=Counts),
        "descriptive":         portally.checktype.CheckClass("Histogram", "descriptive", required=False, type=Descriptive),
        "correlation":         portally.checktype.CheckClass("Histogram", "correlation", required=False, type=Correlation),
        "profile":             portally.checktype.CheckVector("Histogram", "profile", required=False, type=Profile),
        "profile_correlation": portally.checktype.CheckClass("Histogram", "profile_correlation", required=False, type=Correlation),
        "functions":           portally.checktype.CheckVector("Histogram", "functions", required=False, type=Function),
        "title":               portally.checktype.CheckString("Histogram", "title", required=False),
        "metadata":            portally.checktype.CheckClass("Histogram", "metadata", required=False, type=Metadata),
        "decoration":          portally.checktype.CheckClass("Histogram", "decoration", required=False, type=Decoration),
        }

    identifier          = typedproperty(_params["identifier"])
    axis                = typedproperty(_params["axis"])
    counts              = typedproperty(_params["counts"])
    descriptive         = typedproperty(_params["descriptive"])
    correlation         = typedproperty(_params["correlation"])
    profile             = typedproperty(_params["profile"])
    profile_correlation = typedproperty(_params["profile_correlation"])
    functions           = typedproperty(_params["functions"])
    title               = typedproperty(_params["title"])
    metadata            = typedproperty(_params["metadata"])
    decoration          = typedproperty(_params["decoration"])

    def __init__(self, identifier, axis, counts, descriptive=None, correlation=None, profile=None, profile_correlation=None, functions=None, title="", metadata=None, decoration=None):
        self.identifier = identifier
        self.axis = axis
        self.counts = counts
        self.descriptive = descriptive
        self.correlation = correlation
        self.profile = profile
        self.profile_correlation = profile_correlation
        self.functions = functions
        self.title = title
        self.metadata = metadata
        self.decoration = decoration

    def _valid(self, seen, only, shape):
        binshape = shape
        for x in self.axis:
            binshape = binshape + _valid(x, seen, only, shape)

        if only is None or id(self.counts) in only:
            _valid(self.counts, seen, only, binshape)

        if self.descriptive is not None:
            HERE

        if self.correlation is not None:
            HERE

        if self.profile is not None:
            HERE

        if self.profile_correlation is not None:
            HERE

        if self.functions is not None and len(set(x.identifier for x in self.functions)) != len(self.functions):
            raise ValueError("Histogram.functions keys must be unique")
        if only is None or id(self.functions) in only:
            _valid(self.functions, seen, only, binshape)

        if only is None or id(self.metadata) in only:
            _valid(self.metadata, seen, only, shape)
        if only is None or id(self.decoration) in only:
            _valid(self.decoration, seen, only, shape)

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
        self.buffer._valid(seen, only, column.numpy_dtype.itemsize * numentries)
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
            if only is None or id(x) in only:
                x._validtypes()
                x._valid(seen, only, column, self.page_offsets[i + 1] - self.page_offsets[i])

        if self.page_extremes is not None:
            if len(self.page_extremes) != len(self.pages):
                raise ValueError("ColumnChunk.page_extremes length {0} must be equal to ColumnChunk.pages length {1}".format(len(self.page_extremes), len(self.pages)))

            for x in self.page_extremes:
                if only is None or id(x) in only:
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

        if only is None or id(self.metadata) in only:
            _valid(self.metadata, seen, only, ())

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
        "chunks":        portally.checktype.CheckVector("Ntuple", "chunks", required=True, type=Chunk),
        "chunk_offsets": portally.checktype.CheckVector("Ntuple", "chunk_offsets", required=False, type=int, minlen=1),
        "descriptive":   portally.checktype.CheckVector("Ntuple", "descriptive", required=False, type=Descriptive),
        "correlation":   portally.checktype.CheckVector("Ntuple", "correlation", required=False, type=Correlation),
        "functions":     portally.checktype.CheckVector("Ntuple", "functions", required=False, type=FunctionObject),
        }

    chunks        = typedproperty(_params["chunks"])
    chunk_offsets = typedproperty(_params["chunk_offsets"])
    descriptive   = typedproperty(_params["descriptive"])
    correlation   = typedproperty(_params["correlation"])
    functions     = typedproperty(_params["functions"])

    def __init__(self, chunks, chunk_offsets=None, descriptive=None, correlation=None, functions=None):
        self.chunks = chunks
        self.chunk_offsets = chunk_offsets
        self.descriptive = descriptive
        self.correlation = correlation
        self.functions = functions

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

        if self.descriptive is not None:
            for x in self.descriptive:
                if only is None or id(x) in only:
                    _valid(x, seen, only, ())

        if self.correlation is not None:
            raise NotImplementedError

        if self.functions is not None:
            for x in self.functions:
                if only is None or id(x) in only:
                    _valid(x, seen, only, ())

    @property
    def numpy_arrays(self):
        self._valid(set(), None)
        for x in self.chunks:
            yield x.numpy_arrays

################################################# Ntuple

class Ntuple(Object):
    _params = {
        "identifier": portally.checktype.CheckKey("Ntuple", "identifier", required=True, type=str),
        "columns":    portally.checktype.CheckVector("Ntuple", "columns", required=True, type=Column, minlen=1),
        "instances":  portally.checktype.CheckVector("Ntuple", "instances", required=True, type=NtupleInstance, minlen=1),
        "title":      portally.checktype.CheckString("Ntuple", "title", required=False),
        "metadata":   portally.checktype.CheckClass("Ntuple", "metadata", required=False, type=Metadata),
        "decoration": portally.checktype.CheckClass("Ntuple", "decoration", required=False, type=Decoration),
        }

    identifier = typedproperty(_params["identifier"])
    columns    = typedproperty(_params["columns"])
    instances  = typedproperty(_params["instances"])
    title      = typedproperty(_params["title"])
    metadata   = typedproperty(_params["metadata"])
    decoration = typedproperty(_params["decoration"])

    def __init__(self, identifier, columns, instances, title="", metadata=None, decoration=None):
        self.identifier = identifier
        self.columns = columns
        self.instances = instances
        self.title = title
        self.metadata = metadata
        self.decoration = decoration

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

        if only is None or id(self.metadata) in only:
            _valid(self.metadata, seen, only, None)
        if only is None or id(self.decoration) in only:
            _valid(self.decoration, seen, only, None)

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

    def _valid(self, seen, only, shape):
        if len(set(x.identifier for x in self.objects)) != len(self.objects):
            raise ValueError("Collection.objects keys must be unique")

        for x in self.objects:
            if only is None or id(x) in only:
                _valid(x, seen, only, shape)

        if only is None or id(self.metadata) in only:
            _valid(self.metadata, seen, only, shape)
        if only is None or id(self.decoration) in only:
            _valid(self.decoration, seen, only, shape)
        return shape

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
