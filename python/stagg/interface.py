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
import math
import numbers
import operator
import struct
import sys

import numpy
import flatbuffers

import stagg.stagg_generated.MetadataLanguage
import stagg.stagg_generated.Metadata
import stagg.stagg_generated.DecorationLanguage
import stagg.stagg_generated.Decoration
import stagg.stagg_generated.DType
import stagg.stagg_generated.Endianness
import stagg.stagg_generated.DimensionOrder
import stagg.stagg_generated.Filter
import stagg.stagg_generated.Slice
import stagg.stagg_generated.ExternalSource
import stagg.stagg_generated.RawInlineBuffer
import stagg.stagg_generated.RawExternalBuffer
import stagg.stagg_generated.InterpretedInlineBuffer
import stagg.stagg_generated.InterpretedExternalBuffer
import stagg.stagg_generated.RawBuffer
import stagg.stagg_generated.InterpretedBuffer
import stagg.stagg_generated.StatisticFilter
import stagg.stagg_generated.Moments
import stagg.stagg_generated.Extremes
import stagg.stagg_generated.Quantiles
import stagg.stagg_generated.Modes
import stagg.stagg_generated.Statistics
import stagg.stagg_generated.Covariance
import stagg.stagg_generated.BinLocation
import stagg.stagg_generated.IntegerBinning
import stagg.stagg_generated.RealInterval
import stagg.stagg_generated.NonRealMapping
import stagg.stagg_generated.RealOverflow
import stagg.stagg_generated.RegularBinning
import stagg.stagg_generated.HexagonalCoordinates
import stagg.stagg_generated.HexagonalBinning
import stagg.stagg_generated.EdgesBinning
import stagg.stagg_generated.OverlappingFillStrategy
import stagg.stagg_generated.IrregularBinning
import stagg.stagg_generated.CategoryBinning
import stagg.stagg_generated.SparseRegularBinning
import stagg.stagg_generated.FractionLayout
import stagg.stagg_generated.FractionErrorMethod
import stagg.stagg_generated.FractionBinning
import stagg.stagg_generated.PredicateBinning
import stagg.stagg_generated.Assignment
import stagg.stagg_generated.Variation
import stagg.stagg_generated.VariationBinning
import stagg.stagg_generated.Binning
import stagg.stagg_generated.Axis
import stagg.stagg_generated.Profile
import stagg.stagg_generated.UnweightedCounts
import stagg.stagg_generated.WeightedCounts
import stagg.stagg_generated.Counts
import stagg.stagg_generated.Parameter
import stagg.stagg_generated.ParameterizedFunction
import stagg.stagg_generated.EvaluatedFunction
import stagg.stagg_generated.FunctionData
import stagg.stagg_generated.Function
import stagg.stagg_generated.BinnedEvaluatedFunction
import stagg.stagg_generated.FunctionObjectData
import stagg.stagg_generated.FunctionObject
import stagg.stagg_generated.Histogram
import stagg.stagg_generated.Page
import stagg.stagg_generated.ColumnChunk
import stagg.stagg_generated.Chunk
import stagg.stagg_generated.Column
import stagg.stagg_generated.NtupleInstance
import stagg.stagg_generated.Ntuple
import stagg.stagg_generated.ObjectData
import stagg.stagg_generated.Object
import stagg.stagg_generated.Collection

import stagg.checktype

def _name2fb(name):
    return "".join(x.capitalize() for x in name.split("_"))

def typedproperty(check):
    @property
    def prop(self):
        private = "_" + check.paramname
        if not hasattr(self, private):
            assert hasattr(self, "_flatbuffers"), "not derived from a flatbuffer or not properly initialized"
            fbname = _name2fb(check.paramname)
            fbnamelen = fbname + "Length"
            fbnamelookup = fbname + "Lookup"
            fbnametag = fbname + "ByTag"
            if hasattr(self._flatbuffers, fbnametag):
                value = getattr(self._flatbuffers, fbnametag)()
                stagg.checktype.setparent(self, value)
            elif hasattr(self._flatbuffers, fbnamelookup):
                value = stagg.checktype.FBLookup(getattr(self._flatbuffers, fbnamelen)(), getattr(self._flatbuffers, fbnamelookup), getattr(self._flatbuffers, fbname), check, self)
            elif hasattr(self._flatbuffers, fbnamelen):
                value = stagg.checktype.FBVector(getattr(self._flatbuffers, fbnamelen)(), getattr(self._flatbuffers, fbname), check, self)
            else:
                value = check.fromflatbuffers(getattr(self._flatbuffers, fbname)())
                stagg.checktype.setparent(self, value)
            setattr(self, private, value)
        return getattr(self, private)

    @prop.setter
    def prop(self, value):
        value = check(value)
        stagg.checktype.setparent(self, value)
        setattr(self, "_" + check.paramname, value)

    return prop

def _valid(obj, seen, recursive):
    if obj is None:
        pass
    elif isinstance(obj, Stagg):
        if id(obj) in seen:
            raise ValueError("hierarchy is recursively nested")
        seen.add(id(obj))
        obj._validtypes()
        obj._valid(seen, recursive)
    elif isinstance(obj, stagg.checktype.Vector):
        for x in obj:
            _valid(x, seen, recursive)
    elif isinstance(obj, stagg.checktype.Lookup):
        for x in obj.values():
            _valid(x, seen, recursive)
    else:
        raise AssertionError(type(obj))

def _getbykey(self, field, where):
    lookup = "_lookup_" + field
    if not hasattr(self, lookup):
        values = getattr(self, field)
        if isinstance(values, stagg.checktype.Vector):
            setattr(self, lookup, {x.identifier: x for x in values})
            if len(getattr(self, lookup)) != len(values):
                raise ValueError("{0}.{1} keys must be unique".format(type(self).__name__, field))
        elif isinstance(values, stagg.checktype.Lookup):
            setattr(self, lookup, values)
        else:
            raise AssertionError(type(values))
    return getattr(self, lookup)[where]

class _MockFlatbuffers(object):
    class _ByTag(object):
        __slots__ = ["getdata", "gettype", "lookup"]

        def __init__(self, getdata, gettype, lookup):
            self.getdata = getdata
            self.gettype = gettype
            self.lookup = lookup

        def __call__(self):
            data = self.getdata()
            try:
                interface, deserializer = self.lookup[self.gettype()]
            except KeyError:
                return None
            fb = deserializer()
            fb.Init(data.Bytes, data.Pos)
            return interface._fromflatbuffers(fb)

################################################# Stagg

class Stagg(object):
    def __repr__(self):
        if "identifier" in self._params:
            identifier = " " + repr(self.identifier)
        elif hasattr(self, "_identifier"):
            identifier = " " + repr(self._identifier)
        else:
            identifier = ""
        return "<{0}{1} at 0x{2:012x}>".format(type(self).__name__, identifier, id(self))
        
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

    def _validtypes(self):
        for n, x in self._params.items():
            x(getattr(self, n))

    def _valid(self, seen, recursive):
        pass

    def checkvalid(self, recursive=True):
        self._valid(set(), recursive)

    @property
    def isvalid(self):
        try:
            self.checkvalid()
        except ValueError:
            return False
        else:
            return True

    def unattached(self, checkvalid=False):
        out = self._unattached()
        if checkvalid:
            out.checkvalid()
        return out

    def _unattached(self):
        if not hasattr(self, "_parent"):
            return self
        else:
            out = type(self).__new__(type(self))
            if hasattr(self, "_flatbuffers"):
                out._flatbuffers = self._flatbuffers
            for n in self._params:
                private = "_" + n
                if hasattr(self, private):
                    x = getattr(self, private)
                    if isinstance(x, (Stagg, stagg.checktype.Vector, stagg.checktype.Lookup)):
                        x = x._unattached()
                    stagg.checktype.setparent(out, x)
                    setattr(out, private, x)
            return out

    def _add(self, other, shape, noclobber):
        raise NotImplementedError(type(self))

    @classmethod
    def _fromflatbuffers(cls, fb):
        out = cls.__new__(cls)
        out._flatbuffers = fb
        return out

    def _toflatbuffers(self, builder):
        raise NotImplementedError("missing _toflatbuffers implementation in {0}".format(type(self)))

    def __eq__(self, other):
        if type(self) is not type(other):
            return False
        for n in self._params:
            selfn = getattr(self, n)
            othern = getattr(other, n)
            if selfn is None or isinstance(selfn, (Stagg, Enum)):
                if selfn != othern:
                    return False
            else:
                try:
                    if len(selfn) != len(othern):
                        return False
                except TypeError:
                    return selfn == othern
                for x, y in zip(selfn, othern):
                    if x != y:
                        return False
        else:
            return True

    def __ne__(self, other):
        return not self.__eq__(other)
        
################################################# Enum

class Enum(object):
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def __repr__(self):
        return self.base + "." + self.name

    def __str__(self):
        return self.base + "." + self.name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self is other or (isinstance(other, type(self)) and self.value == other.value)

    def __ne__(self, other):
        return not self.__eq__(other)
    
################################################# Object

class Object(Stagg):
    def __init__(self):
        raise TypeError("{0} is an abstract base class; do not construct".format(type(self).__name__))

    def __add__(self, other):
        out = self._unattached()
        out._add(other, (), noclobber=True)
        return out

    def __iadd__(self, other):
        self._add(other, (), noclobber=False)
        return self

    def __radd__(self, other):
        return self.__add__(other)

    @classmethod
    def _fromflatbuffers(cls, fb):
        interface, deserializer = _ObjectData_lookup[fb.DataType()]
        data = fb.Data()
        fb2 = deserializer()
        fb2.Init(data.Bytes, data.Pos)
        return interface._fromflatbuffers(fb, fb2)

    def tobuffer(self):
        self.checkvalid()
        builder = flatbuffers.Builder(1024)
        builder.Finish(self._toflatbuffers(builder))
        return builder.Output()

    def toarray(self):
        return numpy.frombuffer(self.tobuffer(), dtype=numpy.uint8)

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
            file.write(b"StAg")
            builder = flatbuffers.Builder(1024)
            builder.Finish(self._toflatbuffers(builder))
            offset = file.tell()
            file.write(builder.Output())
            file.write(struct.pack("<Q", offset))
            file.write(b"StAg")

        finally:
            if opened:
                file.close()

def frombuffer(buffer, checkvalid=False, offset=0):
    out = Object._fromflatbuffers(stagg.stagg_generated.Object.Object.GetRootAsObject(buffer, offset))
    if checkvalid:
        out.checkvalid()
    return out

def fromarray(array, checkvalid=False):
    return frombuffer(array, checkvalid=checkvalid)

def fromfile(file, mode="r+", checkvalid=False):
    if isinstance(file, str):
        file = numpy.memmap(file, dtype=numpy.uint8, mode=mode)
    if file[:4].tostring() != b"StAg":
        raise OSError("file does not begin with magic 'StAg'")
    if file[-4:].tostring() != b"StAg":
        raise OSError("file does not end with magic 'StAg'")
    offset, = struct.unpack("<Q", file[-12:-4])
    return frombuffer(file[offset:-12], checkvalid=checkvalid)

################################################# Metadata

class MetadataLanguageEnum(Enum):
    base = "Metadata"

class Metadata(Stagg):
    unspecified = MetadataLanguageEnum("unspecified", stagg.stagg_generated.MetadataLanguage.MetadataLanguage.meta_unspecified)
    json = MetadataLanguageEnum("json", stagg.stagg_generated.MetadataLanguage.MetadataLanguage.meta_json)
    language = [unspecified, json]

    _params = {
        "data":     stagg.checktype.CheckString("Metadata", "data", required=True),
        "language": stagg.checktype.CheckEnum("Metadata", "language", required=True, choices=language),
        }

    data     = typedproperty(_params["data"])
    language = typedproperty(_params["language"])

    def __init__(self, data, language=unspecified):
        self.data = data
        self.language = language

    def _toflatbuffers(self, builder):
        data = builder.CreateString(self.data.encode("utf-8"))
        stagg.stagg_generated.Metadata.MetadataStart(builder)
        stagg.stagg_generated.Metadata.MetadataAddData(builder, data)
        if self.language != self.unspecified:
            stagg.stagg_generated.Metadata.MetadataAddLanguage(builder, self.language.value)
        return stagg.stagg_generated.Metadata.MetadataEnd(builder)

################################################# Decoration

class DecorationLanguageEnum(Enum):
    base = "Decoration"

class Decoration(Stagg):
    unspecified = DecorationLanguageEnum("unspecified", stagg.stagg_generated.DecorationLanguage.DecorationLanguage.deco_unspecified)
    css         = DecorationLanguageEnum("css", stagg.stagg_generated.DecorationLanguage.DecorationLanguage.deco_css)
    vega        = DecorationLanguageEnum("vega", stagg.stagg_generated.DecorationLanguage.DecorationLanguage.deco_vega)
    root_json   = DecorationLanguageEnum("root_json", stagg.stagg_generated.DecorationLanguage.DecorationLanguage.deco_root_json)
    language = [unspecified, css, vega, root_json]

    _params = {
        "data":     stagg.checktype.CheckString("Decoration", "data", required=True),
        "language": stagg.checktype.CheckEnum("Decoration", "language", required=True, choices=language),
        }

    data     = typedproperty(_params["data"])
    language = typedproperty(_params["language"])

    def __init__(self, data, language=unspecified):
        self.data = data
        self.language = language

    def _toflatbuffers(self, builder):
        data = builder.CreateString(self.data.encode("utf-8"))
        stagg.stagg_generated.Decoration.DecorationStart(builder)
        stagg.stagg_generated.Decoration.DecorationAddData(builder, data)
        if self.language != self.unspecified:
            stagg.stagg_generated.Decoration.DecorationAddLanguage(builder, self.language.value)
        return stagg.stagg_generated.Decoration.DecorationEnd(builder)

################################################# Buffers

class BufferFilterEnum(Enum):
    base = "Buffer"

class Buffer(Stagg):
    none = BufferFilterEnum("none", stagg.stagg_generated.Filter.Filter.filter_none)
    gzip = BufferFilterEnum("gzip", stagg.stagg_generated.Filter.Filter.filter_gzip)
    lzma = BufferFilterEnum("lzma", stagg.stagg_generated.Filter.Filter.filter_lzma)
    lz4  = BufferFilterEnum("lz4", stagg.stagg_generated.Filter.Filter.filter_lz4)
    filters = [none, gzip, lzma, lz4]

    def __init__(self):
        raise TypeError("{0} is an abstract base class; do not construct".format(type(self).__name__))

class InlineBuffer(object):
    def __init__(self):
        raise TypeError("{0} is an abstract base class; do not construct".format(type(self).__name__))

class ExternalSourceEnum(Enum):
    base = "ExternalBuffer"

class ExternalBuffer(object):
    memory   = ExternalSourceEnum("memory", stagg.stagg_generated.ExternalSource.ExternalSource.external_memory)
    samefile = ExternalSourceEnum("samefile", stagg.stagg_generated.ExternalSource.ExternalSource.external_samefile)
    file     = ExternalSourceEnum("file", stagg.stagg_generated.ExternalSource.ExternalSource.external_file)
    url      = ExternalSourceEnum("url", stagg.stagg_generated.ExternalSource.ExternalSource.external_url)
    types = [memory, samefile, file, url]

    def __init__(self):
        raise TypeError("{0} is an abstract base class; do not construct".format(type(self).__name__))

class RawBuffer(object):
    def __init__(self):
        raise TypeError("{0} is an abstract base class; do not construct".format(type(self).__name__))

class DTypeEnum(Enum):
    base = "Interpretation"

    def __init__(self, name, value, dtype):
        super(DTypeEnum, self).__init__(name, value)
        self.dtype = dtype

class EndiannessEnum(Enum):
    base = "Interpretation"

    def __init__(self, name, value, endianness):
        super(EndiannessEnum, self).__init__(name, value)
        self.endianness = endianness

class Interpretation(object):
    none    = DTypeEnum("none", stagg.stagg_generated.DType.DType.dtype_none, numpy.dtype(numpy.uint8))
    int8    = DTypeEnum("int8", stagg.stagg_generated.DType.DType.dtype_int8, numpy.dtype(numpy.int8))
    uint8   = DTypeEnum("uint8", stagg.stagg_generated.DType.DType.dtype_uint8, numpy.dtype(numpy.uint8))
    int16   = DTypeEnum("int16", stagg.stagg_generated.DType.DType.dtype_int16, numpy.dtype(numpy.int16))
    uint16  = DTypeEnum("uint16", stagg.stagg_generated.DType.DType.dtype_uint16, numpy.dtype(numpy.uint16))
    int32   = DTypeEnum("int32", stagg.stagg_generated.DType.DType.dtype_int32, numpy.dtype(numpy.int32))
    uint32  = DTypeEnum("uint32", stagg.stagg_generated.DType.DType.dtype_uint32, numpy.dtype(numpy.uint32))
    int64   = DTypeEnum("int64", stagg.stagg_generated.DType.DType.dtype_int64, numpy.dtype(numpy.int64))
    uint64  = DTypeEnum("uint64", stagg.stagg_generated.DType.DType.dtype_uint64, numpy.dtype(numpy.uint64))
    float32 = DTypeEnum("float32", stagg.stagg_generated.DType.DType.dtype_float32, numpy.dtype(numpy.float32))
    float64 = DTypeEnum("float64", stagg.stagg_generated.DType.DType.dtype_float64, numpy.dtype(numpy.float64))
    dtypes = [none, int8, uint8, int16, uint16, int32, uint32, int64, uint64, float32, float64]

    little_endian = EndiannessEnum("little_endian", stagg.stagg_generated.Endianness.Endianness.little_endian, "<")
    big_endian    = EndiannessEnum("big_endian", stagg.stagg_generated.Endianness.Endianness.big_endian, ">")
    endiannesses = [little_endian, big_endian]

    def __init__(self):
        raise TypeError("{0} is an abstract base class; do not construct".format(type(self).__name__))

    @property
    def numpy_dtype(self):
        return self.dtype.dtype.newbyteorder(self.endianness.endianness)

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
    base = "InterpretedBuffer"

    def __init__(self, name, value, dimension_order):
        super(DimensionOrderEnum, self).__init__(name, value)
        self.dimension_order = dimension_order

class InterpretedBuffer(Interpretation):
    c_order       = DimensionOrderEnum("c_order", stagg.stagg_generated.DimensionOrder.DimensionOrder.c_order, "C")
    fortran_order = DimensionOrderEnum("fortran", stagg.stagg_generated.DimensionOrder.DimensionOrder.fortran_order, "F")
    orders = [c_order, fortran_order]

    def __init__(self):
        raise TypeError("{0} is an abstract base class; do not construct".format(type(self).__name__))

################################################# RawInlineBuffer

class RawInlineBuffer(Buffer, RawBuffer, InlineBuffer):
    _params = {
        "buffer": stagg.checktype.CheckBuffer("RawInlineBuffer", "buffer", required=True),
        }

    buffer = typedproperty(_params["buffer"])

    def __init__(self, buffer):
        self.buffer = buffer

    @property
    def numbytes(self):
        return len(self.buffer)

    @property
    def array(self):
        return numpy.frombuffer(self.buffer, dtype=InterpretedBuffer.none.dtype)

    @classmethod
    def _fromflatbuffers(cls, fb):
        out = cls.__new__(cls)
        out._flatbuffers = _MockFlatbuffers()
        out._flatbuffers.Buffer = fb.BufferAsNumpy
        return out

    def _toflatbuffers(self, builder):
        stagg.stagg_generated.RawInlineBuffer.RawInlineBufferStartBufferVector(builder, len(self.buffer))
        builder.head = builder.head - len(self.buffer)
        builder.Bytes[builder.head : builder.head + len(self.buffer)] = self.buffer.tostring()
        buffer = builder.EndVector(len(self.buffer))

        stagg.stagg_generated.RawInlineBuffer.RawInlineBufferStart(builder)
        stagg.stagg_generated.RawInlineBuffer.RawInlineBufferAddBuffer(builder, buffer)
        return stagg.stagg_generated.RawInlineBuffer.RawInlineBufferEnd(builder)

################################################# RawExternalBuffer

class RawExternalBuffer(Buffer, RawBuffer, ExternalBuffer):
    _params = {
        "pointer":         stagg.checktype.CheckInteger("RawExternalBuffer", "pointer", required=True, min=0),
        "numbytes":        stagg.checktype.CheckInteger("RawExternalBuffer", "numbytes", required=True, min=0),
        "external_source": stagg.checktype.CheckEnum("RawExternalBuffer", "external_source", required=False, choices=ExternalBuffer.types),
        }

    pointer       = typedproperty(_params["pointer"])
    numbytes      = typedproperty(_params["numbytes"])
    external_source = typedproperty(_params["external_source"])

    def __init__(self, pointer, numbytes, external_source=ExternalBuffer.memory):
        self.pointer = pointer
        self.numbytes = numbytes
        self.external_source = external_source

    @property
    def array(self):
        return numpy.ctypeslib.as_array(ctypes.cast(self.pointer, ctypes.POINTER(ctypes.c_uint8)), shape=(self.numbytes,))

    def _toflatbuffers(self, builder):
        stagg.stagg_generated.RawExternalBuffer.RawExternalBufferStart(builder)
        stagg.stagg_generated.RawExternalBuffer.RawExternalBufferAddPointer(builder, self.pointer)
        stagg.stagg_generated.RawExternalBuffer.RawExternalBufferAddNumbytes(builder, self.numbytes)
        stagg.stagg_generated.RawExternalBuffer.RawExternalBufferAddExternalSource(builder, self.external_source.value)
        return stagg.stagg_generated.RawExternalBuffer.RawExternalBufferEnd(builder)

################################################# InlineBuffer

class InterpretedInlineBuffer(Buffer, InterpretedBuffer, InlineBuffer):
    _params = {
        "buffer":           stagg.checktype.CheckBuffer("InterpretedInlineBuffer", "buffer", required=True),
        "filters":          stagg.checktype.CheckVector("InterpretedInlineBuffer", "filters", required=False, type=Buffer.filters),
        "postfilter_slice": stagg.checktype.CheckSlice("InterpretedInlineBuffer", "postfilter_slice", required=False),
        "dtype":            stagg.checktype.CheckEnum("InterpretedInlineBuffer", "dtype", required=False, choices=InterpretedBuffer.dtypes),
        "endianness":       stagg.checktype.CheckEnum("InterpretedInlineBuffer", "endianness", required=False, choices=InterpretedBuffer.endiannesses),
        "dimension_order":  stagg.checktype.CheckEnum("InterpretedInlineBuffer", "dimension_order", required=False, choices=InterpretedBuffer.orders),
        }

    buffer           = typedproperty(_params["buffer"])
    filters          = typedproperty(_params["filters"])
    postfilter_slice = typedproperty(_params["postfilter_slice"])
    dtype            = typedproperty(_params["dtype"])
    endianness       = typedproperty(_params["endianness"])
    dimension_order  = typedproperty(_params["dimension_order"])

    def __init__(self, buffer, filters=None, postfilter_slice=None, dtype=InterpretedBuffer.none, endianness=InterpretedBuffer.little_endian, dimension_order=InterpretedBuffer.c_order):
        self.buffer = buffer
        self.filters = filters
        self.postfilter_slice = postfilter_slice
        self.dtype = dtype
        self.endianness = endianness
        self.dimension_order = dimension_order

    def _valid(self, seen, recursive):
        if self.postfilter_slice is not None and self.postfilter_slice.has_step and self.postfilter_slice.step == 0:
            raise ValueError("slice step cannot be zero")
        self.array

    @classmethod
    def fromarray(cls, array):
        dtype, endianness = Interpretation.from_numpy_dtype(array.dtype)
        order = InterpretedBuffer.fortran_order if numpy.isfortran(array) else InterpretedBuffer.c_order
        return cls(array, dtype=dtype, endianness=endianness, dimension_order=order)

    @property
    def array(self):
        shape = self._shape((), ())

        if len(self.filters) == 0:
            array = self.buffer
        else:
            raise NotImplementedError(self.filters)

        if array.dtype.itemsize != 1:
            array = array.view(InterpretedBuffer.none.dtype)
        if len(array.shape) != 1:
            array = array.reshape(-1)

        if self.postfilter_slice is not None:
            start = self.postfilter_slice.start if self.postfilter_slice.has_start else None
            stop = self.postfilter_slice.stop if self.postfilter_slice.has_stop else None
            step = self.postfilter_slice.step if self.postfilter_slice.has_step else None
            array = array[start:stop:step]
        
        try:
            array = array.view(self.numpy_dtype)
        except ValueError:
            raise ValueError("InterpretedInlineBuffer.buffer raw length is {0} bytes but this does not fit an itemsize of {1} bytes".format(len(array), self.numpy_dtype.itemsize))

        if len(array) != functools.reduce(operator.mul, shape, 1):
            raise ValueError("InterpretedInlineBuffer.buffer length as {0} is {1} but multiplicity at this position in the hierarchy is {2}".format(self.numpy_dtype, len(array), functools.reduce(operator.mul, shape, 1)))

        return array.reshape(shape, order=self.dimension_order.dimension_order)

    @classmethod
    def _fromflatbuffers(cls, fb):
        out = cls.__new__(cls)
        out._flatbuffers = _MockFlatbuffers()
        out._flatbuffers.Buffer = fb.BufferAsNumpy
        out._flatbuffers.Filters = fb.Filters
        out._flatbuffers.FiltersLength = fb.FiltersLength
        out._flatbuffers.PostfilterSlice = fb.PostfilterSlice
        out._flatbuffers.Dtype = fb.Dtype
        out._flatbuffers.Endianness = fb.Endianness
        out._flatbuffers.DimensionOrder = fb.DimensionOrder
        return out

    def _toflatbuffers(self, builder):
        stagg.stagg_generated.InterpretedInlineBuffer.InterpretedInlineBufferStartBufferVector(builder, len(self.buffer))
        builder.head = builder.head - len(self.buffer)
        builder.Bytes[builder.head : builder.head + len(self.buffer)] = self.buffer.tostring()
        buffer = builder.EndVector(len(self.buffer))

        if len(self.filters) == 0:
            filters = None
        else:
            stagg.stagg_generated.InterpretedInlineBuffer.InterpretedInlineBufferStartFiltersVector(builder, len(self.filters))
            for x in self.filters[::-1]:
                builder.PrependUInt32(x.value)
            filters = builder.EndVector(len(self.filters))

        stagg.stagg_generated.InterpretedInlineBuffer.InterpretedInlineBufferStart(builder)
        stagg.stagg_generated.InterpretedInlineBuffer.InterpretedInlineBufferAddBuffer(builder, buffer)
        if filters is not None:
            stagg.stagg_generated.InterpretedInlineBuffer.InterpretedInlineBufferAddFilters(builder, filters)
        if self.postfilter_slice is not None:
            stagg.stagg_generated.InterpretedInlineBuffer.InterpretedInlineBufferAddPostfilterSlice(builder, stagg.stagg_generated.Slice.CreateSlice(builder, self.postfilter_slice.start, self.postfilter_slice.stop, self.postfilter_slice.step, self.postfilter_slice.hasStart, self.postfilter_slice.hasStop, self.postfilter_slice.hasStep))
        if self.dtype != self.none:
            stagg.stagg_generated.InterpretedInlineBuffer.InterpretedInlineBufferAddDtype(builder, self.dtype.value)
        if self.endianness != self.little_endian:
            stagg.stagg_generated.InterpretedInlineBuffer.InterpretedInlineBufferAddEndianness(builder, self.endianness.value)
        if self.dimension_order != self.c_order:
            stagg.stagg_generated.InterpretedInlineBuffer.InterpretedInlineBufferAddDimensionOrder(builder, self.dimension_order.value)
        return stagg.stagg_generated.InterpretedInlineBuffer.InterpretedInlineBufferEnd(builder)

################################################# ExternalBuffer

class InterpretedExternalBuffer(Buffer, InterpretedBuffer, ExternalBuffer):
    _params = {
        "pointer":          stagg.checktype.CheckInteger("InterpretedExternalBuffer", "pointer", required=True, min=0),
        "numbytes":         stagg.checktype.CheckInteger("InterpretedExternalBuffer", "numbytes", required=True, min=0),
        "external_source":  stagg.checktype.CheckEnum("InterpretedExternalBuffer", "external_source", required=False, choices=ExternalBuffer.types),
        "filters":          stagg.checktype.CheckVector("InterpretedExternalBuffer", "filters", required=False, type=Buffer.filters),
        "postfilter_slice": stagg.checktype.CheckSlice("InterpretedExternalBuffer", "postfilter_slice", required=False),
        "dtype":            stagg.checktype.CheckEnum("InterpretedExternalBuffer", "dtype", required=False, choices=InterpretedBuffer.dtypes),
        "endianness":       stagg.checktype.CheckEnum("InterpretedExternalBuffer", "endianness", required=False, choices=InterpretedBuffer.endiannesses),
        "dimension_order":  stagg.checktype.CheckEnum("InterpretedExternalBuffer", "dimension_order", required=False, choices=InterpretedBuffer.orders),
        "location":         stagg.checktype.CheckString("InterpretedExternalBuffer", "location", required=False),
        }

    pointer          = typedproperty(_params["pointer"])
    numbytes         = typedproperty(_params["numbytes"])
    external_source  = typedproperty(_params["external_source"])
    filters          = typedproperty(_params["filters"])
    postfilter_slice = typedproperty(_params["postfilter_slice"])
    dtype            = typedproperty(_params["dtype"])
    endianness       = typedproperty(_params["endianness"])
    dimension_order  = typedproperty(_params["dimension_order"])
    location         = typedproperty(_params["location"])

    def __init__(self, pointer, numbytes, external_source=ExternalBuffer.memory, filters=None, postfilter_slice=None, dtype=InterpretedBuffer.none, endianness=InterpretedBuffer.little_endian, dimension_order=InterpretedBuffer.c_order, location=None):
        self.pointer = pointer
        self.numbytes = numbytes
        self.external_source = external_source
        self.filters = filters
        self.postfilter_slice = postfilter_slice
        self.dtype = dtype
        self.endianness = endianness
        self.dimension_order = dimension_order
        self.location = location

    def _valid(self, seen, recursive):
        if self.postfilter_slice is not None and self.postfilter_slice.has_step and self.postfilter_slice.step == 0:
            raise ValueError("slice step cannot be zero")
        self.array

    @property
    def array(self):
        shape = self._shape((), ())

        self._buffer = numpy.ctypeslib.as_array(ctypes.cast(self.pointer, ctypes.POINTER(ctypes.c_uint8)), shape=(self.numbytes,))

        if len(self.filters) == 0:
            array = self._buffer
        else:
            raise NotImplementedError(self.filters)

        if self.postfilter_slice is not None:
            start = self.postfilter_slice.start if self.postfilter_slice.has_start else None
            stop = self.postfilter_slice.stop if self.postfilter_slice.has_stop else None
            step = self.postfilter_slice.step if self.postfilter_slice.has_step else None
            array = array[start:stop:step]

        try:
            array = array.view(self.numpy_dtype)
        except ValueError:
            raise ValueError("InterpretedExternalBuffer.buffer raw length is {0} bytes but this does not fit an itemsize of {1} bytes".format(len(array), self.numpy_dtype.itemsize))

        if len(array) != functools.reduce(operator.mul, shape, 1):
            raise ValueError("InterpretedExternalBuffer.buffer length is {0} but multiplicity at this position in the hierarchy is {1}".format(len(array), functools.reduce(operator.mul, shape, 1)))

        return array.reshape(shape, order=self.dimension_order.dimension_order)

    def _toflatbuffers(self, builder):
        location = None if self.location is None else builder.CreateString(location.encode("utf-8"))

        if len(self.filters) == 0:
            filters = None
        else:
            stagg.stagg_generated.InterpretedExternalBuffer.InterpretedExternalBufferStartFiltersVector(builder, len(self.filters))
            for x in self.filters[::-1]:
                builder.PrependUInt32(x.value)
            filters = builder.EndVector(len(self.filters))

        stagg.stagg_generated.InterpretedExternalBuffer.InterpretedExternalBufferStart(builder)
        stagg.stagg_generated.InterpretedExternalBuffer.InterpretedExternalBufferAddPointer(builder, self.pointer)
        stagg.stagg_generated.InterpretedExternalBuffer.InterpretedExternalBufferAddNumbytes(builder, self.numbytes)
        if self.external_source != ExternalBuffer.memory:
            stagg.stagg_generated.InterpretedExternalBuffer.InterpretedExternalBufferAddExternalSource(builder, self.external_source.values)
        if filters is not None:
            stagg.stagg_generated.InterpretedExternalBuffer.InterpretedExternalBufferAddFilters(builder, filters)
        if self.postfilter_slice is not None:
            stagg.stagg_generated.InterpretedExternalBuffer.InterpretedExternalBufferAddPostfilterSlice(builder, stagg.stagg_generated.Slice.CreateSlice(builder, self.postfilter_slice.start, self.postfilter_slice.stop, self.postfilter_slice.step, self.postfilter_slice.hasStart, self.postfilter_slice.hasStop, self.postfilter_slice.hasStep))
        if self.dtype != self.none:
            stagg.stagg_generated.InterpretedExternalBuffer.InterpretedExternalBufferAddDtype(builder, self.dtype.value)
        if self.endianness != self.little_endian:
            stagg.stagg_generated.InterpretedExternalBuffer.InterpretedExternalBufferAddEndianness(builder, self.endianness.value)
        if self.dimension_order != self.c_order:
            stagg.stagg_generated.InterpretedExternalBuffer.InterpretedExternalBufferAddDimensionOrder(builder, self.dimension_order.value)
        if location is not None:
            stagg.stagg_generated.InterpretedExternalBuffer.InterpretedExternalBufferAddLocation(builder, location)
        return stagg.stagg_generated.InterpretedExternalBuffer.InterpretedExternalBufferEnd(builder)

################################################# StatisticFilter

class StatisticFilter(Stagg):
    _params = {
        "min": stagg.checktype.CheckNumber("StatisticFilter", "min", required=False),
        "max": stagg.checktype.CheckNumber("StatisticFilter", "max", required=False),
        "excludes_minf": stagg.checktype.CheckBool("StatisticFilter", "excludes_minf", required=False),
        "excludes_pinf": stagg.checktype.CheckBool("StatisticFilter", "excludes_pinf", required=False),
        "excludes_nan":  stagg.checktype.CheckBool("StatisticFilter", "excludes_nan", required=False),
        }

    min       = typedproperty(_params["min"])
    max       = typedproperty(_params["max"])
    excludes_minf = typedproperty(_params["excludes_minf"])
    excludes_pinf = typedproperty(_params["excludes_pinf"])
    excludes_nan  = typedproperty(_params["excludes_nan"])

    def __init__(self, min=float("-inf"), max=float("inf"), excludes_minf=False, excludes_pinf=False, excludes_nan=False):
        self.min = min
        self.max = max
        self.excludes_minf = excludes_minf
        self.excludes_pinf = excludes_pinf
        self.excludes_nan = excludes_nan

    def _valid(self, seen, recursive):
        if self.min is not None and self.max is not None and self.min >= self.max:
            raise ValueError("StatisticFilter.min ({0}) must be strictly less than StatisticFilter.max ({1})".format(self.min, self.max))

    def _toflatbuffers(self, builder):
        stagg.stagg_generated.StatisticFilter.StatisticFilterStart(builder)
        if self.min != float("-inf"):
            stagg.stagg_generated.StatisticFilter.StatisticFilterAddMin(builder, self.min)
        if self.max != float("inf"):
            stagg.stagg_generated.StatisticFilter.StatisticFilterAddMax(builder, self.max)
        if self.excludes_minf is not False:
            stagg.stagg_generated.StatisticFilter.StatisticFilterAddExcludesMinf(builder, self.excludes_minf)
        if self.excludes_pinf is not False:
            stagg.stagg_generated.StatisticFilter.StatisticFilterAddExcludesPinf(builder, self.excludes_pinf)
        if self.excludes_nan is not False:
            stagg.stagg_generated.StatisticFilter.StatisticFilterAddExcludesNan(builder, self.excludes_nan)
        return stagg.stagg_generated.StatisticFilter.StatisticFilterEnd(builder)

################################################# Moments

class Moments(Stagg):
    _params = {
        "sumwxn":   stagg.checktype.CheckClass("Moments", "sumwxn", required=True, type=InterpretedBuffer),
        "n":        stagg.checktype.CheckInteger("Moments", "n", required=True, min=0),
        "weighted": stagg.checktype.CheckBool("Moments", "weighted", required=False),
        "filter": stagg.checktype.CheckClass("Moments", "filter", required=False, type=StatisticFilter),
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

    def _valid(self, seen, recursive):
        if recursive:
            _valid(self.sumwxn, seen, recursive)
            _valid(self.filter, seen, recursive)

    @classmethod
    def _fromflatbuffers(cls, fb):
        out = cls.__new__(cls)
        out._flatbuffers = _MockFlatbuffers()
        out._flatbuffers.SumwxnByTag = _MockFlatbuffers._ByTag(fb.Sumwxn, fb.SumwxnType, _InterpretedBuffer_lookup)
        out._flatbuffers.N = fb.N
        out._flatbuffers.Weighted = fb.Weighted
        out._flatbuffers.Filter = fb.Filter
        return out

    def _toflatbuffers(self, builder):
        sumwxn = self.sumwxn._toflatbuffers(builder)
        filter = None if self.filter is None else self.filter._toflatbuffers(builder)

        stagg.stagg_generated.Moments.MomentsStart(builder)
        stagg.stagg_generated.Moments.MomentsAddSumwxnType(builder, _InterpretedBuffer_invlookup[type(self.sumwxn)])
        stagg.stagg_generated.Moments.MomentsAddSumwxn(builder, sumwxn)
        stagg.stagg_generated.Moments.MomentsAddN(builder, self.n)
        if self.weighted is not True:
            stagg.stagg_generated.Moments.MomentsAddWeighted(builder, self.weighted)
        if filter is not None:
            stagg.stagg_generated.Moments.MomentsAddFilter(builder, filter)
        return stagg.stagg_generated.Moments.MomentsEnd(builder)

################################################# Extremes

class Extremes(Stagg):
    _params = {
        "values": stagg.checktype.CheckClass("Extremes", "values", required=True, type=InterpretedBuffer),
        "filter": stagg.checktype.CheckClass("Extremes", "filter", required=False, type=StatisticFilter),
        }

    values = typedproperty(_params["values"])
    filter = typedproperty(_params["filter"])

    def __init__(self, values, filter=None):
        self.values = values
        self.filter = filter

    def _valid(self, seen, recursive):
        if recursive:
            _valid(self.values, seen, recursive)
            _valid(self.filter, seen, recursive)

    @classmethod
    def _fromflatbuffers(cls, fb):
        out = cls.__new__(cls)
        out._flatbuffers = _MockFlatbuffers()
        out._flatbuffers.ValuesByTag = _MockFlatbuffers._ByTag(fb.Values, fb.ValuesType, _InterpretedBuffer_lookup)
        out._flatbuffers.Filter = fb.Filter
        return out

    def _toflatbuffers(self, builder):
        values = self.values._toflatbuffers(builder)
        filter = None if self.filter is None else self.filter._toflatbuffers(builder)

        stagg.stagg_generated.Extremes.ExtremesStart(builder)
        stagg.stagg_generated.Extremes.ExtremesAddValuesType(builder, _InterpretedBuffer_invlookup[type(self.values)])
        stagg.stagg_generated.Extremes.ExtremesAddValues(builder, values)
        if filter is not None:
            stagg.stagg_generated.Extremes.ExtremesAddFilter(builder, filter)
        return stagg.stagg_generated.Extremes.ExtremesEnd(builder)

################################################# Quantiles

class Quantiles(Stagg):
    _params = {
        "values": stagg.checktype.CheckClass("Quantiles", "values", required=True, type=InterpretedBuffer),
        "p":      stagg.checktype.CheckNumber("Quantiles", "p", required=True, min=0.0, max=1.0),
        "weighted": stagg.checktype.CheckBool("Quantiles", "weighted", required=False),
        "filter": stagg.checktype.CheckClass("Quantiles", "filter", required=False, type=StatisticFilter),
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

    def _valid(self, seen, recursive):
        if recursive:
            _valid(self.values, seen, recursive)
            _valid(self.filter, seen, recursive)

    @classmethod
    def _fromflatbuffers(cls, fb):
        out = cls.__new__(cls)
        out._flatbuffers = _MockFlatbuffers()
        out._flatbuffers.ValuesByTag = _MockFlatbuffers._ByTag(fb.Values, fb.ValuesType, _InterpretedBuffer_lookup)
        out._flatbuffers.P = fb.P
        out._flatbuffers.Weighted = fb.Weighted
        out._flatbuffers.Filter = fb.Filter
        return out

    def _toflatbuffers(self, builder):
        values = self.values._toflatbuffers(builder)
        filter = None if self.filter is None else self.filter._toflatbuffers(builder)

        stagg.stagg_generated.Quantiles.QuantilesStart(builder)
        stagg.stagg_generated.Quantiles.QuantilesAddValuesType(builder, _InterpretedBuffer_invlookup[type(self.values)])
        stagg.stagg_generated.Quantiles.QuantilesAddValues(builder, values)
        stagg.stagg_generated.Quantiles.QuantilesAddP(builder, self.p)
        if self.weighted is not True:
            stagg.stagg_generated.Quantiles.QuantilesAddWeighted(builder, self.weighted)
        if filter is not None:
            stagg.stagg_generated.Quantiles.QuantilesAddFilter(builder, filter)
        return stagg.stagg_generated.Quantiles.QuantilesEnd(builder)

################################################# Modes

class Modes(Stagg):
    _params = {
        "values": stagg.checktype.CheckClass("Modes", "values", required=True, type=InterpretedBuffer),
        "filter": stagg.checktype.CheckClass("Modes", "filter", required=False, type=StatisticFilter),
        }

    values   = typedproperty(_params["values"])
    filter   = typedproperty(_params["filter"])

    def __init__(self, values, filter=None):
        self.values = values
        self.filter = filter

    def _valid(self, seen, recursive):
        if recursive:
            _valid(self.values, seen, recursive)
            _valid(self.filter, seen, recursive)

    @classmethod
    def _fromflatbuffers(cls, fb):
        out = cls.__new__(cls)
        out._flatbuffers = _MockFlatbuffers()
        out._flatbuffers.ValuesByTag = _MockFlatbuffers._ByTag(fb.Values, fb.ValuesType, _InterpretedBuffer_lookup)
        out._flatbuffers.Filter = fb.Filter
        return out

    def _toflatbuffers(self, builder):
        values = self.values._toflatbuffers(builder)
        filter = None if self.filter is None else self.filter._toflatbuffers(builder)

        stagg.stagg_generated.Modes.ModesStart(builder)
        stagg.stagg_generated.Modes.ModesAddValuesType(builder, _InterpretedBuffer_invlookup[type(self.values)])
        stagg.stagg_generated.Modes.ModesAddValues(builder, values)
        if filter is not None:
            stagg.stagg_generated.Modes.ModesAddFilter(builder, filter)
        return stagg.stagg_generated.Modes.ModesEnd(builder)

################################################# Statistics

class Statistics(Stagg):
    _params = {
        "moments":   stagg.checktype.CheckVector("Statistics", "moments", required=False, type=Moments),
        "quantiles": stagg.checktype.CheckVector("Statistics", "quantiles", required=False, type=Quantiles),
        "mode":     stagg.checktype.CheckClass("Statistics", "mode", required=False, type=Modes),
        "min":    stagg.checktype.CheckClass("Statistics", "min", required=False, type=Extremes),
        "max":    stagg.checktype.CheckClass("Statistics", "max", required=False, type=Extremes),
        }

    moments   = typedproperty(_params["moments"])
    quantiles = typedproperty(_params["quantiles"])
    mode     = typedproperty(_params["mode"])
    min    = typedproperty(_params["min"])
    max    = typedproperty(_params["max"])

    def __init__(self, moments=None, quantiles=None, mode=None, min=None, max=None):
        self.moments = moments
        self.quantiles = quantiles
        self.mode = mode
        self.min = min
        self.max = max

    def _valid(self, seen, recursive):
        if len(set((x.n, x.weighted) for x in self.moments)) != len(self.moments):
            raise ValueError("Statistics.moments must have unique (n, weighted)")
        if len(set((x.p, x.weighted) for x in self.quantiles)) != len(self.quantiles):
            raise ValueError("Statistics.quantiles must have unique (p, weighted)")
        if recursive:
            _valid(self.moments, seen, recursive)
            _valid(self.quantiles, seen, recursive)
            _valid(self.mode, seen, recursive)
            _valid(self.min, seen, recursive)
            _valid(self.max, seen, recursive)

    def _toflatbuffers(self, builder):
        moments = None if len(self.moments) == 0 else [x._toflatbuffers(builder) for x in self.moments]
        quantiles = None if len(self.quantiles) == 0 else [x._toflatbuffers(builder) for x in self.quantiles]
        mode = None if self.mode is None else self.mode._toflatbuffers(builder)
        min = None if self.min is None else self.min._toflatbuffers(builder)
        max = None if self.max is None else self.max._toflatbuffers(builder)

        if moments is not None:
            stagg.stagg_generated.Statistics.StatisticsStartMomentsVector(builder, len(moments))
            for x in moments[::-1]:
                builder.PrependUOffsetTRelative(x)
            moments = builder.EndVector(len(moments))

        if quantiles is not None:
            stagg.stagg_generated.Statistics.StatisticsStartQuantilesVector(builder, len(quantiles))
            for x in quantiles[::-1]:
                builder.PrependUOffsetTRelative(x)
            quantiles = builder.EndVector(len(quantiles))

        stagg.stagg_generated.Statistics.StatisticsStart(builder)
        if moments is not None:
            stagg.stagg_generated.Statistics.StatisticsAddMoments(builder, moments)
        if quantiles is not None:
            stagg.stagg_generated.Statistics.StatisticsAddQuantiles(builder, quantiles)
        if mode is not None:
            stagg.stagg_generated.Statistics.StatisticsAddMode(builder, mode)
        if min is not None:
            stagg.stagg_generated.Statistics.StatisticsAddMin(builder, min)
        if max is not None:
            stagg.stagg_generated.Statistics.StatisticsAddMax(builder, max)
        return stagg.stagg_generated.Statistics.StatisticsEnd(builder)

################################################# Covariance

class Covariance(Stagg):
    _params = {
        "xindex": stagg.checktype.CheckInteger("Covariance", "xindex", required=True, min=0),
        "yindex": stagg.checktype.CheckInteger("Covariance", "yindex", required=True, min=0),
        "sumwxy": stagg.checktype.CheckClass("Covariance", "sumwxy", required=True, type=InterpretedBuffer),
        "weighted": stagg.checktype.CheckBool("Covariance", "weighted", required=False),
        "filter": stagg.checktype.CheckClass("Covariance", "filter", required=False, type=StatisticFilter),
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

    def _valid(self, seen, recursive):
        if recursive:
            _valid(self.sumwxy, seen, recursive)
            _valid(self.filter, seen, recursive)

    @staticmethod
    def _validindexes(covariances, numvars):
        pairs = [(x.xindex, x.yindex, x.weighted) for x in covariances]
        if len(set(pairs)) != len(pairs):
            raise ValueError("Covariance.xindex, yindex pairs must be unique")
        if any(x.xindex >= numvars for x in covariances):
            raise ValueError("Covariance.xindex must all be less than the number of axis or column variables {}".format(numvars))
        if any(x.yindex >= numvars for x in covariances):
            raise ValueError("Covariance.yindex must all be less than the number of axis or column variables {}".format(numvars))

    @classmethod
    def _fromflatbuffers(cls, fb):
        out = cls.__new__(cls)
        out._flatbuffers = _MockFlatbuffers()
        out._flatbuffers.Xindex = fb.Xindex
        out._flatbuffers.Yindex = fb.Yindex
        out._flatbuffers.SumwxyByTag = _MockFlatbuffers._ByTag(fb.Sumwxy, fb.SumwxyType, _InterpretedBuffer_lookup)
        out._flatbuffers.Weighted = fb.Weighted
        out._flatbuffers.Filter = fb.Filter
        return out

    def _toflatbuffers(self, builder):
        sumwxy = self.sumwxy._toflatbuffers(builder)
        filter = None if self.filter is None else self.filter._toflatbuffers(builder)

        stagg.stagg_generated.Covariance.CovarianceStart(builder)
        stagg.stagg_generated.Covariance.CovarianceAddXindex(builder, self.xindex)
        stagg.stagg_generated.Covariance.CovarianceAddYindex(builder, self.yindex)
        stagg.stagg_generated.Covariance.CovarianceAddSumwxyType(builder, _InterpretedBuffer_invlookup[type(self.sumwxy)])
        stagg.stagg_generated.Covariance.CovarianceAddSumwxy(builder, sumwxy)
        if self.weighted is not True:
            stagg.stagg_generated.Covariance.CovarianceAddWeighted(builder, self.weighted)
        if filter is not None:
            stagg.stagg_generated.Covariance.CovarianceAddFilter(builder, filter)
        return stagg.stagg_generated.Covariance.CovarianceEnd(builder)

################################################# Binning

class Binning(Stagg):
    def __init__(self):
        raise TypeError("{0} is an abstract base class; do not construct".format(type(self).__name__))

    @property
    def isnumerical(self):
        return True

    @property
    def dimensions(self):
        return len(self._binshape())

    @staticmethod
    def _promote(one, two):
        if type(one) is type(two):
            return one, two
        elif hasattr(one, "to" + type(two).__name__):
            return getattr(one, "to" + type(two).__name__)(), two
        elif hasattr(two, "to" + type(one).__name__):
            return one, getattr(two, "to" + type(one).__name__)()
        elif hasattr(one, "toIrregularBinning") and hasattr(two, "toIrregularBinning"):
            return one.toIrregularBinning(), two.toIrregularBinning()
        elif hasattr(one, "toCategoryBinning") and hasattr(two, "toCategoryBinning"):
            return one.toCategoryBinning(), two.toCategoryBinning()
        else:
            raise ValueError("{0} and {1} can't be promoted to the same type of Binning".format(one, two))

################################################# BinLocation

class BinLocationEnum(Enum):
    base = "BinLocation"

class BinLocation(object):
    below3      = BinLocationEnum("below3", stagg.stagg_generated.BinLocation.BinLocation.loc_below3)
    below2      = BinLocationEnum("below2", stagg.stagg_generated.BinLocation.BinLocation.loc_below2)
    below1      = BinLocationEnum("below1", stagg.stagg_generated.BinLocation.BinLocation.loc_below1)
    nonexistent = BinLocationEnum("nonexistent", stagg.stagg_generated.BinLocation.BinLocation.loc_nonexistent)
    above1      = BinLocationEnum("above1", stagg.stagg_generated.BinLocation.BinLocation.loc_above1)
    above2      = BinLocationEnum("above2", stagg.stagg_generated.BinLocation.BinLocation.loc_above2)
    above3      = BinLocationEnum("above3", stagg.stagg_generated.BinLocation.BinLocation.loc_above3)
    locations = [below3, below2, below1, nonexistent, above1, above2, above3]

    def __init__(self):
        raise TypeError("{0} is an abstract base class; do not construct".format(type(self).__name__))

    @classmethod
    def _belows(cls, tuples):
        out = [x for x in tuples if x[0].value < cls.nonexistent.value]
        out.sort(key=lambda x: x[0].value)
        return out

    @classmethod
    def _aboves(cls, tuples):
        out = [x for x in tuples if x[0].value > cls.nonexistent.value]
        out.sort(key=lambda x: x[0].value)
        return out

################################################# IntegerBinning

class IntegerBinning(Binning, BinLocation):
    _params = {
        "min":       stagg.checktype.CheckInteger("IntegerBinning", "min", required=True),
        "max":       stagg.checktype.CheckInteger("IntegerBinning", "max", required=True),
        "loc_underflow": stagg.checktype.CheckEnum("IntegerBinning", "loc_underflow", required=False, choices=BinLocation.locations),
        "loc_overflow":  stagg.checktype.CheckEnum("IntegerBinning", "loc_overflow", required=False, choices=BinLocation.locations),
        }

    min       = typedproperty(_params["min"])
    max       = typedproperty(_params["max"])
    loc_underflow = typedproperty(_params["loc_underflow"])
    loc_overflow  = typedproperty(_params["loc_overflow"])

    def __init__(self, min, max, loc_underflow=BinLocation.nonexistent, loc_overflow=BinLocation.nonexistent):
        self.min = min
        self.max = max
        self.loc_underflow = loc_underflow
        self.loc_overflow = loc_overflow

    def _valid(self, seen, recursive):
        if self.min >= self.max:
            raise ValueError("IntegerBinning.min ({0}) must be strictly less than IntegerBinning.max ({1})".format(self.min, self.max))
        if self.loc_underflow != self.nonexistent and self.loc_overflow != self.nonexistent and self.loc_underflow == self.loc_overflow:
            raise ValueError("IntegerBinning.loc_underflow and IntegerBinning.loc_overflow must not be equal unless they are both nonexistent")

    def _binshape(self):
        return (self.max - self.min + 1 + int(self.loc_underflow != self.nonexistent) + int(self.loc_overflow != self.nonexistent),)

    def _toflatbuffers(self, builder):
        stagg.stagg_generated.IntegerBinning.IntegerBinningStart(builder)
        stagg.stagg_generated.IntegerBinning.IntegerBinningAddMin(builder, self.min)
        stagg.stagg_generated.IntegerBinning.IntegerBinningAddMax(builder, self.max)
        if self.loc_underflow != self.nonexistent:
            stagg.stagg_generated.IntegerBinning.IntegerBinningAddLocUnderflow(builder, self.loc_underflow.value)
        if self.loc_overflow != self.nonexistent:
            stagg.stagg_generated.IntegerBinning.IntegerBinningAddLocOverflow(builder, self.loc_overflow.value)
        return stagg.stagg_generated.IntegerBinning.IntegerBinningEnd(builder)

    def toRegularBinning(self):
        if self.loc_underflow == self.nonexistent and self.loc_overflow == self.nonexistent:
            overflow = None
        else:
            overflow = RealOverflow(loc_underflow=self.loc_underflow, loc_overflow=self.loc_overflow, loc_nanflow=RealOverflow.nonexistent)
        return RegularBinning(1 + self.max - self.min, RealInterval(self.min - 0.5, self.max + 0.5), overflow=overflow)

    def toEdgesBinning(self):
        return self.toRegularBinning().toEdgesBinning()

    def toIrregularBinning(self):
        return self.toEdgesBinning().toIrregularBinning()

    def toCategoryBinning(self, format="%g"):
        flows = [(self.loc_underflow, "(-inf, {0}]".format(format % (self.min - 1))), (self.loc_overflow, "[{0}, +inf)".format(format % (self.max + 1)))]
        cats = []
        for loc, cat in BinLocation._belows(flows):
            cats.append(cat)
        cats.extend([format % x for x in range(self.min, self.max + 1)])
        for loc, cat in BinLocation._aboves(flows):
            cats.append(cat)
        return CategoryBinning(cats)

    def toSparseRegularBinning(self):
        return self.toRegularBinning().toSparseRegularBinning()

    def _merger(self, other):
        HERE

################################################# RealInterval

class RealInterval(Stagg):
    _params = {
        "low":            stagg.checktype.CheckNumber("RealInterval", "low", required=True),
        "high":           stagg.checktype.CheckNumber("RealInterval", "high", required=True),
        "low_inclusive":  stagg.checktype.CheckBool("RealInterval", "low_inclusive", required=False),
        "high_inclusive": stagg.checktype.CheckBool("RealInterval", "high_inclusive", required=False),
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

    def _valid(self, seen, recursive):
        if self.low > self.high:
            raise ValueError("RealInterval.low ({0}) must be less than or equal to RealInterval.high ({1})".format(self.low, self.high))
        if self.low == self.high and not self.low_inclusive and not self.high_inclusive:
            raise ValueError("RealInterval describes an empty set ({0} == {1} and both endpoints are exclusive)".format(self.low, self.high))

    def _toflatbuffers(self, builder):
        stagg.stagg_generated.RealInterval.RealIntervalStart(builder)
        stagg.stagg_generated.RealInterval.RealIntervalAddLow(builder, self.low)
        stagg.stagg_generated.RealInterval.RealIntervalAddHigh(builder, self.high)
        if self.low_inclusive is not True:
            stagg.stagg_generated.RealInterval.RealIntervalAddLowInclusive(builder, self.low_inclusive)
        if self.high_inclusive is not False:
            stagg.stagg_generated.RealInterval.RealIntervalAddHighInclusive(builder, self.high_inclusive)
        return stagg.stagg_generated.RealInterval.RealIntervalEnd(builder)

################################################# RealOverflow

class NonRealMappingEnum(Enum):
    base = "RealOverflow"

class RealOverflow(Stagg, BinLocation):
    missing      = NonRealMappingEnum("missing", stagg.stagg_generated.NonRealMapping.NonRealMapping.missing)
    in_underflow = NonRealMappingEnum("in_underflow", stagg.stagg_generated.NonRealMapping.NonRealMapping.in_underflow)
    in_overflow  = NonRealMappingEnum("in_overflow", stagg.stagg_generated.NonRealMapping.NonRealMapping.in_overflow)
    in_nanflow   = NonRealMappingEnum("in_nanflow", stagg.stagg_generated.NonRealMapping.NonRealMapping.in_nanflow)
    mappings = [missing, in_underflow, in_overflow, in_nanflow]

    _params = {
        "loc_underflow": stagg.checktype.CheckEnum("RealOverflow", "loc_underflow", required=False, choices=BinLocation.locations),
        "loc_overflow":  stagg.checktype.CheckEnum("RealOverflow", "loc_overflow", required=False, choices=BinLocation.locations),
        "loc_nanflow":   stagg.checktype.CheckEnum("RealOverflow", "loc_nanflow", required=False, choices=BinLocation.locations),
        "minf_mapping":  stagg.checktype.CheckEnum("RealOverflow", "minf_mapping", required=False, choices=mappings),
        "pinf_mapping":  stagg.checktype.CheckEnum("RealOverflow", "pinf_mapping", required=False, choices=mappings),
        "nan_mapping":   stagg.checktype.CheckEnum("RealOverflow", "nan_mapping", required=False, choices=mappings),
        }

    loc_underflow = typedproperty(_params["loc_underflow"])
    loc_overflow  = typedproperty(_params["loc_overflow"])
    loc_nanflow   = typedproperty(_params["loc_nanflow"])
    minf_mapping  = typedproperty(_params["minf_mapping"])
    pinf_mapping  = typedproperty(_params["pinf_mapping"])
    nan_mapping   = typedproperty(_params["nan_mapping"])

    def __init__(self, loc_underflow=BinLocation.nonexistent, loc_overflow=BinLocation.nonexistent, loc_nanflow=BinLocation.nonexistent, minf_mapping=in_underflow, pinf_mapping=in_overflow, nan_mapping=in_nanflow):
        self.loc_underflow = loc_underflow
        self.loc_overflow = loc_overflow
        self.loc_nanflow = loc_nanflow
        self.minf_mapping = minf_mapping
        self.pinf_mapping = pinf_mapping
        self.nan_mapping = nan_mapping

    def _valid(self, seen, recursive):
        if self.loc_underflow != self.nonexistent and self.loc_overflow != self.nonexistent and self.loc_underflow == self.loc_overflow:
            raise ValueError("RealOverflow.loc_underflow and RealOverflow.loc_overflow must not be equal unless they are both nonexistent")
        if self.loc_underflow != self.nonexistent and self.loc_nanflow != self.nonexistent and self.loc_underflow == self.loc_nanflow:
            raise ValueError("RealOverflow.loc_underflow and RealOverflow.loc_nanflow must not be equal unless they are both nonexistent")
        if self.loc_overflow != self.nonexistent and self.loc_nanflow != self.nonexistent and self.loc_overflow == self.loc_nanflow:
            raise ValueError("RealOverflow.loc_overflow and RealOverflow.loc_nanflow must not be equal unless they are both nonexistent")
        if self.minf_mapping == self.in_overflow:
            raise ValueError("RealOverflow.minf_mapping (-inf mapping) can only be in_underflow or in_nanflow, not in_overflow")
        if self.pinf_mapping == self.in_underflow:
            raise ValueError("RealOverflow.pinf_mapping (+inf mapping) can only be in_overflow or in_nanflow, not in_underflow")

    def _numbins(self):
        return int(self.loc_underflow != self.nonexistent) + int(self.loc_overflow != self.nonexistent) + int(self.loc_nanflow != self.nonexistent)

    def _toflatbuffers(self, builder):
        stagg.stagg_generated.RealOverflow.RealOverflowStart(builder)
        if self.loc_underflow is not self.nonexistent:
            stagg.stagg_generated.RealOverflow.RealOverflowAddLocUnderflow(builder, self.loc_underflow.value)
        if self.loc_overflow is not self.nonexistent:
            stagg.stagg_generated.RealOverflow.RealOverflowAddLocOverflow(builder, self.loc_overflow.value)
        if self.loc_nanflow is not self.nonexistent:
            stagg.stagg_generated.RealOverflow.RealOverflowAddLocNanflow(builder, self.loc_nanflow.value)
        if self.minf_mapping is not self.in_underflow:
            stagg.stagg_generated.RealOverflow.RealOverflowAddMinfMapping(builder, self.minf_mapping.value)
        if self.pinf_mapping is not self.in_overflow:
            stagg.stagg_generated.RealOverflow.RealOverflowAddPinfMapping(builder, self.pinf_mapping.value)
        if self.nan_mapping is not self.in_nanflow:
            stagg.stagg_generated.RealOverflow.RealOverflowAddNanMapping(builder, self.nan_mapping.value)
        return stagg.stagg_generated.RealOverflow.RealOverflowEnd(builder)

################################################# RegularBinning

class RegularBinning(Binning):
    _params = {
        "num":      stagg.checktype.CheckInteger("RegularBinning", "num", required=True, min=1),
        "interval": stagg.checktype.CheckClass("RegularBinning", "interval", required=True, type=RealInterval),
        "overflow": stagg.checktype.CheckClass("RegularBinning", "overflow", required=False, type=RealOverflow),
        "circular": stagg.checktype.CheckBool("RegularBinning", "circular", required=False),
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

    def _valid(self, seen, recursive):
        if math.isinf(self.interval.low):
            raise ValueError("RegularBinning.interval.low must be finite")
        if math.isinf(self.interval.high):
            raise ValueError("RegularBinning.interval.high must be finite")
        if self.interval.low_inclusive and self.interval.high_inclusive:
            raise ValueError("RegularBinning.interval.low_inclusive and RegularBinning.interval.high_inclusive cannot both be True")
        if recursive:
            _valid(self.interval, seen, recursive)
            _valid(self.overflow, seen, recursive)

    def _binshape(self):
        if self.overflow is None:
            numoverflowbins = 0
        else:
            numoverflowbins = self.overflow._numbins()
        return (self.num + numoverflowbins,)

    def _toflatbuffers(self, builder):
        interval = self.interval._toflatbuffers(builder)
        overflow = None if self.overflow is None else self.overflow._toflatbuffers(builder)
        stagg.stagg_generated.RegularBinning.RegularBinningStart(builder)
        stagg.stagg_generated.RegularBinning.RegularBinningAddNum(builder, self.num)
        stagg.stagg_generated.RegularBinning.RegularBinningAddInterval(builder, interval)
        if overflow is not None:
            stagg.stagg_generated.RegularBinning.RegularBinningAddOverflow(builder, overflow)
        if self.circular is not False:
            stagg.stagg_generated.RegularBinning.RegularBinningAddCircular(builder, self.circular)
        return stagg.stagg_generated.RegularBinning.RegularBinningEnd(builder)

    def toEdgesBinning(self):
        edges = numpy.linspace(self.interval.low, self.interval.high, self.num + 1).tolist()
        overflow = None if self.overflow is None else self.overflow.unattached()
        return EdgesBinning(edges, overflow=overflow, low_inclusive=self.interval.low_inclusive, high_inclusive=self.interval.high_inclusive, circular=self.circular)

    def toIrregularBinning(self):
        return self.toEdgesBinning().toIrregularBinning()

    def toCategoryBinning(self, format="%g"):
        return self.toIrregularBinning().toCategoryBinning(format=format)

    def toSparseRegularBinning(self):
        if self.overflow is not None:
            if self.overflow.loc_underflow != self.overflow.nonexistent or self.overflow.loc_overflow != self.overflow.nonexistent:
                raise ValueError("cannot convert RegularBinning with underflow or overflow bins to SparseRegularBinning")
        bin_width = (self.interval.high - self.interval.low) / float(self.num)
        lowindex, origin = divmod(self.interval.low, bin_width)
        lowindex = int(lowindex)
        bins = range(lowindex, lowindex + self.num)
        overflow = None if self.overflow is None else self.overflow.unattached()
        return SparseRegularBinning(bins, bin_width, origin=origin, overflow=overflow, low_inclusive=self.interval.low_inclusive, high_inclusive=self.interval.high_inclusive)

################################################# HexagonalBinning

class HexagonalCoordinatesEnum(Enum):
    base = "HexagonalBinning"

class HexagonalBinning(Binning):
    offset         = HexagonalCoordinatesEnum("offset", stagg.stagg_generated.HexagonalCoordinates.HexagonalCoordinates.hex_offset)
    doubled_offset = HexagonalCoordinatesEnum("doubled_offset", stagg.stagg_generated.HexagonalCoordinates.HexagonalCoordinates.hex_doubled_offset)
    cube_xy        = HexagonalCoordinatesEnum("cube_xy", stagg.stagg_generated.HexagonalCoordinates.HexagonalCoordinates.hex_cube_xy)
    cube_yz        = HexagonalCoordinatesEnum("cube_yz", stagg.stagg_generated.HexagonalCoordinates.HexagonalCoordinates.hex_cube_yz)
    cube_xz        = HexagonalCoordinatesEnum("cube_xz", stagg.stagg_generated.HexagonalCoordinates.HexagonalCoordinates.hex_cube_xz)
    coordinates = [offset, doubled_offset, cube_xy, cube_yz, cube_xz]

    _params = {
        "qmin":        stagg.checktype.CheckInteger("HexagonalBinning", "qmin", required=True),
        "qmax":        stagg.checktype.CheckInteger("HexagonalBinning", "qmax", required=True),
        "rmin":        stagg.checktype.CheckInteger("HexagonalBinning", "rmin", required=True),
        "rmax":        stagg.checktype.CheckInteger("HexagonalBinning", "rmax", required=True),
        "coordinates": stagg.checktype.CheckEnum("HexagonalBinning", "coordinates", required=False, choices=coordinates),
        "xorigin":     stagg.checktype.CheckNumber("HexagonalBinning", "xorigin", required=False, min_inclusive=False, max_inclusive=False),
        "yorigin":     stagg.checktype.CheckNumber("HexagonalBinning", "yorigin", required=False, min_inclusive=False, max_inclusive=False),
        "qangle":      stagg.checktype.CheckNumber("HexagonalBinning", "qangle", required=False, min=-0.5*math.pi, max=0.5*math.pi),
        "qoverflow":   stagg.checktype.CheckClass("HexagonalBinning", "qoverflow", required=False, type=RealOverflow),
        "roverflow":   stagg.checktype.CheckClass("HexagonalBinning", "roverflow", required=False, type=RealOverflow),
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

    def _valid(self, seen, recursive):
        if self.qmin >= self.qmax:
            raise ValueError("HexagonalBinning.qmin ({0}) must be strictly less than HexagonalBinning.qmax ({1})".format(self.qmin, self.qmax))
        if self.rmin >= self.rmax:
            raise ValueError("HexagonalBinning.rmin ({0}) must be strictly less than HexagonalBinning.rmax ({1})".format(self.rmin, self.rmax))
        if recursive:
            _valid(self.qoverflow, seen, recursive)
            _valid(self.roverflow, seen, recursive)

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

    def _toflatbuffers(self, builder):
        qoverflow = None if self.qoverflow is None else self.qoverflow._toflatbuffers(builder)
        roverflow = None if self.roverflow is None else self.roverflow._toflatbuffers(builder)

        stagg.stagg_generated.HexagonalBinning.HexagonalBinningStart(builder)
        stagg.stagg_generated.HexagonalBinning.HexagonalBinningAddQmin(builder, self.qmin)
        stagg.stagg_generated.HexagonalBinning.HexagonalBinningAddQmax(builder, self.qmax)
        stagg.stagg_generated.HexagonalBinning.HexagonalBinningAddRmin(builder, self.rmin)
        stagg.stagg_generated.HexagonalBinning.HexagonalBinningAddRmax(builder, self.rmax)
        if self.coordinates != self.offset:
            stagg.stagg_generated.HexagonalBinning.HexagonalBinningAddOffset(builder, self.coordinates)
        if self.xorigin != 0.0:
            stagg.stagg_generated.HexagonalBinning.HexagonalBinningAddXorigin(builder, self.xorigin)
        if self.yorigin != 0.0:
            stagg.stagg_generated.HexagonalBinning.HexagonalBinningAddYorigin(builder, self.yorigin)
        if self.qangle != 0.0:
            stagg.stagg_generated.HexagonalBinning.HexagonalBinningAddQangle(builder, self.qangle)
        if qoverflow is not None:
            stagg.stagg_generated.HexagonalBinning.HexagonalBinningAddQoverflow(builder, qoverflow)
        if roverflow is not None:
            stagg.stagg_generated.HexagonalBinning.HexagonalBinningAddRoverflow(builder, roverflow)
        return stagg.stagg_generated.HexagonalBinning.HexagonalBinningEnd(builder)

################################################# EdgesBinning

class EdgesBinning(Binning):
    _params = {
        "edges":          stagg.checktype.CheckVector("EdgesBinning", "edges", required=True, type=float, minlen=1),
        "overflow":       stagg.checktype.CheckClass("EdgesBinning", "overflow", required=False, type=RealOverflow),
        "low_inclusive":  stagg.checktype.CheckBool("EdgesBinning", "low_inclusive", required=False),
        "high_inclusive": stagg.checktype.CheckBool("EdgesBinning", "high_inclusive", required=False),
        "circular":       stagg.checktype.CheckBool("EdgesBinning", "circular", required=False),
        }

    edges          = typedproperty(_params["edges"])
    overflow       = typedproperty(_params["overflow"])
    low_inclusive  = typedproperty(_params["low_inclusive"])
    high_inclusive = typedproperty(_params["high_inclusive"])
    circular       = typedproperty(_params["circular"])

    def __init__(self, edges, overflow=None, low_inclusive=True, high_inclusive=False, circular=False):
        self.edges = edges
        self.overflow = overflow
        self.low_inclusive = low_inclusive
        self.high_inclusive = high_inclusive
        self.circular = circular

    def _valid(self, seen, recursive):
        if numpy.isinf(self.edges).any():
            raise ValueError("EdgesBinning.edges must all be finite")
        if not numpy.greater(self.edges[1:], self.edges[:-1]).all():
            raise ValueError("EdgesBinning.edges must be strictly increasing")
        if len(self.edges) == 1 and (self.overflow is None or self.overflow._numbins() == 0):
            raise ValueError("EdgesBinning.edges must have more than one edge if EdgesBinning.overflow is missing or has zero bins")
        if self.low_inclusive and self.high_inclusive:
            raise ValueError("EdgesBinning.low_inclusive and EdgesBinning.high_inclusive cannot both be True")
        if recursive:
            _valid(self.overflow, seen, recursive)

    def _binshape(self):
        if self.overflow is None:
            numoverflowbins = 0
        else:
            numoverflowbins = self.overflow._numbins()
        return (len(self.edges) - 1 + numoverflowbins,)

    def _toflatbuffers(self, builder):
        stagg.stagg_generated.EdgesBinning.EdgesBinningStartEdgesVector(builder, len(self.edges))
        for x in self.edges[::-1]:
            builder.PrependFloat64(x)
        edges = builder.EndVector(len(self.edges))
        overflow = None if self.overflow is None else self.overflow._toflatbuffers(builder)

        stagg.stagg_generated.EdgesBinning.EdgesBinningStart(builder)
        stagg.stagg_generated.EdgesBinning.EdgesBinningAddEdges(builder, edges)
        if overflow is not None:
            stagg.stagg_generated.EdgesBinning.EdgesBinningAddOverflow(builder, overflow)
        if self.low_inclusive is not True:
            stagg.stagg_generated.EdgesBinning.EdgesBinningAddLowInclusive(builder, self.low_inclusive)
        if self.high_inclusive is not False:
            stagg.stagg_generated.EdgesBinning.EdgesBinningAddHighInclusive(builder, self.high_inclusive)
        if self.circular is not False:
            stagg.stagg_generated.EdgesBinning.EdgesBinningAddCircular(builder, self.circular)
        return stagg.stagg_generated.EdgesBinning.EdgesBinningEnd(builder)

    def toIrregularBinning(self):
        if self.low_inclusive and self.high_inclusive:
            raise ValueError("EdgesBinning.interval.low_inclusive and EdgesBinning.interval.high_inclusive cannot both be True")
        intervals = []
        for i in range(len(self.edges) - 1):
            intervals.append(RealInterval(self.edges[i], self.edges[i + 1], low_inclusive=self.low_inclusive, high_inclusive=self.high_inclusive))
        overflow = None if self.overflow is None else self.overflow.unattached()
        return IrregularBinning(intervals, overflow=overflow)

    def toCategoryBinning(self, format="%g"):
        return self.toIrregularBinning().toCategoryBinning(format=format)

################################################# IrregularBinning

class OverlappingFillStrategyEnum(Enum):
    base = "IrregularBinning"

class OverlappingFill(object):
    undefined = OverlappingFillStrategyEnum("undefined", stagg.stagg_generated.OverlappingFillStrategy.OverlappingFillStrategy.overfill_undefined)
    all       = OverlappingFillStrategyEnum("all", stagg.stagg_generated.OverlappingFillStrategy.OverlappingFillStrategy.overfill_all)
    first     = OverlappingFillStrategyEnum("first", stagg.stagg_generated.OverlappingFillStrategy.OverlappingFillStrategy.overfill_first)
    last      = OverlappingFillStrategyEnum("last", stagg.stagg_generated.OverlappingFillStrategy.OverlappingFillStrategy.overfill_last)
    overlapping_fill_strategies = [undefined, all, first, last]

class IrregularBinning(Binning, OverlappingFill):
    _params = {
        "intervals":        stagg.checktype.CheckVector("IrregularBinning", "intervals", required=True, type=RealInterval, minlen=1),
        "overflow":         stagg.checktype.CheckClass("IrregularBinning", "overflow", required=False, type=RealOverflow),
        "overlapping_fill": stagg.checktype.CheckEnum("IrregularBinning", "overlapping_fill", required=False, choices=OverlappingFill.overlapping_fill_strategies),
        }

    intervals        = typedproperty(_params["intervals"])
    overflow         = typedproperty(_params["overflow"])
    overlapping_fill = typedproperty(_params["overlapping_fill"])

    def __init__(self, intervals, overflow=None, overlapping_fill=OverlappingFill.undefined):
        self.intervals = intervals
        self.overflow = overflow
        self.overlapping_fill = overlapping_fill

    def _valid(self, seen, recursive):
        if recursive:
            _valid(self.intervals, seen, recursive)
            _valid(self.overflow, seen, recursive)

    def _binshape(self):
        if self.overflow is None:
            numoverflowbins = 0
        else:
            numoverflowbins = self.overflow._numbins()
        return (len(self.intervals) + numoverflowbins,)

    def _toflatbuffers(self, builder):
        intervals = [x._toflatbuffers(builder) for x in self.intervals]
        overflow = None if self.overflow is None else self.overflow._toflatbuffers(builder)

        stagg.stagg_generated.IrregularBinning.IrregularBinningStartIntervalsVector(builder, len(intervals))
        for x in intervals[::-1]:
            builder.PrependUOffsetTRelative(x)
        intervals = builder.EndVector(len(intervals))

        stagg.stagg_generated.IrregularBinning.IrregularBinningStart(builder)
        stagg.stagg_generated.IrregularBinning.IrregularBinningAddIntervals(builder, intervals)
        if overflow is not None:
            stagg.stagg_generated.IrregularBinning.IrregularBinningAddOverflow(builder, overflow)
        if self.overlapping_fill != self.undefined:
            stagg.stagg_generated.IrregularBinning.IrregularBinningAddOverlappingFill(builder, self.overlapping_fill.value)
        return stagg.stagg_generated.IrregularBinning.IrregularBinningEnd(builder)

    def toCategoryBinning(self, format="%g"):
        flows = []
        if self.overflow is not None:
            flows.append((self.overflow.loc_underflow, "{0}-inf, {1}{2}".format(
                "[" if self.overflow.minf_mapping == self.overflow.in_underflow else "(",
                format % self.intervals[0].low,
                ")" if self.intervals[0].low_inclusive else "]")))
            flows.append((self.overflow.loc_overflow, "{0}{1}, +inf{2}".format(
                "(" if self.intervals[-1].high_inclusive else "[",
                format % self.intervals[-1].high,
                "]" if self.overflow.pinf_mapping == self.overflow.in_overflow else ")")))
            nanflow = []
            if self.overflow.minf_mapping == self.overflow.in_nanflow:
                nanflow.append("-inf")
            if self.overflow.pinf_mapping == self.overflow.in_nanflow:
                nanflow.append("+inf")
            if self.overflow.nan_mapping == self.overflow.in_nanflow:
                nanflow.append("nan")
            flows.append((self.overflow.loc_nanflow, "{" + ", ".join(nanflow) + "}"))

        cats = []
        for loc, cat in BinLocation._belows(flows):
            cats.append(cat)

        for interval in self.intervals:
            cats.append("{0}{1}, {2}{3}".format(
                "[" if interval.low_inclusive else "(",
                format % interval.low,
                format % interval.high,
                "]" if interval.high_inclusive else ")"))

        for loc, cat in BinLocation._aboves(flows):
            cats.append(cat)

        return CategoryBinning(cats)
        
################################################# CategoryBinning

class CategoryBinning(Binning, BinLocation):
    _params = {
        "categories": stagg.checktype.CheckVector("CategoryBinning", "categories", required=True, type=str),
        "loc_overflow":  stagg.checktype.CheckEnum("CategoryBinning", "loc_overflow", required=False, choices=BinLocation.locations),
        }

    categories = typedproperty(_params["categories"])
    loc_overflow = typedproperty(_params["loc_overflow"])

    def __init__(self, categories, loc_overflow=BinLocation.nonexistent):
        self.categories = categories
        self.loc_overflow = loc_overflow

    def _valid(self, seen, recursive):
        if len(self.categories) != len(set(self.categories)):
            raise ValueError("CategoryBinning.categories must be unique")

    @property
    def isnumerical(self):
        return False

    def _binshape(self):
        return (len(self.categories) + (self.loc_overflow != self.nonexistent),)

    def _toflatbuffers(self, builder):
        categories = [builder.CreateString(x.encode("utf-8")) for x in self.categories]

        stagg.stagg_generated.CategoryBinning.CategoryBinningStartCategoriesVector(builder, len(categories))
        for x in categories[::-1]:
            builder.PrependUOffsetTRelative(x)
        categories = builder.EndVector(len(categories))

        stagg.stagg_generated.CategoryBinning.CategoryBinningStart(builder)
        stagg.stagg_generated.CategoryBinning.CategoryBinningAddCategories(builder, categories)
        if self.loc_overflow != self.nonexistent:
            stagg.stagg_generated.CategoryBinning.CategoryBinningAddLocOverflow(builder, self.loc_overflow.value)
        return stagg.stagg_generated.CategoryBinning.CategoryBinningEnd(builder)

################################################# SparseRegularBinning

class SparseRegularBinning(Binning, BinLocation):
    _params = {
        "bins":           stagg.checktype.CheckVector("SparseRegularBinning", "bins", required=True, type=int),
        "bin_width":      stagg.checktype.CheckNumber("SparseRegularBinning", "bin_width", required=True, min=0, min_inclusive=False),
        "origin":         stagg.checktype.CheckNumber("SparseRegularBinning", "origin", required=False),
        "overflow":       stagg.checktype.CheckClass("SparseRegularBinning", "overflow", required=False, type=RealOverflow),
        "low_inclusive":  stagg.checktype.CheckBool("SparseRegularBinning", "low_inclusive", required=False),
        "high_inclusive": stagg.checktype.CheckBool("SparseRegularBinning", "high_inclusive", required=False),
        }

    bins           = typedproperty(_params["bins"])
    bin_width      = typedproperty(_params["bin_width"])
    origin         = typedproperty(_params["origin"])
    overflow       = typedproperty(_params["overflow"])
    low_inclusive  = typedproperty(_params["low_inclusive"])
    high_inclusive = typedproperty(_params["high_inclusive"])

    def __init__(self, bins, bin_width, origin=0.0, overflow=None, low_inclusive=True, high_inclusive=False):
        self.bins = bins
        self.bin_width = bin_width
        self.origin = origin
        self.overflow = overflow
        self.low_inclusive = low_inclusive
        self.high_inclusive = high_inclusive

    def _valid(self, seen, recursive):
        if len(self.bins) != len(numpy.unique(self.bins)):
            raise ValueError("SparseRegularBinning.bins must be unique")
        if self.overflow is not None:
            if self.overflow.loc_underflow != self.overflow.nonexistent and self.overflow.minf_mapping != self.overflow.in_underflow:
                raise ValueError("SparseRegularBinning underflow bin exists but -inf (its only possible value) is not mapped to it")
            if self.overflow.loc_overflow != self.overflow.nonexistent and self.overflow.pinf_mapping != self.overflow.in_overflow:
                raise ValueError("SparseRegularBinning overflow bin exists but +inf (its only possible value) is not mapped to it")
        if self.low_inclusive and self.high_inclusive:
            raise ValueError("SparseRegularBinning.low_inclusive and SparseRegularBinning.high_inclusive cannot both be True")
        if recursive:
            _valid(self.overflow, seen, recursive)

    def _binshape(self):
        if self.overflow is None:
            numoverflowbins = 0
        else:
            numoverflowbins = self.overflow._numbins()
        return (len(self.bins) + numoverflowbins,)

    def _toflatbuffers(self, builder):
        stagg.stagg_generated.SparseRegularBinning.SparseRegularBinningStartBinsVector(builder, len(self.bins))
        for x in self.bins[::-1]:
            builder.PrependInt64(x)
        bins = builder.EndVector(len(self.bins))
        overflow = None if self.overflow is None else self.overflow._toflatbuffers(builder)

        stagg.stagg_generated.SparseRegularBinning.SparseRegularBinningStart(builder)
        stagg.stagg_generated.SparseRegularBinning.SparseRegularBinningAddBins(builder, bins)
        stagg.stagg_generated.SparseRegularBinning.SparseRegularBinningAddBinWidth(builder, self.bin_width)
        if self.origin != 0.0:
            stagg.stagg_generated.SparseRegularBinning.SparseRegularBinningAddOrigin(builder, self.origin)
        if overflow is not None:
            stagg.stagg_generated.EdgesBinning.EdgesBinningAddOverflow(builder, overflow)
        if self.low_inclusive is not True:
            stagg.stagg_generated.EdgesBinning.EdgesBinningAddLowInclusive(builder, self.low_inclusive)
        if self.high_inclusive is not False:
            stagg.stagg_generated.EdgesBinning.EdgesBinningAddHighInclusive(builder, self.high_inclusive)
        return stagg.stagg_generated.SparseRegularBinning.SparseRegularBinningEnd(builder)

    def toIrregularBinning(self):
        if self.low_inclusive and self.high_inclusive:
            raise ValueError("SparseRegularBinning.interval.low_inclusive and SparseRegularBinning.interval.high_inclusive cannot both be True")
        if self.overflow is not None:
            if self.overflow.loc_underflow != self.overflow.nonexistent or self.overflow.loc_overflow != self.overflow.nonexistent:
                raise ValueError("SparseRegularBinning with underflow or overflow (infinity singletons) cannot be converted to IrregularBinning")
        intervals = []
        for x in self.bins:
            intervals.append(RealInterval(self.bin_width*(x) + self.origin, self.bin_width*(x + 1) + self.origin))
        overflow = None if self.overflow is None else self.overflow.unattached()
        return IrregularBinning(intervals, overflow=overflow)

    def toCategoryBinning(self, format="%g"):
        flows = []
        if self.overflow is not None:
            if self.overflow.loc_underflow != self.overflow.nonexistent and self.overflow.minf_mapping != self.overflow.in_underflow:
                raise ValueError("SparseRegularBinning underflow bin exists but -inf (its only possible value) is not mapped to it")
            flows.append((self.overflow.loc_underflow, "{-inf}"))
            if self.overflow.loc_overflow != self.overflow.nonexistent and self.overflow.pinf_mapping != self.overflow.in_overflow:
                raise ValueError("SparseRegularBinning overflow bin exists but +inf (its only possible value) is not mapped to it")
            flows.append((self.overflow.loc_overflow, "{+inf}"))
            nanflow = []
            if self.overflow.minf_mapping == self.overflow.in_nanflow:
                nanflow.append("-inf")
            if self.overflow.pinf_mapping == self.overflow.in_nanflow:
                nanflow.append("+inf")
            if self.overflow.nan_mapping == self.overflow.in_nanflow:
                nanflow.append("nan")
            flows.append((self.overflow.loc_nanflow, "{" + ", ".join(nanflow) + "}"))

        cats = []
        for loc, cat in BinLocation._belows(flows):
            cats.append(cat)

        formatted = [(format % (self.bin_width*(x) + self.origin), format % (self.bin_width*(x + 1) + self.origin)) for x in self.bins]

        if self.low_inclusive and self.high_inclusive:
            raise ValueError("SparseRegularBinning.low_inclusive and SparseRegularBinning.high_inclusive cannot both be True")
        elif not self.low_inclusive and not self.high_inclusive:
            cats.extend(["[{0}, {1}]".format(x, y) for x, y in formatted])
        elif self.low_inclusive:
            cats.extend(["[{0}, {1})".format(x, y) for x, y in formatted])
        elif self.high_inclusive:
            cats.extend(["({0}, {1}]".format(x, y) for x, y in formatted])

        for loc, cat in BinLocation._aboves(flows):
            cats.append(cat)

        return CategoryBinning(cats)

################################################# FractionBinning

class FractionLayoutEnum(Enum):
    base = "FractionBinning"

class FractionErrorMethodEnum(Enum):
    base = "FractionBinning"

class FractionBinning(Binning):
    passall  = FractionLayoutEnum("passall", stagg.stagg_generated.FractionLayout.FractionLayout.frac_passall)
    failall  = FractionLayoutEnum("failall", stagg.stagg_generated.FractionLayout.FractionLayout.frac_failall)
    passfail = FractionLayoutEnum("passfail", stagg.stagg_generated.FractionLayout.FractionLayout.frac_passfail)
    layouts = [passall, failall, passfail]

    normal           = FractionErrorMethodEnum("normal", stagg.stagg_generated.FractionErrorMethod.FractionErrorMethod.frac_normal)
    clopper_pearson  = FractionErrorMethodEnum("clopper_pearson", stagg.stagg_generated.FractionErrorMethod.FractionErrorMethod.frac_clopper_pearson)
    wilson           = FractionErrorMethodEnum("wilson", stagg.stagg_generated.FractionErrorMethod.FractionErrorMethod.frac_wilson)
    agresti_coull    = FractionErrorMethodEnum("agresti_coull", stagg.stagg_generated.FractionErrorMethod.FractionErrorMethod.frac_agresti_coull)
    feldman_cousins  = FractionErrorMethodEnum("feldman_cousins", stagg.stagg_generated.FractionErrorMethod.FractionErrorMethod.frac_feldman_cousins)
    jeffrey          = FractionErrorMethodEnum("jeffrey", stagg.stagg_generated.FractionErrorMethod.FractionErrorMethod.frac_jeffrey)
    bayesian_uniform = FractionErrorMethodEnum("bayesian_uniform", stagg.stagg_generated.FractionErrorMethod.FractionErrorMethod.frac_bayesian_uniform)
    error_methods = [normal, clopper_pearson, wilson, agresti_coull, feldman_cousins, jeffrey, bayesian_uniform]

    _params = {
        "layout": stagg.checktype.CheckEnum("FractionBinning", "layout", required=False, choices=layouts),
        "layout_reversed": stagg.checktype.CheckBool("FractionBinning", "layout_reversed", required=False),
        "error_method": stagg.checktype.CheckEnum("FractionBinning", "error_method", required=False, choices=error_methods),
        }

    layout          = typedproperty(_params["layout"])
    layout_reversed = typedproperty(_params["layout_reversed"])
    error_method    = typedproperty(_params["error_method"])

    def __init__(self, layout=passall, layout_reversed=False, error_method=normal):
        self.layout = layout
        self.layout_reversed = layout_reversed
        self.error_method = error_method

    @property
    def isnumerical(self):
        return False

    def _binshape(self):
        return (2,)

    def _toflatbuffers(self, builder):
        stagg.stagg_generated.FractionBinning.FractionBinningStart(builder)
        if self.layout != self.passall:
            stagg.stagg_generated.FractionBinning.FractionBinningAddLayout(builder, self.layout.value)
        if self.layout_reversed is not False:
            stagg.stagg_generated.FractionBinning.FractionBinningAddLayoutReversed(builder, self.layout_reversed)
        if self.error_method != self.normal:
            stagg.stagg_generated.FractionBinning.FractionBinningAddErrorMethod(builder, self.error_method.value)
        return stagg.stagg_generated.FractionBinning.FractionBinningEnd(builder)

################################################# PredicateBinning

class PredicateBinning(Binning, OverlappingFill):
    _params = {
        "predicates":       stagg.checktype.CheckVector("PredicateBinning", "predicates", required=True, type=str, minlen=1),
        "overlapping_fill": stagg.checktype.CheckEnum("PredicateBinning", "overlapping_fill", required=False, choices=OverlappingFill.overlapping_fill_strategies),
        }

    predicates       = typedproperty(_params["predicates"])
    overlapping_fill = typedproperty(_params["overlapping_fill"])

    def __init__(self, predicates, overlapping_fill=OverlappingFill.undefined):
        self.predicates = predicates
        self.overlapping_fill = overlapping_fill

    def _binshape(self):
        return (len(self.predicates),)

    def _toflatbuffers(self, builder):
        predicates = [builder.CreateString(x.encode("utf-8")) for x in self.predicates]

        stagg.stagg_generated.PredicateBinning.PredicateBinningStartPredicatesVector(builder, len(predicates))
        for x in predicates[::-1]:
            builder.PrependUOffsetTRelative(x)
        predicates = builder.EndVector(len(predicates))

        stagg.stagg_generated.PredicateBinning.PredicateBinningStart(builder)
        stagg.stagg_generated.PredicateBinning.PredicateBinningAddPredicates(builder, predicates)
        if self.overlapping_fill != self.undefined:
            stagg.stagg_generated.PredicateBinning.PredicateBinningAddOverlappingFill(builder, self.overlapping_fill.value)
        return stagg.stagg_generated.PredicateBinning.PredicateBinningEnd(builder)

################################################# Assignment

class Assignment(Stagg):
    _params = {
        "identifier": stagg.checktype.CheckKey("Assignment", "identifier", required=True, type=str),
        "expression": stagg.checktype.CheckString("Assignment", "expression", required=True),
        }

    identifier = typedproperty(_params["identifier"])
    expression = typedproperty(_params["expression"])

    def __init__(self, identifier, expression):
        self.identifier = identifier
        self.expression  = expression 

    def _toflatbuffers(self, builder):
        identifier = builder.CreateString(self.identifier.encode("utf-8"))
        expression = builder.CreateString(self.expression.encode("utf-8"))
        stagg.stagg_generated.Assignment.AssignmentStart(builder)
        stagg.stagg_generated.Assignment.AssignmentAddIdentifier(builder, identifier)
        stagg.stagg_generated.Assignment.AssignmentAddExpression(builder, expression)
        return stagg.stagg_generated.Assignment.AssignmentEnd(builder)

################################################# Variation

class Variation(Stagg):
    _params = {
        "assignments":         stagg.checktype.CheckVector("Variation", "assignments", required=True, type=Assignment),
        "systematic":          stagg.checktype.CheckVector("Variation", "systematic", required=False, type=float),
        "category_systematic": stagg.checktype.CheckVector("Variation", "category_systematic", required=False, type=str),
        }

    assignments         = typedproperty(_params["assignments"])
    systematic          = typedproperty(_params["systematic"])
    category_systematic = typedproperty(_params["category_systematic"])

    def __init__(self, assignments, systematic=None, category_systematic=None):
        self.assignments = assignments
        self.systematic = systematic
        self.category_systematic = category_systematic

    def _valid(self, seen, recursive):
        if len(set(x.identifier for x in self.assignments)) != len(self.assignments):
            raise ValueError("Variation.assignments keys must be unique")
        if recursive:
            _valid(self.assignments, seen, recursive)

    def __getitem__(self, where):
        if where == ():
            return self
        elif isinstance(where, tuple):
            head, tail = where[0], where[1:]
        else:
            head, tail = where, ()
        out = _getbykey(self, "assignments", head)
        if tail == ():
            return out
        else:
            return out[tail]

    def _toflatbuffers(self, builder):
        assignments = [x._toflatbuffers(builder) for x in self.assignments]
        category_systematic = None if len(self.category_systematic) == 0 else [builder.CreateString(x.encode("utf-8")) for x in self.category_systematic]

        stagg.stagg_generated.Variation.VariationStartAssignmentsVector(builder, len(assignments))
        for x in assignments[::-1]:
            builder.PrependUOffsetTRelative(x)
        assignments = builder.EndVector(len(assignments))

        if len(self.systematic) == 0:
            systematic = None
        else:
            stagg.stagg_generated.Variation.VariationStartSystematicVector(builder, len(self.systematic))
            for x in self.systematic[::-1]:
                builder.PrependFloat64(x)
            systematic = builder.EndVector(len(self.systematic))

        if category_systematic is not None:
            stagg.stagg_generated.Variation.VariationStartCategorySystematicVector(builder, len(category_systematic))
            for x in category_systematic[::-1]:
                builder.PrependUOffsetTRelative(x)
            category_systematic = builder.EndVector(len(category_systematic))

        stagg.stagg_generated.Variation.VariationStart(builder)
        stagg.stagg_generated.Variation.VariationAddAssignments(builder, assignments)
        if systematic is not None:
            stagg.stagg_generated.Variation.VariationAddSystematic(builder, systematic)
        if category_systematic is not None:
            stagg.stagg_generated.Variation.VariationAddCategorySystematic(builder, category_systematic)
        return stagg.stagg_generated.Variation.VariationEnd(builder)

################################################# VariationBinning

class VariationBinning(Binning):
    _params = {
        "variations": stagg.checktype.CheckVector("VariationBinning", "variations", required=True, type=Variation, minlen=1),
        }

    variations = typedproperty(_params["variations"])

    def __init__(self, variations):
        self.variations = variations

    def _valid(self, seen, recursive):
        if recursive:
            _valid(self.variations, seen, recursive)

    def _binshape(self):
        return (len(self.variations),)

    def _toflatbuffers(self, builder):
        variations = [x._toflatbuffers(builder) for x in self.variations]

        stagg.stagg_generated.VariationBinning.VariationBinningStartVariationsVector(builder, len(variations))
        for x in variations[::-1]:
            builder.PrependUOffsetTRelative(x)
        variations = builder.EndVector(len(variations))

        stagg.stagg_generated.VariationBinning.VariationBinningStart(builder)
        stagg.stagg_generated.VariationBinning.VariationBinningAddVariations(builder, variations)
        return stagg.stagg_generated.VariationBinning.VariationBinningEnd(builder)

################################################# Axis

class Axis(Stagg):
    _params = {
        "binning":    stagg.checktype.CheckClass("Axis", "binning", required=False, type=Binning),
        "expression": stagg.checktype.CheckString("Axis", "expression", required=False),
        "statistics": stagg.checktype.CheckClass("Axis", "statistics", required=False, type=Statistics),
        "title":      stagg.checktype.CheckString("Axis", "title", required=False),
        "metadata":   stagg.checktype.CheckClass("Axis", "metadata", required=False, type=Metadata),
        "decoration": stagg.checktype.CheckClass("Axis", "decoration", required=False, type=Decoration),
        }

    binning    = typedproperty(_params["binning"])
    expression = typedproperty(_params["expression"])
    statistics = typedproperty(_params["statistics"])
    title      = typedproperty(_params["title"])
    metadata   = typedproperty(_params["metadata"])
    decoration = typedproperty(_params["decoration"])

    def __init__(self, binning=None, expression=None, statistics=None, title=None, metadata=None, decoration=None):
        self.binning = binning
        self.expression = expression
        self.statistics = statistics
        self.title = title
        self.metadata = metadata
        self.decoration = decoration

    def _valid(self, seen, recursive):
        if recursive:
            _valid(self.binning, seen, recursive)
            _valid(self.statistics, seen, recursive)
            _valid(self.metadata, seen, recursive)
            _valid(self.decoration, seen, recursive)

    def _binshape(self):
        if self.binning is None:
            return (1,)
        else:
            return self.binning._binshape()

    @classmethod
    def _fromflatbuffers(cls, fb):
        out = cls.__new__(cls)
        out._flatbuffers = _MockFlatbuffers()
        out._flatbuffers.BinningByTag = _MockFlatbuffers._ByTag(fb.Binning, fb.BinningType, _Binning_lookup)
        out._flatbuffers.Expression = fb.Expression
        out._flatbuffers.Statistics = fb.Statistics
        out._flatbuffers.Title = fb.Title
        out._flatbuffers.Metadata = fb.Metadata
        out._flatbuffers.Decoration = fb.Decoration
        return out

    def _toflatbuffers(self, builder):
        binning = None if self.binning is None else self.binning._toflatbuffers(builder)
        expression = None if self.expression is None else builder.CreateString(self.expression.encode("utf-8"))
        statistics = None if self.statistics is None else self.statistics._toflatbuffers(builder)
        title = None if self.title is None else builder.CreateString(self.title.encode("utf-8"))
        metadata = None if self.metadata is None else self.metadata._toflatbuffers(builder)
        decoration = None if self.decoration is None else self.decoration._toflatbuffers(builder)

        stagg.stagg_generated.Axis.AxisStart(builder)
        if binning is not None:
            stagg.stagg_generated.Axis.AxisAddBinningType(builder, _Binning_invlookup[type(self.binning)])
            stagg.stagg_generated.Axis.AxisAddBinning(builder, binning)
        if expression is not None:
            stagg.stagg_generated.Axis.AxisAddExpression(builder, expression)
        if statistics is not None:
            stagg.stagg_generated.Axis.AxisAddStatistics(builder, statistics)
        if title is not None:
            stagg.stagg_generated.Axis.AxisAddTitle(builder, title)
        if metadata is not None:
            stagg.stagg_generated.Axis.AxisAddMetadata(builder, metadata)
        if decoration is not None:
            stagg.stagg_generated.Axis.AxisAddDecoration(builder, decoration)
        return stagg.stagg_generated.Axis.AxisEnd(builder)

################################################# Profile

class Profile(Stagg):
    _params = {
        "expression": stagg.checktype.CheckString("Profile", "expression", required=True),
        "statistics": stagg.checktype.CheckClass("Profile", "statistics", required=True, type=Statistics),
        "title":      stagg.checktype.CheckString("Profile", "title", required=False),
        "metadata":   stagg.checktype.CheckClass("Profile", "metadata", required=False, type=Metadata),
        "decoration": stagg.checktype.CheckClass("Profile", "decoration", required=False, type=Decoration),
        }

    expression = typedproperty(_params["expression"])
    statistics = typedproperty(_params["statistics"])
    title      = typedproperty(_params["title"])
    metadata   = typedproperty(_params["metadata"])
    decoration = typedproperty(_params["decoration"])

    def __init__(self, expression, statistics, title=None, metadata=None, decoration=None):
        self.expression = expression
        self.statistics = statistics
        self.title = title
        self.metadata = metadata
        self.decoration = decoration

    def _valid(self, seen, recursive):
        if recursive:
            _valid(self.statistics, seen, recursive)
            _valid(self.metadata, seen, recursive)
            _valid(self.decoration, seen, recursive)

    def _toflatbuffers(self, builder):
        expression = builder.CreateString(self.expression.encode("utf-8"))
        statistics = self.statistics._toflatbuffers(builder)
        title = None if self.title is None else builder.CreateString(self.title.encode("utf-8"))
        metadata = None if self.metadata is None else self.metadata._toflatbuffers(builder)
        decoration = None if self.decoration is None else self.decoration._toflatbuffers(builder)

        stagg.stagg_generated.Profile.ProfileStart(builder)
        stagg.stagg_generated.Profile.ProfileAddExpression(builder, expression)
        stagg.stagg_generated.Profile.ProfileAddStatistics(builder, statistics)
        if title is not None:
            stagg.stagg_generated.Profile.ProfileAddTitle(builder, title)
        if metadata is not None:
            stagg.stagg_generated.Profile.ProfileAddMetadata(builder, metadata)
        if decoration is not None:
            stagg.stagg_generated.Profile.ProfileAddDecoration(builder, decoration)
        return stagg.stagg_generated.Profile.ProfileEnd(builder)

################################################# Counts

class Counts(Stagg):
    def __init__(self):
        raise TypeError("{0} is an abstract base class; do not construct".format(type(self).__name__))

################################################# UnweightedCounts

class UnweightedCounts(Counts):
    _params = {
        "counts": stagg.checktype.CheckClass("UnweightedCounts", "counts", required=True, type=InterpretedBuffer),
        }

    counts = typedproperty(_params["counts"])

    def __init__(self, counts):
        self.counts = counts

    def _valid(self, seen, recursive):
        if recursive:
            _valid(self.counts, seen, recursive)

    @classmethod
    def _fromflatbuffers(cls, fb):
        out = cls.__new__(cls)
        out._flatbuffers = _MockFlatbuffers()
        out._flatbuffers.CountsByTag = _MockFlatbuffers._ByTag(fb.Counts, fb.CountsType, _InterpretedBuffer_lookup)
        return out

    def _toflatbuffers(self, builder):
        counts = self.counts._toflatbuffers(builder)
        stagg.stagg_generated.UnweightedCounts.UnweightedCountsStart(builder)
        stagg.stagg_generated.UnweightedCounts.UnweightedCountsAddCountsType(builder, _InterpretedBuffer_invlookup[type(self.counts)])
        stagg.stagg_generated.UnweightedCounts.UnweightedCountsAddCounts(builder, counts)
        return stagg.stagg_generated.UnweightedCounts.UnweightedCountsEnd(builder)
    
################################################# WeightedCounts

class WeightedCounts(Counts):
    _params = {
        "sumw":       stagg.checktype.CheckClass("WeightedCounts", "sumw", required=True, type=InterpretedBuffer),
        "sumw2":      stagg.checktype.CheckClass("WeightedCounts", "sumw2", required=False, type=InterpretedBuffer),
        "unweighted": stagg.checktype.CheckClass("WeightedCounts", "unweighted", required=False, type=UnweightedCounts),
        }

    sumw       = typedproperty(_params["sumw"])
    sumw2      = typedproperty(_params["sumw2"])
    unweighted = typedproperty(_params["unweighted"])

    def __init__(self, sumw, sumw2=None, unweighted=None):
        self.sumw = sumw
        self.sumw2 = sumw2
        self.unweighted = unweighted

    def _valid(self, seen, recursive):
        if recursive:
            _valid(self.sumw, seen, recursive)
            _valid(self.sumw2, seen, recursive)
            _valid(self.unweighted, seen, recursive)

    @classmethod
    def _fromflatbuffers(cls, fb):
        out = cls.__new__(cls)
        out._flatbuffers = _MockFlatbuffers()
        out._flatbuffers.SumwByTag = _MockFlatbuffers._ByTag(fb.Sumw, fb.SumwType, _InterpretedBuffer_lookup)
        out._flatbuffers.Sumw2ByTag = _MockFlatbuffers._ByTag(fb.Sumw2, fb.Sumw2Type, _InterpretedBuffer_lookup)
        out._flatbuffers.Unweighted = fb.Unweighted
        return out

    def _toflatbuffers(self, builder):
        sumw = self.sumw._toflatbuffers(builder)
        sumw2 = None if self.sumw2 is None else self.sumw2._toflatbuffers(builder)
        unweighted = None if self.unweighted is None else self.unweighted._toflatbuffers(builder)

        stagg.stagg_generated.WeightedCounts.WeightedCountsStart(builder)
        stagg.stagg_generated.WeightedCounts.WeightedCountsAddSumwType(builder, _InterpretedBuffer_invlookup[type(self.sumw)])
        stagg.stagg_generated.WeightedCounts.WeightedCountsAddSumw(builder, sumw)
        if sumw2 is not None:
            stagg.stagg_generated.WeightedCounts.WeightedCountsAddSumw2Type(builder, _InterpretedBuffer_invlookup[type(self.sumw2)])
            stagg.stagg_generated.WeightedCounts.WeightedCountsAddSumw2(builder, sumw2)
        if unweighted is not None:
            stagg.stagg_generated.WeightedCounts.WeightedCountsAddUnweighted(builder, unweighted)
        return stagg.stagg_generated.WeightedCounts.WeightedCountsEnd(builder)

################################################# Parameter

class Parameter(Stagg):
    _params = {
        "identifier": stagg.checktype.CheckKey("Parameter", "identifier", required=True, type=str),
        "values":     stagg.checktype.CheckClass("Parameter", "values", required=True, type=InterpretedBuffer),
        }

    identifier = typedproperty(_params["identifier"])
    values     = typedproperty(_params["values"])

    def __init__(self, identifier, values):
        self.identifier = identifier
        self.values = values

    def _valid(self, seen, recursive):
        if recursive:
            _valid(self.values, seen, recursive)

    @classmethod
    def _fromflatbuffers(cls, fb):
        out = cls.__new__(cls)
        out._flatbuffers = _MockFlatbuffers()
        out._flatbuffers.Identifier = fb.Identifier
        out._flatbuffers.ValuesByTag = _MockFlatbuffers._ByTag(fb.Values, fb.ValuesType, _InterpretedBuffer_lookup)
        return out

    def _toflatbuffers(self, builder):
        values = self.values._toflatbuffers(builder)
        identifier = builder.CreateString(self.identifier.encode("utf-8"))
        
        stagg.stagg_generated.Parameter.ParameterStart(builder)
        stagg.stagg_generated.Parameter.ParameterAddIdentifier(builder, identifier)
        stagg.stagg_generated.Parameter.ParameterAddValuesType(builder, _InterpretedBuffer_invlookup[type(self.values)])
        stagg.stagg_generated.Parameter.ParameterAddValues(builder, values)
        return stagg.stagg_generated.Parameter.ParameterEnd(builder)

################################################# Function

class Function(Stagg):
    def __init__(self):
        raise TypeError("{0} is an abstract base class; do not construct".format(type(self).__name__))

    @classmethod
    def _fromflatbuffers(cls, fb):
        interface, deserializer = _FunctionData_lookup[fb.DataType()]
        data = fb.Data()
        fb2 = deserializer()
        fb2.Init(data.Bytes, data.Pos)
        return interface._fromflatbuffers(fb, fb2)

################################################# FunctionObject

class FunctionObject(Object):
    def __init__(self):
        raise TypeError("{0} is an abstract base class; do not construct".format(type(self).__name__))

    @classmethod
    def _fromflatbuffers(cls, fb, fb2):
        interface, deserializer = _FunctionObjectData_lookup[fb2.DataType()]
        data = fb2.Data()
        fb3 = deserializer()
        fb3.Init(data.Bytes, data.Pos)
        return interface._fromflatbuffers(fb, fb2, fb3)

################################################# ParameterizedFunction

class ParameterizedFunction(Function, FunctionObject):
    _params = {
        "expression": stagg.checktype.CheckString("ParameterizedFunction", "expression", required=True),
        "parameters": stagg.checktype.CheckVector("ParameterizedFunction", "parameters", required=False, type=Parameter),
        "title":      stagg.checktype.CheckString("ParameterizedFunction", "title", required=False),
        "metadata":   stagg.checktype.CheckClass("ParameterizedFunction", "metadata", required=False, type=Metadata),
        "decoration": stagg.checktype.CheckClass("ParameterizedFunction", "decoration", required=False, type=Decoration),
        "script":     stagg.checktype.CheckString("ParameterizedFunction", "script", required=False),
        }

    expression = typedproperty(_params["expression"])
    parameters = typedproperty(_params["parameters"])
    title      = typedproperty(_params["title"])
    metadata   = typedproperty(_params["metadata"])
    decoration = typedproperty(_params["decoration"])
    script     = typedproperty(_params["script"])

    def __init__(self, expression, parameters=None, title=None, metadata=None, decoration=None, script=None):
        self.expression = expression
        self.parameters = parameters
        self.title = title
        self.metadata = metadata
        self.decoration = decoration
        self.script = script

    def _valid(self, seen, recursive):
        if len(set(x.identifier for x in self.parameters)) != len(self.parameters):
            raise ValueError("ParameterizedFunction.parameters keys must be unique")
        if recursive:
            _valid(self.parameters, seen, recursive)
            _valid(self.metadata, seen, recursive)
            _valid(self.decoration, seen, recursive)

    def __getitem__(self, where):
        if where == ():
            return self
        elif isinstance(where, tuple):
            head, tail = where[0], where[1:]
        else:
            head, tail = where, ()
        out = _getbykey(self, "parameters", head)
        if tail == ():
            return out
        else:
            return out[tail]

    @classmethod
    def _fromflatbuffers(cls, *args):
        out = cls.__new__(cls)
        out._flatbuffers = _MockFlatbuffers()
        out._flatbuffers.Expression = args[-1].Expression
        out._flatbuffers.Parameters = args[-1].Parameters
        out._flatbuffers.ParametersLength = args[-1].ParametersLength
        out._flatbuffers.Title = args[0].Title
        out._flatbuffers.Metadata = args[0].Metadata
        out._flatbuffers.Decoration = args[0].Decoration
        out._flatbuffers.Script = args[0].Script
        return out

    def _toflatbuffers(self, builder):
        script = None if self.script is None else builder.CreateString(self.script.encode("utf-8"))
        decoration = None if self.decoration is None else self.decoration._toflatbuffers(builder)
        metadata = None if self.metadata is None else self.metadata._toflatbuffers(builder)
        title = None if self.title is None else builder.CreateString(self.title.encode("utf-8"))
        parameters = None if len(self.parameters) == 0 else [x._toflatbuffers(builder) for x in self.parameters]
        expression = builder.CreateString(self.expression.encode("utf-8"))

        if parameters is not None:
            stagg.stagg_generated.ParameterizedFunction.ParameterizedFunctionStartParametersVector(builder, len(parameters))
            for x in parameters[::-1]:
                builder.PrependUOffsetTRelative(x)
            parameters = builder.EndVector(len(parameters))

        stagg.stagg_generated.ParameterizedFunction.ParameterizedFunctionStart(builder)
        stagg.stagg_generated.ParameterizedFunction.ParameterizedFunctionAddExpression(builder, expression)
        if parameters is not None:
            stagg.stagg_generated.ParameterizedFunction.ParameterizedFunctionAddParameters(builder, parameters)
        parameterized = stagg.stagg_generated.ParameterizedFunction.ParameterizedFunctionEnd(builder)

        if isinstance(getattr(self, "_parent", None), Histogram):
            stagg.stagg_generated.Function.FunctionStart(builder)
            stagg.stagg_generated.Function.FunctionAddDataType(builder, stagg.stagg_generated.FunctionData.FunctionData.ParameterizedFunction)
            stagg.stagg_generated.Function.FunctionAddData(builder, parameterized)
            if title is not None:
                stagg.stagg_generated.Function.FunctionAddTitle(builder, title)
            if metadata is not None:
                stagg.stagg_generated.Function.FunctionAddMetadata(builder, metadata)
            if decoration is not None:
                stagg.stagg_generated.Function.FunctionAddDecoration(builder, decoration)
            if script is not None:
                stagg.stagg_generated.Function.FunctionAddScript(builder, script)
            return stagg.stagg_generated.Function.FunctionEnd(builder)

        else:
            stagg.stagg_generated.FunctionObject.FunctionObjectStart(builder)
            stagg.stagg_generated.FunctionObject.FunctionObjectAddDataType(builder, stagg.stagg_generated.FunctionObjectData.FunctionObjectData.ParameterizedFunction)
            stagg.stagg_generated.FunctionObject.FunctionObjectAddData(builder, parameterized)
            function_object = stagg.stagg_generated.FunctionObject.FunctionObjectEnd(builder)

            stagg.stagg_generated.Object.ObjectStart(builder)
            stagg.stagg_generated.Object.ObjectAddDataType(builder, stagg.stagg_generated.ObjectData.ObjectData.FunctionObject)
            stagg.stagg_generated.Object.ObjectAddData(builder, function_object)
            if title is not None:
                stagg.stagg_generated.Object.ObjectAddTitle(builder, title)
            if metadata is not None:
                stagg.stagg_generated.Object.ObjectAddMetadata(builder, metadata)
            if decoration is not None:
                stagg.stagg_generated.Object.ObjectAddDecoration(builder, decoration)
            if script is not None:
                stagg.stagg_generated.Object.ObjectAddScript(builder, script)
            return stagg.stagg_generated.Object.ObjectEnd(builder)
       
################################################# EvaluatedFunction

class EvaluatedFunction(Function):
    _params = {
        "values":      stagg.checktype.CheckClass("EvaluatedFunction", "values", required=True, type=InterpretedBuffer),
        "derivatives": stagg.checktype.CheckClass("EvaluatedFunction", "derivatives", required=False, type=InterpretedBuffer),
        "errors":      stagg.checktype.CheckVector("EvaluatedFunction", "errors", required=False, type=Quantiles),
        "title":       stagg.checktype.CheckString("EvaluatedFunction", "title", required=False),
        "metadata":    stagg.checktype.CheckClass("EvaluatedFunction", "metadata", required=False, type=Metadata),
        "decoration":  stagg.checktype.CheckClass("EvaluatedFunction", "decoration", required=False, type=Decoration),
        "script":      stagg.checktype.CheckString("EvaluatedFunction", "script", required=False),
        }

    values      = typedproperty(_params["values"])
    derivatives = typedproperty(_params["derivatives"])
    errors      = typedproperty(_params["errors"])
    title       = typedproperty(_params["title"])
    metadata    = typedproperty(_params["metadata"])
    decoration  = typedproperty(_params["decoration"])
    script      = typedproperty(_params["script"])

    def __init__(self, values, derivatives=None, errors=None, title=None, metadata=None, decoration=None, script=None):
        self.values = values
        self.derivatives = derivatives
        self.errors = errors
        self.title = title
        self.metadata = metadata
        self.decoration = decoration
        self.script = script

    def _valid(self, seen, recursive):
        if recursive:
            _valid(self.values, seen, recursive)
            _valid(self.derivatives, seen, recursive)
            _valid(self.errors, seen, recursive)
            _valid(self.metadata, seen, recursive)
            _valid(self.decoration, seen, recursive)

    @classmethod
    def _fromflatbuffers(cls, fbfunction, fbevaluated):
        out = cls.__new__(cls)
        out._flatbuffers = _MockFlatbuffers()
        out._flatbuffers.ValuesByTag = _MockFlatbuffers._ByTag(fbevaluated.Values, fbevaluated.ValuesType, _InterpretedBuffer_lookup)
        out._flatbuffers.DerivativesByTag = _MockFlatbuffers._ByTag(fbevaluated.Derivatives, fbevaluated.DerivativesType, _InterpretedBuffer_lookup)
        out._flatbuffers.Errors = fbevaluated.Errors
        out._flatbuffers.ErrorsLength = fbevaluated.ErrorsLength
        out._flatbuffers.Title = fbfunction.Title
        out._flatbuffers.Metadata = fbfunction.Metadata
        out._flatbuffers.Decoration = fbfunction.Decoration
        out._flatbuffers.Script = fbfunction.Script
        return out

    def _toflatbuffers(self, builder):
        values = self.values._toflatbuffers(builder)
        derivatives = None if self.derivatives is None else self.derivatives._toflatbuffers(builder)
        errors = None if len(self.errors) == 0 else [x._toflatbuffers(builder) for x in self.errors]
        script = None if self.script is None else builder.CreateString(self.script.encode("utf-8"))
        decoration = None if self.decoration is None else self.decoration._toflatbuffers(builder)
        metadata = None if self.metadata is None else self.metadata._toflatbuffers(builder)
        title = None if self.title is None else builder.CreateString(self.title.encode("utf-8"))

        if errors is not None:
            stagg.stagg_generated.EvaluatedFunction.EvaluatedFunctionStartErrorsVector(builder, len(errors))
            for x in errors[::-1]:
                builder.PrependUOffsetTRelative(x)
            errors = builder.EndVector(len(errors))

        stagg.stagg_generated.EvaluatedFunction.EvaluatedFunctionStart(builder)
        stagg.stagg_generated.EvaluatedFunction.EvaluatedFunctionAddValuesType(builder, _InterpretedBuffer_invlookup[type(self.values)])
        stagg.stagg_generated.EvaluatedFunction.EvaluatedFunctionAddValues(builder, values)
        if derivatives is not None:
            stagg.stagg_generated.EvaluatedFunction.EvaluatedFunctionAddDerivativesType(builder, _InterpretedBuffer_invlookup[type(self.derivatives)])
            stagg.stagg_generated.EvaluatedFunction.EvaluatedFunctionAddDerivatives(builder, derivatives)
        if errors is not None:
            stagg.stagg_generated.EvaluatedFunction.EvaluatedFunctionAddErrors(builder, errors)
        evaluated = stagg.stagg_generated.EvaluatedFunction.EvaluatedFunctionEnd(builder)

        stagg.stagg_generated.Function.FunctionStart(builder)
        stagg.stagg_generated.Function.FunctionAddDataType(builder, stagg.stagg_generated.FunctionData.FunctionData.EvaluatedFunction)
        stagg.stagg_generated.Function.FunctionAddData(builder, evaluated)
        if title is not None:
            stagg.stagg_generated.Function.FunctionAddTitle(builder, title)
        if metadata is not None:
            stagg.stagg_generated.Function.FunctionAddMetadata(builder, metadata)
        if decoration is not None:
            stagg.stagg_generated.Function.FunctionAddDecoration(builder, decoration)
        if script is not None:
            stagg.stagg_generated.Function.FunctionAddScript(builder, script)
        return stagg.stagg_generated.Function.FunctionEnd(builder)

################################################# BinnedEvaluatedFunction

class BinnedEvaluatedFunction(FunctionObject):
    _params = {
        "axis":        stagg.checktype.CheckVector("BinnedEvaluatedFunction", "axis", required=True, type=Axis, minlen=1),
        "values":      stagg.checktype.CheckClass("BinnedEvaluatedFunction", "values", required=True, type=InterpretedBuffer),
        "derivatives": stagg.checktype.CheckClass("BinnedEvaluatedFunction", "derivatives", required=False, type=InterpretedBuffer),
        "errors":      stagg.checktype.CheckVector("BinnedEvaluatedFunction", "errors", required=False, type=Quantiles),
        "title":       stagg.checktype.CheckString("BinnedEvaluatedFunction", "title", required=False),
        "metadata":    stagg.checktype.CheckClass("BinnedEvaluatedFunction", "metadata", required=False, type=Metadata),
        "decoration":  stagg.checktype.CheckClass("BinnedEvaluatedFunction", "decoration", required=False, type=Decoration),
        "script":      stagg.checktype.CheckString("BinnedEvaluatedFunction", "script", required=False),
        }

    axis        = typedproperty(_params["axis"])
    values      = typedproperty(_params["values"])
    derivatives = typedproperty(_params["derivatives"])
    errors      = typedproperty(_params["errors"])
    title       = typedproperty(_params["title"])
    metadata    = typedproperty(_params["metadata"])
    decoration  = typedproperty(_params["decoration"])
    script      = typedproperty(_params["script"])

    def __init__(self, axis, values, derivatives=None, errors=None, title=None, metadata=None, decoration=None, script=None):
        self.axis = axis
        self.values = values
        self.derivatives = derivatives
        self.errors = errors
        self.title = title
        self.metadata = metadata
        self.decoration = decoration
        self.script = script

    def _valid(self, seen, recursive):
        if recursive:
            _valid(self.axis, seen, recursive)
            _valid(self.values, seen, recursive)
            _valid(self.derivatives, seen, recursive)
            _valid(self.errors, seen, recursive)
            _valid(self.metadata, seen, recursive)
            _valid(self.decoration, seen, recursive)

    def _shape(self, path, shape):
        shape = ()
        if len(path) > 0 and isinstance(path[0], (InterpretedBuffer, Quantiles)):
            for x in self.axis:
                shape = shape + x._binshape()
        return super(BinnedEvaluatedFunction, self)._shape(path, shape)

    @classmethod
    def _fromflatbuffers(cls, fbobject, fbfunction, fbbinned):
        fbevaluated = fbbinned.Data()
        out = cls.__new__(cls)
        out._flatbuffers = _MockFlatbuffers()
        out._flatbuffers.Axis = fbbinned.Axis
        out._flatbuffers.AxisLength = fbbinned.AxisLength
        out._flatbuffers.ValuesByTag = _MockFlatbuffers._ByTag(fbevaluated.Values, fbevaluated.ValuesType, _InterpretedBuffer_lookup)
        out._flatbuffers.DerivativesByTag = _MockFlatbuffers._ByTag(fbevaluated.Derivatives, fbevaluated.DerivativesType, _InterpretedBuffer_lookup)
        out._flatbuffers.Errors = fbevaluated.Errors
        out._flatbuffers.ErrorsLength = fbevaluated.ErrorsLength
        out._flatbuffers.Title = fbobject.Title
        out._flatbuffers.Metadata = fbobject.Metadata
        out._flatbuffers.Decoration = fbobject.Decoration
        out._flatbuffers.Script = fbobject.Script
        return out

    def _toflatbuffers(self, builder):
        values = self.values._toflatbuffers(builder)
        derivatives = None if self.derivatives is None else self.derivatives._toflatbuffers(builder)
        errors = None if len(self.errors) == 0 else [x._toflatbuffers(builder) for x in self.errors]
        script = None if self.script is None else builder.CreateString(self.script.encode("utf-8"))
        decoration = None if self.decoration is None else self.decoration._toflatbuffers(builder)
        metadata = None if self.metadata is None else self.metadata._toflatbuffers(builder)
        title = None if self.title is None else builder.CreateString(self.title.encode("utf-8"))
        axis = [x._toflatbuffers(builder) for x in self.axis]
        
        if errors is not None:
            stagg.stagg_generated.EvaluatedFunction.EvaluatedFunctionStartErrorsVector(builder, len(errors))
            for x in errors[::-1]:
                builder.PrependUOffsetTRelative(x)
            errors = builder.EndVector(len(errors))

        stagg.stagg_generated.EvaluatedFunction.EvaluatedFunctionStart(builder)
        stagg.stagg_generated.EvaluatedFunction.EvaluatedFunctionAddValuesType(builder, _InterpretedBuffer_invlookup[type(self.values)])
        stagg.stagg_generated.EvaluatedFunction.EvaluatedFunctionAddValues(builder, values)
        if derivatives is not None:
            stagg.stagg_generated.EvaluatedFunction.EvaluatedFunctionAddDerivativesType(builder, _InterpretedBuffer_invlookup[type(self.derivatives)])
            stagg.stagg_generated.EvaluatedFunction.EvaluatedFunctionAddDerivatives(builder, derivatives)
        if errors is not None:
            stagg.stagg_generated.EvaluatedFunction.EvaluatedFunctionAddErrors(builder, errors)
        evaluated = stagg.stagg_generated.EvaluatedFunction.EvaluatedFunctionEnd(builder)

        stagg.stagg_generated.BinnedEvaluatedFunction.BinnedEvaluatedFunctionStartAxisVector(builder, len(axis))
        for x in axis[::-1]:
            builder.PrependUOffsetTRelative(x)
        axis = builder.EndVector(len(axis))

        stagg.stagg_generated.BinnedEvaluatedFunction.BinnedEvaluatedFunctionStart(builder)
        stagg.stagg_generated.BinnedEvaluatedFunction.BinnedEvaluatedFunctionAddAxis(builder, axis)
        stagg.stagg_generated.BinnedEvaluatedFunction.BinnedEvaluatedFunctionAddData(builder, evaluated)
        binned_evaluated = stagg.stagg_generated.BinnedEvaluatedFunction.BinnedEvaluatedFunctionEnd(builder)

        stagg.stagg_generated.FunctionObject.FunctionObjectStart(builder)
        stagg.stagg_generated.FunctionObject.FunctionObjectAddDataType(builder, stagg.stagg_generated.FunctionObjectData.FunctionObjectData.BinnedEvaluatedFunction)
        stagg.stagg_generated.FunctionObject.FunctionObjectAddData(builder, binned_evaluated)
        function_object = stagg.stagg_generated.FunctionObject.FunctionObjectEnd(builder)

        stagg.stagg_generated.Object.ObjectStart(builder)
        stagg.stagg_generated.Object.ObjectAddDataType(builder, stagg.stagg_generated.ObjectData.ObjectData.FunctionObject)
        stagg.stagg_generated.Object.ObjectAddData(builder, function_object)
        if title is not None:
            stagg.stagg_generated.Object.ObjectAddTitle(builder, title)
        if metadata is not None:
            stagg.stagg_generated.Object.ObjectAddMetadata(builder, metadata)
        if decoration is not None:
            stagg.stagg_generated.Object.ObjectAddDecoration(builder, decoration)
        if script is not None:
            stagg.stagg_generated.Object.ObjectAddScript(builder, script)
        return stagg.stagg_generated.Object.ObjectEnd(builder)

################################################# Histogram

class Histogram(Object):
    _params = {
        "axis":                stagg.checktype.CheckVector("Histogram", "axis", required=True, type=Axis, minlen=1),
        "counts":              stagg.checktype.CheckClass("Histogram", "counts", required=True, type=Counts),
        "profile":             stagg.checktype.CheckVector("Histogram", "profile", required=False, type=Profile),
        "axis_covariances":    stagg.checktype.CheckVector("Histogram", "axis_covariances", required=False, type=Covariance),
        "profile_covariances": stagg.checktype.CheckVector("Histogram", "profile_covariances", required=False, type=Covariance),
        "functions":           stagg.checktype.CheckLookup("Histogram", "functions", required=False, type=Function),
        "title":               stagg.checktype.CheckString("Histogram", "title", required=False),
        "metadata":            stagg.checktype.CheckClass("Histogram", "metadata", required=False, type=Metadata),
        "decoration":          stagg.checktype.CheckClass("Histogram", "decoration", required=False, type=Decoration),
        "script":              stagg.checktype.CheckString("Histogram", "script", required=False),
        }

    axis                 = typedproperty(_params["axis"])
    counts               = typedproperty(_params["counts"])
    profile              = typedproperty(_params["profile"])
    axis_covariances    = typedproperty(_params["axis_covariances"])
    profile_covariances = typedproperty(_params["profile_covariances"])
    functions            = typedproperty(_params["functions"])
    title                = typedproperty(_params["title"])
    metadata             = typedproperty(_params["metadata"])
    decoration           = typedproperty(_params["decoration"])
    script               = typedproperty(_params["script"])

    def __init__(self, axis, counts, profile=None, axis_covariances=None, profile_covariances=None, functions=None, title=None, metadata=None, decoration=None, script=None):
        self.axis = axis
        self.counts = counts
        self.profile = profile
        self.axis_covariances = axis_covariances
        self.profile_covariances = profile_covariances
        self.functions = functions
        self.title = title
        self.metadata = metadata
        self.decoration = decoration
        self.script = script

    def _valid(self, seen, recursive):
        if len(self.axis_covariances) != 0:
            Covariance._validindexes(self.axis_covariances, len(self.axis))
        if len(self.profile_covariances) != 0:
            Covariance._validindexes(self.profile_covariances, len(self.profile))
        if recursive:
            _valid(self.axis, seen, recursive)
            _valid(self.counts, seen, recursive)
            _valid(self.profile, seen, recursive)
            _valid(self.axis_covariances, seen, recursive)
            _valid(self.profile_covariances, seen, recursive)
            _valid(self.functions, seen, recursive)
            _valid(self.metadata, seen, recursive)
            _valid(self.decoration, seen, recursive)

    def __getitem__(self, where):
        if where == ():
            return self
        elif isinstance(where, tuple):
            head, tail = where[0], where[1:]
        else:
            head, tail = where, ()
        out = _getbykey(self, "functions", head)
        if tail == ():
            return out
        else:
            return out[tail]

    def _shape(self, path, shape):
        shape = ()
        if len(path) > 0 and isinstance(path[0], (Counts, Profile, Function)):
            for x in self.axis:
                shape = shape + x._binshape()
        return super(Histogram, self)._shape(path, shape)

    @classmethod
    def _fromflatbuffers(cls, fbobject, fbhistogram):
        out = cls.__new__(cls)
        out._flatbuffers = _MockFlatbuffers()
        out._flatbuffers.Axis = fbhistogram.Axis
        out._flatbuffers.AxisLength = fbhistogram.AxisLength
        out._flatbuffers.CountsByTag = _MockFlatbuffers._ByTag(fbhistogram.Counts, fbhistogram.CountsType, _Counts_lookup)
        out._flatbuffers.Profile = fbhistogram.Profile
        out._flatbuffers.ProfileLength = fbhistogram.ProfileLength
        out._flatbuffers.AxisCovariances = fbhistogram.AxisCovariances
        out._flatbuffers.AxisCovariancesLength = fbhistogram.AxisCovariancesLength
        out._flatbuffers.ProfileCovariances = fbhistogram.ProfileCovariances
        out._flatbuffers.ProfileCovariancesLength = fbhistogram.ProfileCovariancesLength
        out._flatbuffers.Functions = fbhistogram.Functions
        out._flatbuffers.FunctionsLength = fbhistogram.FunctionsLength
        out._flatbuffers.FunctionsLookup = fbhistogram.FunctionsLookup
        out._flatbuffers.Title = fbobject.Title
        out._flatbuffers.Metadata = fbobject.Metadata
        out._flatbuffers.Decoration = fbobject.Decoration
        out._flatbuffers.Script = fbobject.Script
        return out

    def _toflatbuffers(self, builder):
        counts = self.counts._toflatbuffers(builder)
        functions = None if len(self.functions) == 0 else [x._toflatbuffers(builder) for x in self.functions.values()]
        script = None if self.script is None else builder.CreateString(self.script.encode("utf-8"))
        decoration = None if self.decoration is None else self.decoration._toflatbuffers(builder)
        metadata = None if self.metadata is None else self.metadata._toflatbuffers(builder)
        profile_covariances = None if len(self.profile_covariances) == 0 else [x._toflatbuffers(builder) for x in self.profile_covariances]
        axis_covariances = None if len(self.axis_covariances) == 0 else [x._toflatbuffers(builder) for x in self.axis_covariances]
        title = None if self.title is None else builder.CreateString(self.title.encode("utf-8"))
        profile = None if len(self.profile) == 0 else [x._toflatbuffers(builder) for x in self.profile]
        axis = [x._toflatbuffers(builder) for x in self.axis]

        stagg.stagg_generated.Histogram.HistogramStartAxisVector(builder, len(axis))
        for x in axis[::-1]:
            builder.PrependUOffsetTRelative(x)
        axis = builder.EndVector(len(axis))

        if profile is not None:
            stagg.stagg_generated.Histogram.HistogramStartProfileVector(builder, len(profile))
            for x in profile[::-1]:
                builder.PrependUOffsetTRelative(x)
            profile = builder.EndVector(len(profile))

        if axis_covariances is not None:
            stagg.stagg_generated.Histogram.HistogramStartAxisCovariancesVector(builder, len(axis_covariances))
            for x in axis_covariances[::-1]:
                builder.PrependUOffsetTRelative(x)
            axis_covariances = builder.EndVector(len(axis_covariances))

        if profile_covariances is not None:
            stagg.stagg_generated.Histogram.HistogramStartProfileCovariancesVector(builder, len(profile_covariances))
            for x in profile_covariances[::-1]:
                builder.PrependUOffsetTRelative(x)
            profile_covariances = builder.EndVector(len(profile_covariances))

        if functions is not None:
            stagg.stagg_generated.Histogram.HistogramStartFunctionsVector(builder, len(functions))
            for x in functions[::-1]:
                builder.PrependUOffsetTRelative(x)
            functions = builder.EndVector(len(functions))

        functions_lookup = None if len(self.functions) == 0 else [builder.CreateString(n.encode("utf-8")) for n in self.functions.keys()]
        if functions_lookup is not None:
            stagg.stagg_generated.Histogram.HistogramStartFunctionsLookupVector(builder, len(functions_lookup))
            for x in functions_lookup[::-1]:
                builder.PrependUOffsetTRelative(x)
            functions_lookup = builder.EndVector(len(functions_lookup))

        stagg.stagg_generated.Histogram.HistogramStart(builder)
        stagg.stagg_generated.Histogram.HistogramAddAxis(builder, axis)
        stagg.stagg_generated.Histogram.HistogramAddCountsType(builder, _Counts_invlookup[type(self.counts)])
        stagg.stagg_generated.Histogram.HistogramAddCounts(builder, counts)
        if profile is not None:
            stagg.stagg_generated.Histogram.HistogramAddProfile(builder, profile)
        if axis_covariances is not None:
            stagg.stagg_generated.Histogram.HistogramAddAxisCovariances(builder, axis_covariances)
        if profile_covariances is not None:
            stagg.stagg_generated.Histogram.HistogramAddProfileCovariances(builder, profile_covariances)
        if functions is not None:
            stagg.stagg_generated.Histogram.HistogramAddFunctionsLookup(builder, functions_lookup)
            stagg.stagg_generated.Histogram.HistogramAddFunctions(builder, functions)
        data = stagg.stagg_generated.Histogram.HistogramEnd(builder)

        stagg.stagg_generated.Object.ObjectStart(builder)
        stagg.stagg_generated.Object.ObjectAddDataType(builder, stagg.stagg_generated.ObjectData.ObjectData.Histogram)
        stagg.stagg_generated.Object.ObjectAddData(builder, data)
        if title is not None:
            stagg.stagg_generated.Object.ObjectAddTitle(builder, title)
        if metadata is not None:
            stagg.stagg_generated.Object.ObjectAddMetadata(builder, metadata)
        if decoration is not None:
            stagg.stagg_generated.Object.ObjectAddDecoration(builder, decoration)
        if script is not None:
            stagg.stagg_generated.Object.ObjectAddScript(builder, script)
        return stagg.stagg_generated.Object.ObjectEnd(builder)

    def _add(self, other, shape, noclobber):
        if not isinstance(other, Histogram):
            raise ValueError("cannot add {0} and {1}".format(self, other))
        # FIXME: check axis
        # FIXME: add counts
        # FIXME: add profile, if applicable
        # FIXME: add axis_covariances
        # FIXME: add profile_covariances
        # FIXME: add functions, if applicable

################################################# Page

class Page(Stagg):
    _params = {
        "buffer": stagg.checktype.CheckClass("Page", "buffer", required=True, type=RawBuffer),
        }

    buffer = typedproperty(_params["buffer"])

    def __init__(self, buffer):
        self.buffer = buffer

    def _valid(self, seen, recursive):
        self.array

    def numentries(self):
        if not hasattr(self, "_parent"):
            raise ValueError("Page not attached to a hierarchy")
        for pageid, page in enumerate(self._parent.pages):
            if self is page:
                break
        else:
            raise AssertionError("Page not in its own parent's pages list")

        return self._parent.numentries(pageid)

    @property
    def column(self):
        if not hasattr(self, "_parent"):
            raise ValueError("Page not attached to a hierarchy")
        return self._parent.column

    @property
    def array(self):
        array = self.buffer.array
        column = self.column

        if len(column.filters) != 0:
            raise NotImplementedError("handle column.filters")

        if column.postfilter_slice is not None:
            start = column.postfilter_slice.start if column.postfilter_slice.has_start else None
            stop = column.postfilter_slice.stop if column.postfilter_slice.has_stop else None
            step = column.postfilter_slice.step if column.postfilter_slice.has_step else None
            if step == 0:
                raise ValueError("slice step cannot be zero")
            array = array[start:stop:step]

        numentries = self.numentries()
        itemsize = self.column.numpy_dtype.itemsize
        if len(array) != numentries * itemsize:
            raise ValueError("Page array has {0} bytes but this page has {1} entries with {2} bytes each".format(len(array), numentries, itemsize))

        return array.view(column.numpy_dtype).reshape((numentries,))

    @classmethod
    def _fromflatbuffers(cls, fb):
        out = cls.__new__(cls)
        out._flatbuffers = _MockFlatbuffers()
        out._flatbuffers.BufferByTag = _MockFlatbuffers._ByTag(fb.Buffer, fb.BufferType, _RawBuffer_lookup)
        return out

    def _toflatbuffers(self, builder):
        buffer = self.buffer._toflatbuffers(builder)
        stagg.stagg_generated.Page.PageStart(builder)
        stagg.stagg_generated.Page.PageAddBufferType(builder, _RawBuffer_invlookup[type(self.buffer)])
        stagg.stagg_generated.Page.PageAddBuffer(builder, buffer)
        return stagg.stagg_generated.Page.PageEnd(builder)

################################################# ColumnChunk

class ColumnChunk(Stagg):
    _params = {
        "pages":        stagg.checktype.CheckVector("ColumnChunk", "pages", required=True, type=Page),
        "page_offsets": stagg.checktype.CheckVector("ColumnChunk", "page_offsets", required=True, type=int, minlen=1),
        "page_min":  stagg.checktype.CheckVector("ColumnChunk", "page_min", required=False, type=Extremes),
        "page_max":  stagg.checktype.CheckVector("ColumnChunk", "page_max", required=False, type=Extremes),
        }

    pages        = typedproperty(_params["pages"])
    page_offsets = typedproperty(_params["page_offsets"])
    page_min  = typedproperty(_params["page_min"])
    page_max  = typedproperty(_params["page_max"])

    def __init__(self, pages, page_offsets, page_min=None, page_max=None):
        self.pages = pages
        self.page_offsets = page_offsets
        self.page_min = page_min
        self.page_max = page_max

    def _valid(self, seen, recursive):
        if self.page_offsets[0] != 0:
            raise ValueError("ColumnChunk.page_offsets must start with 0")
        if not numpy.greater_equal(self.page_offsets[1:], self.page_offsets[:-1]).all():
            raise ValueError("ColumnChunk.page_offsets must be monotonically increasing")
        if len(self.page_offsets) != len(self.pages) + 1:
            raise ValueError("ColumnChunk.page_offsets length is {0}, but it must be one longer than ColumnChunk.pages, which is {1}".format(len(self.page_offsets), len(self.pages)))
        if len(self.page_min) != 0:
            if len(self.page_min) != len(self.pages):
                raise ValueError("ColumnChunk.page_extremes length {0} must be equal to ColumnChunk.pages length {1}".format(len(self.page_min), len(self.pages)))
            raise NotImplementedError("check min")
        if len(self.page_max) != 0:
            if len(self.page_max) != len(self.pages):
                raise ValueError("ColumnChunk.page_extremes length {0} must be equal to ColumnChunk.pages length {1}".format(len(self.page_max), len(self.pages)))
            raise NotImplementedError("check max")
        if recursive:
            _valid(self.pages, seen, recursive)
            _valid(self.page_min, seen, recursive)
            _valid(self.page_max, seen, recursive)
        
    def numentries(self, pageid=None):
        if pageid is None:
            return self.page_offsets[-1]
        elif isinstance(pageid, (numbers.Integral, numpy.integer)):
            original_pageid = pageid
            if pageid < 0:
                pageid += len(self.page_offsets) - 1
            if not 0 <= pageid < len(self.page_offsets) - 1:
                raise IndexError("pageid {0} out of range for {1} pages".format(original_pageid, len(self.page_offsets) - 1))
            return self.page_offsets[pageid + 1] - self.page_offsets[pageid]
        else:
            raise TypeError("pageid must be None (for total number of entries) or an integer (for number of entries in a page)")

    @property
    def column(self):
        if not hasattr(self, "_parent"):
            raise ValueError("ColumnChunk not attached to a hierarchy")
        if not hasattr(self._parent, "_parent"):
            raise ValueError("{0} not attached to a hierarchy".format(type(self._parent)))
        if not hasattr(self._parent._parent, "_parent"):
            raise ValueError("{0} not attached to a hierarchy".format(type(self._parent._parent)))
        
        for columnid, columnchunk in enumerate(self._parent.column_chunks):
            if self is columnchunk:
                break
        else:
            raise AssertionError("ColumnChunk not in its own parent's columns list")

        return self._parent._parent._parent.columns[columnid]   # FIXME: go through intermediate columns properties

    @property
    def array(self):
        out = [x.array for x in self.pages]
        if len(out) == 0:
            return numpy.empty(0, self.column.numpy_dtype)
        elif len(out) == 1:
            return out[0]
        else:
            return numpy.concatenate(out)

    def _toflatbuffers(self, builder):
        pages = [x._toflatbuffers(builder) for x in self.pages]
        page_min = None if len(self.page_min) == 0 else [x._toflatbuffers(builder) for x in self.page_min]
        page_max = None if len(self.page_max) == 0 else [x._toflatbuffers(builder) for x in self.page_max]

        stagg.stagg_generated.ColumnChunk.ColumnChunkStartPagesVector(builder, len(pages))
        for x in pages[::-1]:
            builder.PrependUOffsetTRelative(x)
        pages = builder.EndVector(len(pages))

        stagg.stagg_generated.ColumnChunk.ColumnChunkStartPageOffsetsVector(builder, len(self.page_offsets))
        for x in self.page_offsets[::-1]:
            builder.PrependInt64(x)
        page_offsets = builder.EndVector(len(self.page_offsets))

        if page_min is not None:
            stagg.stagg_generated.ColumnChunk.ColumnChunkStartPageMinVector(builder, len(page_min))
            for x in page_min[::-1]:
                builder.PrependUOffsetTRelative(x)
            page_min = builder.EndVector(len(page_min))

        if page_max is not None:
            stagg.stagg_generated.ColumnChunk.ColumnChunkStartPageMaxVector(builder, len(page_max))
            for x in page_max[::-1]:
                builder.PrependUOffsetTRelative(x)
            page_max = builder.EndVector(len(page_max))

        stagg.stagg_generated.ColumnChunk.ColumnChunkStart(builder)
        stagg.stagg_generated.ColumnChunk.ColumnChunkAddPages(builder, pages)
        stagg.stagg_generated.ColumnChunk.ColumnChunkAddPageOffsets(builder, page_offsets)
        if page_min is not None:
            stagg.stagg_generated.ColumnChunk.ColumnChunkAddPageMin(builder, page_min)
        if page_max is not None:
            stagg.stagg_generated.ColumnChunk.ColumnChunkAddPageMax(builder, page_max)
        return stagg.stagg_generated.ColumnChunk.ColumnChunkEnd(builder)
        
################################################# Chunk

class Chunk(Stagg):
    _params = {
        "column_chunks": stagg.checktype.CheckVector("Chunk", "column_chunks", required=True, type=ColumnChunk),
        "metadata":      stagg.checktype.CheckClass("Chunk", "metadata", required=False, type=Metadata),
        }

    column_chunks = typedproperty(_params["column_chunks"])
    metadata      = typedproperty(_params["metadata"])

    def __init__(self, column_chunks, metadata=None):
        self.column_chunks = column_chunks
        self.metadata = metadata

    def _valid(self, seen, recursive):
        if len(self.column_chunks) != len(self.columns):
            raise ValueError("Chunk.column_chunks has length {0}, but Ntuple.columns has length {1}".format(len(self.column_chunks), len(self.columns)))
        if recursive:
            _valid(self.column_chunks, seen, recursive)
            _valid(self.metadata, seen, recursive)

    @property
    def columns(self):
        if not hasattr(self, "_parent"):
            raise ValueError("Chunk not attached to a hierarchy")
        return self._parent.columns

    @property
    def arrays(self):
        if not isinstance(getattr(self, "_parent", None), NtupleInstance) or not isinstance(getattr(self._parent, "_parent", None), Ntuple):
            raise ValueError("{0} object is not nested in a hierarchy".format(type(self).__name__))
        if len(self.column_chunks) != len(self._parent._parent.columns):
            raise ValueError("Chunk.columns has length {0}, but Ntuple.columns has length {1}".format(len(self.column_chunks), len(self._parent._parent.columns)))
        return {y.identifier: x.array for x, y in zip(self.column_chunks, self._parent._parent.columns)}

    def _toflatbuffers(self, builder):
        column_chunks = [x._toflatbuffers(builder) for x in self.column_chunks]
        metadata = None if self.metadata is None else self.metadata._toflatbuffers(builder)

        stagg.stagg_generated.Chunk.ChunkStartColumnChunksVector(builder, len(column_chunks))
        for x in column_chunks[::-1]:
            builder.PrependUOffsetTRelative(x)
        column_chunks = builder.EndVector(len(column_chunks))

        stagg.stagg_generated.Chunk.ChunkStart(builder)
        stagg.stagg_generated.Chunk.ChunkAddColumnChunks(builder, column_chunks)
        if metadata is not None:
            stagg.stagg_generated.Chunk.ChunkAddMetadata(builder, metadata)
        return stagg.stagg_generated.Chunk.ChunkEnd(builder)

################################################# Column

class Column(Stagg, Interpretation):
    _params = {
        "identifier":       stagg.checktype.CheckKey("Column", "identifier", required=True, type=str),
        "dtype":            stagg.checktype.CheckEnum("Column", "dtype", required=True, choices=Interpretation.dtypes),
        "endianness":       stagg.checktype.CheckEnum("Column", "endianness", required=False, choices=Interpretation.endiannesses),
        "filters":          stagg.checktype.CheckVector("Column", "filters", required=False, type=Buffer.filters),
        "postfilter_slice": stagg.checktype.CheckSlice("Column", "postfilter_slice", required=False),
        "title":            stagg.checktype.CheckString("Column", "title", required=False),
        "metadata":         stagg.checktype.CheckClass("Column", "metadata", required=False, type=Metadata),
        "decoration":       stagg.checktype.CheckClass("Column", "decoration", required=False, type=Decoration),
        }

    identifier       = typedproperty(_params["identifier"])
    dtype            = typedproperty(_params["dtype"])
    endianness       = typedproperty(_params["endianness"])
    filters          = typedproperty(_params["filters"])
    postfilter_slice = typedproperty(_params["postfilter_slice"])
    title            = typedproperty(_params["title"])
    metadata         = typedproperty(_params["metadata"])
    decoration       = typedproperty(_params["decoration"])

    def __init__(self, identifier, dtype, endianness=InterpretedBuffer.little_endian, dimension_order=InterpretedBuffer.c_order, filters=None, postfilter_slice=None, title=None, metadata=None, decoration=None):
        self.identifier = identifier
        self.dtype = dtype
        self.endianness = endianness
        self.filters = filters
        self.postfilter_slice = postfilter_slice
        self.title = title
        self.metadata = metadata
        self.decoration = decoration

    def _valid(self, seen, recursive):
        if self.postfilter_slice is not None and self.postfilter_slice.has_step and self.postfilter_slice.step == 0:
            raise ValueError("slice step cannot be zero")
        if recursive:
            _valid(self.metadata, seen, recursive)
            _valid(self.decoration, seen, recursive)

    def _toflatbuffers(self, builder):
        decoration = None if self.decoration is None else self.decoration._toflatbuffers(builder)
        metadata = None if self.metadata is None else self.metadata._toflatbuffers(builder)
        title = None if self.title is None else builder.CreateString(self.title.encode("utf-8"))
        identifier = builder.CreateString(self.identifier.encode("utf-8"))

        if len(self.filters) == 0:
            filters = None
        else:
            stagg.stagg_generated.Column.ColumnStartFiltersVector(builder, len(self.filters))
            for x in self.filters[::-1]:
                builder.PrependUInt32(x.value)
            filters = builder.EndVector(len(self.filters))

        stagg.stagg_generated.Column.ColumnStart(builder)
        stagg.stagg_generated.Column.ColumnAddIdentifier(builder, identifier)
        stagg.stagg_generated.Column.ColumnAddDtype(builder, self.dtype.value)
        if self.endianness != InterpretedBuffer.little_endian:
            stagg.stagg_generated.Column.ColumnAddEndianness(builder, self.endianness.value)
        if filters is not None:
            stagg.stagg_generated.Column.ColumnAddFilters(builder, self.filters)
        if self.postfilter_slice is not None:
            stagg.stagg_generated.Column.ColumnAddPostfilterSlice(builder, stagg.stagg_generated.Slice.CreateSlice(builder, self.postfilter_slice.start, self.postfilter_slice.stop, self.postfilter_slice.step, self.postfilter_slice.hasStart, self.postfilter_slice.hasStop, self.postfilter_slice.hasStep))
        if title is not None:
            stagg.stagg_generated.Column.ColumnAddTitle(builder, title)
        if metadata is not None:
            stagg.stagg_generated.Column.ColumnAddMetadata(builder, metadata)
        if decoration is not None:
            stagg.stagg_generated.Column.ColumnAddDecoration(builder, decoration)
        return stagg.stagg_generated.Column.ColumnEnd(builder)

################################################# NtupleInstance

class NtupleInstance(Stagg):
    _params = {
        "chunks":        stagg.checktype.CheckVector("NtupleInstance", "chunks", required=True, type=Chunk),
        "chunk_offsets": stagg.checktype.CheckVector("NtupleInstance", "chunk_offsets", required=False, type=int),
        }

    chunks              = typedproperty(_params["chunks"])
    chunk_offsets       = typedproperty(_params["chunk_offsets"])

    def __init__(self, chunks, chunk_offsets=None):
        self.chunks = chunks
        self.chunk_offsets = chunk_offsets

    def _valid(self, seen, recursive):
        if not isinstance(getattr(self, "_parent", None), Ntuple):
            raise ValueError("{0} object is not nested in a hierarchy".format(type(self).__name__))
        if len(self.chunk_offsets) != 0:
            if self.chunk_offsets[0] != 0:
                raise ValueError("Ntuple.chunk_offsets must start with 0")
            if not numpy.greater_equal(self.chunk_offsets[1:], self.chunk_offsets[:-1]).all():
                raise ValueError("Ntuple.chunk_offsets must be monotonically increasing")

            if len(self.chunk_offsets) != len(self.chunks) + 1:
                raise ValueError("Ntuple.chunk_offsets length is {0}, but it must be one longer than Ntuple.chunks, which is {1}".format(len(self.chunk_offsets), len(self.chunks)))
        if recursive:
            _valid(self.chunks, seen, recursive)

    @property
    def columns(self):
        if not hasattr(self, "_parent"):
            raise ValueError("NtupleInstance not attached to a hierarchy")
        return self._parent.columns

    def numentries(self, chunkid=None):
        if len(self.chunk_offsets) == 0:
            if chunkid is None:
                return sum(x.numentries() for x in self.chunks)
            else:
                return self.chunks[chunkid].numentries()

        else:
            if chunkid is None:
                return self.chunk_offsets[-1]
            elif isinstance(chunkid, (numbers.Integral, numpy.integer)):
                original_chunkid = chunkid
                if chunkid < 0:
                    chunkid += len(self.chunk_offsets) - 1
                if not 0 <= chunkid < len(self.chunk_offsets) - 1:
                    raise IndexError("chunkid {0} out of range for {1} chunks".format(original_chunkid, len(self.chunk_offsets) - 1))
                return self.chunk_offsets[chunkid + 1] - self.chunk_offsets[chunkid]
            else:
                raise TypeError("chunkid must be None (for total number of entries) or an integer (for number of entries in a chunk)")

    @property
    def arrays(self):
        for x in self.chunks:
            yield x.arrays

    def _toflatbuffers(self, builder):
        chunks = [x._toflatbuffers(builder) for x in self.chunks]

        stagg.stagg_generated.NtupleInstance.NtupleInstanceStartChunksVector(builder, len(chunks))
        for x in chunks[::-1]:
            builder.PrependUOffsetTRelative(x)
        chunks = builder.EndVector(len(chunks))

        if len(self.chunk_offsets) == 0:
            chunk_offsets = None
        else:
            stagg.stagg_generated.NtupleInstance.NtupleInstanceStartChunkOffsetsVector(builder, len(self.chunk_offsets))
            for x in self.chunk_offsets[::-1]:
                builder.PrependInt64(x)
            chunk_offsets = builder.EndVector(len(self.chunk_offsets))

        stagg.stagg_generated.NtupleInstance.NtupleInstanceStart(builder)
        stagg.stagg_generated.NtupleInstance.NtupleInstanceAddChunks(builder, chunks)
        if chunk_offsets is not None:
            stagg.stagg_generated.NtupleInstance.NtupleInstanceAddChunkOffsets(builder, chunk_offsets)
        return stagg.stagg_generated.NtupleInstance.NtupleInstanceEnd(builder)

################################################# Ntuple

class Ntuple(Object):
    _params = {
        "columns":            stagg.checktype.CheckVector("Ntuple", "columns", required=True, type=Column, minlen=1),
        "instances":          stagg.checktype.CheckVector("Ntuple", "instances", required=True, type=NtupleInstance, minlen=1),
        "column_statistics":  stagg.checktype.CheckVector("Ntuple", "column_statistics", required=False, type=Statistics),
        "column_covariances": stagg.checktype.CheckVector("Ntuple", "column_covariances", required=False, type=Covariance),
        "functions":          stagg.checktype.CheckLookup("Ntuple", "functions", required=False, type=FunctionObject),
        "title":              stagg.checktype.CheckString("Ntuple", "title", required=False),
        "metadata":           stagg.checktype.CheckClass("Ntuple", "metadata", required=False, type=Metadata),
        "decoration":         stagg.checktype.CheckClass("Ntuple", "decoration", required=False, type=Decoration),
        "script":             stagg.checktype.CheckString("Ntuple", "script", required=False),
        }

    columns             = typedproperty(_params["columns"])
    instances           = typedproperty(_params["instances"])
    column_statistics   = typedproperty(_params["column_statistics"])
    column_covariances = typedproperty(_params["column_covariances"])
    functions           = typedproperty(_params["functions"])
    title               = typedproperty(_params["title"])
    metadata            = typedproperty(_params["metadata"])
    decoration          = typedproperty(_params["decoration"])
    script              = typedproperty(_params["script"])

    def __init__(self, columns, instances, column_statistics=None, column_covariances=None, functions=None, title=None, metadata=None, decoration=None, script=None):
        self.columns = columns
        self.instances = instances
        self.column_statistics = column_statistics
        self.column_covariances = column_covariances
        self.functions = functions
        self.title = title
        self.metadata = metadata
        self.decoration = decoration
        self.script = script

    def _valid(self, seen, recursive):
        shape = self._shape((), ())
        if len(set(x.identifier for x in self.columns)) != len(self.columns):
            raise ValueError("Ntuple.columns keys must be unique")
        if len(self.instances) != functools.reduce(operator.mul, shape, 1):
            raise ValueError("Ntuple.instances length is {0} but multiplicity at this location in the hierarchy is {1}".format(len(self.instances), functools.reduce(operator.mul, shape, 1)))
        if len(self.column_covariances) != 0:
            Covariance._validindexes(self.column_covariances, len(self.columns))
        if recursive:
            _valid(self.columns, seen, recursive)
            _valid(self.instances, seen, recursive)
            _valid(self.column_statistics, seen, recursive)
            _valid(self.column_covariances, seen, recursive)
            _valid(self.functions, seen, recursive)
            _valid(self.metadata, seen, recursive)
            _valid(self.decoration, seen, recursive)

    def __getitem__(self, where):
        if where == ():
            return self
        elif isinstance(where, tuple):
            head, tail = where[0], where[1:]
        else:
            head, tail = where, ()
        out = _getbykey(self, "columns", head)
        if tail == ():
            return out
        else:
            return out[tail]

    @classmethod
    def _fromflatbuffers(cls, fbobject, fbntuple):
        out = cls.__new__(cls)
        out._flatbuffers = _MockFlatbuffers()
        out._flatbuffers.Columns = fbntuple.Columns
        out._flatbuffers.ColumnsLength = fbntuple.ColumnsLength
        out._flatbuffers.Instances = fbntuple.Instances
        out._flatbuffers.InstancesLength = fbntuple.InstancesLength
        out._flatbuffers.ColumnStatistics = fbntuple.ColumnStatistics
        out._flatbuffers.ColumnStatisticsLength = fbntuple.ColumnStatisticsLength
        out._flatbuffers.ColumnCovariances = fbntuple.ColumnCovariances
        out._flatbuffers.ColumnCovariancesLength = fbntuple.ColumnCovariancesLength
        out._flatbuffers.Functions = fbntuple.Functions
        out._flatbuffers.FunctionsLength = fbntuple.FunctionsLength
        out._flatbuffers.FunctionsLookup = fbntuple.FunctionsLookup
        out._flatbuffers.Title = fbobject.Title
        out._flatbuffers.Metadata = fbobject.Metadata
        out._flatbuffers.Decoration = fbobject.Decoration
        out._flatbuffers.Script = fbobject.Script
        return out

    def _toflatbuffers(self, builder):
        instances = [x._toflatbuffers(builder) for x in self.instances]
        functions = None if len(self.functions) == 0 else [x._toflatbuffers(builder) for x in self.functions.values()]
        column_statistics = None if len(self.column_statistics) == 0 else [x._toflatbuffers(builder) for x in self.column_statistics]
        column_covariances = None if len(self.column_covariances) == 0 else [x._toflatbuffers(builder) for x in self.column_covariances]
        script = None if self.script is None else builder.CreateString(self.script.encode("utf-8"))
        decoration = None if self.decoration is None else self.decoration._toflatbuffers(builder)
        metadata = None if self.metadata is None else self.metadata._toflatbuffers(builder)
        title = None if self.title is None else builder.CreateString(self.title.encode("utf-8"))
        columns = [x._toflatbuffers(builder) for x in self.columns]

        stagg.stagg_generated.Ntuple.NtupleStartColumnsVector(builder, len(columns))
        for x in columns[::-1]:
            builder.PrependUOffsetTRelative(x)
        columns = builder.EndVector(len(columns))

        stagg.stagg_generated.Ntuple.NtupleStartInstancesVector(builder, len(instances))
        for x in instances[::-1]:
            builder.PrependUOffsetTRelative(x)
        instances = builder.EndVector(len(instances))

        if column_statistics is not None:
            stagg.stagg_generated.Ntuple.NtupleStartColumnStatisticsVector(builder, len(column_statistics))
            for x in column_statistics[::-1]:
                builder.PrependUOffsetTRelative(x)
            column_statistics = builder.EndVector(len(column_statistics))

        if column_covariances is not None:
            stagg.stagg_generated.Ntuple.NtupleStartColumnCovariancesVector(builder, len(column_covariances))
            for x in column_covariances[::-1]:
                builder.PrependUOffsetTRelative(x)
            column_covariances = builder.EndVector(len(column_covariances))

        if functions is not None:
            stagg.stagg_generated.Ntuple.NtupleStartFunctionsVector(builder, len(functions))
            for x in functions[::-1]:
                builder.PrependUOffsetTRelative(x)
            functions = builder.EndVector(len(functions))

        functions_lookup = None if len(self.functions) == 0 else [builder.CreateString(n.encode("utf-8")) for n in self.functions.keys()]
        if functions_lookup is not None:
            stagg.stagg_generated.Ntuple.NtupleStartFunctionsLookupVector(builder, len(functions_lookup))
            for x in functions_lookup[::-1]:
                builder.PrependUOffsetTRelative(x)
            functions_lookup = builder.EndVector(len(functions_lookup))

        stagg.stagg_generated.Ntuple.NtupleStart(builder)
        stagg.stagg_generated.Ntuple.NtupleAddColumns(builder, columns)
        stagg.stagg_generated.Ntuple.NtupleAddInstances(builder, instances)
        if column_statistics is not None:
            stagg.stagg_generated.Ntuple.NtupleAddColumnStatistics(builder, column_statistics)
        if column_covariances is not None:
            stagg.stagg_generated.Ntuple.NtupleAddColumnCovariances(builder, column_covariances)
        if functions is not None:
            stagg.stagg_generated.Ntuple.NtupleAddFunctionsLookup(builder, functions_lookup)
            stagg.stagg_generated.Ntuple.NtupleAddFunctions(builder, functions)
        data = stagg.stagg_generated.Ntuple.NtupleEnd(builder)

        stagg.stagg_generated.Object.ObjectStart(builder)
        stagg.stagg_generated.Object.ObjectAddDataType(builder, stagg.stagg_generated.ObjectData.ObjectData.Ntuple)
        stagg.stagg_generated.Object.ObjectAddData(builder, data)
        if title is not None:
            stagg.stagg_generated.Object.ObjectAddTitle(builder, title)
        if metadata is not None:
            stagg.stagg_generated.Object.ObjectAddMetadata(builder, metadata)
        if decoration is not None:
            stagg.stagg_generated.Object.ObjectAddDecoration(builder, decoration)
        if script is not None:
            stagg.stagg_generated.Object.ObjectAddScript(builder, script)
        return stagg.stagg_generated.Object.ObjectEnd(builder)

################################################# Collection

class Collection(Object):
    _params = {
        "objects":    stagg.checktype.CheckLookup("Collection", "objects", required=False, type=Object),
        "axis":       stagg.checktype.CheckVector("Collection", "axis", required=False, type=Axis),
        "title":      stagg.checktype.CheckString("Collection", "title", required=False),
        "metadata":   stagg.checktype.CheckClass("Collection", "metadata", required=False, type=Metadata),
        "decoration": stagg.checktype.CheckClass("Collection", "decoration", required=False, type=Decoration),
        "script":     stagg.checktype.CheckString("Collection", "script", required=False),
        }

    objects        = typedproperty(_params["objects"])
    axis           = typedproperty(_params["axis"])
    title          = typedproperty(_params["title"])
    metadata       = typedproperty(_params["metadata"])
    decoration     = typedproperty(_params["decoration"])
    script         = typedproperty(_params["script"])

    def __init__(self, objects=None, axis=None, title=None, metadata=None, decoration=None, script=None):
        self.objects = objects
        self.axis = axis
        self.title = title
        self.metadata = metadata
        self.decoration = decoration
        self.script = script

    def _valid(self, seen, recursive):
        if recursive:
            _valid(self.objects, seen, recursive)
            _valid(self.axis, seen, recursive)
            _valid(self.metadata, seen, recursive)
            _valid(self.decoration, seen, recursive)

    def __getitem__(self, where):
        if where == ():
            return self
        elif isinstance(where, tuple):
            head, tail = where[0], where[1:]
        else:
            head, tail = where, ()
        out = _getbykey(self, "objects", head)
        if tail == ():
            return out
        else:
            return out[tail]

    def _shape(self, path, shape):
        axisshape = ()
        if len(path) > 0 and isinstance(path[0], Object):
            for x in self.axis:
                axisshape = axisshape + x._binshape()
        shape = axisshape + shape
        return super(Collection, self)._shape(path, shape)

    @classmethod
    def _fromflatbuffers(cls, fbobject, fbcollection):
        out = cls.__new__(cls)
        out._flatbuffers = _MockFlatbuffers()
        out._flatbuffers.Objects = fbcollection.Objects
        out._flatbuffers.ObjectsLength = fbcollection.ObjectsLength
        out._flatbuffers.ObjectsLookup = fbcollection.Lookup
        out._flatbuffers.Axis = fbcollection.Axis
        out._flatbuffers.AxisLength = fbcollection.AxisLength
        out._flatbuffers.Title = fbobject.Title
        out._flatbuffers.Metadata = fbobject.Metadata
        out._flatbuffers.Decoration = fbobject.Decoration
        out._flatbuffers.Script = fbobject.Script
        return out

    def _toflatbuffers(self, builder):
        objects = None if len(self.objects) == 0 else [x._toflatbuffers(builder) for x in self.objects.values()]
        script = None if self.script is None else builder.CreateString(self.script.encode("utf-8"))
        decoration = None if self.decoration is None else self.decoration._toflatbuffers(builder)
        metadata = None if self.metadata is None else self.metadata._toflatbuffers(builder)
        title = None if self.title is None else builder.CreateString(self.title.encode("utf-8"))
        axis = None if len(self.axis) == 0 else [x._toflatbuffers(builder) for x in self.axis]

        if axis is not None:
            stagg.stagg_generated.Collection.CollectionStartAxisVector(builder, len(axis))
            for x in axis[::-1]:
                builder.PrependUOffsetTRelative(x)
            axis = builder.EndVector(len(axis))

        if objects is not None:
            stagg.stagg_generated.Collection.CollectionStartObjectsVector(builder, len(objects))
            for x in objects[::-1]:
                builder.PrependUOffsetTRelative(x)
            objects = builder.EndVector(len(objects))

        lookup = None if len(self.objects) == 0 else [builder.CreateString(n.encode("utf-8")) for n in self.objects.keys()]
        if lookup is not None:
            stagg.stagg_generated.Collection.CollectionStartLookupVector(builder, len(lookup))
            for x in lookup[::-1]:
                builder.PrependUOffsetTRelative(x)
            lookup = builder.EndVector(len(lookup))

        stagg.stagg_generated.Collection.CollectionStart(builder)
        if objects is not None:
            stagg.stagg_generated.Collection.CollectionAddLookup(builder, lookup)
            stagg.stagg_generated.Collection.CollectionAddObjects(builder, objects)
        if axis is not None:
            stagg.stagg_generated.Collection.CollectionAddAxis(builder, axis)
        data = stagg.stagg_generated.Collection.CollectionEnd(builder)

        stagg.stagg_generated.Object.ObjectStart(builder)
        stagg.stagg_generated.Object.ObjectAddDataType(builder, stagg.stagg_generated.ObjectData.ObjectData.Collection)
        stagg.stagg_generated.Object.ObjectAddData(builder, data)
        if title is not None:
            stagg.stagg_generated.Object.ObjectAddTitle(builder, title)
        if metadata is not None:
            stagg.stagg_generated.Object.ObjectAddMetadata(builder, metadata)
        if decoration is not None:
            stagg.stagg_generated.Object.ObjectAddDecoration(builder, decoration)
        if script is not None:
            stagg.stagg_generated.Object.ObjectAddScript(builder, script)
        return stagg.stagg_generated.Object.ObjectEnd(builder)

_RawBuffer_lookup = {
    stagg.stagg_generated.RawBuffer.RawBuffer.RawInlineBuffer: (RawInlineBuffer, stagg.stagg_generated.RawInlineBuffer.RawInlineBuffer),
    stagg.stagg_generated.RawBuffer.RawBuffer.RawExternalBuffer: (RawExternalBuffer, stagg.stagg_generated.RawExternalBuffer.RawExternalBuffer),
    }
_RawBuffer_invlookup = {x[0]: n for n, x in _RawBuffer_lookup.items()}

_InterpretedBuffer_lookup = {
    stagg.stagg_generated.InterpretedBuffer.InterpretedBuffer.InterpretedInlineBuffer: (InterpretedInlineBuffer, stagg.stagg_generated.InterpretedInlineBuffer.InterpretedInlineBuffer),
    stagg.stagg_generated.InterpretedBuffer.InterpretedBuffer.InterpretedExternalBuffer: (InterpretedExternalBuffer, stagg.stagg_generated.InterpretedExternalBuffer.InterpretedExternalBuffer),
    }
_InterpretedBuffer_invlookup = {x[0]: n for n, x in _InterpretedBuffer_lookup.items()}

_ObjectData_lookup = {
    stagg.stagg_generated.ObjectData.ObjectData.Histogram: (Histogram, stagg.stagg_generated.Histogram.Histogram),
    stagg.stagg_generated.ObjectData.ObjectData.Ntuple: (Ntuple, stagg.stagg_generated.Ntuple.Ntuple),
    stagg.stagg_generated.ObjectData.ObjectData.FunctionObject: (FunctionObject, stagg.stagg_generated.FunctionObject.FunctionObject),
    stagg.stagg_generated.ObjectData.ObjectData.Collection: (Collection, stagg.stagg_generated.Collection.Collection),
    }
_ObjectData_invlookup = {x[0]: n for n, x in _ObjectData_lookup.items()}

_FunctionObjectData_lookup = {
    stagg.stagg_generated.FunctionObjectData.FunctionObjectData.ParameterizedFunction: (ParameterizedFunction, stagg.stagg_generated.ParameterizedFunction.ParameterizedFunction),
    stagg.stagg_generated.FunctionObjectData.FunctionObjectData.BinnedEvaluatedFunction: (BinnedEvaluatedFunction, stagg.stagg_generated.BinnedEvaluatedFunction.BinnedEvaluatedFunction),
    }
_FunctionObjectData_invlookup = {x[0]: n for n, x in _FunctionObjectData_lookup.items()}

_FunctionData_lookup = {
    stagg.stagg_generated.FunctionData.FunctionData.ParameterizedFunction: (ParameterizedFunction, stagg.stagg_generated.ParameterizedFunction.ParameterizedFunction),
    stagg.stagg_generated.FunctionData.FunctionData.EvaluatedFunction: (EvaluatedFunction, stagg.stagg_generated.EvaluatedFunction.EvaluatedFunction),
    }
_FunctionData_invlookup = {x[0]: n for n, x in _FunctionData_lookup.items()}

_Binning_lookup = {
    stagg.stagg_generated.Binning.Binning.IntegerBinning: (IntegerBinning, stagg.stagg_generated.IntegerBinning.IntegerBinning),
    stagg.stagg_generated.Binning.Binning.RegularBinning: (RegularBinning, stagg.stagg_generated.RegularBinning.RegularBinning),
    stagg.stagg_generated.Binning.Binning.HexagonalBinning: (HexagonalBinning, stagg.stagg_generated.HexagonalBinning.HexagonalBinning),
    stagg.stagg_generated.Binning.Binning.EdgesBinning: (EdgesBinning, stagg.stagg_generated.EdgesBinning.EdgesBinning),
    stagg.stagg_generated.Binning.Binning.IrregularBinning: (IrregularBinning, stagg.stagg_generated.IrregularBinning.IrregularBinning),
    stagg.stagg_generated.Binning.Binning.CategoryBinning: (CategoryBinning, stagg.stagg_generated.CategoryBinning.CategoryBinning),
    stagg.stagg_generated.Binning.Binning.SparseRegularBinning: (SparseRegularBinning, stagg.stagg_generated.SparseRegularBinning.SparseRegularBinning),
    stagg.stagg_generated.Binning.Binning.FractionBinning: (FractionBinning, stagg.stagg_generated.FractionBinning.FractionBinning),
    stagg.stagg_generated.Binning.Binning.PredicateBinning: (PredicateBinning, stagg.stagg_generated.PredicateBinning.PredicateBinning),
    stagg.stagg_generated.Binning.Binning.VariationBinning: (VariationBinning, stagg.stagg_generated.VariationBinning.VariationBinning),
    }
_Binning_invlookup = {x[0]: n for n, x in _Binning_lookup.items()}
    
_Counts_lookup = {
    stagg.stagg_generated.Counts.Counts.UnweightedCounts: (UnweightedCounts, stagg.stagg_generated.UnweightedCounts.UnweightedCounts),
    stagg.stagg_generated.Counts.Counts.WeightedCounts: (WeightedCounts, stagg.stagg_generated.WeightedCounts.WeightedCounts),
    }
_Counts_invlookup = {x[0]: n for n, x in _Counts_lookup.items()}
