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
import ctypes
import functools
import math
import numbers
import operator
import struct
import sys
try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

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
import stagg.stagg_generated.InterpretedInlineInt64Buffer
import stagg.stagg_generated.InterpretedInlineFloat64Buffer
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
import stagg.stagg_generated.SystematicUnits
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

MININT64 = -9223372036854775808
MAXINT64 = 9223372036854775807

def _sameedges(one, two):
    assert isinstance(one, numpy.ndarray) and isinstance(two, numpy.ndarray)
    if len(one) != len(two):
        return False
    if len(one) == 1:
        return one[0] - two[0] < 1e-10
    gap = min((one[1:] - one[:-1]).min(), (two[1:] - two[:-1]).min())
    if gap <= 0:
        return False
    return (numpy.absolute(one - two) / gap < 1e-10).all()

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

class _LocIndexer(object):
    def __init__(self, obj, isiloc):
        self.obj = obj
        self._isiloc = isiloc

    def __getitem__(self, where):
        if not isinstance(where, tuple):
            where = (where,)
            
        node = self.obj
        binnings = ()
        while hasattr(node, "_parent"):
            node = node._parent
            binnings = tuple(x.binning for x in node.axis) + binnings

        return self.obj._getloc(self._isiloc, where, binnings)

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

    def detached(self, reclaim=False, exceptions=()):
        if reclaim:
            if hasattr(self, "_parent"):
                del self._parent
            return self
        else:
            return self._detached(True, exceptions=exceptions)

    def _detached(self, top, exceptions=()):
        if not top and not hasattr(self, "_parent"):
            return self
        else:
            out = type(self).__new__(type(self))
            if hasattr(self, "_flatbuffers"):
                out._flatbuffers = self._flatbuffers
            for n in self._params:
                if n not in exceptions:
                    private = "_" + n
                    if hasattr(self, private):
                        x = getattr(self, private)
                        if isinstance(x, (Stagg, stagg.checktype.Vector, stagg.checktype.Lookup)):
                            x = x._detached(False)
                        stagg.checktype.setparent(out, x)
                        setattr(out, private, x)
            return out

    @classmethod
    def _fromflatbuffers(cls, fb):
        out = cls.__new__(cls)
        out._flatbuffers = fb
        return out

    def _toflatbuffers(self, builder):
        raise NotImplementedError("missing _toflatbuffers implementation in {0}".format(type(self)))

    def dump(self, indent="", width=100, end="\n", file=sys.stdout, flush=False):
        file.write(self._dump(indent, width, end))
        file.write(end)
        if flush:
            file.flush()

    def __eq__(self, other):
        if self is other:
            return True
        if getattr(self, "_flatbuffers", None) is not None and self._flatbuffers is getattr(other, "_flatbuffers", None):
            return True
        if type(self) is not type(other):
            return False
        for n in self._params:
            selfn = getattr(self, n)
            othern = getattr(other, n)
            if selfn is None or isinstance(selfn, (Stagg, Enum)):
                if selfn != othern:
                    return False
            elif isinstance(selfn, numpy.ndarray) and isinstance(othern, numpy.ndarray):
                return selfn.shape == othern.shape and (selfn == othern).all()
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
        pairs, triples = Collection._pairs_triples(getattr(self, "_parent", None), getattr(other, "_parent", None))
        out = self.detached()
        out._add(other, pairs, triples, noclobber=True)
        return out

    def __iadd__(self, other):
        pairs, triples = Collection._pairs_triples(getattr(self, "_parent", None), getattr(other, "_parent", None))
        self._add(other, pairs, triples, noclobber=False)
        return self

    @property
    def loc(self):
        return _LocIndexer(self, False)

    @property
    def iloc(self):
        return _LocIndexer(self, True)

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

def _dumpstring(obj):
    if obj.count("\n") > 0:
        return "''" + repr(obj).replace("\\n", end) + "''"
    else:
        return repr(obj)

def _dumpline(obj, args, indent, width, end):
    preamble = type(obj).__name__ + "("
    linear = ", ".join(args)
    if len(indent) + len(preamble) + len(linear) + 1 > width:
        return preamble + ",".join(end + indent + "  " + x for x in args) + ")"
    else:
        return preamble + linear + ")"

def _dumplist(objs, indent, width, end):
    out = ("," + end + indent + "    ").join(objs)
    if len(objs) > 0 or objs[0].counts("\n") > 0:
        return out + end + indent + "  "
    return out

def _dumpeq(data, indent, end):
    if data.count(end) > 0:
        return end + indent + "    " + data
    else:
        return data

def _dumparray(obj, indent, end):
    asarray = str(obj)
    if asarray.count("\n") > 0:
        return end + indent + "  " + asarray.replace("\n", end + indent + "  ")
    else:
        return "[" + " ".join("%g" % x for x in obj) + "]"
    
################################################# Metadata

class MetadataLanguageEnum(Enum):
    base = "Metadata"

class Metadata(Stagg):
    unspecified = MetadataLanguageEnum("unspecified", stagg.stagg_generated.MetadataLanguage.MetadataLanguage.meta_unspecified)
    json        = MetadataLanguageEnum("json", stagg.stagg_generated.MetadataLanguage.MetadataLanguage.meta_json)
    language = [unspecified, json]

    _params = {
        "data":     stagg.checktype.CheckString("Metadata", "data", required=True),
        "language": stagg.checktype.CheckEnum("Metadata", "language", required=True, choices=language),
        }

    data     = typedproperty(_params["data"])
    language = typedproperty(_params["language"])

    description = ""
    validity_rules = ()
    long_description = """
"""

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

    def _dump(self, indent, width, end):
        args = ["data={0}".format(_dumpstring(self.data))]
        if self.language != self.unspecified:
            args.append("language={0}".format(repr(self.language)))
        return _dumpline(self, args, indent, width, end)

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

    description = ""
    validity_rules = ()
    long_description = """
"""

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

    def _dump(self, indent, width, end):
        args = ["data={0}".format(_dumpstring(self.data))]
        if self.language != self.unspecified:
            args.append("language={0}".format(repr(self.language)))
        return _dumpline(self, args, indent, width, end)

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
    sources = [memory, samefile, file, url]

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
    bool    = DTypeEnum("bool", stagg.stagg_generated.DType.DType.dtype_bool, numpy.dtype(numpy.bool_))
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
    dtypes = [none, bool, int8, uint8, int16, uint16, int32, uint32, int64, uint64, float32, float64]

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
        elif dtype.byteorder == "<" or dtype.byteorder == "|":
            endianness = cls.little_endian

        if dtype.kind == "b":
            return cls.bool, endianness

        elif dtype.kind == "i":
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

    def _reindex(self, oldshape, indexes):
        order = "c" if self.dimension_order == self.c_order else "f"
        dtype = self.numpy_dtype

        buf = self.flatarray.reshape(oldshape, order=order)

        for i, index in list(enumerate(indexes))[::-1]:
            if index is None:
                buf = buf.sum(axis=i)
            if not isinstance(index, (bool, numpy.bool, numpy.bool_)) and isinstance(index, (numbers.Integral, numpy.integer)):
                buf = buf[(slice(None),)*i + (index,)]

        indexes = [x for x in indexes if isinstance(x, (numpy.ndarray, slice))]
        for i, index in enumerate(indexes):
            if isinstance(index, numpy.ndarray) and issubclass(index.dtype.type, (numpy.bool, numpy.bool_)):
                buf = numpy.compress(index, buf, axis=i)
            elif isinstance(index, numpy.ndarray) and issubclass(index.dtype.type, numpy.integer):
                buf = numpy.take(buf, index, axis=i)
            else:
                buf = buf[(slice(None),)*i + (index,)]

        return buf

    def _rebin(self, oldshape, pairs):
        order = "c" if self.dimension_order == self.c_order else "f"
        dtype = self.numpy_dtype

        buf = self.flatarray.reshape(oldshape, order=order)
        original = buf

        i = len(oldshape)
        newshape = ()
        for binning, selfmap in pairs[::-1]:
            assert isinstance(selfmap, tuple)
            i -= len(selfmap)
            if binning is None:
                buf = buf.sum(axis=i)
            else:
                newshape = binning._binshape() + newshape
                newbuf = numpy.zeros(oldshape[:i] + newshape, dtype=dtype, order=order)
                for j in range(len(selfmap)):
                    if isinstance(selfmap[j], numpy.ndarray):
                        clear = (selfmap[j] < 0)
                        if clear.any():
                            if buf is original:
                                buf = buf.copy()
                            buf[(Ellipsis, clear) + (slice(None),)*(len(selfmap) - j - 1)] = 0
                            selfmap[j][clear] = 0
                numpy.add.at(newbuf, i*(slice(None),) + selfmap, buf)
                buf = newbuf

        if len(buf.shape) == 0:
            buf = buf.reshape(1)

        if self.dtype == InterpretedBuffer.int64 and self.endianness == InterpretedBuffer.little_endian and self.dimension_order == self.dimension_order == InterpretedBuffer.c_order:
            return InterpretedInlineInt64Buffer(buf.view(numpy.uint8))

        elif self.dtype == InterpretedBuffer.float64 and self.endianness == InterpretedBuffer.little_endian and self.dimension_order == self.dimension_order == InterpretedBuffer.c_order:
            return InterpretedInlineFloat64Buffer(buf.view(numpy.uint8))

        else:
            return InterpretedInlineBuffer(buf.view(numpy.uint8),
                                           filters=None,
                                           postfilter_slice=None,
                                           dtype=self.dtype,
                                           endianness=self.endianness,
                                           dimension_order=self.dimension_order)

    def _remap(self, newshape, selfmap):
        order = "c" if self.dimension_order == self.c_order else "f"
        dtype = self.numpy_dtype

        buf = self.flatarray

        oldshape = tuple(len(sm) if sm is not None else ns for ns, sm in zip(newshape, selfmap))

        for i in range(len(newshape) - 1, -1, -1):
            if selfmap[i] is not None:
                newbuf = numpy.zeros(oldshape[:i] + newshape[i:], dtype=dtype, order=order)
                newbuf[i*(slice(None),) + (selfmap[i],)] = buf.reshape((-1, len(selfmap[i])) + newshape[i + 1 :], order=order)
                buf = newbuf

        if self.dtype == InterpretedBuffer.int64 and self.endianness == InterpretedBuffer.little_endian and self.dimension_order == self.dimension_order == InterpretedBuffer.c_order:
            return InterpretedInlineInt64Buffer(buf.view(numpy.uint8))

        elif self.dtype == InterpretedBuffer.float64 and self.endianness == InterpretedBuffer.little_endian and self.dimension_order == self.dimension_order == InterpretedBuffer.c_order:
            return InterpretedInlineFloat64Buffer(buf.view(numpy.uint8))

        else:
            return InterpretedInlineBuffer(buf.view(numpy.uint8),
                                           filters=None,
                                           postfilter_slice=None,
                                           dtype=self.dtype,
                                           endianness=self.endianness,
                                           dimension_order=self.dimension_order)

    def _add(self, other, noclobber):
        if noclobber:
            if isinstance(self, InterpretedInlineBuffer) or isinstance(other, InterpretedInlineBuffer):
                return InterpretedInlineBuffer((self.flatarray + other.flatarray).view(numpy.uint8),
                                               filters=self.filters,
                                               postfilter_slice=self.postfilter_slice,
                                               dtype=self.dtype,
                                               endianness=self.endianness,
                                               dimension_order=self.dimension_order)

            elif isinstance(self, InterpretedInlineFloat64Buffer) or isinstance(other, InterpretedInlineFloat64Buffer):
                return InterpretedInlineFloat64Buffer((self.flatarray + other.flatarray).view(numpy.uint8))

            elif isinstance(self, InterpretedInlineInt64Buffer) or isinstance(other, InterpretedInlineInt64Buffer):
                return InterpretedInlineInt64Buffer((self.flatarray + other.flatarray).view(numpy.uint8))

            else:
                raise AssertionError((type(self), type(other)))

        else:
            self.flatarray += other.flatarray
            return self

################################################# RawInlineBuffer

class RawInlineBuffer(Buffer, RawBuffer, InlineBuffer):
    _params = {
        "buffer": stagg.checktype.CheckBuffer("RawInlineBuffer", "buffer", required=True),
        }

    buffer = typedproperty(_params["buffer"])

    description = "A generic, uninterpreted array in the Flatbuffers hierarchy; used for small buffers, like <<Ntuple>> pages, that are interpreted centrally, as in an <<Ntuple>> column."
    validity_rules = ()
    long_description = """
This array class does not provide its own interpretation in terms of data type and dimension order. The interpretation must be provided elsewhere, such as in an ntuple's <<Column>>. This is to avoid repeating (and possibly introduce conflicting) interpretation metadata for many buffers whose type is identical but are stored in pages for performance reasons.

The *buffer* is the actual data, encoded in Flatbuffers as an array of bytes with known length.
"""

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

    def _dump(self, indent, width, end):
        args = ["buffer={0}".format(repr(self.buffer.tostring()))]
        return _dumpline(self, args, indent, width, end)

################################################# RawExternalBuffer

class RawExternalBuffer(Buffer, RawBuffer, ExternalBuffer):
    _params = {
        "pointer":         stagg.checktype.CheckInteger("RawExternalBuffer", "pointer", required=True, min=0),
        "numbytes":        stagg.checktype.CheckInteger("RawExternalBuffer", "numbytes", required=True, min=0),
        "external_source": stagg.checktype.CheckEnum("RawExternalBuffer", "external_source", required=False, choices=ExternalBuffer.sources),
        }

    pointer       = typedproperty(_params["pointer"])
    numbytes      = typedproperty(_params["numbytes"])
    external_source = typedproperty(_params["external_source"])

    description = "A generic, uninterpreted array stored outside the Flatbuffers hierarchy; used for small buffers, like <<Ntuple>> pages, that are interpreted centrally, as in an <<Ntuple>> column."
    validity_rules = ()
    long_description = """
This array class is like <<RawInlineBuffer>>, but its contents are outside of the Flatbuffers hierarchy. Instead of a *buffer* property, it has a *pointer* and a *numbytes* to specify the source of bytes.

If the *external_source* is `memory`, then the *pointer* and *numbytes* are interpreted as a raw array in memory. If the *external_source* is `samefile`, then the *pointer* is taken to be a seek position in the same file that stores the Flatbuffer (assuming the Flatbuffer resides in a file). If *external_source* is `file`, then the *location* property is taken to be a file path, and the *pointer* is taken to be a seek position in that file. If *external_source* is `url`, then the *location* property is taken to be a URL and the bytes are requested by HTTP.
"""

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
        if self.external_source != ExternalBuffer.memory:
            stagg.stagg_generated.RawExternalBuffer.RawExternalBufferAddExternalSource(builder, self.external_source.value)
        return stagg.stagg_generated.RawExternalBuffer.RawExternalBufferEnd(builder)

    def _dump(self, indent, width, end):
        args = ["pointer={0}".format(repr(self.pointer)), "numbytes={0}".format(repr(self.numbytes))]
        if self.external_source != ExternalBuffer.memory:
            args.append("external_source={0}".format(repr(self.external_source)))
        return _dumpline(self, args, indent, width, end)

################################################# InterpretedInlineBuffer

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

    description = "A generic array in the Flatbuffers hierarchy; used for any quantity that can have different values in different <<Histogram>> or <<BinnedEvaluatedFunction>> bins."
    validity_rules = ("The *postfilter_slice*'s *step* cannot be zero.",
                      "The number of items in the *buffer* must be equal to the number of bins at this level of the hierarchy.")
    long_description = """
This array class provides its own interpretation in terms of data type and dimension order. It does not specify its own shape, the number of bins in each dimension, because that is given by its position in the hierarchy. If it is the <<UnweightedCounts>> of a <<Histogram>>, for instance, it must be reshapable to fit the number of bins implied by the <<Histogram>> *axis*.

The *buffer* is the actual data, encoded in Flatbuffers as an array of bytes with known length.

The list of *filters* are applied to convert bytes in the *buffer* into an array. Typically, *filters* are compression algorithms such as `gzip`, `lzma`, and `lz4`, but they may be any predefined transformation (e.g. zigzag deencoding of integers or affine mappings from integers to floating point numbers may be added in the future). If there is more than one filter, the output of each step is provided as input to the next.

The *postfilter_slice*, if provided, selects a subset of the bytes returned by the last filter (or directly in the *buffer* if there are no *filters*). A slice has the following structure:

    struct Slice {
      start: long;
      stop: long;
      step: int;
      has_start: bool;
      has_stop: bool;
      has_step: bool;
    }

though in Python, a builtin `slice` object should be provided to this class's constructor. The *postfilter_slice* is interpreted according to Python's rules (negative indexes, start-inclusive and stop-exclusive, clipping-not-errors if beyond the range, etc.).

The *dtype* is the numeric type of the array, which includes `bool`, all signed and unsigned integers from 8 bits to 64 bits, and IEEE 754 floating point types with 32 or 64 bits. The `none` interpretation is presumed, if necessary, to be unsigned, 8 bit integers.

The *endianness* may be `little_endian` or `big_endian`; the former is used by most recent architectures.

The *dimension_order* may be `c_order` to follow the C programming language's convention or `fortran` to follow the FORTRAN programming language's convention. The *dimension_order* only has an effect when shaping an array with more than one dimension.
"""

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
        if not isinstance(array, numpy.ndarray):
            array = numpy.array(array)
        dtype, endianness = Interpretation.from_numpy_dtype(array.dtype)
        order = InterpretedBuffer.fortran_order if numpy.isfortran(array) else InterpretedBuffer.c_order
        if dtype == InterpretedBuffer.int64 and endianness == InterpretedBuffer.little_endian and order == InterpretedBuffer.c_order:
            return InterpretedInlineInt64Buffer(array)
        elif dtype == InterpretedBuffer.float64 and endianness == InterpretedBuffer.little_endian and order == InterpretedBuffer.c_order:
            return InterpretedInlineFloat64Buffer(array)
        else:
            return cls(array, dtype=dtype, endianness=endianness, dimension_order=order)

    @property
    def flatarray(self):
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

        return array

    @property
    def array(self):
        array = self.flatarray
        shape = self._shape((), ())
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
        stagg.stagg_generated.InterpretedInlineBuffer.InterpretedInlineBufferStartBufferVector(builder, self.buffer.nbytes)
        builder.head = builder.head - self.buffer.nbytes
        builder.Bytes[builder.head : builder.head + self.buffer.nbytes] = self.buffer.tostring()
        buffer = builder.EndVector(self.buffer.nbytes)

        if len(self.filters) == 0:
            filters = None
        else:
            stagg.stagg_generated.InterpretedInlineBuffer.InterpretedInlineBufferStartFiltersVector(builder, len(self.filters))
            for x in self.filters[::-1]:
                builder.PrependUint32(x.value)
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

    def _dump(self, indent, width, end):
        args = ["buffer={0}".format(_dumparray(self.flatarray, indent, end))]
        if len(self.filters) != 0:
            args.append("filters=[{0}]".format(", ".format(repr(x) for x in self.filters)))
        if self.postfilter_slice is not None:
            args.append("postfilter_slice=slice({0}, {1}, {2})".format(self.postfilter_slice.start if self.postfilter_slice.hasStart else "None",
                                                                       self.postfilter_slice.stop if self.postfilter_slice.hasStop else "None",
                                                                       self.postfilter_slice.step if self.postfilter_slice.hasStep else "None"))
        if self.dtype != InterpretedBuffer.none:
            args.append("dtype={0}".format(repr(self.dtype)))
        if self.endianness != InterpretedBuffer.little_endian:
            args.append("endianness={0}".format(repr(self.endianness)))
        if self.dimension_order != InterpretedBuffer.c_order:
            args.append("dimension_order={0}".format(repr(self.dimension_order)))
        return _dumpline(self, args, indent, width, end)
 
################################################# InterpretedInlineInt64Buffer

class InterpretedInlineInt64Buffer(Buffer, InterpretedBuffer, InlineBuffer):
    _params = {
        "buffer": stagg.checktype.CheckBuffer("InterpretedInlineInt64Buffer", "buffer", required=True),
        }

    buffer = typedproperty(_params["buffer"])

    description = "An integer array in the Flatbuffers hierarchy; used for integer-valued quantities that can have different values in different <<Histogram>> or <<BinnedEvaluatedFunction>> bins."
    validity_rules = ("The number of items in the *buffer* must be equal to the number of bins at this level of the hierarchy.",)
    long_description = """
This class is equivalent to an <<InterpretedInlineBuffer>> with no *filters*, no *postfilter_slice*, a *dtype* of `int64`, an *endianness* of `little_endian`, and a *dimension_order* of `c_order`. It is provided as an optimization because many small arrays should avoid unnecessary Flatbuffers lookup overhead.
"""

    def __init__(self, buffer):
        self.buffer = buffer

    def _valid(self, seen, recursive):
        self.array

    @property
    def flatarray(self):
        try:
            return self.buffer.reshape(-1).view(self.numpy_dtype)
        except ValueError:
            raise ValueError("InterpretedInlineInt64Buffer.buffer raw length is {0} bytes but this does not fit an itemsize of {1} bytes".format(len(array), self.numpy_dtype.itemsize))

    @property
    def array(self):
        array = self.flatarray
        shape = self._shape((), ())
        if len(array) != functools.reduce(operator.mul, shape, 1):
            raise ValueError("InterpretedInlineInt64Buffer.buffer length as {0} is {1} but multiplicity at this position in the hierarchy is {2}".format(self.numpy_dtype, len(array), functools.reduce(operator.mul, shape, 1)))
        return array.reshape(shape, order=self.dimension_order.dimension_order)

    @property
    def filters(self):
        return None

    @property
    def postfilter_slice(self):
        return None

    @property
    def dtype(self):
        return InterpretedBuffer.int64

    @property
    def numpy_dtype(self):
        return numpy.dtype(numpy.int64)

    @property
    def endianness(self):
        return InterpretedBuffer.little_endian

    @property
    def dimension_order(self):
        return InterpretedBuffer.c_order

    @classmethod
    def _fromflatbuffers(cls, fb):
        out = cls.__new__(cls)
        out._flatbuffers = _MockFlatbuffers()
        out._flatbuffers.Buffer = fb.BufferAsNumpy
        return out

    def _toflatbuffers(self, builder):
        stagg.stagg_generated.InterpretedInlineInt64Buffer.InterpretedInlineInt64BufferStartBufferVector(builder, self.buffer.nbytes)
        builder.head = builder.head - self.buffer.nbytes
        builder.Bytes[builder.head : builder.head + self.buffer.nbytes] = self.buffer.tostring()
        buffer = builder.EndVector(self.buffer.nbytes)

        stagg.stagg_generated.InterpretedInlineInt64Buffer.InterpretedInlineInt64BufferStart(builder)
        stagg.stagg_generated.InterpretedInlineInt64Buffer.InterpretedInlineInt64BufferAddBuffer(builder, buffer)
        return stagg.stagg_generated.InterpretedInlineInt64Buffer.InterpretedInlineInt64BufferEnd(builder)

    def _dump(self, indent, width, end):
        args = ["buffer={0}".format(_dumparray(self.flatarray, indent, end))]
        return _dumpline(self, args, indent, width, end)

################################################# InterpretedInlineFloat64Buffer

class InterpretedInlineFloat64Buffer(Buffer, InterpretedBuffer, InlineBuffer):
    _params = {
        "buffer": stagg.checktype.CheckBuffer("InterpretedInlineFloat64Buffer", "buffer", required=True),
        }

    buffer = typedproperty(_params["buffer"])

    description = "A floating point array in the Flatbuffers hierarchy; used for real-valued quantities that can have different values in different <<Histogram>> or <<BinnedEvaluatedFunction>> bins."
    validity_rules = ("The number of items in the *buffer* must be equal to the number of bins at this level of the hierarchy.",)
    long_description = """
This class is equivalent to an <<InterpretedInlineBuffer>> with no *filters*, no *postfilter_slice*, a *dtype* of `float64`, an *endianness* of `little_endian`, and a *dimension_order* of `c_order`. It is provided as an optimization because many small arrays should avoid unnecessary Flatbuffers lookup overhead.
"""

    def __init__(self, buffer):
        self.buffer = buffer

    def _valid(self, seen, recursive):
        self.array

    @property
    def flatarray(self):
        try:
            return self.buffer.reshape(-1).view(self.numpy_dtype)
        except ValueError:
            raise ValueError("InterpretedInlineFloat64Buffer.buffer raw length is {0} bytes but this does not fit an itemsize of {1} bytes".format(len(array), self.numpy_dtype.itemsize))

    @property
    def array(self):
        array = self.flatarray
        shape = self._shape((), ())
        if len(array) != functools.reduce(operator.mul, shape, 1):
            raise ValueError("InterpretedInlineFloat64Buffer.buffer length as {0} is {1} but multiplicity at this position in the hierarchy is {2}".format(self.numpy_dtype, len(array), functools.reduce(operator.mul, shape, 1)))
        return array.reshape(shape, order=self.dimension_order.dimension_order)

    @property
    def filters(self):
        return None

    @property
    def postfilter_slice(self):
        return None

    @property
    def dtype(self):
        return InterpretedBuffer.float64

    @property
    def numpy_dtype(self):
        return numpy.dtype(numpy.float64)

    @property
    def endianness(self):
        return InterpretedBuffer.little_endian

    @property
    def dimension_order(self):
        return InterpretedBuffer.c_order

    @classmethod
    def _fromflatbuffers(cls, fb):
        out = cls.__new__(cls)
        out._flatbuffers = _MockFlatbuffers()
        out._flatbuffers.Buffer = fb.BufferAsNumpy
        return out

    def _toflatbuffers(self, builder):
        stagg.stagg_generated.InterpretedInlineFloat64Buffer.InterpretedInlineFloat64BufferStartBufferVector(builder, self.buffer.nbytes)
        builder.head = builder.head - self.buffer.nbytes
        builder.Bytes[builder.head : builder.head + self.buffer.nbytes] = self.buffer.tostring()
        buffer = builder.EndVector(self.buffer.nbytes)

        stagg.stagg_generated.InterpretedInlineFloat64Buffer.InterpretedInlineFloat64BufferStart(builder)
        stagg.stagg_generated.InterpretedInlineFloat64Buffer.InterpretedInlineFloat64BufferAddBuffer(builder, buffer)
        return stagg.stagg_generated.InterpretedInlineFloat64Buffer.InterpretedInlineFloat64BufferEnd(builder)

    def _dump(self, indent, width, end):
        args = ["buffer={0}".format(_dumparray(self.flatarray, indent, end))]
        return _dumpline(self, args, indent, width, end)

################################################# InterpretedExternalBuffer

class InterpretedExternalBuffer(Buffer, InterpretedBuffer, ExternalBuffer):
    _params = {
        "pointer":          stagg.checktype.CheckInteger("InterpretedExternalBuffer", "pointer", required=True, min=0),
        "numbytes":         stagg.checktype.CheckInteger("InterpretedExternalBuffer", "numbytes", required=True, min=0),
        "external_source":  stagg.checktype.CheckEnum("InterpretedExternalBuffer", "external_source", required=False, choices=ExternalBuffer.sources),
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

    description = "A generic array stored outside the Flatbuffers hierarchy; used for any quantity that can have different values in different <<Histogram>> or <<BinnedEvaluatedFunction>> bins."
    validity_rules = ("The *postfilter_slice*'s *step* cannot be zero.",
                      "The number of items in the *buffer* must be equal to the number of bins at this level of the hierarchy.")
    long_description = """
This array class is like <<InterpretedInlineBuffer>>, but its contents are outside of the Flatbuffers hierarchy. Instead of a *buffer* property, it has a *pointer* and a *numbytes* to specify the source of bytes.

If the *external_source* is `memory`, then the *pointer* and *numbytes* are interpreted as a raw array in memory. If the *external_source* is `samefile`, then the *pointer* is taken to be a seek position in the same file that stores the Flatbuffer (assuming the Flatbuffer resides in a file). If *external_source* is `file`, then the *location* property is taken to be a file path, and the *pointer* is taken to be a seek position in that file. If *external_source* is `url`, then the *location* property is taken to be a URL and the bytes are requested by HTTP.

Like <<InterpretedInlineBuffer>>, this array class provides its own interpretation in terms of data type and dimension order. It does not specify its own shape, the number of bins in each dimension, because that is given by its position in the hierarchy. If it is the <<UnweightedCounts>> of a <<Histogram>>, for instance, it must be reshapable to fit the number of bins implied by the <<Histogram>> *axis*.

The list of *filters* are applied to convert bytes in the *buffer* into an array. Typically, *filters* are compression algorithms such as `gzip`, `lzma`, and `lz4`, but they may be any predefined transformation (e.g. zigzag deencoding of integers or affine mappings from integers to floating point numbers may be added in the future). If there is more than one filter, the output of each step is provided as input to the next.

The *postfilter_slice*, if provided, selects a subset of the bytes returned by the last filter (or directly in the *buffer* if there are no *filters*). A slice has the following structure:

    struct Slice {
      start: long;
      stop: long;
      step: int;
      has_start: bool;
      has_stop: bool;
      has_step: bool;
    }

though in Python, a builtin `slice` object should be provided to this class's constructor. The *postfilter_slice* is interpreted according to Python's rules (negative indexes, start-inclusive and stop-exclusive, clipping-not-errors if beyond the range, etc.).

The *dtype* is the numeric type of the array, which includes `bool`, all signed and unsigned integers from 8 bits to 64 bits, and IEEE 754 floating point types with 32 or 64 bits. The `none` interpretation is presumed, if necessary, to be unsigned, 8 bit integers.

The *endianness* may be `little_endian` or `big_endian`; the former is used by most recent architectures.

The *dimension_order* may be `c_order` to follow the C programming language's convention or `fortran` to follow the FORTRAN programming language's convention. The *dimension_order* only has an effect when shaping an array with more than one dimension.
"""

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
    def flatarray(self):
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

        return array

    @property
    def array(self):
        array = self.flatarray
        shape = self._shape((), ())
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
                builder.PrependUint32(x.value)
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

    def _dump(self, indent, width, end):
        args = ["pointer={0}".format(repr(self.pointer)), "numbytes={0}".format(repr(self.numbytes))]
        if self.external_source != ExternalBuffer.memory:
            args.append("external_source={0}".format(repr(self.external_source)))
        if len(self.filters) != 0:
            args.append("filters=[{0}]".format(", ".format(repr(x) for x in self.filters)))
        if self.postfilter_slice is not None:
            args.append("postfilter_slice=slice({0}, {1}, {2})".format(self.postfilter_slice.start if self.postfilter_slice.hasStart else "None",
                                                                       self.postfilter_slice.stop if self.postfilter_slice.hasStop else "None",
                                                                       self.postfilter_slice.step if self.postfilter_slice.hasStep else "None"))
        if self.dtype != InterpretedBuffer.none:
            args.append("dtype={0}".format(repr(self.dtype)))
        if self.endianness != InterpretedBuffer.little_endian:
            args.append("endianness={0}".format(repr(self.endianness)))
        if self.dimension_order != InterpretedBuffer.c_order:
            args.append("dimension_order={0}".format(repr(self.dimension_order)))
        if self.location is not None:
            args.append("location={0}".format(_dumpstring(self.location)))
        return _dumpline(self, args, indent, width, end)

    def _add(self, other, noclobber):
        if noclobber or self.external_source != self.memory or len(self.filters) != 0:
            return super(InterpretedExternalBuffer, self)._add(other, noclobber)

        else:
            self.flatarray += other.flatarray
            return self

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

    description = ""
    validity_rules = ()
    long_description = """
"""

    def __init__(self, min=-numpy.inf, max=numpy.inf, excludes_minf=False, excludes_pinf=False, excludes_nan=False):
        self.min = min
        self.max = max
        self.excludes_minf = excludes_minf
        self.excludes_pinf = excludes_pinf
        self.excludes_nan = excludes_nan

    def _valid(self, seen, recursive):
        if self.min is not None and self.max is not None and self.min >= self.max:
            raise ValueError("StatisticFilter.min ({0}) must be strictly less than StatisticFilter.max ({1})".format(self.min, self.max))

    def _toflatbuffers(self, builder):
        return stagg.stagg_generated.StatisticFilter.CreateStatisticFilter(builder, self.min, self.max, self.excludes_minf, self.excludes_pinf, self.excludes_nan)

    def _dump(self, indent, width, end):
        args = []
        if self.min != -numpy.inf:
            args.append("min={0}".format(self.min))
        if self.max != numpy.inf:
            args.append("max={0}".format(self.max))
        if self.excludes_minf is not False:
            args.append("excludes_minf={0}".format(self.excludes_minf))
        if self.excludes_pinf is not False:
            args.append("excludes_pinf={0}".format(self.excludes_pinf))
        if self.excludes_nan is not False:
            args.append("excludes_nan={0}".format(self.excludes_nan))
        return _dumpline(self, args, indent, width, end)

################################################# Moments

class Moments(Stagg):
    _params = {
        "sumwxn":      stagg.checktype.CheckClass("Moments", "sumwxn", required=True, type=InterpretedBuffer),
        "n":           stagg.checktype.CheckInteger("Moments", "n", required=True, min=-128, max=127),
        "weightpower": stagg.checktype.CheckInteger("Moments", "weightpower", required=False, min=-128, max=127),
        "filter":      stagg.checktype.CheckClass("Moments", "filter", required=False, type=StatisticFilter),
        }

    sumwxn      = typedproperty(_params["sumwxn"])
    n           = typedproperty(_params["n"])
    weightpower = typedproperty(_params["weightpower"])
    filter      = typedproperty(_params["filter"])

    description = ""
    validity_rules = ()
    long_description = """
"""

    def __init__(self, sumwxn, n, weightpower=0, filter=None):
        self.sumwxn = sumwxn
        self.n = n
        self.weightpower = weightpower
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
        out._flatbuffers.Weightpower = fb.Weightpower
        out._flatbuffers.Filter = fb.Filter
        return out

    def _toflatbuffers(self, builder):
        sumwxn = self.sumwxn._toflatbuffers(builder)

        stagg.stagg_generated.Moments.MomentsStart(builder)
        stagg.stagg_generated.Moments.MomentsAddSumwxnType(builder, _InterpretedBuffer_invlookup[type(self.sumwxn)])
        stagg.stagg_generated.Moments.MomentsAddSumwxn(builder, sumwxn)
        stagg.stagg_generated.Moments.MomentsAddN(builder, self.n)
        if self.weightpower != 0:
            stagg.stagg_generated.Moments.MomentsAddWeightpower(builder, self.weightpower)
        if self.filter is not None:
            stagg.stagg_generated.Moments.MomentsAddFilter(builder, self.filter._toflatbuffers(builder))
        return stagg.stagg_generated.Moments.MomentsEnd(builder)

    def _dump(self, indent, width, end):
        args = ["sumwxn={0}".format(_dumpeq(self.sumwxn._dump(indent + "    ", width, end), indent, end)), "n={0}".format(repr(self.n))]
        if self.weightpower != 0:
            args.append("weightpower={0}".format(self.weightpower))
        if self.filter is not None:
            args.append("filter={0}".format(_dumpeq(self.filter._dump(indent + "  ", end, flie, flush), indent, end)))
        return _dumpline(self, args, indent, width, end)

################################################# Extremes

class Extremes(Stagg):
    _params = {
        "values": stagg.checktype.CheckClass("Extremes", "values", required=True, type=InterpretedBuffer),
        "filter": stagg.checktype.CheckClass("Extremes", "filter", required=False, type=StatisticFilter),
        }

    values = typedproperty(_params["values"])
    filter = typedproperty(_params["filter"])

    description = ""
    validity_rules = ()
    long_description = """
"""

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

        stagg.stagg_generated.Extremes.ExtremesStart(builder)
        stagg.stagg_generated.Extremes.ExtremesAddValuesType(builder, _InterpretedBuffer_invlookup[type(self.values)])
        stagg.stagg_generated.Extremes.ExtremesAddValues(builder, values)
        if self.filter is not None:
            stagg.stagg_generated.Extremes.ExtremesAddFilter(builder, self.filter._toflatbuffers(builder))
        return stagg.stagg_generated.Extremes.ExtremesEnd(builder)

    def _dump(self, indent, width, end):
        args = ["values={0}".format(_dumpeq(self.values._dump(indent + "    ", width, end), indent, end))]
        if self.filter is not None:
            args.append("filter={0}".format(_dumpeq(self.filter._dump(indent + "  ", end, flie, flush), indent, end)))
        return _dumpline(self, args, indent, width, end)

################################################# Quantiles

class Quantiles(Stagg):
    _params = {
        "values":      stagg.checktype.CheckClass("Quantiles", "values", required=True, type=InterpretedBuffer),
        "p":           stagg.checktype.CheckNumber("Quantiles", "p", required=True, min=0.0, max=1.0),
        "weightpower": stagg.checktype.CheckInteger("Quantiles", "weightpower", required=False, min=-128, max=127),
        "filter":      stagg.checktype.CheckClass("Quantiles", "filter", required=False, type=StatisticFilter),
        }

    values      = typedproperty(_params["values"])
    p           = typedproperty(_params["p"])
    weightpower = typedproperty(_params["weightpower"])
    filter      = typedproperty(_params["filter"])

    description = ""
    validity_rules = ()
    long_description = """
"""

    def __init__(self, values, p=0.5, weightpower=0, filter=None):
        self.values = values
        self.p = p
        self.weightpower = weightpower
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
        out._flatbuffers.Weightpower = fb.Weightpower
        out._flatbuffers.Filter = fb.Filter
        return out

    def _toflatbuffers(self, builder):
        values = self.values._toflatbuffers(builder)

        stagg.stagg_generated.Quantiles.QuantilesStart(builder)
        stagg.stagg_generated.Quantiles.QuantilesAddValuesType(builder, _InterpretedBuffer_invlookup[type(self.values)])
        stagg.stagg_generated.Quantiles.QuantilesAddValues(builder, values)
        stagg.stagg_generated.Quantiles.QuantilesAddP(builder, self.p)
        if self.weightpower != 0:
            stagg.stagg_generated.Quantiles.QuantilesAddWeightpower(builder, self.weightpower)
        if self.filter is not None:
            stagg.stagg_generated.Quantiles.QuantilesAddFilter(builder, self.filter._toflatbuffers(builder))
        return stagg.stagg_generated.Quantiles.QuantilesEnd(builder)

    def _dump(self, indent, width, end):
        args = ["values={0}".format(_dumpeq(self.values._dump(indent + "    ", width, end), indent, end)), "p={0}".format(repr(self.p))]
        if self.weightpower != 0:
            args.append("weightpower={0}".format(self.weightpower))
        if self.filter is not None:
            args.append("filter={0}".format(_dumpeq(self.filter._dump(indent + "  ", end, flie, flush), indent, end)))
        return _dumpline(self, args, indent, width, end)

################################################# Modes

class Modes(Stagg):
    _params = {
        "values": stagg.checktype.CheckClass("Modes", "values", required=True, type=InterpretedBuffer),
        "filter": stagg.checktype.CheckClass("Modes", "filter", required=False, type=StatisticFilter),
        }

    values = typedproperty(_params["values"])
    filter = typedproperty(_params["filter"])

    description = ""
    validity_rules = ()
    long_description = """
"""

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

        stagg.stagg_generated.Modes.ModesStart(builder)
        stagg.stagg_generated.Modes.ModesAddValuesType(builder, _InterpretedBuffer_invlookup[type(self.values)])
        stagg.stagg_generated.Modes.ModesAddValues(builder, values)
        if self.filter is not None:
            stagg.stagg_generated.Modes.ModesAddFilter(builder, self.filter._toflatbuffers(builder))
        return stagg.stagg_generated.Modes.ModesEnd(builder)

    def _dump(self, indent, width, end):
        args = ["values={0}".format(_dumpeq(self.values._dump(indent + "    ", width, end), indent, end))]
        if self.filter is not None:
            args.append("filter={0}".format(_dumpeq(self.filter._dump(indent + "  ", end, flie, flush), indent, end)))
        return _dumpline(self, args, indent, width, end)

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

    description = "Represents summary statistics for a <<Histogram>> axis or for each bin in a <<Profile>>."
    validity_rules = ("All *moments* must have unique *n* and *weightpower* properties.",
                      "All *quantiles* must have unique *n* and *weightpower* properties.")
    long_description = """
This object provides a statistical summary of a distribution without binning it as a histogram does. Examples include mean, standard deviation, median, and mode.

Anything that can be computed from moments, such as the mean and standard deviation, are stored as raw moments, in the *moments* property. Concepts like "`mean`" and "`standard deviation`" are not explicitly called out by the structure; they must be constructed.

Medians, quartiles, and quintiles are all stored in the *quantiles* property.

If the mode of the distribution was computed, it is stored in the *mode* property.

The minimum and maximum of a distribution are special cases of quantiles, but quantiles can't in general be combined from preaggregated subsets of the data. The *min* and *max* can be combined (they are monadic calculations, like the sums that are *moments*), so they are stored separately as <<Extremes>>.
"""

    def __init__(self, moments=None, quantiles=None, mode=None, min=None, max=None):
        self.moments = moments
        self.quantiles = quantiles
        self.mode = mode
        self.min = min
        self.max = max

    def _valid(self, seen, recursive):
        if len(set((x.n, x.weightpower) for x in self.moments)) != len(self.moments):
            raise ValueError("Statistics.moments must have unique (n, weightpower)")
        if len(set((x.p, x.weightpower) for x in self.quantiles)) != len(self.quantiles):
            raise ValueError("Statistics.quantiles must have unique (p, weightpower)")
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

    def _dump(self, indent, width, end):
        args = []
        if len(self.moments) != 0:
            args.append("moments=[{0}]".format(_dumpeq(_dumplist([x._dump(indent + "    ", width, end) for x in self.moments], indent, width, end), indent, end)))
        if len(self.quantiles) != 0:
            args.append("quantiles=[{0}]".format(_dumpeq(_dumplist([x._dump(indent + "    ", width, end) for x in self.quantiles], indent, width, end), indent, end)))
        if self.mode is not None:
            args.append("mode={0}".format(_dumpeq(self.mode._dump(indent + "    ", width, end), indent, end)))
        if self.min is not None:
            args.append("min={0}".format(_dumpeq(self.min._dump(indent + "    ", width, end), indent, end)))
        if self.max is not None:
            args.append("max={0}".format(_dumpeq(self.max._dump(indent + "    ", width, end), indent, end)))
        return _dumpline(self, args, indent, width, end)

################################################# Covariance

class Covariance(Stagg):
    _params = {
        "xindex":      stagg.checktype.CheckInteger("Covariance", "xindex", required=True, min=0),
        "yindex":      stagg.checktype.CheckInteger("Covariance", "yindex", required=True, min=0),
        "sumwxy":      stagg.checktype.CheckClass("Covariance", "sumwxy", required=True, type=InterpretedBuffer),
        "weightpower": stagg.checktype.CheckInteger("Covariance", "weightpower", required=False, min=-128, max=127),
        "filter":      stagg.checktype.CheckClass("Covariance", "filter", required=False, type=StatisticFilter),
        }

    xindex      = typedproperty(_params["xindex"])
    yindex      = typedproperty(_params["yindex"])
    sumwxy      = typedproperty(_params["sumwxy"])
    weightpower = typedproperty(_params["weightpower"])
    filter      = typedproperty(_params["filter"])

    description = ""
    validity_rules = ()
    long_description = """
"""

    def __init__(self, xindex, yindex, sumwxy, weightpower=0, filter=None):
        self.xindex = xindex
        self.yindex = yindex
        self.sumwxy = sumwxy
        self.weightpower = weightpower
        self.filter = filter

    def _valid(self, seen, recursive):
        if recursive:
            _valid(self.sumwxy, seen, recursive)
            _valid(self.filter, seen, recursive)

    @staticmethod
    def _validindexes(covariances, numvars):
        triples = [(x.xindex, x.yindex, x.weightpower) for x in covariances]
        if len(set(triples)) != len(triples):
            raise ValueError("Covariance.xindex, yindex, weightpower triples must be unique")
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
        out._flatbuffers.Weightpower = fb.Weightpower
        out._flatbuffers.Filter = fb.Filter
        return out

    def _toflatbuffers(self, builder):
        sumwxy = self.sumwxy._toflatbuffers(builder)

        stagg.stagg_generated.Covariance.CovarianceStart(builder)
        stagg.stagg_generated.Covariance.CovarianceAddXindex(builder, self.xindex)
        stagg.stagg_generated.Covariance.CovarianceAddYindex(builder, self.yindex)
        stagg.stagg_generated.Covariance.CovarianceAddSumwxyType(builder, _InterpretedBuffer_invlookup[type(self.sumwxy)])
        stagg.stagg_generated.Covariance.CovarianceAddSumwxy(builder, sumwxy)
        if self.weightpower != 0:
            stagg.stagg_generated.Covariance.CovarianceAddWeightpower(builder, self.weightpower)
        if self.filter is not None:
            stagg.stagg_generated.Covariance.CovarianceAddFilter(builder, self.filter._toflatbuffers(builder))
        return stagg.stagg_generated.Covariance.CovarianceEnd(builder)

    def _dump(self, indent, width, end):
        args = ["xindex={0}".format(repr(self.xindex)), "yindex={0}".format(repr(self.yindex)), "sumwxy={0}".format(_dumpeq(self.sumwxy._dump(indent + "    ", width, end), indent, end))]
        if self.weightpower != 0:
            args.append("weightpower={0}".format(self.weightpower))
        if self.filter is not None:
            args.append("filter={0}".format(_dumpeq(self.filter._dump(indent + "  ", end, flie, flush), indent, end)))
        return _dumpline(self, args, indent, width, end)

################################################# Binning

class Binning(Stagg):
    def __init__(self):
        raise TypeError("{0} is an abstract base class; do not construct".format(type(self).__name__))

    @property
    def isnumerical(self):
        return True

    @staticmethod
    def _promote(one, two):
        if type(one) is type(two):
            if isinstance(two, RegularBinning) and (one.num != two.num or one.interval != two.interval):
                return one.toIrregularBinning(), two.toIrregularBinning()
            elif isinstance(two, EdgesBinning) and (not _sameedges(one.edges, two.edges) or one.low_inclusive != two.low_inclusive or one.high_inclusive != two.high_inclusive):
                return one.toIrregularBinning(), two.toIrregularBinning()
            elif isinstance(two, SparseRegularBinning) and (one.bin_width != two.bin_width or one.origin != two.origin or one.low_inclusive != two.low_inclusive or one.high_inclusive != two.high_inclusive):
                return one.toIrregularBinning(), two.toIrregularBinning()
            else:
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

    def _selfmap(self, flows, index):
        selfmap = numpy.empty(self._binshape(), dtype=numpy.int64)
        belows = BinLocation._belows(flows)
        aboves = BinLocation._aboves(flows)
        selfmap[len(belows) : len(belows) + len(index)] = index
        i = 0
        for loc, pos in belows:
            selfmap[i] = pos
            i += 1
        i += len(index)
        for loc, pos in aboves:
            selfmap[i] = pos
            i += 1
        return selfmap

    def _getindex_general(self, where, length, loc_underflow, loc_overflow, loc_nanflow):
        if where is None:
            return (None,)

        elif isinstance(where, slice) and where.start is None and where.stop is None and where.step is None:
            start, stop, step = where.indices(length)
            shift = len(BinLocation._belows([(loc_underflow,), (loc_overflow,), (loc_nanflow,)]))
            return (slice(start + shift, stop + shift, step),)

        elif isinstance(where, slice):
            include_underflow, include_overflow = False, False
            if where.step is None or where.step > 0:
                if where.start == -numpy.inf:
                    include_underflow = True
                    where = slice(None, where.stop, where.step)
                if where.stop == numpy.inf:
                    include_overflow = True
                    where = slice(where.start, None, where.step)
            else:
                if where.start == numpy.inf:
                    include_overflow = True
                    where = slice(None, where.stop, where.step)
                if where.stop == -numpy.inf:
                    include_underflow = True
                    where = slice(where.start, None, where.step)

            start, stop, step = where.indices(length)
            shift = len(BinLocation._belows([(loc_underflow,), (loc_overflow,), (loc_nanflow,)]))

            if not include_underflow and not include_overflow:
                return (slice(start + shift, stop + shift, step),)
            else:
                if step > 0:
                    underspot, overspot = 0, -1
                    indexshift = int(include_underflow)
                    start = max(start, 0)
                    stop = min(stop, length)
                else:
                    underspot, overspot = -1, 0
                    indexshift = int(include_overflow)
                    start = min(start, length - 1)
                    stop = max(stop, -1)

                values = numpy.arange(start + shift, stop + shift, step)
                index = numpy.empty(len(values) + int(include_underflow) + int(include_overflow), dtype=numpy.int64)
                under, over, nan = BinLocation._positions(loc_underflow, loc_overflow, loc_nanflow, length)

                if include_underflow:
                    if under is None:
                        raise IndexError("index -inf requested but this {0} has no underflow".format(type(self).__name__))
                    index[underspot] = under
                if include_overflow:
                    if over is None:
                        raise IndexError("index +inf requested but this {0} has no overflow".format(type(self).__name__))
                    index[overspot] = over
                index[indexshift : indexshift + len(values)] = values
                return (index,)

        elif isinstance(where, (numbers.Real, numpy.floating)) and where == -numpy.inf:
            under, over, nan = BinLocation._positions(loc_underflow, loc_overflow, loc_nanflow, length)
            if under is None:
                raise IndexError("index -inf requested but this {0} has no underflow".format(type(self).__name__))
            return (under,)

        elif isinstance(where, (numbers.Real, numpy.floating)) and where == numpy.inf:
            under, over, nan = BinLocation._positions(loc_underflow, loc_overflow, loc_nanflow, length)
            if over is None:
                raise IndexError("index +inf requested but this {0} has no overflow".format(type(self).__name__))
            return (over,)

        elif isinstance(where, (numbers.Real, numpy.floating)) and numpy.isnan(where):
            under, over, nan = BinLocation._positions(loc_underflow, loc_overflow, loc_nanflow, length)
            if nan is None:
                raise IndexError("index nan requested but this {0} has no nanflow".format(type(self).__name__))
            return (nan,)

        elif not isinstance(where, (bool, numpy.bool, numpy.bool_)) and isinstance(where, (numbers.Integral, numpy.integer)):
            i = where
            if i < 0:
                i += length
            if not 0 <= i < length:
                raise IndexError("index out of bounds for {0}: {1}".format(type(self).__name__, where))
            shift = len(BinLocation._belows([(loc_underflow,), (loc_overflow,), (loc_nanflow,)]))
            return (i + shift,)

        else:
            where = numpy.array(where, copy=False)

            if len(where.shape) == 1 and issubclass(where.dtype.type, numpy.floating):
                flows = [(loc_underflow, -numpy.inf), (loc_overflow, numpy.inf), (loc_nanflow, numpy.nan)]
                index = numpy.empty(where.shape, dtype=numpy.int64)
                okay = numpy.isfinite(where)
                pos = 0
                for loc, tag in BinLocation._belows(flows):
                    if numpy.isnan(tag):
                        this = numpy.isnan(where)
                    else:
                        this = (where == tag)
                    index[this] = pos
                    okay[this] = True
                    pos += 1
                shift = pos
                pos += length
                for loc, tag in BinLocation._aboves(flows):
                    if numpy.isnan(tag):
                        this = numpy.isnan(where)
                    else:
                        this = (where == tag)
                    index[this] = pos
                    okay[this] = True
                    pos += 1
                if not okay.all():
                    raise IndexError("forbidden indexes for this {0}: {1}".format(type(self).__name__, ", ".join(set(repr(x) for x in where[~okay]))))
                mask = numpy.isfinite(where)
                where = where[mask].astype(numpy.int64)
                index[mask] = numpy.where(where < 0, where + length, where) + shift
                return (index,)

            elif len(where.shape) == 1 and issubclass(where.dtype.type, numpy.integer):
                shift = len(BinLocation._belows([(loc_underflow,), (loc_overflow,), (loc_nanflow,)]))
                index = numpy.where(where < 0, where + length, where) + shift
                return (index,)

            elif len(where.shape) == 1 and issubclass(where.dtype.type, (numpy.bool, numpy.bool_)):
                if len(where) != length:
                    raise IndexError("boolean index of length {0} does not match this {1} of length {2}".format(len(where), type(self).__name__, length))
                flows = [(loc_underflow,), (loc_overflow,), (loc_nanflow,)]
                shiftbelow = len(BinLocation._belows(flows))
                shiftabove = len(BinLocation._aboves(flows))
                if shiftbelow + shiftabove == 0:
                    return (where,)
                else:
                    belowmask = numpy.zeros(shiftbelow, dtype=where.dtype)
                    abovemask = numpy.zeros(shiftabove, dtype=where.dtype)
                    return (numpy.concatenate([belowmask, where, abovemask]),)

            else:
                raise TypeError("{0} accepts an integer, -inf/inf/nan, an integer/-inf/inf/nan slice (`:`), ellipsis (`...`), projection (`None`), an array of integers/-inf/inf/nan, or an array of booleans, not {1}".format(type(self).__name__, repr(where)))

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
    locations  = [below3, below2, below1, nonexistent, above1, above2, above3]
    _locations = {-3: below3, -2: below2, -1: below1, 0: nonexistent, 1: above1, 2: above2, 3: above3}

    def __init__(self):
        raise TypeError("{0} is an abstract base class; do not construct".format(type(self).__name__))

    @classmethod
    def _belows(cls, tuples):
        out = [x for x in tuples if x[0] is not None and x[0].value < cls.nonexistent.value]
        out.sort(key=lambda x: x[0].value)
        return out

    @classmethod
    def _aboves(cls, tuples):
        out = [x for x in tuples if x[0] is not None and x[0].value > cls.nonexistent.value]
        out.sort(key=lambda x: x[0].value)
        return out

    @classmethod
    def _positions(cls, loc_underflow, loc_overflow, loc_nanflow, length):
        flows = []
        if loc_underflow is not None:
            flows.append((loc_underflow, "u"))
        if loc_overflow is not None:
            flows.append((loc_overflow, "o"))
        if loc_nanflow is not None:
            flows.append((loc_nanflow, "n"))

        under, over, nan = None, None, None
        pos = 0
        for loc, letter in cls._belows(flows):
            if letter == "u":
                under = pos
            elif letter == "o":
                over = pos
            else:
                nan = pos
            pos += 1

        pos += length
        for loc, letter in cls._aboves(flows):
            if letter == "u":
                under = pos
            elif letter == "o":
                over = pos
            else:
                nan = pos
            pos += 1
        
        return under, over, nan

################################################# IntegerBinning

class IntegerBinning(Binning, BinLocation):
    _params = {
        "min":       stagg.checktype.CheckInteger("IntegerBinning", "min", required=True),
        "max":       stagg.checktype.CheckInteger("IntegerBinning", "max", required=True),
        "loc_underflow": stagg.checktype.CheckEnum("IntegerBinning", "loc_underflow", required=False, choices=BinLocation.locations, intlookup=BinLocation._locations),
        "loc_overflow":  stagg.checktype.CheckEnum("IntegerBinning", "loc_overflow", required=False, choices=BinLocation.locations, intlookup=BinLocation._locations),
        }

    min       = typedproperty(_params["min"])
    max       = typedproperty(_params["max"])
    loc_underflow = typedproperty(_params["loc_underflow"])
    loc_overflow  = typedproperty(_params["loc_overflow"])

    description = "Splits a one-dimensional axis into a contiguous set of integer-valued bins."
    validity_rules = ("The *min* must be strictly less than the *max*.",
                      "The *loc_underflow* and *loc_overflow* must not be equal unless they are `nonexistent`.")
    long_description = """
This binning is intended for one-dimensional, integer-valued data in a compact range. The *min* and *max* values are both inclusive, so the number of bins is `+1 + max - min+`.

If *loc_underflow* and *loc_overflow* are `nonexistent`, then there are no slots in the <<Histogram>> counts or <<BinnedEvaluatedFunction>> values for underflow or overflow. If they are `below`, then their slots precede the normal bins, if `above`, then their slots follow the normal bins, and their order is in sequence: `below3`, `below2`, `below1`, (normal bins), `above1`, `above2`, `above3`.
"""

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
        return (1 + self.max - self.min + int(self.loc_underflow != self.nonexistent) + int(self.loc_overflow != self.nonexistent),)

    @property
    def dimensions(self):
        return 1

    def _toflatbuffers(self, builder):
        stagg.stagg_generated.IntegerBinning.IntegerBinningStart(builder)
        stagg.stagg_generated.IntegerBinning.IntegerBinningAddMin(builder, self.min)
        stagg.stagg_generated.IntegerBinning.IntegerBinningAddMax(builder, self.max)
        if self.loc_underflow != self.nonexistent:
            stagg.stagg_generated.IntegerBinning.IntegerBinningAddLocUnderflow(builder, self.loc_underflow.value)
        if self.loc_overflow != self.nonexistent:
            stagg.stagg_generated.IntegerBinning.IntegerBinningAddLocOverflow(builder, self.loc_overflow.value)
        return stagg.stagg_generated.IntegerBinning.IntegerBinningEnd(builder)

    def _dump(self, indent, width, end):
        args = ["min={0}".format(repr(self.min)), "max={0}".format(repr(self.max))]
        if self.loc_underflow != BinLocation.nonexistent:
            args.append("loc_underflow={0}".format(self.loc_underflow))
        if self.loc_overflow != BinLocation.nonexistent:
            args.append("loc_overflow={0}".format(self.loc_overflow))
        return _dumpline(self, args, indent, width, end)

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

    def _getindex(self, where):
        return self._getindex_general(where, 1 + self.max - self.min, self.loc_underflow, self.loc_overflow, None)

    def _getloc_flows(self, length, start, stop):
        loc = 0
        pos = length
        if self.loc_underflow != self.nonexistent or start > 0:
            loc += 1
            loc_underflow = BinLocation._locations[loc]
            pos_underflow = pos
            pos += 1
        else:
            loc_underflow = self.nonexistent
            pos_underflow = None
        if self.loc_overflow != self.nonexistent or stop < 1 + self.max - self.min:
            loc += 1
            loc_overflow = BinLocation._locations[loc]
            pos_overflow = pos
            pos += 1
        else:
            loc_overflow = self.nonexistent
            pos_overflow = None
        return loc_underflow, pos_underflow, loc_overflow, pos_overflow

    def _getloc(self, isiloc, where):
        if where is None:
            return None, (slice(None),)

        elif isinstance(where, slice) and where.start is None and where.stop is None and where.step is None:
            return self, (slice(None),)

        elif isinstance(where, slice):
            if isiloc:
                start, stop, step = where.indices(1 + self.max - self.min)
            else:
                start = self.min if where.start is None else where.start
                stop = self.max + 1 if where.stop is None else where.stop
                stop += 1    # inclusive because loc (not iloc) selections are inclusive
                start -= self.min
                stop -= self.min
                step = 1 if where.step is None else where.step
            if step != 1:
                raise IndexError("IntegerBinning slice step can only be 1 or None")
            start = max(start, 0)
            stop = min(stop, 1 + self.max - self.min)

            if stop - start <= 0:
                raise IndexError("slice {0}:{1} would result in no bins".format(where.start, where.stop))

            loc_underflow, pos_underflow, loc_overflow, pos_overflow = self._getloc_flows(stop - start, start, stop)

            binning = IntegerBinning(start + self.min, stop + self.min - 1, loc_underflow=loc_underflow, loc_overflow=loc_overflow)

            index = numpy.empty(1 + self.max - self.min, dtype=numpy.int64)
            index[:start] = pos_underflow
            index[start:stop] = numpy.arange(stop - start)
            index[stop:] = pos_overflow
            selfmap = self._selfmap([(self.loc_underflow, pos_underflow), (self.loc_overflow, pos_overflow)], index)

            return binning, (selfmap,)
                
        elif not isinstance(where, (bool, numpy.bool, numpy.bool_)) and isinstance(where, (numbers.Integral, numpy.integer)):
            i = where
            if i < 0:
                i += 1 + self.max - self.min
            if not 0 <= i < 1 + self.max - self.min:
                raise IndexError("index {0} is out of bounds".format(where))
            return self._getloc(isiloc, slice(i, i))

        elif isiloc:
            where = numpy.array(where, copy=False)
            if len(where.shape) == 1 and issubclass(where.dtype.type, (numpy.integer, numpy.bool, numpy.bool_)):
                return self.toCategoryBinning()._getloc(True, where)

        if isiloc:
            raise TypeError("IntegerBinning.iloc accepts an integer, an integer slice (`:`), ellipsis (`...`), projection (`None`), or an array of integers/booleans, not {0}".format(repr(where)))
        else:
            raise TypeError("IntegerBinning.loc accepts an integer, an integer slice (`:`), ellipsis (`...`), or projection (`None`), not {0}".format(repr(where)))

    def _restructure(self, other):
        assert isinstance(other, IntegerBinning)

        if self.min == other.min and self.max == other.max and self.loc_underflow == other.loc_underflow and self.loc_overflow == other.loc_overflow:
            return self, (None,), (None,)

        else:
            newmin = min(self.min, other.min)
            newmax = max(self.max, other.max)

            loc = 0
            pos = 1 + newmax - newmin
            if self.loc_underflow != self.nonexistent or other.loc_underflow != other.nonexistent:
                loc += 1
                loc_underflow = BinLocation._locations[loc]
                pos_underflow = pos
                pos += 1
            else:
                loc_underflow = self.nonexistent
                pos_underflow = None

            if self.loc_overflow != self.nonexistent or other.loc_overflow != other.nonexistent:
                loc += 1
                loc_overflow = BinLocation._locations[loc]
                pos_overflow = pos
                pos += 1
            else:
                loc_overflow = self.nonexistent
                pos_overflow = None

            othermap = other._selfmap([(other.loc_underflow, pos_underflow), (other.loc_overflow, pos_overflow)], numpy.arange(other.min - newmin, 1 + other.max - newmin, dtype=numpy.int64))

            if newmin == self.min and newmax == self.max and loc_underflow == self.loc_underflow and loc_overflow == self.loc_overflow:
                return self, (None,), (othermap,)
            else:
                selfmap = self._selfmap([(self.loc_underflow, pos_underflow), (self.loc_overflow, pos_overflow)], numpy.arange(self.min - newmin, 1 + self.max - newmin, dtype=numpy.int64))
                return IntegerBinning(newmin, newmax, loc_underflow=loc_underflow, loc_overflow=loc_overflow), (selfmap,), (othermap,)

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

    description = "Represents a real interval with inclusive (closed) or exclusive (open) endpoints."
    validity_rules = ("The *low* limit must be less than or equal to the *high* limit.",
                      "The *low* limit may only be equal to the *high* limit if at least one endpoint is inclusive (*low_inclusive* or *high_inclusive* is true). Such an interval would represent a single real value.")
    long_description = """
The position and size of the real interval is defined by *low* and *high*, and each endpoint is inclusive (closed) if *low_inclusive* or *high_inclusive*, respectively, is true. Otherwise, the endpoint is exclusive (open).

A single interval defines a <<RegularBinning>> and a set of intervals defines an <<IrregularBinning>>.
"""

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
        return stagg.stagg_generated.RealInterval.CreateRealInterval(builder, self.low, self.high, self.low_inclusive, self.high_inclusive)

    def _dump(self, indent, width, end):
        args = ["low={0}".format(repr(self.low)), "high={0}".format(repr(self.high))]
        if self.low_inclusive is not True:
            args.append("low_inclusive={0}".format(self.low_inclusive))
        if self.high_inclusive is not True:
            args.append("high_inclusive={0}".format(self.high_inclusive))
        return _dumpline(self, args, indent, width, end)

    def __hash__(self):
        return hash((RealInterval, self.low, self.high, self.low_inclusive, self.high_inclusive))

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
        "loc_underflow": stagg.checktype.CheckEnum("RealOverflow", "loc_underflow", required=False, choices=BinLocation.locations, intlookup=BinLocation._locations),
        "loc_overflow":  stagg.checktype.CheckEnum("RealOverflow", "loc_overflow", required=False, choices=BinLocation.locations, intlookup=BinLocation._locations),
        "loc_nanflow":   stagg.checktype.CheckEnum("RealOverflow", "loc_nanflow", required=False, choices=BinLocation.locations, intlookup=BinLocation._locations),
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

    description = "Underflow, overflow, and nanflow configuration for one-dimensional, real-valued data."
    validity_rules = ("The *loc_underflow*, *loc_overflow*, and *loc_nanflow* must not be equal unless they are `nonexistent`.",
                      u"The *minf_mapping* (\u2012\u221e mapping) can only be `missing`, `in_underflow`, or `in_nanflow`, not `in_overflow`.",
                      u"The *pinf_mapping* (+\u221e mapping) can only be `missing`, `in_overflow`, or `in_nanflow`, not `in_underflow`.")
    long_description = u"""
If *loc_underflow*, *loc_overflow*, and *loc_nanflow* are `nonexistent`, then there are no slots in the <<Histogram>> counts or <<BinnedEvaluatedFunction>> values for underflow, overflow, or nanflow. Underflow represents values smaller than the lower limit of the binning, overflow represents values larger than the upper limit of the binning, and nanflow represents floating point values that are `nan` (not a number). With the normal bins, underflow, overflow, and nanflow, every possible input value corresponds to some bin.

If any of the *loc_underflow*, *loc_overflow*, and *loc_nanflow* are `below`, then their slots precede the normal bins, if `above`, then their slots follow the normal bins, and their order is in sequence: `below3`, `below2`, `below1`, (normal bins), `above1`, `above2`, `above3`. It is possible to represent a histogram counts buffer with the three special bins in any position relative to the normal bins.

The *minf_mapping* specifies whether \u2012\u221e values were ignored when the histogram was filled (`missing`), are in the underflow bin (`in_underflow`) or are in the nanflow bin (`in_nanflow`). The *pinf_mapping* specifies whether +\u221e values were ignored when the histogram was filled (`missing`), are in the overflow bin (`in_overflow`) or are in the nanflow bin (`in_nanflow`). Thus, it would be possible to represent a histogram that was filled with finite underflow/overflow bins and a generic bin for all three non-finite floating point states.
"""

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
            raise ValueError("RealOverflow.minf_mapping (-inf mapping) can only be missing, in_underflow, or in_nanflow, not in_overflow")
        if self.pinf_mapping == self.in_underflow:
            raise ValueError("RealOverflow.pinf_mapping (+inf mapping) can only be missing, in_overflow, or in_nanflow, not in_underflow")

    def _numbins(self):
        return int(self.loc_underflow != self.nonexistent) + int(self.loc_overflow != self.nonexistent) + int(self.loc_nanflow != self.nonexistent)

    def _toflatbuffers(self, builder):
        return stagg.stagg_generated.RealOverflow.CreateRealOverflow(builder, self.loc_underflow.value, self.loc_overflow.value, self.loc_nanflow.value, self.minf_mapping.value, self.pinf_mapping.value, self.nan_mapping.value)

    def _dump(self, indent, width, end):
        args = []
        if self.loc_underflow != BinLocation.nonexistent:
            args.append("loc_underflow={0}".format(self.loc_underflow))
        if self.loc_overflow != BinLocation.nonexistent:
            args.append("loc_overflow={0}".format(self.loc_overflow))
        if self.loc_nanflow != BinLocation.nonexistent:
            args.append("loc_nanflow={0}".format(self.loc_nanflow))
        if self.minf_mapping != RealOverflow.in_underflow:
            args.append("minf_mapping={0}".format(self.minf_mapping))
        if self.pinf_mapping != RealOverflow.in_overflow:
            args.append("pinf_mapping={0}".format(self.pinf_mapping))
        if self.nan_mapping != RealOverflow.in_nanflow:
            args.append("nan_mapping={0}".format(self.nan_mapping))
        return _dumpline(self, args, indent, width, end)

    @staticmethod
    def _getloc(overflow, yes_underflow, yes_overflow, length):
        loc = 0
        pos = length
        if yes_underflow or (overflow is not None and overflow.loc_underflow != BinLocation.nonexistent):
            loc += 1
            loc_underflow = BinLocation._locations[loc]
            pos_underflow = pos
            pos += 1
        else:
            loc_underflow = BinLocation.nonexistent
            pos_underflow = None
        if yes_overflow or (overflow is not None and overflow.loc_overflow != BinLocation.nonexistent):
            loc += 1
            loc_overflow = BinLocation._locations[loc]
            pos_overflow = pos
            pos += 1
        else:
            loc_overflow = BinLocation.nonexistent
            pos_overflow = None
        if (overflow is not None and overflow.loc_nanflow != BinLocation.nonexistent):
            loc += 1
            loc_nanflow = BinLocation._locations[loc]
            pos_nanflow = pos
            pos += 1
        else:
            loc_nanflow = BinLocation.nonexistent
            pos_nanflow = None

        if overflow is None:
            minf_mapping = RealOverflow.missing
            pinf_mapping = RealOverflow.missing
            nan_mapping = RealOverflow.missing
        else:
            minf_mapping = overflow.minf_mapping
            pinf_mapping = overflow.pinf_mapping
            nan_mapping = overflow.nan_mapping

        if overflow is not None or yes_underflow or yes_overflow:
            overflow = RealOverflow(loc_underflow=loc_underflow,
                                    loc_overflow=loc_overflow,
                                    loc_nanflow=loc_nanflow,
                                    minf_mapping=minf_mapping,
                                    pinf_mapping=pinf_mapping,
                                    nan_mapping=nan_mapping)
        else:
            overflow = None

        return overflow, loc_underflow, pos_underflow, loc_overflow, pos_overflow, loc_nanflow, pos_nanflow

    @staticmethod
    def _common(one, two, pos):
        if one is not None and two is not None:
            if one.minf_mapping != two.minf_mapping:
                if (one.minf_mapping == one.in_underflow or two.minf_mapping == two.in_underflow) and (one.loc_underflow != one.nonexistent and two.loc_underflow != two.nonexistent):
                    raise ValueError("cannot combine RealOverflows in which one maps -inf to underflow, the other doesn't, and both underflows exist")
                if (one.minf_mapping == one.in_nanflow or two.minf_mapping == two.in_nanflow) and (one.loc_nanflow != one.nonexistent and two.loc_nanflow != two.nonexistent):
                    raise ValueError("cannot combine RealOverflows in which one maps -inf to nanflow, the other doesn't, and both nanflows exist")

            if one.pinf_mapping != two.pinf_mapping:
                if (one.pinf_mapping == one.in_overflow or two.pinf_mapping == two.in_overflow) and (one.loc_overflow != one.nonexistent and two.loc_overflow != two.nonexistent):
                    raise ValueError("cannot combine RealOverflows in which one maps +inf to overflow, the other doesn't, and both overflows exist")
                if (one.pinf_mapping == one.in_nanflow or two.pinf_mapping == two.in_nanflow) and (one.loc_nanflow != one.nonexistent and two.loc_nanflow != two.nonexistent):
                    raise ValueError("cannot combine RealOverflows in which one maps +inf to nanflow, the other doesn't, and both nanflows exist")

            if one.nan_mapping != two.nan_mapping:
                if (one.nan_mapping == one.in_underflow or two.nan_mapping == two.in_underflow) and (one.loc_underflow != one.nonexistent and two.loc_underflow != two.nonexistent):
                    raise ValueError("cannot combine RealOverflows in which one maps nan to underflow, the other doesn't, and both underflows exist")
                if (one.nan_mapping == one.in_overflow or two.nan_mapping == two.in_overflow) and (one.loc_overflow != one.nonexistent and two.loc_overflow != two.nonexistent):
                    raise ValueError("cannot combine RealOverflows in which one maps nan to overflow, the other doesn't, and both overflows exist")
                if (one.nan_mapping == one.in_nanflow or two.nan_mapping == two.in_nanflow) and (one.loc_nanflow != one.nonexistent and two.loc_nanflow != two.nonexistent):
                    raise ValueError("cannot combine RealOverflows in which one maps nan to nanflow, the other doesn't, and both nanflows exist")

        loc = 0
        if (one is not None and one.loc_underflow != one.nonexistent) or (two is not None and two.loc_underflow != two.nonexistent):
            loc += 1
            loc_underflow = BinLocation._locations[loc]
            pos_underflow = pos
            pos += 1
        else:
            loc_underflow = BinLocation.nonexistent
            pos_underflow = None

        if (one is not None and one.loc_overflow != one.nonexistent) or (two is not None and two.loc_overflow != two.nonexistent):
            loc += 1
            loc_overflow = BinLocation._locations[loc]
            pos_overflow = pos
            pos += 1
        else:
            loc_overflow = BinLocation.nonexistent
            pos_overflow = None

        if (one is not None and one.loc_nanflow != one.nonexistent) or (two is not None and two.loc_nanflow != two.nonexistent):
            loc += 1
            loc_nanflow = BinLocation._locations[loc]
            pos_nanflow = pos
            pos += 1
        else:
            loc_nanflow = BinLocation.nonexistent
            pos_nanflow = None

        if loc_underflow == BinLocation.nonexistent and loc_overflow == BinLocation.nonexistent and loc_nanflow == BinLocation.nonexistent:
            overflow = None
        else:
            if one is not None:
                minf_mapping = one.minf_mapping
                pinf_mapping = one.pinf_mapping
                nan_mapping  = one.nan_mapping
            else:
                minf_mapping = two.minf_mapping
                pinf_mapping = two.pinf_mapping
                nan_mapping  = two.nan_mapping
            overflow = RealOverflow(loc_underflow=loc_underflow, loc_overflow=loc_overflow, loc_nanflow=loc_nanflow, minf_mapping=minf_mapping, pinf_mapping=pinf_mapping, nan_mapping=nan_mapping)

        return overflow, pos_underflow, pos_overflow, pos_nanflow

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

    description = "Splits a one-dimensional axis into an ordered, abutting set of equal-sized real intervals."
    validity_rules = ("The *interval.low* and *interval.high* limits must both be finite.",
                      "The *interval.low_inclusive* and *interval.high_inclusive* cannot both be true. (They can both be false, which allows for infinitesimal gaps between bins.)")
    long_description = """
This binning is intended for one-dimensional, real-valued data in a compact range. The limits of this range are specified in a single <<RealInterval>>, and the number of subdivisions is *num*.

The existence and positions of any underflow, overflow, and nanflow bins, as well as how non-finite values were handled during filling, are contained in the <<RealOverflow>>.

If the binning is *circular*, then it represents a finite segment in which *interval.low* is topologically identified with *interval.high*. This could be used to convert [\u2012\u03c0, \u03c0) intervals into [0, 2\u03c0) intervals, for instance.

*See also:*

   * <<RegularBinning>>: for ordered, equal-sized, abutting real intervals.
   * <<EdgesBinning>>: for ordered, any-sized, abutting real intervals.
   * <<IrregularBinning>>: for unordered, any-sized real intervals (that may even overlap).
   * <<SparseRegularBinning>>: for unordered, equal-sized real intervals aligned to a regular grid, but only need to be defined if the bin content is not empty.
"""

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

    @property
    def dimensions(self):
        return 1

    def _toflatbuffers(self, builder):
        stagg.stagg_generated.RegularBinning.RegularBinningStart(builder)
        stagg.stagg_generated.RegularBinning.RegularBinningAddNum(builder, self.num)
        stagg.stagg_generated.RegularBinning.RegularBinningAddInterval(builder, self.interval._toflatbuffers(builder))
        if self.overflow is not None:
            stagg.stagg_generated.RegularBinning.RegularBinningAddOverflow(builder, self.overflow._toflatbuffers(builder))
        if self.circular is not False:
            stagg.stagg_generated.RegularBinning.RegularBinningAddCircular(builder, self.circular)
        return stagg.stagg_generated.RegularBinning.RegularBinningEnd(builder)

    def _dump(self, indent, width, end):
        args = ["num={0}".format(repr(self.num)), "interval={0}".format(_dumpeq(self.interval._dump(indent + "    ", width, end), indent, end))]
        if self.overflow is not None:
            args.append("overflow={0}".format(_dumpeq(self.overflow._dump(indent + "    ", width, end), indent, end)))
        if self.circular is not False:
            args.append("circular={0}".format(self.circular))
        return _dumpline(self, args, indent, width, end)

    def toEdgesBinning(self):
        edges = numpy.linspace(self.interval.low, self.interval.high, self.num + 1).tolist()
        overflow = None if self.overflow is None else self.overflow.detached()
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
        overflow = None if self.overflow is None else self.overflow.detached()
        return SparseRegularBinning(bins, bin_width, origin=origin, overflow=overflow, low_inclusive=self.interval.low_inclusive, high_inclusive=self.interval.high_inclusive)

    def _getindex(self, where):
        loc_underflow = None if self.overflow is None else self.overflow.loc_underflow
        loc_overflow = None if self.overflow is None else self.overflow.loc_overflow
        loc_nanflow = None if self.overflow is None else self.overflow.loc_nanflow
        return self._getindex_general(where, self.num, loc_underflow, loc_overflow, loc_nanflow)

    def _getloc(self, isiloc, where):
        if where is None:
            return None, (slice(None),)

        elif isinstance(where, slice) and where.start is None and where.stop is None and where.step is None:
            return self, (slice(None),)

        elif isinstance(where, slice):
            bin_width = (self.interval.high - self.interval.low) / float(self.num)
            if isiloc:
                start, stop, step = where.indices(self.num)
            else:
                start = 0 if where.start is None else int(math.trunc((where.start - self.interval.low) / bin_width))
                stop = self.num if where.stop is None else int(math.ceil((where.stop - self.interval.low) / bin_width))
                step = 1 if where.step is None else int(round(where.step / bin_width))
            if step <= 0:
                raise IndexError("slice step cannot be zero or negative")
            start = max(start, 0)
            stop = min(stop, self.num)
            length = (stop - start) // step
            stop = start + step*length

            if stop - start <= 0:
                raise IndexError("slice {0}:{1} would result in no bins".format(where.start, where.stop))

            yes_underflow = (start != 0)
            yes_overflow = (stop != self.num)
            overflow, loc_underflow, pos_underflow, loc_overflow, pos_overflow, loc_nanflow, pos_nanflow = RealOverflow._getloc(self.overflow, yes_underflow, yes_overflow, length)
            if yes_underflow and self.interval.low == -numpy.inf and self.interval.low_inclusive:
                overflow.minf_mapping = RealOverflow.in_underflow
            if yes_overflow and self.interval.high == numpy.inf and self.interval.high_inclusive:
                overflow.pinf_mapping = RealOverflow.in_overflow

            interval = RealInterval(self.interval.low + start*bin_width,
                                    self.interval.low + stop*bin_width,
                                    low_inclusive=self.interval.low_inclusive,
                                    high_inclusive=self.interval.high_inclusive)
            circular = self.circular and not yes_underflow and not yes_overflow
            binning = RegularBinning(length, interval, overflow=overflow, circular=circular)

            index = numpy.empty(self.num, dtype=numpy.int64)
            index[:start] = pos_underflow
            index[start:stop] = numpy.repeat(numpy.arange(length), step)
            index[stop:] = pos_overflow
            flows = [] if self.overflow is None else [(self.overflow.loc_underflow, pos_underflow), (self.overflow.loc_overflow, pos_overflow), (self.overflow.loc_nanflow, pos_nanflow)]
            selfmap = self._selfmap(flows, index)

            return binning, (selfmap,)

        elif not isiloc and not isinstance(where, (bool, numpy.bool, numpy.bool_)) and isinstance(where, (numbers.Real, numpy.integer, numpy.floating)):
            i = int(math.trunc((where - self.interval.low) / bin_width))
            if not 0 <= i < self.num:
                raise IndexError("index {0} is out of bounds".format(where))
            return self._getloc(True, slice(i, i))

        elif isiloc and not isinstance(where, (bool, numpy.bool, numpy.bool_)) and isinstance(where, (numbers.Integral, numpy.integer)):
            i = where
            if i < 0:
                i += self.num
            if not 0 <= i < self.num:
                raise IndexError("index {0} is out of bounds".format(where))
            return self._getloc(True, slice(i, i))

        elif isiloc:
            where = numpy.array(where, copy=False)
            if len(where.shape) == 1 and issubclass(where.dtype.type, (numpy.integer, numpy.bool, numpy.bool_)):
                return self.toIrregularBinning()._getloc(True, where)

        if isiloc:
            raise TypeError("RegularBinning.iloc accepts an integer, an integer slice (`:`), ellipsis (`...`), projection (`None`), or an array of integers/booleans, not {0}".format(repr(where)))
        else:
            raise TypeError("RegularBinning.loc accepts a number, a real-valued slice (`:`), ellipsis (`...`), or projection (`None`), not {0}".format(repr(where)))

    def _restructure(self, other):
        assert isinstance(other, RegularBinning)
        if self.num != other.num:
            raise ValueError("cannot add RegularBinnings because they have different nums: {0} vs {1}".format(self.num, other.num))
        if self.interval != other.interval:
            raise ValueError("cannot add RegularBinnings because they have different intervals: {0} vs {1}".format(self.interval, other.interval))

        circular = self.circular and other.circular

        if self.overflow == other.overflow:
            if self.circular == circular:
                return self, (None,), (None,)
            else:
                return RegularBinning(self.num, self.interval.detached(reclaim=True), overflow=(None if self.overflow is None else self.overflow.detached(reclaim=True)), circular=circular), (None,), (None,)

        else:
            overflow, pos_underflow, pos_overflow, pos_nanflow = RealOverflow._common(self.overflow, other.overflow, self.num)

            othermap = other._selfmap([] if other.overflow is None else [(other.overflow.loc_underflow, pos_underflow), (other.overflow.loc_overflow, pos_overflow), (other.overflow.loc_nanflow, pos_nanflow)],
                                      numpy.arange(other.num, dtype=numpy.int64))

            if (self.overflow is None and overflow is None) or (self.overflow is not None and self.overflow.loc_underflow == overflow.loc_underflow and self.overflow.loc_overflow == overflow.loc_overflow and self.overflow.loc_nanflow == overflow.loc_nanflow):
                if self.circular == circular:
                    return self, (None,), (othermap,)
                else:
                    return RegularBinning(self.num, self.interval.detached(reclaim=True), overflow=(None if self.overflow is None else self.overflow.detached(reclaim=True)), circular=circular), (None,), (othermap,)

            else:
                selfmap = self._selfmap([] if self.overflow is None else [(self.overflow.loc_underflow, pos_underflow), (self.overflow.loc_overflow, pos_overflow), (self.overflow.loc_nanflow, pos_nanflow)],
                                        numpy.arange(self.num, dtype=numpy.int64))
                return RegularBinning(self.num, self.interval.detached(reclaim=True), overflow=overflow, circular=circular), (selfmap,), (othermap,)

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
        "bin_width":   stagg.checktype.CheckNumber("HexagonalBinning", "bin_width", required=False, min=0.0, min_inclusive=False, max_inclusive=False),
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
    bin_width   = typedproperty(_params["bin_width"])
    qangle      = typedproperty(_params["qangle"])
    qoverflow   = typedproperty(_params["qoverflow"])
    roverflow   = typedproperty(_params["roverflow"])

    description = "Splits a two-dimensional axis into a tiling of equal-sized hexagons."
    validity_rules = ("The *qmin* must be strictly less than the *qmax*.",
                      "The *rmin* must be strictly less than the *rmax*.")
    long_description = u"""
This binning is intended for two-dimensional, real-valued data in a compact region. Hexagons tile a two-dimensional plane, just as rectangles do, but whereas a rectangular tiling can be represented by two <<RegularBinning>> axes, hexagonal binning requires a special binning. Some advantages of hexagonal binning are https://www.meccanismocomplesso.org/hexagonal-binning[described here].

As with any other binning, integer-valued indexes in the <<Histogram>> counts or <<BinnedEvaluatedFunction>> values are mapped to values in the data space. However, rather than mapping a single integer slot position to an integer, real interval, or categorical data value, two integers from a rectangular integer grid are mapped to hexagonal tiles. The integers are labeled `q` and `r`, with `q` values between *qmin* and *qmax* (inclusive) and `r` values between *rmin* and *rmax* (inclusive). The total number of bins is `(1 + qmax - qmin)*(1 + rmax - rmin)`. Data coordinates are labeled `x` and `y`.

There are several different schemes for mapping integer rectangles to hexagonal tiles; we use the ones https://www.redblobgames.com/grids/hexagons[defined here]: `offset`, `doubled_offset`, `cube_xy`, `cube_yz`, `cube_xz`, specified by the *coordinates* property. The center of the `q = 0, r = 0` tile is at *xorigin*, *yorigin*.

In "`pointy topped`" coordinates, *qangle* is zero if increasing `q` is collinear with increasing `x`, and this angle ranges from \u2012\u03c0/2, if increasing `q` is collinear with decreasing `y`, to \u03c0/2, if increasing `q` is collinear with increasing `y`. The *bin_width* is the shortest distance between adjacent tile centers: the line between tile centers crosses the border between tiles at a right angle.

A roughly but not exactly rectangular region of `x` and `y` fall within a slot in `q` and `r`. Overflows, underflows, and nanflows, converted to floating point `q` and `r`, are represented by overflow, underflow, and nanflow bins in *qoverflow* and *roverflow*. Note that the total number of bins is strictly multiplicative (as it would be for a rectangular with two <<RegularBinning>> axes): the total number of bins is the number of normal `q` bins plus any overflows times the number of normal `r` bins plus any overflows. That is, all `r` bins are represented for each `q` bin, even overflow `q` bins.
"""

    def __init__(self, qmin, qmax, rmin, rmax, coordinates=offset, xorigin=0.0, yorigin=0.0, qangle=0.0, bin_width=1.0, qoverflow=None, roverflow=None):
        self.qmin = qmin
        self.qmax = qmax
        self.rmin = rmin
        self.rmax = rmax
        self.coordinates = coordinates
        self.xorigin = xorigin
        self.yorigin = yorigin
        self.qangle = qangle
        self.bin_width = bin_width
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

    @property
    def dimensions(self):
        return 2

    def _toflatbuffers(self, builder):
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
        if self.bin_width != 1.0:
            stagg.stagg_generated.HexagonalBinning.HexagonalBinningAddBinWidth(builder, self.bin_width)
        if self.qoverflow is not None:
            stagg.stagg_generated.HexagonalBinning.HexagonalBinningAddQoverflow(builder, self.qoverflow._toflatbuffers(builder))
        if self.roverflow is not None:
            stagg.stagg_generated.HexagonalBinning.HexagonalBinningAddRoverflow(builder, self.roverflow._toflatbuffers(builder))
        return stagg.stagg_generated.HexagonalBinning.HexagonalBinningEnd(builder)

    def _dump(self, indent, width, end):
        args = ["qmin={0}".format(repr(self.qmin)), "qmax={0}".format(repr(self.qmax)), "rmin={0}".format(repr(self.rmin)), "rmax={0}".format(repr(self.rmax))]
        if self.coordinates != HexagonalBinning.offset:
            args.append("coordinates={0}".format(repr(self.coordinates)))
        if self.xorigin != 0.0:
            args.append("xorigin={0}".format(repr(self.xorigin)))
        if self.yorigin != 0.0:
            args.append("yorigin={0}".format(repr(self.yorigin)))
        if self.qangle != 0.0:
            args.append("qangle={0}".format(repr(self.qangle)))
        if self.bin_width != 1.0:
            args.append("bin_width={0}".format(repr(self.bin_width)))
        if self.qoverflow is not None:
            args.append("qoverflow={0}".format(_dumpeq(self.qoverflow._dump(indent + "    ", width, end), indent, end)))
        if self.roverflow is not None:
            args.append("roverflow={0}".format(_dumpeq(self.roverflow._dump(indent + "    ", width, end), indent, end)))
        return _dumpline(self, args, indent, width, end)

    def _getindex(self, where1, where2):
        raise NotImplementedError

    def _getloc(self, isiloc, where1, where2):
        if where1 is None and where2 is None:
            return None, (slice(None), slice(None))

        elif isinstance(where1, slice) and where1.start is None and where1.stop is None and where1.step is None and isinstance(where2, slice) and where2.start is None and where2.stop is None and where2.step is None:
            return self, (slice(None), slice(None))

        elif isinstance(where1, slice) and isinstance(where2, slice):
            raise NotImplementedError

        elif not isiloc and not isinstance(where1, (bool, numpy.bool, numpy.bool_)) and not isinstance(where2, (bool, numpy.bool, numpy.bool_)) and isinstance(where1, (numbers.Real, numpy.integer, numpy.floating)) and isinstance(where2, (numbers.Real, numpy.integer, numpy.floating)):
            raise NotImplementedError

        elif isiloc and not isinstance(where1, (bool, numpy.bool, numpy.bool_)) and not isinstance(where2, (bool, numpy.bool, numpy.bool_)) and isinstance(where1, (numbers.Integral, numpy.integer)) and isinstance(where2, (numbers.Integral, numpy.integer)):
            raise NotImplementedError

        elif isiloc:
            where1 = numpy.array(where1, copy=False)
            where2 = numpy.array(where2, copy=False)
            if len(where1.shape) == 1 and len(where2.shape) == 1 and issubclass(where1.dtype.type, (numpy.integer, numpy.bool, numpy.bool_)) and issubclass(where2.dtype.type, (numpy.integer, numpy.bool, numpy.bool_)):
                raise NotImplementedError

        if isiloc:
            raise TypeError("HexagonalBinning.iloc accepts two integers, two integer slices (`:`), ellipsis (`...`), two projections (`None`), or two arrays of integers/booleans, not {0} and {1}".format(repr(where1), repr(where2)))
        else:
            raise TypeError("HexagonalBinning.loc accepts two numbers, two real-valued slices (`:`), ellipsis (`...`), or two projections (`None`), not {0} and {1}".format(repr(where1), repr(where2)))

    def _restructure(self, other):
        assert isinstance(other, HexagonalBinning)
        if self == other:
            return self, (None,), (None,)
        else:
            raise NotImplementedError

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

    description = "Splits a one-dimensional axis into an ordered, abutting set of any-sized real intervals."
    validity_rules = ("All *edges* must be finite and strictly increasing.",
                      "An *edges* of length 1 is only allowed if *overflow* is non-null with at least one underflow, overflow, or nanflow bin.",
                      "The *low_inclusive* and *high_inclusive* cannot both be true. (They can both be false, which allows for infinitesimal gaps between bins.)")
    long_description = """
This binning is intended for one-dimensional, real-valued data in a compact range. The limits of this range and the size of each bin are defined by *edges*, which are the edges _between_ the bins. Since they are edges between bins, the number of non-overflow bins is `len(edges) - 1`. The degenerate case of exactly one edge is only allowed if there are any underflow, overflow, or nanflow bins.

The existence and positions of any underflow, overflow, and nanflow bins, as well as how non-finite values were handled during filling, are contained in the <<RealOverflow>>.

If *low_inclusive* is true, then all intervals between pairs of edges include the low edge. If *high_inclusive* is true, then all intervals between pairs of edges include the high edge.

If the binning is *circular*, then it represents a finite segment in which *interval.low* is topologically identified with *interval.high*. This could be used to convert [\u2012\u03c0, \u03c0) intervals into [0, 2\u03c0) intervals, for instance.

*See also:*

   * <<RegularBinning>>: for ordered, equal-sized, abutting real intervals.
   * <<EdgesBinning>>: for ordered, any-sized, abutting real intervals.
   * <<IrregularBinning>>: for unordered, any-sized real intervals (that may even overlap).
   * <<SparseRegularBinning>>: for unordered, equal-sized real intervals aligned to a regular grid, but only need to be defined if the bin content is not empty.
"""

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

    @property
    def dimensions(self):
        return 1

    @classmethod
    def _fromflatbuffers(cls, fb):
        out = cls.__new__(cls)
        out._flatbuffers = _MockFlatbuffers()
        out._flatbuffers.Edges = fb.EdgesAsNumpy
        out._flatbuffers.Overflow = fb.Overflow
        out._flatbuffers.LowInclusive = fb.LowInclusive
        out._flatbuffers.HighInclusive = fb.HighInclusive
        out._flatbuffers.Circular = fb.Circular
        return out

    def _toflatbuffers(self, builder):
        edgesbuf = self.edges.tostring()
        stagg.stagg_generated.EdgesBinning.EdgesBinningStartEdgesVector(builder, len(self.edges))
        builder.head = builder.head - len(edgesbuf)
        builder.Bytes[builder.head : builder.head + len(edgesbuf)] = edgesbuf
        edges = builder.EndVector(len(self.edges))

        stagg.stagg_generated.EdgesBinning.EdgesBinningStart(builder)
        stagg.stagg_generated.EdgesBinning.EdgesBinningAddEdges(builder, edges)
        if self.overflow is not None:
            stagg.stagg_generated.EdgesBinning.EdgesBinningAddOverflow(builder, self.overflow._toflatbuffers(builder))
        if self.low_inclusive is not True:
            stagg.stagg_generated.EdgesBinning.EdgesBinningAddLowInclusive(builder, self.low_inclusive)
        if self.high_inclusive is not False:
            stagg.stagg_generated.EdgesBinning.EdgesBinningAddHighInclusive(builder, self.high_inclusive)
        if self.circular is not False:
            stagg.stagg_generated.EdgesBinning.EdgesBinningAddCircular(builder, self.circular)
        return stagg.stagg_generated.EdgesBinning.EdgesBinningEnd(builder)

    def _dump(self, indent, width, end):
        args = ["edges={0}".format(_dumparray(self.edges, indent, end))]
        if self.overflow is not None:
            args.append("overflow={0}".format(_dumpeq(self.overflow._dump(indent + "    ", width, end), indent, end)))
        if self.low_inclusive is not True:
            args.append("low_inclusive={0}".format(repr(self.low_inclusive)))
        if self.high_inclusive is not False:
            args.append("high_inclusive={0}".format(repr(self.high_inclusive)))
        if self.circular is not False:
            args.append("circular={0}".format(repr(self.circular)))
        return _dumpline(self, args, indent, width, end)

    def toIrregularBinning(self):
        if self.low_inclusive and self.high_inclusive:
            raise ValueError("EdgesBinning.interval.low_inclusive and EdgesBinning.interval.high_inclusive cannot both be True")
        intervals = []
        for i in range(len(self.edges) - 1):
            intervals.append(RealInterval(self.edges[i], self.edges[i + 1], low_inclusive=self.low_inclusive, high_inclusive=self.high_inclusive))
        overflow = None if self.overflow is None else self.overflow.detached()
        return IrregularBinning(intervals, overflow=overflow)

    def toCategoryBinning(self, format="%g"):
        return self.toIrregularBinning().toCategoryBinning(format=format)

    def _getindex(self, where):
        loc_underflow = None if self.overflow is None else self.overflow.loc_underflow
        loc_overflow = None if self.overflow is None else self.overflow.loc_overflow
        loc_nanflow = None if self.overflow is None else self.overflow.loc_nanflow
        return self._getindex_general(where, len(self.edges) - 1, loc_underflow, loc_overflow, loc_nanflow)

    def _getloc(self, isiloc, where):
        if where is None:
            return None, (slice(None),)

        elif isinstance(where, slice) and where.start is None and where.stop is None and where.step is None:
            return self, (slice(None),)

        elif isinstance(where, slice):
            if isiloc:
                start, stop, step = where.indices(len(self.edges) - 1)
                if step <= 0:
                    raise IndexError("slice step cannot be zero or negative")
            else:
                if where.step is not None:
                    raise IndexError("EdgesBinning.loc slice cannot have a step")
                xstart = self.edges[0] if where.start is None else where.start
                xstop = self.edges[-1] if where.stop is None else where.stop
                start, stop = numpy.searchsorted(self.edges, (xstart, xstop), side="left")
                if self.edges[start] > xstart:
                    start -= 1
                step = 1
            start = max(start, 0)
            stop = min(stop, len(self.edges) - 1)
            length = (stop - start) // step
            stop = start + step*length

            if stop - start <= 0:
                raise IndexError("slice {0}:{1} would result in no bins".format(where.start, where.stop))

            yes_underflow = (start != 0)
            yes_overflow  = (stop != len(self.edges) - 1)
            overflow, loc_underflow, pos_underflow, loc_overflow, pos_overflow, loc_nanflow, pos_nanflow = RealOverflow._getloc(self.overflow, yes_underflow, yes_overflow, length)
            if yes_underflow and self.edges[0] == -numpy.inf and self.low_inclusive:
                overflow.minf_mapping = RealOverflow.in_underflow
            if yes_overflow and self.edges[-1] == numpy.inf and self.high_inclusive:
                overflow.pinf_mapping = RealOverflow.in_overflow

            circular = self.circular and not yes_underflow and not yes_overflow
            binning = EdgesBinning(self.edges[start:(stop + 1):step], overflow=overflow, low_inclusive=self.low_inclusive, high_inclusive=self.high_inclusive, circular=circular)

            index = numpy.empty(len(self.edges) - 1, dtype=numpy.int64)
            index[:start] = pos_underflow
            index[start:stop] = numpy.repeat(numpy.arange(length), step)
            index[stop:] = pos_overflow
            flows = [] if self.overflow is None else [(self.overflow.loc_underflow, pos_underflow), (self.overflow.loc_overflow, pos_overflow), (self.overflow.loc_nanflow, pos_nanflow)]
            selfmap = self._selfmap(flows, index)

            return binning, (selfmap,)

        elif not isiloc and not isinstance(where, (bool, numpy.bool, numpy.bool_)) and isinstance(where, (numbers.Real, numpy.integer, numpy.floating)):
            i = numpy.searchsorted(self.edges, where, side="left")
            if self.edges[i] > where:
                i -= 1
            if not 0 <= i < len(self.edges) - 1:
                raise IndexError("index {0} is out of bounds".format(where))
            return self._getloc(True, slice(i, i))

        elif isiloc and not isinstance(where, (bool, numpy.bool, numpy.bool_)) and isinstance(where, (numbers.Integral, numpy.integer)):
            i = where
            if i < 0:
                i += len(self.edges)
            if not 0 <= i < len(self.edges) - 1:
                raise IndexError("index {0} is out of bounds".format(where))
            return self._getloc(True, slice(i, i))

        else:
            raise IndexError("EdgesBinning.loc and .iloc only allow an integer, slice (`:`), ellipsis (`...`), or projection (`None`) as an index; .loc additionally allows floating point numbers")

    def _restructure(self, other):
        assert isinstance(other, EdgesBinning)
        if not _sameedges(self.edges, other.edges):
            raise ValueError("cannot add EdgesBinnings because they have different edges: {0} vs {1}".format(self.edges, other.edges))
        if self.low_inclusive != other.low_inclusive:
            raise ValueError("cannot add EdgesBinnings because they have different low_inclusives: {0} vs {1}".format(self.low_inclusive, other.low_inclusive))
        if self.high_inclusive != other.high_inclusive:
            raise ValueError("cannot add EdgesBinnings because they have different high_inclusives: {0} vs {1}".format(self.high_inclusive, other.high_inclusive))

        circular = self.circular and other.circular

        if self.overflow == other.overflow:
            if self.circular == circular:
                return self, (None,), (None,)
            else:
                return EdgesBinning(self.edges, overflow=(None if self.overflow is None else self.overflow.detached(reclaim=True)), low_inclusive=self.low_inclusive, high_inclusive=self.high_inclusive, circular=circular), (None,), (None,)

        else:
            overflow, pos_underflow, pos_overflow, pos_nanflow = RealOverflow._common(self.overflow, other.overflow, len(self.edges) - 1)

            othermap = other._selfmap([] if other.overflow is None else [(other.overflow.loc_underflow, pos_underflow), (other.overflow.loc_overflow, pos_overflow), (other.overflow.loc_nanflow, pos_nanflow)],
                                      numpy.arange(len(other.edges) - 1, dtype=numpy.int64))

            if (self.overflow is None and overflow is None) or (self.overflow is not None and self.overflow.loc_underflow == overflow.loc_underflow and self.overflow.loc_overflow == overflow.loc_overflow and self.overflow.loc_nanflow == overflow.loc_nanflow):
                if self.circular == circular:
                    return self, (None,), (othermap,)
                else:
                    return EdgesBinning(self.edges, overflow=(None if self.overflow is None else self.overflow.detached(reclaim=True)), low_inclusive=self.low_inclusive, high_inclusive=self.high_inclusive, circular=circular), (None,), (othermap,)

            else:
                selfmap = self._selfmap([] if self.overflow is None else [(self.overflow.loc_underflow, pos_underflow), (self.overflow.loc_overflow, pos_overflow), (self.overflow.loc_nanflow, pos_nanflow)],
                                        numpy.arange(len(self.edges) - 1, dtype=numpy.int64))
                return EdgesBinning(self.edges, overflow=overflow, low_inclusive=self.low_inclusive, high_inclusive=self.high_inclusive, circular=circular), (selfmap,), (othermap,)

################################################# IrregularBinning

class OverlappingFillStrategyEnum(Enum):
    base = "IrregularBinning"

class OverlappingFill(object):
    unspecified = OverlappingFillStrategyEnum("unspecified", stagg.stagg_generated.OverlappingFillStrategy.OverlappingFillStrategy.overfill_unspecified)
    all         = OverlappingFillStrategyEnum("all", stagg.stagg_generated.OverlappingFillStrategy.OverlappingFillStrategy.overfill_all)
    first       = OverlappingFillStrategyEnum("first", stagg.stagg_generated.OverlappingFillStrategy.OverlappingFillStrategy.overfill_first)
    last        = OverlappingFillStrategyEnum("last", stagg.stagg_generated.OverlappingFillStrategy.OverlappingFillStrategy.overfill_last)
    overlapping_fill_strategies = [unspecified, all, first, last]

class IrregularBinning(Binning, OverlappingFill):
    _params = {
        "intervals":        stagg.checktype.CheckVector("IrregularBinning", "intervals", required=True, type=RealInterval, minlen=1),
        "overflow":         stagg.checktype.CheckClass("IrregularBinning", "overflow", required=False, type=RealOverflow),
        "overlapping_fill": stagg.checktype.CheckEnum("IrregularBinning", "overlapping_fill", required=False, choices=OverlappingFill.overlapping_fill_strategies),
        }

    intervals        = typedproperty(_params["intervals"])
    overflow         = typedproperty(_params["overflow"])
    overlapping_fill = typedproperty(_params["overlapping_fill"])

    description = "Splits a one-dimensional axis into unordered, any-sized real intervals (that may even overlap)."
    validity_rules = ("The intervals, as defined by their *low*, *high*, *low_inclusive*, *high_inclusive* fields, must be unique.",)
    long_description = """
This binning is intended for one-dimensional, real-valued data. Unlike <<EdgesBinning>>, the any-sized intervals do not need to be abutting, so this binning can describe a distribution with large gaps.

The existence and positions of any underflow, overflow, and nanflow bins, as well as how non-finite values were handled during filling, are contained in the <<RealOverflow>>.

In fact, the intervals are not even required to be non-overlapping. A data value may correspond to zero, one, or more than one bin. The latter case raises the question of which bin was filled by a value that corresponds to multiple bins: the *overlapping_fill* strategy may be `unspecified` if we don't know, `all` if every corresponding bin was filled, `first` if only the first match was filled, and `last` if only the last match was filled.

Irregular bins are usually not directly created by histogramming libraries, but they may come about as a result of merging histograms with different binnings.

*See also:*

   * <<RegularBinning>>: for ordered, equal-sized, abutting real intervals.
   * <<EdgesBinning>>: for ordered, any-sized, abutting real intervals.
   * <<IrregularBinning>>: for unordered, any-sized real intervals (that may even overlap).
   * <<SparseRegularBinning>>: for unordered, equal-sized real intervals aligned to a regular grid, but only need to be defined if the bin content is not empty.
"""

    def __init__(self, intervals, overflow=None, overlapping_fill=OverlappingFill.unspecified):
        self.intervals = intervals
        self.overflow = overflow
        self.overlapping_fill = overlapping_fill

    def _valid(self, seen, recursive):
        if len(self.intervals) != len(set(self.intervals)):
            raise ValueError("IrregularBinning.intervals must be unique")
        if recursive:
            _valid(self.intervals, seen, recursive)
            _valid(self.overflow, seen, recursive)

    def _binshape(self):
        if self.overflow is None:
            numoverflowbins = 0
        else:
            numoverflowbins = self.overflow._numbins()
        return (len(self.intervals) + numoverflowbins,)

    @property
    def dimensions(self):
        return 1

    def _toflatbuffers(self, builder):
        stagg.stagg_generated.IrregularBinning.IrregularBinningStartIntervalsVector(builder, len(self.intervals))
        for x in self.intervals[::-1]:
            x._toflatbuffers(builder)
        intervals = builder.EndVector(len(self.intervals))

        stagg.stagg_generated.IrregularBinning.IrregularBinningStart(builder)
        stagg.stagg_generated.IrregularBinning.IrregularBinningAddIntervals(builder, intervals)
        if self.overflow is not None:
            stagg.stagg_generated.IrregularBinning.IrregularBinningAddOverflow(builder, self.overflow._toflatbuffers(builder))
        if self.overlapping_fill != self.unspecified:
            stagg.stagg_generated.IrregularBinning.IrregularBinningAddOverlappingFill(builder, self.overlapping_fill.value)
        return stagg.stagg_generated.IrregularBinning.IrregularBinningEnd(builder)

    def _dump(self, indent, width, end):
        args = ["intervals=[" + _dumpeq(_dumplist([x._dump(indent + "    ", width, end) for x in self.intervals], indent, width, end), indent, end) + "]"]
        if self.overflow is not None:
            args.append("overflow={0}".format(_dumpeq(self.overflow._dump(indent + "    ", width, end), indent, end)))
        if self.overlapping_fill is not True:
            args.append("overlapping_fill={0}".format(repr(self.overlapping_fill)))
        return _dumpline(self, args, indent, width, end)

    def toCategoryBinning(self, format="%g"):
        flows = []
        if self.overflow is not None:
            low = numpy.inf
            low_inclusive = False
            high = -numpy.inf
            high_inclusive = False
            for interval in self.intervals:
                if interval.low <= low:
                    low = interval.low
                    low_inclusive = interval.low_inclusive
                if interval.high >= high:
                    high = interval.high
                    high_inclusive = interval.high_inclusive

            flows.append((self.overflow.loc_underflow, "{0}-inf, {1}{2}".format(
                "[" if self.overflow.minf_mapping == self.overflow.in_underflow else "(",
                format % low,
                ")" if low_inclusive else "]")))
            flows.append((self.overflow.loc_overflow, "{0}{1}, +inf{2}".format(
                "(" if high_inclusive else "[",
                format % high,
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

    def _getindex(self, where):
        loc_underflow = None if self.overflow is None else self.overflow.loc_underflow
        loc_overflow = None if self.overflow is None else self.overflow.loc_overflow
        loc_nanflow = None if self.overflow is None else self.overflow.loc_nanflow
        return self._getindex_general(where, len(self.intervals), loc_underflow, loc_overflow, loc_nanflow)

    def _getloc(self, isiloc, where):
        if where is None:
            return None, (slice(None),)

        elif isinstance(where, slice) and where.start is None and where.stop is None and where.step is None:
            return self, (slice(None),)

        elif isinstance(where, slice):
            if isiloc:
                start, stop, step = where.indices(len(self.intervals))
                if step <= 0:
                    raise IndexError("slice step cannot be zero or negative")
                start = max(start, 0)
                stop = min(stop, len(self.intervals))
                d, m = divmod(stop - start, step)
                length = d + (1 if m != 0 else 0)
                stop = start + step*length

                yes_underflow, yes_overflow = False, False
                index = numpy.full(len(self.intervals), -1, dtype=numpy.int64)
                index[start:stop:step] = numpy.arange(length)
                intervals = [x.detached() for x in self.intervals[start:stop:step]]

            else:
                if where.step is not None:
                    raise IndexError("IrregularBinning.loc slice cannot have a step")
                yes_underflow, yes_overflow = False, False
                index = numpy.empty(len(self.intervals), dtype=numpy.int64)
                length = 0
                intervals = []
                for i, interval in enumerate(self.intervals):
                    if where.start is not None and where.start >= interval.high:
                        yes_underflow = True
                        index[i] = -3
                    elif where.stop is not None and where.stop <= interval.low:
                        yes_overflow = True
                        index[i] = -2
                    else:
                        index[i] = length
                        length += 1
                        intervals.append(interval.detached())

            if length == 0:
                raise IndexError("slice {0}:{1} would result in no bins".format(where.start, where.stop))

            overflow, loc_underflow, pos_underflow, loc_overflow, pos_overflow, loc_nanflow, pos_nanflow = RealOverflow._getloc(self.overflow, yes_underflow, yes_overflow, length)
            binning = IrregularBinning(intervals, overflow=overflow, overlapping_fill=self.overlapping_fill)

            if pos_underflow is not None:
                index[index == -3] = pos_underflow
            if pos_overflow is not None:
                index[index == -2] = pos_overflow
            flows = [] if self.overflow is None else [(self.overflow.loc_underflow, pos_underflow), (self.overflow.loc_overflow, pos_overflow), (self.overflow.loc_nanflow, pos_nanflow)]
            selfmap = self._selfmap(flows, index)

            return binning, (selfmap,)
                
        elif not isiloc and not isinstance(where, (bool, numpy.bool, numpy.bool_)) and isinstance(where, (numbers.Real, numpy.integer, numpy.floating)):
            for interval in self.intervals:
                if interval.low <= where <= interval.high:
                    break
            else:
                raise IndexError("index {0} is out of bounds".format(where))
            return self._getloc(False, slice(where, where))

        elif isiloc and not isinstance(where, (bool, numpy.bool, numpy.bool_)) and isinstance(where, (numbers.Integral, numpy.integer)):
            i = where
            if i < 0:
                i += len(self.intervals)
            if not 0 <= i < len(self.intervals):
                raise IndexError("index {0} is out of bounds".format(where))
            return self._getloc(True, slice(i, i))

        elif isiloc:
            where = numpy.array(where, copy=False)
            if len(where.shape) == 1 and issubclass(where.dtype.type, (numpy.integer, numpy.bool, numpy.bool_)):
                intervals = numpy.array(self.intervals, dtype=numpy.object)[where]
                if len(intervals) == 0:
                    raise IndexError("index {0} would result in no bins".format(where))
                index = self.full(len(self.intervals), -1, dtype=numpy.int64)
                index[where] = numpy.arange(len(intervals))
                binning = IrregularBinning(intervals, overflow=overflow, overlapping_fill=self.overlapping_fill)
                flows = [] if self.overflow is None else [(self.overflow.loc_underflow, -1), (self.overflow.loc_overflow, -1), (self.overflow.loc_nanflow, -1)]
                selfmap = self._selfmap(flows, index)
                return binning, (selfmap,)

        if isiloc:
            raise TypeError("IrregularBinning.iloc accepts an integer, an integer slice (`:`), ellipsis (`...`), projection (`None`), or an array of integers/booleans, not {0}".format(repr(where)))
        else:
            raise TypeError("IrregularBinning.loc accepts a number, a real-valued slice (`:`), ellipsis (`...`), or projection (`None`), not {0}".format(repr(where)))

    def _restructure(self, other):
        assert isinstance(other, IrregularBinning)

        if self.overlapping_fill == other.overlapping_fill:
            overlapping_fill = self.overlapping_fill
        else:
            overlapping_fill = self.unspecified

        selfints = list(self.intervals)
        otherints = list(other.intervals)

        lookup = {x: i for i, x in enumerate(selfints)}
        intervals = selfints + [x for x in otherints if x not in lookup]

        if len(intervals) == len(selfints) == len(otherints) and selfints == otherints and self.overflow == other.overflow:
            if overlapping_fill == self.overlapping_fill:
                return self, (None,), (None,)
            else:
                return IrregularBinning([x.detached(reclaim=True) for x in intervals], overflow=self.overflow.detached(reclaim=True), overlapping_fill=overlapping_fill), (None,), (None,)

        else:
            overflow, pos_underflow, pos_overflow, pos_nanflow = RealOverflow._common(self.overflow, other.overflow, len(intervals))

            lookup = {x: i for i, x in enumerate(intervals)}
            othermap = other._selfmap([] if other.overflow is None else [(other.overflow.loc_underflow, pos_underflow), (other.overflow.loc_overflow, pos_overflow), (other.overflow.loc_nanflow, pos_nanflow)],
                                      numpy.array([lookup[x] for x in otherints], dtype=numpy.int64))

            if ((self.overflow is None and overflow is None) or (self.overflow is not None and self.overflow.loc_underflow == overflow.loc_underflow and self.overflow.loc_overflow == overflow.loc_overflow and self.overflow.loc_nanflow == overflow.loc_nanflow)) and len(selfints) == len(intervals):
                return self, (None,), (othermap,)

            else:
                selfmap = self._selfmap([] if self.overflow is None else [(self.overflow.loc_underflow, pos_underflow), (self.overflow.loc_overflow, pos_overflow), (self.overflow.loc_nanflow, pos_nanflow)],
                                        numpy.arange(len(selfints), dtype=numpy.int64))
                return IrregularBinning([x.detached(reclaim=True) for x in intervals], overflow=overflow, overlapping_fill=overlapping_fill), (selfmap,), (othermap,)

################################################# CategoryBinning

class CategoryBinning(Binning, BinLocation):
    _params = {
        "categories":   stagg.checktype.CheckVector("CategoryBinning", "categories", required=True, type=str),
        "loc_overflow": stagg.checktype.CheckEnum("CategoryBinning", "loc_overflow", required=False, choices=BinLocation.locations, intlookup=BinLocation._locations),
        }

    categories = typedproperty(_params["categories"])
    loc_overflow = typedproperty(_params["loc_overflow"])

    description = "Associates disjoint categories from a categorical dataset with bins."
    validity_rules = ("The *categories* must be unique.",)
    long_description = """
This binning is intended for string-valued categorical data (or values that can be converted to strings without losing uniqueness). Each named category in *categories* corresponds to one bin.

If *loc_overflow* is `nonexistent`, unspecified strings were ignored in the filling procedure. Otherwise, the overflow bin corresponds to unspecified strings, and it can be `below` or `above` the normal bins. Unlike <<RealOverflow>>, which has up to three overflow bins (underflow, overflow, and nanflow), no distinction is made among `below3`, `below2`, `below1` or `above1`, `above2`, `above3`.

*See also:*

   * <<CategoryBinning>>: for disjoint categories with a possible overflow bin.
   * <<PredicateBinning>>: for possibly overlapping regions defined by predicate functions.
   * <<VariationBinning>>: for completely overlapping input data, with derived features computed different ways.
"""

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

    @property
    def dimensions(self):
        return 1

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

    def _dump(self, indent, width, end):
        args = ["categories=[" + ", ".join(repr(x) for x in self.categories) + "]"]
        if self.loc_overflow != BinLocation.nonexistent:
            args.append("loc_overflow={0}".format(repr(self.loc_overflow)))
        return _dumpline(self, args, indent, width, end)

    def _getindex(self, where):
        return self._getindex_general(where, len(self.categories), None, self.loc_overflow, None)

    def _getloc(self, isiloc, where):
        if where is None:
            return None, (slice(None),)

        elif isinstance(where, slice) and where.start is None and where.stop is None and where.step is None:
            return self, (slice(None),)

        elif isiloc and isinstance(where, slice):
            start, stop, step = where.indices(len(self.categories))
            if step <= 0:
                raise IndexError("slice step cannot be zero or negative")
            start = max(start, 0)
            stop = min(stop, len(self.categories))
            d, m = divmod(stop - start, step)
            length = d + (1 if m != 0 else 0)
            stop = start + step*length

            if length == 0:
                raise IndexError("slice {0}:{1} would result in no bins".format(where.start, where.stop))

            categories = self.categories[start:stop:step]
            index = numpy.full(len(self.categories), len(categories), dtype=numpy.int64)
            index[start:stop:step] = numpy.arange(length)

            if self.loc_overflow != BinLocation.nonexistent or (index == len(categories)).any():
                loc_overflow = BinLocation.above1
                pos_overflow = len(categories)
            else:
                loc_overflow = BinLocation.nonexistent
                pos_overflow = None

            binning = CategoryBinning(categories, loc_overflow=loc_overflow)
            selfmap = self._selfmap([(self.loc_overflow, pos_overflow)], index)

            return binning, (selfmap,)
            
        elif isiloc and not isinstance(where, (bool, numpy.bool, numpy.bool_)) and isinstance(where, (numbers.Integral, numpy.integer)):
            i = where
            if i < 0:
                i += len(self.categories)
            if not 0 <= i < len(self.categories):
                raise IndexError("index {0} is out of bounds".format(where))
            return self._getloc(True, slice(i, i))

        elif not isiloc and isinstance(where, str):
            return self._getloc(False, [where])

        elif not isiloc and isinstance(where, Iterable) and all(isinstance(x, str) for x in where):
            lookup = {x: j for j, x in enumerate(self.categories)}
            index = numpy.full(len(lookup), len(where), dtype=numpy.int64)
            for i, x in enumerate(where):
                j = lookup.get(x, None)
                if j is None:
                    raise IndexError("CategoryBinning does not have category {0}".format(repr(x)))
                index[j] = i

            if self.loc_overflow != BinLocation.nonexistent or (index == len(where)).any():
                loc_overflow = BinLocation.above1
                pos_overflow = len(where)
            else:
                loc_overflow = BinLocation.nonexistent
                pos_overflow = None

            binning = CategoryBinning(where, loc_overflow=loc_overflow)
            selfmap = self._selfmap([(self.loc_overflow, pos_overflow)], index)

            return binning, (selfmap,)
            
        elif isiloc:
            where = numpy.array(where, copy=False)
            if len(where.shape) == 1 and issubclass(where.dtype.type, (numpy.integer, numpy.bool, numpy.bool_)):
                categories = numpy.array(self.categories, dtype=numpy.object)[where]
                if len(categories) == 0:
                    raise IndexError("index {0} would result in no bins".format(where))
                index = self.full(len(self.categories), -1, dtype=numpy.int64)
                index[where] = numpy.arange(len(categories))
                binning = CategoryBinning(categories, loc_overflow=self.loc_overflow)
                selfmap = self._selfmap([(self.loc_overflow, -1)], index)
                return binning, (selfmap,)

        if isiloc:
            raise TypeError("CategoryBinning.iloc accepts an integer, an integer slice (`:`), ellipsis (`...`), projection (`None`), or an array of integers/booleans, not {0}".format(repr(where)))
        else:
            raise TypeError("CategoryBinning.loc accepts a string, an iterable of strings, an empty slice (`:` only), ellipsis (`...`), or projection (`None`), not {0}".format(repr(where)))

    def _restructure(self, other):
        assert isinstance(other, CategoryBinning)

        selfcat = list(self.categories)
        othercat = list(other.categories)

        lookup = {x: i for i, x in enumerate(selfcat)}
        categories = selfcat + [x for x in othercat if x not in lookup]

        if len(categories) == len(selfcat) == len(othercat) and selfcat == othercat and self.loc_overflow == other.loc_overflow:
            return self, (None,), (None,)

        else:
            if self.loc_overflow != self.nonexistent or other.loc_overflow != other.nonexistent:
                loc_overflow = self.above1
                pos_overflow = len(categories)
            else:
                loc_overflow = self.nonexistent
                pos_overflow = None

            lookup = {x: i for i, x in enumerate(categories)}
            othermap = other._selfmap([(other.loc_overflow, pos_overflow)],
                                      numpy.array([lookup[x] for x in othercat], dtype=numpy.int64))

            if self.loc_overflow == loc_overflow and len(selfcat) == len(categories):
                return self, (None,), (othermap,)
            else:
                selfmap = self._selfmap([(self.loc_overflow, pos_overflow)],
                                        numpy.arange(len(selfcat), dtype=numpy.int64))
                return CategoryBinning(categories, loc_overflow=loc_overflow), (selfmap,), (othermap,)

################################################# SparseRegularBinning

class SparseRegularBinning(Binning, BinLocation):
    _params = {
        "bins":           stagg.checktype.CheckVector("SparseRegularBinning", "bins", required=True, type=int),
        "bin_width":      stagg.checktype.CheckNumber("SparseRegularBinning", "bin_width", required=True, min=0, min_inclusive=False),
        "origin":         stagg.checktype.CheckNumber("SparseRegularBinning", "origin", required=False),
        "overflow":       stagg.checktype.CheckClass("SparseRegularBinning", "overflow", required=False, type=RealOverflow),
        "low_inclusive":  stagg.checktype.CheckBool("SparseRegularBinning", "low_inclusive", required=False),
        "high_inclusive": stagg.checktype.CheckBool("SparseRegularBinning", "high_inclusive", required=False),
        "minbin":         stagg.checktype.CheckInteger("SparseRegularBinning", "minbin", required=False, min=MININT64, max=MAXINT64),
        "maxbin":         stagg.checktype.CheckInteger("SparseRegularBinning", "maxbin", required=False, min=MININT64, max=MAXINT64),
        }

    bins           = typedproperty(_params["bins"])
    bin_width      = typedproperty(_params["bin_width"])
    origin         = typedproperty(_params["origin"])
    overflow       = typedproperty(_params["overflow"])
    low_inclusive  = typedproperty(_params["low_inclusive"])
    high_inclusive = typedproperty(_params["high_inclusive"])
    minbin         = typedproperty(_params["minbin"])
    maxbin         = typedproperty(_params["maxbin"])

    description = "Splits a one-dimensional axis into unordered, equal-sized real intervals aligned to a regular grid, which only need to be defined if the bin content is not empty."
    validity_rules = ()
    long_description = u"""
This binning is intended for one-dimensional, real-valued data. Unlike <<RegularBinning>> and <<EdgesBinning>>, the intervals do not need to be abutting. Unlike <<IrregularBinning>>, they must be equal-sized, non-overlapping, and aligned to a grid.

Integer-valued bin indexes `i` are mapped to real intervals using *bin_width* and *origin*: each interval starts at `bin_width*(i) + origin` and stops at `bin_width*(i + 1) + origin`. The *bins* property is an unordered list of bin indexes, with the same length and order as the <<Histogram>> bins or <<BinnedEvaluatedFunction>> values. Unspecified bins are empty: for counts or sums of weights, this means zero; for minima, this means +\u221e; for maxima, this meanss \u2012\u221e; for all other values, `nan` (not a number).

There is a degeneracy between *bins* and *origin*: adding an integer multiple of *bin_width* to *origin* and subtracting that integer from all bins yields an equivalent binning.

If *low_inclusive* is true, then all intervals between pairs of edges include the low edge. If *high_inclusive* is true, then all intervals between pairs of edges include the high edge.

Although this binning can reach a very wide range of values without using much memory, there is a limit. The *bins* array values are 64-bit signed integers, so they are in principle limited to [\u20122\u2076\u00b3, 2\u2076\u00b3 \u2012 1]. Changing the *origin* moves this window, and chaning the *bin_width* widens its coverage of real values at the expense of detail. In some cases, the meaningful range is narrower than this. For instance, if a binning is shifted to a higher *origin* (e.g. to align two histograms to add them), some values below 2\u2076\u00b3 \u2012 1 in the shifted histogram were out of range in the unshifted histogram, so we cannot say that they are in range in the new histogram. For this, the *maxbin* would be less than 2\u2076\u00b3 \u2012 1. By a similar argument, the *minbin* can be greater than \u20122\u2076\u00b3.

Therefore, even though this binning is sparse, it can have underflow and overflow bins for values below *minbin* or above *maxbin*. Since `nan` (not a number) values don't map to any integer, this binning may also need a nanflow. The existence and positions of any underflow, overflow, and nanflow bins, as well as how non-finite values were handled during filling, are contained in the <<RealOverflow>>.

*See also:*

   * <<RegularBinning>>: for ordered, equal-sized, abutting real intervals.
   * <<EdgesBinning>>: for ordered, any-sized, abutting real intervals.
   * <<IrregularBinning>>: for unordered, any-sized real intervals (that may even overlap).
   * <<SparseRegularBinning>>: for unordered, equal-sized real intervals aligned to a regular grid, but only need to be defined if the bin content is not empty.
"""

    def __init__(self, bins, bin_width, origin=0.0, overflow=None, low_inclusive=True, high_inclusive=False, minbin=MININT64, maxbin=MAXINT64):
        self.bins = bins
        self.bin_width = bin_width
        self.origin = origin
        self.overflow = overflow
        self.low_inclusive = low_inclusive
        self.high_inclusive = high_inclusive
        self.minbin = minbin
        self.maxbin = maxbin

    def _valid(self, seen, recursive):
        if len(self.bins) != len(numpy.unique(self.bins)):
            raise ValueError("SparseRegularBinning.bins must be unique")
        if self.low_inclusive and self.high_inclusive:
            raise ValueError("SparseRegularBinning.low_inclusive and SparseRegularBinning.high_inclusive cannot both be True")
        if self.minbin >= self.maxbin:
            raise ValueError("SparseRegularBinning.minbin must be less than SparseRegularBinning.maxbin")
        if (self.bins < self.minbin).any():
            raise ValueError("SparseRegularBinning.bins must be greater than or equal to SparseRegularBinning.minbin")
        if (self.bins > self.maxbin).any():
            raise ValueError("SparseRegularBinning.bins must be less than or equal to SparseRegularBinning.maxbin")

        if recursive:
            _valid(self.overflow, seen, recursive)

    def _binshape(self):
        if self.overflow is None:
            numoverflowbins = 0
        else:
            numoverflowbins = self.overflow._numbins()
        return (len(self.bins) + numoverflowbins,)

    @property
    def dimensions(self):
        return 1

    @classmethod
    def _fromflatbuffers(cls, fb):
        out = cls.__new__(cls)
        out._flatbuffers = _MockFlatbuffers()
        out._flatbuffers.Bins = fb.BinsAsNumpy
        out._flatbuffers.BinWidth = fb.BinWidth
        out._flatbuffers.Origin = fb.Origin
        out._flatbuffers.Overflow = fb.Overflow
        out._flatbuffers.LowInclusive = fb.LowInclusive
        out._flatbuffers.HighInclusive = fb.HighInclusive
        out._flatbuffers.Minbin = fb.Minbin
        out._flatbuffers.Maxbin = fb.Maxbin
        return out

    def _toflatbuffers(self, builder):
        binsbuf = self.bins.tostring()
        stagg.stagg_generated.SparseRegularBinning.SparseRegularBinningStartBinsVector(builder, len(self.bins))
        builder.head = builder.head - len(binsbuf)
        builder.Bytes[builder.head : builder.head + len(binsbuf)] = binsbuf
        bins = builder.EndVector(len(self.bins))

        stagg.stagg_generated.SparseRegularBinning.SparseRegularBinningStart(builder)
        stagg.stagg_generated.SparseRegularBinning.SparseRegularBinningAddBins(builder, bins)
        stagg.stagg_generated.SparseRegularBinning.SparseRegularBinningAddBinWidth(builder, self.bin_width)
        if self.origin != 0.0:
            stagg.stagg_generated.SparseRegularBinning.SparseRegularBinningAddOrigin(builder, self.origin)
        if self.overflow is not None:
            stagg.stagg_generated.EdgesBinning.EdgesBinningAddOverflow(builder, self.overflow._toflatbuffers(builder))
        if self.low_inclusive is not True:
            stagg.stagg_generated.EdgesBinning.EdgesBinningAddLowInclusive(builder, self.low_inclusive)
        if self.high_inclusive is not False:
            stagg.stagg_generated.EdgesBinning.EdgesBinningAddHighInclusive(builder, self.high_inclusive)
        if self.minbin != MININT64:
            stagg.stagg_generated.EdgesBinning.EdgesBinningAddMinbin(builder, self.minbin)
        if self.maxbin != MAXINT64:
            stagg.stagg_generated.EdgesBinning.EdgesBinningAddMaxbin(builder, self.maxbin)
        return stagg.stagg_generated.SparseRegularBinning.SparseRegularBinningEnd(builder)

    def _dump(self, indent, width, end):
        args = ["bins={0}".format(_dumparray(self.bins, indent, end)), "bin_width={0}".format(repr(self.bin_width))]
        if self.origin != 0.0:
            args.append("origin={0}".format(repr(self.origin)))
        if self.low_inclusive is not True:
            args.append("low_inclusive={0}".format(repr(self.low_inclusive)))
        if self.high_inclusive is not True:
            args.append("high_inclusive={0}".format(repr(self.high_inclusive)))
        if self.minbin != MININT64:
            args.append("minbin={0}".format(repr(self.minbin)))
        if self.maxbin != MAXINT64:
            args.append("maxbin={0}".format(repr(self.maxbin)))
        return _dumpline(self, args, indent, width, end)

    def originto(self, origin):
        numbins = int(round((origin - self.origin) / self.bin_width))
        origin = numbins*self.bin_width + self.origin
        bins = self.bins + numbins
        minbin = max(self.minbin + numbins, MININT64)
        maxbin = min(self.maxbin + numbins, MAXINT64)
        return SparseRegularBinning(bins, self.bin_width, origin, overflow=self.overflow.detached(), low_inclusive=self.low_inclusive, high_inclusive=self.high_inclusive, minbin=minbin, maxbin=maxbin)

    def toIrregularBinning(self):
        if self.low_inclusive and self.high_inclusive:
            raise ValueError("SparseRegularBinning.interval.low_inclusive and SparseRegularBinning.interval.high_inclusive cannot both be True")
        overflow = None if self.overflow is None else self.overflow.detached()
        flows = [] if overflow is None else [(overflow.loc_underflow, -numpy.inf), (overflow.loc_overflow, numpy.inf), (overflow.loc_nanflow, numpy.nan)]
        intervals = []
        for loc, val in BinLocation._belows(flows):
            if val == -numpy.inf:
                intervals.append(RealInterval(-numpy.inf, self.bin_width*(self.minbin) + self.origin, low_inclusive=(overflow.minf_mapping == RealOverflow.in_underflow), high_inclusive=(not self.low_inclusive)))
                overflow.loc_underflow = BinLocation.nonexistent
            if val == numpy.inf:
                intervals.append(RealInterval(self.bin_width*(self.maxbin + 1) + self.origin, numpy.inf, low_inclusive=(not self.high_inclusive), high_inclusive=(overflow.pinf_mapping == RealOverflow.in_overflow)))
                overflow.loc_overflow = BinLocation.nonexistent
        for x in self.bins:
            intervals.append(RealInterval(self.bin_width*(x) + self.origin, self.bin_width*(x + 1) + self.origin))
        for loc, val in BinLocation._aboves(flows):
            if val == -numpy.inf:
                intervals.append(RealInterval(-numpy.inf, self.bin_width*(self.minbin) + self.origin, low_inclusive=(overflow.minf_mapping == RealOverflow.in_underflow), high_inclusive=(not self.low_inclusive)))
                overflow.loc_underflow = BinLocation.nonexistent
            if val == numpy.inf:
                intervals.append(RealInterval(self.bin_width*(self.maxbin + 1) + self.origin, numpy.inf, low_inclusive=(not self.high_inclusive), high_inclusive=(overflow.pinf_mapping == RealOverflow.in_overflow)))
                overflow.loc_overflow = BinLocation.nonexistent
        return IrregularBinning(intervals, overflow=overflow)

    def toCategoryBinning(self, format="%g"):
        flows = []
        if self.overflow is not None:
            if self.overflow.loc_underflow != BinLocation.nonexistent:
                if self.minbin == MININT64 and self.overflow.minf_mapping == RealOverflow.in_underflow:
                    flows.append((self.overflow.loc_underflow, "{-inf}"))
                else:
                    flows.append((self.overflow.loc_underflow, "{0}-inf, {1}{2}".format("[" if self.overflow.minf_mapping == RealOverflow.in_underflow else "(", format % (self.bin_width*(self.minbin) + self.origin), "]" if not self.low_inclusive else ")")))

            if self.overflow.loc_overflow != BinLocation.nonexistent:
                if self.maxbin == MAXINT64 and self.overflow.pinf_mapping == RealOverflow.in_overflow:
                    flows.append((self.overflow.loc_overflow, "{+inf}"))
                else:
                    flows.append((self.overflow.loc_overflow, "{0}{1}, +inf{2}".format("[" if not self.high_inclusive else ")", format % (self.bin_width*(self.maxbin + 1) + self.origin), "]" if self.overflow.pinf_mapping == RealOverflow.in_overflow else ")")))

            if self.overflow.loc_nanflow != BinLocation.nonexistent:
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

    def _getindex(self, where):
        loc_underflow = None if self.overflow is None else self.overflow.loc_underflow
        loc_overflow = None if self.overflow is None else self.overflow.loc_overflow
        loc_nanflow = None if self.overflow is None else self.overflow.loc_nanflow
        return self._getindex_general(where, len(self.bins), loc_underflow, loc_overflow, loc_nanflow)

    def _getloc(self, isiloc, where):
        if where is None:
            return None, (slice(None),)

        elif isinstance(where, slice) and where.start is None and where.stop is None and where.step is None:
            return self, (slice(None),)

        elif isinstance(where, slice):
            if isiloc:
                if where.step is not None and where.step != 1:
                    raise IndexError("SparseRegularBinning.iloc slice cannot have a step")
                start, stop, step = where.indices(len(self.bins))
                start = max(start, 0)
                stop = min(stop, len(self.bins))
                overflow, loc_underflow, pos_underflow, loc_overflow, pos_overflow, loc_nanflow, pos_nanflow = RealOverflow._getloc(self.overflow, False, False, stop - start)
                index = numpy.full(len(self.bins), -1, dtype=numpy.int64)
                index[start:stop] = numpy.arange(stop - start)

                binning = SparseRegularBinning(self.bins[start:stop], self.bin_width, origin=self.origin, overflow=overflow, low_inclusive=self.low_inclusive, high_inclusive=self.high_inclusive, minbin=self.minbin, maxbin=self.maxbin)

            else:
                step = 1 if where.step is None else int(round(where.step / self.bin_width))
                if step <= 0:
                    raise IndexError("slice step cannot be zero or negative")
                lowest  = (int(self.bins.min()) // step) * step
                highest = int(self.bins.max()) + 1
                exactstart = lowest if where.start is None else int(math.trunc((where.start - self.origin) / self.bin_width))
                exactstop = highest if where.stop is None else int(math.ceil((where.stop - self.origin) / self.bin_width))
                start = max(exactstart, lowest)
                stop = min(exactstop, highest)

                d, m = divmod(stop - start, step)
                length = d + (1 if m != 0 else 0)
                stop = start + step*length

                origin = self.bin_width*start + self.origin
                bins = (self.bins - start) // step
                below = (bins < 0)
                above = (bins >= length)
                good = numpy.logical_not(below | above)
                bins = bins[good]

                if len(bins) == 0:
                    bins = numpy.array([0], dtype=bins.dtype)
                    origin = exactstart*self.bin_width + self.origin

                overflow, loc_underflow, pos_underflow, loc_overflow, pos_overflow, loc_nanflow, pos_nanflow = RealOverflow._getloc(self.overflow, below.any(), above.any(), len(bins))

                index = numpy.full(len(self.bins), -1, dtype=numpy.int64)
                if pos_underflow is not None:
                    index[below] = pos_underflow
                index[good] = numpy.arange(len(bins))
                if pos_overflow is not None:
                    index[above] = pos_overflow

                minbin = self.minbin if where.start is None else exactstart
                maxbin = self.maxbin if where.stop is None else exactstop - 1
                minbin = (minbin - start) // step
                maxbin = (maxbin - start) // step
                minbin = max(minbin, MININT64)
                maxbin = min(maxbin, MAXINT64)

                binning = SparseRegularBinning(bins, self.bin_width*step, origin=origin, overflow=overflow, low_inclusive=self.low_inclusive, high_inclusive=self.high_inclusive, minbin=minbin, maxbin=maxbin)

            flows = [] if self.overflow is None else [(self.overflow.loc_underflow, pos_underflow), (self.overflow.loc_overflow, pos_overflow), (self.overflow.loc_nanflow, pos_nanflow)]
            selfmap = self._selfmap(flows, index)

            return binning, (selfmap,)

        elif not isiloc and not isinstance(where, (bool, numpy.bool, numpy.bool_)) and isinstance(where, (numbers.Real, numpy.integer, numpy.floating)):
            i = int(math.trunc((where - self.origin) / self.bin_width))
            if not self.minbin <= i <= self.maxbin:
                raise IndexError("index {0} is out of bounds".format(where))
            return self._getloc(False, slice(where, where))

        elif isiloc and not isinstance(where, (bool, numpy.bool, numpy.bool_)) and isinstance(where, (numbers.Integral, numpy.integer)):
            i = where
            if i < 0:
                i += len(self.bins)
            if not 0 <= i < len(self.bins):
                raise IndexError("index {0} is out of bounds".format(where))
            return self._getloc(True, slice(i, i))

        elif isiloc:
            where = numpy.array(where, copy=False)
            if len(where.shape) == 1 and issubclass(where.dtype.type, (numpy.integer, numpy.bool, numpy.bool_)):
                bins = self.bins[where]
                if len(bins) == 0:
                    raise IndexError("index {0} would result in no bins".format(where))
                index = self.full(len(self.bins), -1, dtype=numpy.int64)
                index[where] = numpy.arange(len(bins))
                binning = SparseRegularBinning(bins, self.bin_width, origin=self.origin, overflow=self.overflow, low_inclusive=self.low_inclusive, high_inclusive=self.high_inclusive, minbin=self.minbin, maxbin=self.maxbin)
                flows = [] if self.overflow is None else [(self.overflow.loc_underflow, -1), (self.overflow.loc_overflow, -1), (self.overflow.loc_nanflow, -1)]
                selfmap = self._selfmap(flows, index)
                return binning, (selfmap,)

        if isiloc:
            raise TypeError("SparseRegularbinning.iloc accepts an integer, an integer slice (`:`), ellipsis (`...`), projection (`None`), or an array of integers/booleans, not {0}".format(repr(where)))
        else:
            raise TypeError("SparseRegularbinning.loc accepts a number, a real-valued slice (`:`), ellipsis (`...`), or projection (`None`), not {0}".format(repr(where)))

    def _restructure(self, other):
        assert isinstance(other, SparseRegularBinning)
        if self.bin_width != other.bin_width:
            raise ValueError("cannot add SparseRegularBinnings because they have different bin_widths: {0} vs {1}".format(self.bin_width, other.bin_width))
        if self.origin != other.origin:
            original = other.origin
            other = other.originto(self.origin)
            if self.origin != other.origin:
                raise ValueError("cannot add SparseRegularBinnings because they have different origins: {0} vs {1}".format(self.origin, original))
        if self.low_inclusive != other.low_inclusive:
            raise ValueError("cannot add SparseRegularBinnings because they have different low_inclusives: {0} vs {1}".format(self.low_inclusive, other.low_inclusive))
        if self.high_inclusive != other.high_inclusive:
            raise ValueError("cannot add SparseRegularBinnings because they have different high_inclusives: {0} vs {1}".format(self.high_inclusive, other.high_inclusive))

        bins = numpy.concatenate((self.bins, other.bins[~numpy.isin(other.bins, self.bins, assume_unique=True)]))
        minbin = max(self.minbin, other.minbin)
        maxbin = min(self.maxbin, other.maxbin)

        if len(bins) == len(self.bins) == len(other.bins) and (self.bins == other.bins).all() and self.overflow == other.overflow:
            if self.minbin == other.minbin and self.maxbin == other.maxbin:
                return self, (None,), (None,)
            else:
                return SparseRegularBinning(bins, self.bin_width, self.origin, overflow=self.overflow.detached(), low_inclusive=self.low_inclusive, high_inclusive=self.high_inclusive, minbin=minbin, maxbin=maxbin), (None,), (None,)

        else:
            overflow, pos_underflow, pos_overflow, pos_nanflow = RealOverflow._common(self.overflow, other.overflow, len(bins))

            lookup = numpy.argsort(bins)
            othermap = other._selfmap([] if other.overflow is None else [(other.overflow.loc_underflow, pos_underflow), (other.overflow.loc_overflow, pos_overflow), (other.overflow.loc_nanflow, pos_nanflow)],
                                      lookup[numpy.searchsorted(bins[lookup], other.bins, side="left")])

            if ((self.overflow is None and overflow is None) or (self.overflow is not None and self.overflow.loc_underflow == overflow.loc_underflow and self.overflow.loc_overflow == overflow.loc_overflow and self.overflow.loc_nanflow == overflow.loc_nanflow)) and len(self.bins) == len(bins):
                if self.minbin == other.minbin and self.maxbin == other.maxbin:
                    return self, (None,), (othermap,)
                else:
                    return SparseRegularBinning(bins, self.bin_width, self.origin, overflow=self.overflow.detached(), low_inclusive=self.low_inclusive, high_inclusive=self.high_inclusive, minbin=minbin, maxbin=maxbin), (None,), (othermap,)

            else:
                selfmap = self._selfmap([] if self.overflow is None else [(self.overflow.loc_underflow, pos_underflow), (self.overflow.loc_overflow, pos_overflow), (self.overflow.loc_nanflow, pos_nanflow)],
                                        numpy.arange(len(self.bins), dtype=numpy.int64))
                return SparseRegularBinning(bins, self.bin_width, self.origin, overflow=overflow, low_inclusive=self.low_inclusive, high_inclusive=self.high_inclusive, minbin=minbin, maxbin=maxbin), (selfmap,), (othermap,)

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

    unspecified      = FractionErrorMethodEnum("unspecified", stagg.stagg_generated.FractionErrorMethod.FractionErrorMethod.frac_unspecified)
    normal           = FractionErrorMethodEnum("normal", stagg.stagg_generated.FractionErrorMethod.FractionErrorMethod.frac_normal)
    clopper_pearson  = FractionErrorMethodEnum("clopper_pearson", stagg.stagg_generated.FractionErrorMethod.FractionErrorMethod.frac_clopper_pearson)
    wilson           = FractionErrorMethodEnum("wilson", stagg.stagg_generated.FractionErrorMethod.FractionErrorMethod.frac_wilson)
    agresti_coull    = FractionErrorMethodEnum("agresti_coull", stagg.stagg_generated.FractionErrorMethod.FractionErrorMethod.frac_agresti_coull)
    feldman_cousins  = FractionErrorMethodEnum("feldman_cousins", stagg.stagg_generated.FractionErrorMethod.FractionErrorMethod.frac_feldman_cousins)
    jeffrey          = FractionErrorMethodEnum("jeffrey", stagg.stagg_generated.FractionErrorMethod.FractionErrorMethod.frac_jeffrey)
    bayesian_uniform = FractionErrorMethodEnum("bayesian_uniform", stagg.stagg_generated.FractionErrorMethod.FractionErrorMethod.frac_bayesian_uniform)
    error_methods = [unspecified, normal, clopper_pearson, wilson, agresti_coull, feldman_cousins, jeffrey, bayesian_uniform]

    _params = {
        "layout": stagg.checktype.CheckEnum("FractionBinning", "layout", required=False, choices=layouts),
        "layout_reversed": stagg.checktype.CheckBool("FractionBinning", "layout_reversed", required=False),
        "error_method": stagg.checktype.CheckEnum("FractionBinning", "error_method", required=False, choices=error_methods),
        }

    layout          = typedproperty(_params["layout"])
    layout_reversed = typedproperty(_params["layout_reversed"])
    error_method    = typedproperty(_params["error_method"])

    description = "Splits a boolean (true/false) axis into two bins."
    validity_rules = ()
    long_description = """
This binning is intended for predicate data, values that can only be true or false. It can be combined with other axis types to compute fractions as a function of some other binned variable, such as efficiency (probability of some condition) versus a real value or categories. For example,

    Histogram([Axis(FractionBinning(), "pass cuts"),
               Axis(RegularBinning(10, RealInterval(-5, 5)), "x")],
              UnweightedCounts(InterpretedInlineInt64Buffer(
                  [[  9,  25,  29,  35,  54,  67,  60,  84,  80,  94],
                   [ 99, 119, 109, 109,  95, 104, 102, 106, 112, 122]])))

could represent a rising probability of passing cuts versus `"x"`. The first axis has two bins, number passing and total, and the second axis has 10 bins, values of `x`. Fraction binnings are also a good choice for a <<Collection>> axis, because only one set of histograms need to be defined to construct all numerators and denominators.

The *layout* and *layout_reversed* specify what the two bins mean. With a false *layout_reversed*, if *layout* is `passall`, the first bin is the number of inputs that pass a condition (the predicate evaluates to true) and the second is the total number of inputs. If *layout* is `failall`, the first bin is the number of inputs that fail the condition (the predicate evaluates to false). If *layout* is `passfail`, the first bin is the number that pass and the second bin is the number tha fail. These three types of layout can easily be converted to one another, but doing so requires a change to the <<Histogram>> bins or <<BinnedEvaluatedFunction>> values. If *layout_reversed* is true, the order of the two bins is reversed. (Thus, six layouts are possible.)

The *error_method* does not specify how the histograms or functions were filled, but how the fraction should be interpreted statistically. It may be `unspecified`, leaving that interpretation unspecified. The `normal` method (sometimes called "`Wald`") is a naive binomial interpretation, in which zero passing or zero failing values are taken to have zero uncertainty. The `clopper_pearson` method (sometimes called "`exact`") is a common choice, though it fails in some statistical criteria. The computation and meaning of the methods are described in the references below.

*See also:*

   * Newcombe, R. "`Two-Sided Confidence Intervals for the Single
Proportion: Comparison of Seven Methods`" [https://doi.org/10.1002/(SICI)1097-0258(19980430)17:8%3C857::AID-SIM777%3E3.0.CO;2-E[doi]] [http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.408.7107&rep=rep1&type=pdf[pdf]]
   * Dunnigan, K. "`Confidence Interval Calculation for Binomial Proportion`" [http://www.mwsug.org/proceedings/2008/pharma/MWSUG-2008-P08.pdf[pdf]]
   * Mayfield, P. "`Understanding Binomial Confidence Intervals`" [http://sigmazone.com/binomial-confidence-intervals[pdf]]
   * ATLAS collaboration http://www.pp.rhul.ac.uk/~cowan/atlas/ErrorBars.pdf[efficiency error bar recommendations]
   * ROOT https://root.cern.ch/doc/master/classTEfficiency.html[TEfficiency class] documentation
   * R `binom` package [https://cran.r-project.org/web/packages/binom/index.html[CRAN]] [https://cran.r-project.org/web/packages/binom/binom.pdf[pdf]]
   * Wikipedia https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval[Binomial proportion confidence interval]
"""

    def __init__(self, layout=passall, layout_reversed=False, error_method=unspecified):
        self.layout = layout
        self.layout_reversed = layout_reversed
        self.error_method = error_method

    @property
    def isnumerical(self):
        return False

    def _binshape(self):
        return (2,)

    @property
    def dimensions(self):
        return 1

    def _toflatbuffers(self, builder):
        stagg.stagg_generated.FractionBinning.FractionBinningStart(builder)
        if self.layout != self.passall:
            stagg.stagg_generated.FractionBinning.FractionBinningAddLayout(builder, self.layout.value)
        if self.layout_reversed is not False:
            stagg.stagg_generated.FractionBinning.FractionBinningAddLayoutReversed(builder, self.layout_reversed)
        if self.error_method != self.unspecified:
            stagg.stagg_generated.FractionBinning.FractionBinningAddErrorMethod(builder, self.error_method.value)
        return stagg.stagg_generated.FractionBinning.FractionBinningEnd(builder)

    def _dump(self, indent, width, end):
        args = []
        if self.layout != self.passall:
            args.append("layout={0}".format(repr(self.layout)))
        if self.layout_reversed is not False:
            args.append("layout_reversed={0}".format(repr(self.layout_reversed)))
        if self.error_method != self.unspecified:
            args.append("error_method={0}".format(repr(self.error_method)))
        return _dumpline(self, args, indent, width, end)

    def toCategoryBinning(self, format="%g"):
        if self.layout == self.passall:
            categories = ["pass", "all"]
        elif self.layout == self.failall:
            categories = ["fail", "all"]
        elif self.layout == self.passfail:
            categories = ["pass", "fail"]
        else:
            raise AssertionError(self.layout)
        if self.layout_reversed:
            categories = categories[::-1]
        return CategoryBinning(categories)

    def _getindex(self, where):
        return self._getindex_general(where, 2, None, None, None)

    def _getloc(self, isiloc, where):
        if where is None:
            return None, (slice(None),)

        elif isinstance(where, slice) and where.start is None and where.stop is None and where.step is None:
            return self, (slice(None),)

        elif not isiloc and isinstance(where, (bool, numpy.bool_, numpy.bool)):
            raise NotImplementedError

        if isiloc:
            raise TypeError("FractionBinning.iloc accepts an empty slice (`:` only), ellipsis (`...`), or projection (`None`), not {0}".format(repr(where)))
        else:
            raise TypeError("FractionBinning.loc accepts a boolean, an empty slice (`:` only), ellipsis (`...`), or projection (`None`), not {0}".format(repr(where)))

    def _restructure(self, other):
        assert isinstance(other, FractionBinning)

        if self.error_method == other.error_method:
            error_method = self.error_method
        else:
            error_method = self.unspecified

        if self.layout == other.layout and self.reversed == other.reversed:
            if self.error_method == error_method:
                return self, (None,), (None,)
            else:
                return FractionBinning(layout=self.layout, layout_reversed=self.layout_reversed, error_method=error_method)

        elif self.layout == other.layout:
            return self, (None,), (numpy.array([1, 0], dtype=numpy.int64),)

        else:
            raise NotImplementedError("{0}, {1}".format(self.layout, other.layout))

################################################# PredicateBinning

class PredicateBinning(Binning, OverlappingFill):
    _params = {
        "predicates":       stagg.checktype.CheckVector("PredicateBinning", "predicates", required=True, type=str, minlen=1),
        "overlapping_fill": stagg.checktype.CheckEnum("PredicateBinning", "overlapping_fill", required=False, choices=OverlappingFill.overlapping_fill_strategies),
        }

    predicates       = typedproperty(_params["predicates"])
    overlapping_fill = typedproperty(_params["overlapping_fill"])

    description = "Associates predicates (derived boolean features), which may represent different data \"`regions,`\" with bins."
    validity_rules = ()
    long_description = """
This binning is intended to represent data "`regions,`" such as signal and control regions, defined by boolean functions of some input variables. The details of the predicate function are not captured by this class; they are expressed as strings in the *predicates* property. It is up to the user or application to associate string-valued *predicates* with data regions or predicate functions, as executable code, as keys in a lookup function, or as human-readable titles.

Unlike <<CategoryBinning>>, this binning has no possibility of an overflow bin and a single input datum could pass multiple predicates. As with <<IrregularBinning>>, there is an *overlapping_fill* property to specify whether such a value is in `all` matching predicates, the `first`, the `last`, or if this is unknown (`unspecified`).

Use a <<CategoryBinning>> if the data regions are strictly disjoint, have string-valued labels computed in the filling procedure, or could produce strings that are not known before filling. Use a <<PredicateBinning>> if the data regions overlap or are identified by a fixed set of predicate functions. There are some cases in which a <<CategoryBinning>> and a <<PredicateBinning>> are both appropriate.

*See also:*

   * <<CategoryBinning>>: for disjoint categories with a possible overflow bin.
   * <<PredicateBinning>>: for possibly overlapping regions defined by predicate functions.
   * <<VariationBinning>>: for completely overlapping input data, with derived features computed different ways.
"""

    def __init__(self, predicates, overlapping_fill=OverlappingFill.unspecified):
        self.predicates = predicates
        self.overlapping_fill = overlapping_fill

    def _binshape(self):
        return (len(self.predicates),)

    @property
    def dimensions(self):
        return 1

    def _toflatbuffers(self, builder):
        predicates = [builder.CreateString(x.encode("utf-8")) for x in self.predicates]

        stagg.stagg_generated.PredicateBinning.PredicateBinningStartPredicatesVector(builder, len(predicates))
        for x in predicates[::-1]:
            builder.PrependUOffsetTRelative(x)
        predicates = builder.EndVector(len(predicates))

        stagg.stagg_generated.PredicateBinning.PredicateBinningStart(builder)
        stagg.stagg_generated.PredicateBinning.PredicateBinningAddPredicates(builder, predicates)
        if self.overlapping_fill != self.unspecified:
            stagg.stagg_generated.PredicateBinning.PredicateBinningAddOverlappingFill(builder, self.overlapping_fill.value)
        return stagg.stagg_generated.PredicateBinning.PredicateBinningEnd(builder)

    def _dump(self, indent, width, end):
        args = ["predicates=[" + ", ".join(repr(x) for x in self.predicates) + "]"]
        if self.overlapping_fill != OverlappingFill.unspecified:
            args.append("overlapping_fill={0}".format(repr(self.overlapping_fill)))
        return _dumpline(self, args, indent, width, end)

    def toCategoryBinning(self, format="%g"):
        return CategoryBinning(self.predicates)

    def _getindex(self, where):
        return self._getindex_general(where, len(self.predicates), None, None, None)

    def _getloc(self, isiloc, where):
        if where is None:
            return None, (slice(None),)

        elif isinstance(where, slice) and where.start is None and where.stop is None and where.step is None:
            return self, (slice(None),)

        elif isiloc and isinstance(where, slice):
            start, stop, step = where.indices(len(self.predicates))
            if step <= 0:
                raise IndexError("slice step cannot be zero or negative")
            start = max(start, 0)
            stop = min(stop, len(self.predicates))
            d, m = divmod(stop - start, step)
            length = d + (1 if m != 0 else 0)
            stop = start + step*length

            if length == 0:
                raise IndexError("slice {0}:{1} would result in no bins".format(where.start, where.stop))

            predicates = self.predicates[start:stop:step]
            index = numpy.full(len(self.predicates), -1, dtype=numpy.int64)
            index[start:stop:step] = numpy.arange(length)

            binning = PredicateBinning(predicates, overlapping_fill=self.overlapping_fill)

            return binning, (index,)

        elif isiloc and not isinstance(where, (bool, numpy.bool, numpy.bool_)) and isinstance(where, (numbers.Integral, numpy.integer)):
            i = where
            if i < 0:
                i += len(self.predicates)
            if not 0 <= i < len(self.predicates):
                raise IndexError("index {0} is out of bounds".format(where))
            return self._getloc(True, slice(i, i))

        elif not isiloc and isinstance(where, str):
            return self._getloc(False, [where])

        elif not isiloc and isinstance(where, Iterable) and all(isinstance(x, str) for x in where):
            lookup = {x: j for j, x in enumerate(self.predicates)}
            index = numpy.full(len(lookup), -1, dtype=numpy.int64)
            for i, x in enumerate(where):
                j = lookup.get(x, None)
                if j is None:
                    raise IndexError("PredicateBinning does not have predicate {0}".format(repr(x)))
                index[j] = i

            binning = PredicateBinning(where, overlapping_fill=self.overlapping_fill)

            return binning, (index,)

        elif isiloc:
            where = numpy.array(where, copy=False)
            if len(where.shape) == 1 and issubclass(where.dtype.type, (numpy.integer, numpy.bool, numpy.bool_)):
                predicates = numpy.array(self.predicates, dtype=numpy.object)[where]
                if len(predicates) == 0:
                    raise IndexError("index {0} would result in no bins".format(where))
                index = self.full(len(self.predicates), -1, dtype=numpy.int64)
                index[where] = numpy.arange(len(predicates))
                binning = PredicateBinning(predicates, overlapping_fill=self.overlapping_fill)
                return binning, (index,)

        if isiloc:
            raise TypeError("PredicateBinning.iloc accepts an integer, an integer slice (`:`), ellipsis (`...`), projection (`None`), or an array of integers/booleans, not {0}".format(repr(where)))
        else:
            raise TypeError("PredicateBinning.loc accepts a string, an iterable of strings, an empty slice (`:` only), ellipsis (`...`), or projection (`None`), not {0}".format(repr(where)))

    def _restructure(self, other):
        assert isinstance(other, PredicateBinning)

        if self.overlapping_fill == other.overlapping_fill:
            overlapping_fill = self.overlapping_fill
        else:
            overlapping_fill = self.unspecified

        selfpred = list(self.predicates)
        otherpred = list(other.predicates)

        lookup = {x: i for i, x in enumerate(selfpred)}
        predicates = selfpred + [x for x in otherpred if x not in lookup]

        if len(predicates) == len(selfpred) == len(otherpred) and selfpred == otherpred:
            if overlapping_fill == self.overlapping_fill:
                return self, (None,), (None,)
            else:
                return PredicateBinning(selfpred, overlapping_fill=overlapping_fill), (None,), (None,)

        else:
            lookup = {x: i for i, x in enumerate(predicates)}
            othermap = numpy.array([lookup[x] for x in otherpred], dtype=numpy.int64)

            if overlapping_fill == self.overlapping_fill:
                return self, (None,), (othermap,)
            else:
                return PredicateBinning(selfpred, overlapping_fill=overlapping_fill), (None,), (othermap,)

################################################# Assignment

class Assignment(Stagg):
    _params = {
        "identifier": stagg.checktype.CheckKey("Assignment", "identifier", required=True, type=str),
        "expression": stagg.checktype.CheckString("Assignment", "expression", required=True),
        }

    identifier = typedproperty(_params["identifier"])
    expression = typedproperty(_params["expression"])

    description = "Represents one derived feature in a <<Variation>>."
    validity_rules = ()
    long_description = """
The *identifier* is the name of the derived feature that gets recomputed in this <<Variation>>, and *expression* is what it is assigned to. No constraints are placed on the *expression* syntax; it may even be a key to a lookup function or a human-readable description.
"""

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

    def _dump(self, indent, width, end):
        args = ["identifier={0}".format(repr(self.identifier)), "expression={0}".format(_dumpeq(self.expression._dump(indent + "    ", width, end), indent, end))]
        return _dumpline(self, args, indent, width, end)

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

    description = "Represents one systematic variation, which is one bin of a <<VariationBinning>>."
    validity_rules = ("The *identifier* in each of the *assignments* must be unique.",)
    long_description = """
The *assignments* specify how the derived features were computed when filling this bin. The <<Assignment>> class is defined below.

Variations may be labeled as representing systematic errors. For instance, one bin may be "`one sigma high`" and another "`one sigma low.`" In general, several types of systematic error may be varied at once, and they may be varied by any amount in any direction. Therefore, this object describes a point in a vector space: the number of dimensions in this space is the number of types of systematic errors and the basis vectors are variations of each type of systematic error separately.

Some systematic errors are quantitative (e.g. misalignment) and others are categorical (e.g. choice of simulation algorithm). There are therefore two vectors: *systematic* is real-valued and *category_systematic* is string-valued.
"""

    def __init__(self, assignments, systematic=None, category_systematic=None):
        self.assignments = assignments
        self.systematic = systematic
        self.category_systematic = category_systematic

    def _valid(self, seen, recursive):
        if len(set(x.identifier for x in self.assignments)) != len(self.assignments):
            raise ValueError("Variation.assignments keys must be unique")
        if recursive:
            _valid(self.assignments, seen, recursive)

    @classmethod
    def _fromflatbuffers(cls, fb):
        out = cls.__new__(cls)
        out._flatbuffers = _MockFlatbuffers()
        out._flatbuffers.Assignments = fb.Assignments
        out._flatbuffers.AssignmentsLength = fb.AssignmentsLength
        out._flatbuffers.Systematic = lambda: numpy.empty(0, dtype="<f8") if fb.SystematicLength() == 0 else fb.SystematicAsNumpy()
        out._flatbuffers.CategorySystematic = fb.CategorySystematic
        out._flatbuffers.CategorySystematicLength = fb.CategorySystematicLength
        return out

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
            systematicbuf = systematic.tostring()
            stagg.stagg_generated.Variation.VariationStartSystematicVector(builder, len(self.systematic))
            builder.head = builder.head - len(systematicbuf)
            builder.Bytes[builder.head : builder.head + len(systematicbuf)] = systematicbuf
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

    def _dump(self, indent, width, end):
        args = ["assignments=[" + _dumpeq(_dumplist([x._dump(indent + "    ", width, end) for x in self.assignments], indent, width, end), indent, end) + "]"]
        if self.systematic is not None:
            args.append("systematic=[{0}]".format(", ".join(repr(x) for x in self.systematic)))
        if self.category_systematic is not None:
            args.append("category_systematic=[{0}]".format(", ".join(repr(x) for x in self.category_systematic)))
        return _dumpline(self, args, indent, width, end)

################################################# VariationBinning

class SystematicUnitsEnum(Enum):
    base = "VariationBinning"

class VariationBinning(Binning):
    unspecified = SystematicUnitsEnum("unspecified", stagg.stagg_generated.SystematicUnits.SystematicUnits.syst_unspecified)
    confidence  = SystematicUnitsEnum("confidence", stagg.stagg_generated.SystematicUnits.SystematicUnits.syst_confidence)
    sigmas      = SystematicUnitsEnum("sigmas", stagg.stagg_generated.SystematicUnits.SystematicUnits.syst_sigmas)
    units = [unspecified, confidence, sigmas]

    _params = {
        "variations":                stagg.checktype.CheckVector("VariationBinning", "variations", required=True, type=Variation, minlen=1),
        "systematic_units":          stagg.checktype.CheckEnum("VariationBinning", "systematic_units", required=False, choices=units),
        "systematic_names":          stagg.checktype.CheckVector("VariationBinning", "category_systematic_names", required=False, type=str),
        "category_systematic_names": stagg.checktype.CheckVector("VariationBinning", "category_systematic_names", required=False, type=str),
        }

    variations                = typedproperty(_params["variations"])
    systematic_units          = typedproperty(_params["systematic_units"])
    systematic_names          = typedproperty(_params["systematic_names"])
    category_systematic_names = typedproperty(_params["category_systematic_names"])

    description = "Associates alternative derived features of the same input data, which may represent systematic variations of the data, with bins."
    validity_rules = ("All *variations* must define the same set of *identifiers* in its *assignments*.",
                      "All *variations* must have the same lengh *systematic* vector as this binning has *systematic_names* and the same length *category_systematic* vector as this binning has *category_systematic_names*.")
    long_description = """
This binning is intended to represent systematic variations of the same data. A filling procedure should fill every bin with derived features computed in different ways. In this way, the relevance of a systematic error can be estimated.

Each of the *variations* are <<Variation>> objects, which are defined below.

Variations may be labeled as representing systematic errors. For instance, one bin may be "`one sigma high`" and another "`one sigma low.`" In general, several types of systematic error may be varied at once, and they may be varied by any amount in any direction. Each <<Variation>> therefore describes a point in a vector space: the number of dimensions in this space is the number of types of systematic errors and the basis vectors are variations of each type of systematic error separately.

Some systematic errors are quantitative (e.g. misalignment) and others are categorical (e.g. choice of simulation algorithm). There are therefore two vectors in each <<Variation>>, one real-valued, the other string-valued. The *systematic_units* defines the units of the real-valued systematics vector.

The *systematic_names* labels the dimensions of the <<Variation>> *systematic* vectors; they must all have the same number of dimensions. The *category_systematic_names* labels the dimensions of the <<Variation>> *category_systematic* vectors; they, too, must all have the same number of dimensions.

*See also:*

   * <<CategoryBinning>>: for disjoint categories with a possible overflow bin.
   * <<PredicateBinning>>: for possibly overlapping regions defined by predicate functions.
   * <<VariationBinning>>: for completely overlapping input data, with derived features computed different ways.
"""

    def __init__(self, variations, systematic_units=unspecified, systematic_names=None, category_systematic_names=None):
        self.variations = variations
        self.systematic_units = systematic_units
        self.systematic_names = systematic_names
        self.category_systematic_names = category_systematic_names

    def _valid(self, seen, recursive):
        idset = None
        for variation in self.variations:
            thisidset = set(x.identifier for x in variation.assignments)
            if idset is None:
                idset = thisidset
            if idset != thisidset:
                raise ValueError("variation defines identifiers {{{0}}} while another defines {{{1}}}".format(", ".join(sorted(idset)), ", ".join(sorted(thisidset))))
            if len(variation.systematic) != len(self.systematic_names):
                raise ValueError("variation has systematic [{0}] (length {1}) but systematic_names has length {2}".format(", ".join("%g" % x for x in variation.systematic), len(variation.systematic), len(self.systematic_names)))
            if len(variation.category_systematic) != len(self.category_systematic_names):
                raise ValueError("variation has category_systematic [{0}] (length {1}) but category_systematic_names has length {2}".format(", ".join(repr(x) for x in variation.category_systematic), len(variation.category_systematic), len(self.category_systematic_names)))
        if recursive:
            _valid(self.variations, seen, recursive)

    def _binshape(self):
        return (len(self.variations),)

    @property
    def dimensions(self):
        return 1

    def _toflatbuffers(self, builder):
        variations = [x._toflatbuffers(builder) for x in self.variations]
        systematic_names = None if len(self.systematic_names) == 0 else [builder.CreateString(x.encode("utf-8")) for x in self.systematic_names]
        category_systematic_names = None if len(self.category_systematic_names) == 0 else [builder.CreateString(x.encode("utf-8")) for x in self.category_systematic_names]

        stagg.stagg_generated.VariationBinning.VariationBinningStartVariationsVector(builder, len(variations))
        for x in variations[::-1]:
            builder.PrependUOffsetTRelative(x)
        variations = builder.EndVector(len(variations))

        if systematic_names is not None:
            stagg.stagg_generated.VariationBinning.VariationBinningStartSystematicNamesVector(builder, len(systematic_names))
            for x in systematic_names[::-1]:
                builder.PrependUOffsetTRelative(x)
            systematic_names = builder.EndVector(len(systematic_names))

        if category_systematic_names is not None:
            stagg.stagg_generated.VariationBinning.VariationBinningStartCategorySystematicNamesVector(builder, len(category_systematic_names))
            for x in category_systematic_names[::-1]:
                builder.PrependUOffsetTRelative(x)
            category_systematic_names = builder.EndVector(len(category_systematic_names))

        stagg.stagg_generated.VariationBinning.VariationBinningStart(builder)
        stagg.stagg_generated.VariationBinning.VariationBinningAddVariations(builder, variations)
        if self.systematic_units != self.unspecified:
            stagg.stagg_generated.VariationBinning.VariationBinningAddSystematicUnits(builder, self.systematic_units.value)
        if systematic_names is not None:
            stagg.stagg_generated.VariationBinning.VariationBinningAddSystematicNames(builder, systematic_names)
        if category_systematic_names is not None:
            stagg.stagg_generated.VariationBinning.VariationBinningAddCategorySystematicNames(builder, category_systematic_names)
        return stagg.stagg_generated.VariationBinning.VariationBinningEnd(builder)

    def _dump(self, indent, width, end):
        args = ["variations=[" + _dumpeq(_dumplist([x._dump(indent + "    ", width, end) for x in self.variations], indent, width, end), indent, end) + "]"]
        if self.systematic_units != self.unspecified:
            args.append("systematic_units={0}".format(repr(self.systematic_units)))
        if self.systematic_names is not None:
            args.append("systematic_names=[{0}]".format(", ".join(repr(x) for x in self.systematic_names)))
        if self.category_systematic_names is not None:
            args.append("category_systematic_names=[{0}]".format(", ".join(repr(x) for x in self.category_systematic_names)))
        return _dumpline(self, args, indent, width, end)

    def toCategoryBinning(self, format="%g"):
        categories = []
        for variation in self.variations:
            categories.append("; ".join("{0} := {1}".format(x.identifier, x.expression) for x in variation.assignments))
        return CategoryBinning(categories)

    def _getindex(self, where):
        return self._getindex_general(where, len(self.variations), None, None, None)

    def _getloc(self, isiloc, where):
        if where is None:
            return None, (slice(None),)

        elif isinstance(where, slice) and where.start is None and where.stop is None and where.step is None:
            return self, (slice(None),)

        elif isiloc and isinstance(where, slice):
            start, stop, step = where.indices(len(self.variations))
            if step <= 0:
                raise IndexError("slice step cannot be zero or negative")
            start = max(start, 0)
            stop = min(stop, len(self.variations))
            d, m = divmod(stop - start, step)
            length = d + (1 if m != 0 else 0)
            stop = start + step*length

            if length == 0:
                raise IndexError("slice {0}:{1} would result in no bins".format(where.start, where.stop))

            variations = self.variations[start:stop:step]
            index = numpy.full(len(self.variations), -1, dtype=numpy.int64)
            index[start:stop:step] = numpy.arange(length)

            binning = VariationBinning([x.detached() for x in variations], systematic_units=self.systematic_units, systematic_names=self.systematic_names, category_systematic_names=self.category_systematic_names)

            return binning, (index,)

        elif isiloc and not isinstance(where, (bool, numpy.bool, numpy.bool_)) and isinstance(where, (numbers.Integral, numpy.integer)):
            i = where
            if i < 0:
                i += len(self.variations)
            if not 0 <= i < len(self.variations):
                raise IndexError("index {0} is out of bounds".format(where))
            return self._getloc(True, slice(i, i))

        elif isiloc:
            where = numpy.array(where, copy=False)
            if len(where.shape) == 1 and issubclass(where.dtype.type, (numpy.integer, numpy.bool, numpy.bool_)):
                variations = numpy.array(self.variations, dtype=numpy.object)[where]
                if len(variations) == 0:
                    raise IndexError("index {0} would result in no bins".format(where))
                index = self.full(len(self.variations), -1, dtype=numpy.int64)
                index[where] = numpy.arange(len(variations))
                binning = VariationBinning([x.detached() for x in variations], systematic_units=self.systematic_units, systematic_names=self.systematic_names, category_systematic_names=self.category_systematic_names)
                return binning, (index,)

        if isiloc:
            raise TypeError("VariationBinning.iloc accepts an integer, an integer slice (`:`), ellipsis (`...`), projection (`None`), or an array of integers/booleans, not {0}".format(repr(where)))
        else:
            raise TypeError("VariationBinning.loc accepts an empty slice (`:` only), ellipsis (`...`), or projection (`None`), not {0}".format(repr(where)))

    def _restructure(self, other):
        assert isinstance(other, VariationBinning)

        if self.variations != other.variations:
            raise ValueError("cannot add VariationBinnings with different sets of variations")

        return self, (None,), (None,)

################################################# Axis

class Axis(Stagg):
    _params = {
        "binning":    stagg.checktype.CheckClass("Axis", "binning", required=False, type=Binning),
        "expression": stagg.checktype.CheckString("Axis", "expression", required=False),
        "statistics": stagg.checktype.CheckVector("Axis", "statistics", required=False, type=Statistics),
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

    description = "Axis of a histogram or binned function representing one or more binned dimensions."
    validity_rules = ("The *statistics* must be empty or have a length equal to the number of dimensions in the *binning* (no binning is one-dimensional).",)
    long_description = """
The dimension or dimensions are subdivided by the *binning* property; all other properties provide additional information.

If the axis represents a computed *expression* (derived feature), it may be encoded here as a string. The *title* is a human-readable description.

A <<Statistics>> object (one per dimension) summarizes the data separately from the histogram counts. For instance, it may contain the mean and standard deviation of all data along a dimension, which is more accurate than a mean and standard deviation derived from the counts.

The *expression*, *title*, *metadata*, and *decoration* properties have no semantic constraints.
"""

    def __init__(self, binning=None, expression=None, statistics=None, title=None, metadata=None, decoration=None):
        self.binning = binning
        self.expression = expression
        self.statistics = statistics
        self.title = title
        self.metadata = metadata
        self.decoration = decoration

    def _valid(self, seen, recursive):
        if len(self.statistics) != 0:
            if self.binning is None and len(self.statistics) != 1:
                raise ValueError("one-dimensional axis (binning=None) must have 0 or 1 Statistics objects")
            if self.binning is not None and len(self.statistics) != self.binning.dimensions:
                raise ValueError("axis with dimension {0} (binning={1}) must have 0 or {0} Statistics objects".format(self.binning.dimensions, type(self.binning).__name__))
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
        out._flatbuffers.StatisticsLength = fb.StatisticsLength
        out._flatbuffers.Title = fb.Title
        out._flatbuffers.Metadata = fb.Metadata
        out._flatbuffers.Decoration = fb.Decoration
        return out

    def _toflatbuffers(self, builder):
        statistics = None if len(self.statistics) == 0 else [x._toflatbuffers(builder) for x in self.statistics]

        if statistics is not None:
            stagg.stagg_generated.Axis.AxisStartStatisticsVector(builder, len(statistics))
            for x in statistics[::-1]:
                builder.PrependUOffsetTRelative(x)
            statistics = builder.EndVector(len(statistics))

        binning = None if self.binning is None else self.binning._toflatbuffers(builder)
        expression = None if self.expression is None else builder.CreateString(self.expression.encode("utf-8"))
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

    def _dump(self, indent, width, end):
        args = []
        if self.binning is not None:
            args.append("binning={0}".format(_dumpeq(self.binning._dump(indent + "    ", width, end), indent, end)))
        if self.expression is not None:
            args.append("expression={0}".format(_dumpstring(self.expression)))
        if len(self.statistics) != 0:
            args.append("statistics=[{0}]".format(_dumpeq(_dumplist([x._dump(indent + "    ", width, end) for x in self.statistics], indent, width, end), indent, end)))
        if self.title is not None:
            args.append("title={0}".format(_dumpstring(self.title)))
        if self.metadata is not None:
            args.append("metadata={0}".format(_dumpeq(self.metadata._dump(indent + "    ", width, end), indent, end)))
        if self.decoration is not None:
            args.append("decoration={0}".format(_dumpeq(self.decoration._dump(indent + "    ", width, end), indent, end)))
        return _dumpline(self, args, indent, width, end)

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

    description = "Summarizes a dependent variable in a <<Histogram>>, binned by the <<Histogram>> axis (independent variables)."
    validity_rules = ()
    long_description = """
Although a statistician's histogram strictly represents a distribution, it is often useful to store a few more values per bin to estimate average values for an empirical function from a dataset. This practice is common in particle physics, from HPROF in CERNLIB to https://root.cern.ch/doc/master/classTProfile.html[TProfile] in ROOT.

To estimate an unweighted mean and standard deviation of `x`, one needs the *counts* from <<UnweightedCounts>> as well as a sum of `x` and a sum of squares of `x`. For a weighted mean and standard deviation of `x`, one needs the *sumw* (sum of weights) and *sumw2* (sum of squared weights) from <<WeightedCounts>> as well as a sum of weights times `x` and a sum of weights times squares of `x`.

Rather than making profile a separate class from histograms, as is commonly done in particle physics, we can add profiled quantities to a <<Histogram>> object. If we have many profiles with the same binning, this avoids duplication of the *counts* or *sumw* and *sumw2*. We can also generalize from storing only moments (to compute mean and standard deviation) to also storing quantiles (to compute a box-and-whiskers plot, for instance).

If the profile represents a computed *expression* (derived feature), it may be encoded here as a string. The *title* is a human-readable description.

All of the *moments*, *quantiles*, and any *mode*, *min*, or *max* are in the required *statistics* object. See below for a definition of the <<Statistics>> class.

The *title*, *metadata*, and *decoration* properties have no semantic constraints.
"""

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

    def _dump(self, indent, width, end):
        args = ["expression={0}".format(repr(self.expression)), "statistics={0}".format(_dumpeq(self.statistics._dump(indent + "    ", width, end), indent, end))]
        if self.title is not None:
            args.append("title={0}".format(_dumpstring(self.title)))
        if self.metadata is not None:
            args.append("metadata={0}".format(_dumpeq(self.metadata._dump(indent + "    ", width, end), indent, end)))
        if self.decoration is not None:
            args.append("decoration={0}".format(_dumpeq(self.decoration._dump(indent + "    ", width, end), indent, end)))
        return _dumpline(self, args, indent, width, end)

################################################# Counts

class Counts(Stagg):
    def __init__(self):
        raise TypeError("{0} is an abstract base class; do not construct".format(type(self).__name__))

    @staticmethod
    def _promote(one, two):
        if isinstance(one, UnweightedCounts) and isinstance(two, UnweightedCounts):
            return one, two

        has_sumw2 = (isinstance(one, WeightedCounts) and one.sumw2 is not None) and (isinstance(two, WeightedCounts) and two.sumw2 is not None)
        has_unweighted = (isinstance(one, UnweightedCounts) or one.unweighted is not None) and (isinstance(two, UnweightedCounts) or two.unweighted is not None)

        if isinstance(one, UnweightedCounts) or has_sumw2 != (one.sumw2 is not None) or has_unweighted != (one.unweighted is not None):
            sumw = one.counts.detached() if isinstance(one, UnweightedCounts) else one.sumw.detached()
            sumw2 = one.sumw2.detached() if has_sumw2 else None
            unweighted = one.detached() if isinstance(one, UnweightedCounts) else None if one.unweighted is None else one.unweighted.detached()
            one = WeightedCounts(sumw, sumw2=sumw2, unweighted=unweighted)

        if isinstance(two, UnweightedCounts) or has_sumw2 != (two.sumw2 is not None) or has_unweighted != (two.unweighted is not None):
            sumw = two.counts.detached() if isinstance(two, UnweightedCounts) else two.sumw.detached()
            sumw2 = two.sumw2.detached() if has_sumw2 else None
            unweighted = two.detached() if isinstance(two, UnweightedCounts) else None if two.unweighted is None else two.unweighted.detached()
            two = WeightedCounts(sumw, sumw2=sumw2, unweighted=unweighted)

        return one, two

    def __getitem__(self, where):
        if not isinstance(where, tuple):
            where = (where,)

        if not isinstance(getattr(self, "_parent", None), Histogram):
            raise ValueError("this {0} is not attached to a Histogram".format(type(self).__name__))

        node = self._parent
        binnings = ()
        while hasattr(node, "_parent"):
            node = node._parent
            binnings = tuple(x.binning for x in node.axis) + binnings
        binnings = binnings + tuple(x.binning for x in self._parent.axis)
        oldshape = sum((x._binshape() for x in binnings), ())
        where = self._parent._expand_ellipsis(where, len(oldshape))

        i = 0
        indexes = ()
        for binning in binnings:
            indexes = indexes + binning._getindex(*where[i : i + binning.dimensions])
            i += binning.dimensions

        return self._reindex(oldshape, indexes)

################################################# UnweightedCounts

class UnweightedCounts(Counts):
    _params = {
        "counts": stagg.checktype.CheckClass("UnweightedCounts", "counts", required=True, type=InterpretedBuffer),
        }

    counts = typedproperty(_params["counts"])

    description = "Represents counts in a <<Histogram>> that were filled without weighting. (All inputs increase bin values by one unit.)"
    validity_rules = ()
    long_description = """
The *counts* buffer contains the actual values. Since these counts are unweighted, they could have unsigned integer type, but no such constraint is applied.

A <<Histogram>> bin count is typically interpreted as an estimate of the probability of a data value falling into that bin times the total number of input values. It is therefore estimating a probability distribution, and that estimate has uncertainty. The uncertainty for unweighted counts follows a Poisson distribution. In the limit of large counts, the uncertainty approaches the square root of the number of counts, with deviations from this for small counts. A separate statistic to quantify this uncertainty is unnecessary because it can be fully determined from the number of counts.

To be valid, the length of the *counts* buffer (in number of items, not number of bytes) must be equal to the number of bins in this <<Histogram>>, including any axes inherited by nesting the <<Histogram>> in a <<Collection>>. The number of bins in the <<Histogram>> is the product of the number of bins in each <<Axis>>, including any underflow, overflow, or nanflow bins. That is, it must be possible to reshape the buffer into a multidimensional array, in which every dimension corresponds to one <<Axis>>.
"""

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

    def _dump(self, indent, width, end):
        args = ["counts={0}".format(_dumpeq(self.counts._dump(indent + "    ", width, end), indent, end))]
        return _dumpline(self, args, indent, width, end)

    def _reindex(self, oldshape, indexes):
        return self.counts._reindex(oldshape, indexes)

    def _rebin(self, oldshape, pairs):
        return UnweightedCounts(self.counts._rebin(oldshape, pairs))

    def _remap(self, newshape, selfmap):
        return UnweightedCounts(self.counts._remap(newshape, selfmap))

    def _add(self, other, noclobber):
        assert isinstance(other, UnweightedCounts)
        self.counts = self.counts._add(other.counts, noclobber)
        return self

    @property
    def flatarray(self):
        return self.counts.flatarray

    @property
    def array(self):
        return self.counts.array

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

    description = "Represents counts in a <<Histogram>> that were filled with weights. (Some inputs may increase bin values more than others, or even by a negative amount.)"
    validity_rules = ()
    long_description = """
The *sumw* (sum of weights) buffer contains the actual values. Since these values are weighted, they might need a floating point or even signed type.

A <<Histogram>> bin count is typically interpreted as an estimate of the probability of a data value falling into that bin times the total number of input values. It is therefore estimating a probability distribution, and that estimate has uncertainty. The uncertainty for weighted counts is approximately the square root of the sum of squared weights, so this object can optionally store *sumw2*, the sum of squared weights, to compute this uncertainty.

It may also be necessary to know the unweighted counts, as well as the weighted counts, so there is an *unweighted* property for that.

To be valid, the length of all of these buffers (in number of items, not number of bytes) must be equal to the number of bins in this <<Histogram>>, including any axes inherited by nesting the <<Histogram>> in a <<Collection>>. The number of bins in the <<Histogram>> is the product of the number of bins in each <<Axis>>, including any underflow, overflow, or nanflow bins. That is, it must be possible to reshape these buffers into multidimensional arrays of the same shape, in which every dimension corresponds to one <<Axis>>.
"""

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

    def _dump(self, indent, width, end):
        args = ["sumw={0}".format(_dumpeq(self.sumw._dump(indent + "    ", width, end), indent, end))]
        if self.sumw2 is not None:
            args.append("sumw2={0}".format(_dumpeq(self.sumw2._dump(indent + "    ", width, end), indent, end)))
        if self.unweighted is not None:
            args.append("unweighted={0}".format(_dumpeq(self.unweighted._dump(indent + "    ", width, end), indent, end)))
        return _dumpline(self, args, indent, width, end)

    def _reindex(self, oldshape, indexes):
        out = {"sumw": self.sumw._reindex(oldshape, indexes)}
        if self.sumw2 is not None:
            out["sumw2"] = self.sumw2._reindex(oldshape, indexes)
        if self.unweighted is not None:
            out["unweighted"] = self.unweighted._reindex(oldshape, indexes)
        return out

    def _rebin(self, oldshape, pairs):
        return WeightedCounts(self.sumw._rebin(oldshape, pairs),
                              None if self.sumw2 is None else self.sumw2._rebin(oldshape, pairs),
                              None if self.unweighted is None else self.unweighted._rebin(oldshape, pairs))

    def _remap(self, newshape, selfmap):
        return WeightedCounts(self.sumw._remap(newshape, selfmap),
                              None if self.sumw2 is None else self.sumw2._remap(newshape, selfmap),
                              None if self.unweighted is None else self.unweighted._remap(newshape, selfmap))

    def _add(self, other, noclobber):
        assert isinstance(other, WeightedCounts)

        self.sumw = self.sumw._add(other.sumw, noclobber)

        if self.sumw2 is not None and other.sumw2 is not None:
            self.sumw2 = self.sumw2._add(other.sumw2, noclobber)
        else:
            self.sumw2 = None

        if self.unweighted is not None and other.unweighted is not None:
            self.unweighted = self.unweighted._add(other.unweighted, noclobber)
        else:
            self.unweighted = None

        return self

    @property
    def flatarray(self):
        out = {"sumw": self.sumw.flatarray}
        if self.sumw2 is not None:
            out["sumw2"] = self.sumw2.flatarray
        if self.unweighted is not None:
            out["unweighted"] = self.unweighted.flatarray
        return out

    @property
    def array(self):
        out = {"sumw": self.sumw.array}
        if self.sumw2 is not None:
            out["sumw2"] = self.sumw2.array
        if self.unweighted is not None:
            out["unweighted"] = self.unweighted.array
        return out

################################################# Parameter

class Parameter(Stagg):
    _params = {
        "identifier": stagg.checktype.CheckKey("Parameter", "identifier", required=True, type=str),
        "values":     stagg.checktype.CheckClass("Parameter", "values", required=True, type=InterpretedBuffer),
        }

    identifier = typedproperty(_params["identifier"])
    values     = typedproperty(_params["values"])

    description = ""
    validity_rules = ()
    long_description = """
"""

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

    def _dump(self, indent, width, end):
        args = ["parameter={0}".format(repr(self.parameter)), "values={0}".format(_dumpeq(self.values._dump(indent + "    ", width, end), indent, end))]
        return _dumpline(self, args, indent, width, end)

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

    description = ""
    validity_rules = ()
    long_description = """
"""

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

    def _dump(self, indent, width, end):
        args = ["expression={0}".format(_dumpstring(self.expression))]
        if len(self.parameters) != 0:
            args.append("parameters=[{0}]".format(_dumpeq(_dumplist([x._dump(indent + "    ", width, end) for x in self.parameters], indent, width, end), indent, end)))
        if self.title is not None:
            args.append("title={0}".format(_dumpstring(self.title)))
        if self.metadata is not None:
            args.append("metadata={0}".format(_dumpeq(self.metadata._dump(indent + "    ", width, end), indent, end)))
        if self.decoration is not None:
            args.append("decoration={0}".format(_dumpeq(self.decoration._dump(indent + "    ", width, end), indent, end)))
        if self.script is not None:
            args.append("script={0}".format(_dumpstring(self.script)))
        return _dumpline(self, args, indent, width, end)

    def _add(self, other, pairs, triples, noclobber):
        raise NotImplementedError

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

    description = ""
    validity_rules = ()
    long_description = """
"""

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

    def _dump(self, indent, width, end):
        args = ["values={0}".format(_dumpeq(self.values._dump(indent + "    ", width, end), indent, end))]
        if self.derivatives is not None:
            args.append("derivatives={0}".format(_dumpeq(self.derivatives._dump(indent + "    ", width, end), indent, end)))
        if len(self.errors) != 0:
            args.append("errors=[{0}]".format(_dumpeq(_dumplist([x._dump(indent + "    ", width, end) for x in self.quantiles], indent, width, end), indent, end)))
        if self.title is not None:
            args.append("title={0}".format(_dumpstring(self.title)))
        if self.metadata is not None:
            args.append("metadata={0}".format(_dumpeq(self.metadata._dump(indent + "    ", width, end), indent, end)))
        if self.decoration is not None:
            args.append("decoration={0}".format(_dumpeq(self.decoration._dump(indent + "    ", width, end), indent, end)))
        if self.script is not None:
            args.append("script={0}".format(_dumpstring(self.script)))
        return _dumpline(self, args, indent, width, end)

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

    description = ""
    validity_rules = ()
    long_description = """
"""

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

    def _dump(self, indent, width, end):
        args = ["axis[" + _dumpeq(_dumplist(x._dump(indent + "    ", width, end) for x in self.axis), indent, end) + "]", "values={0}".format(_dumpeq(self.values._dump(indent + "    ", width, end), indent, end))]
        if self.derivatives is not None:
            args.append("derivatives={0}".format(_dumpeq(self.derivatives._dump(indent + "    ", width, end), indent, end)))
        if len(self.errors) != 0:
            args.append("errors=[{0}]".format(_dumpeq(_dumplist([x._dump(indent + "    ", width, end) for x in self.errors], indent, width, end), indent, end)))
        if self.title is not None:
            args.append("title={0}".format(_dumpstring(self.title)))
        if self.metadata is not None:
            args.append("metadata={0}".format(_dumpeq(self.metadata._dump(indent + "    ", width, end), indent, end)))
        if self.decoration is not None:
            args.append("decoration={0}".format(_dumpeq(self.decoration._dump(indent + "    ", width, end), indent, end)))
        if self.script is not None:
            args.append("script={0}".format(_dumpstring(self.script)))
        return _dumpline(self, args, indent, width, end)

    def _add(self, other, pairs, triples, noclobber):
        raise NotImplementedError

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

    axis                = typedproperty(_params["axis"])
    counts              = typedproperty(_params["counts"])
    profile             = typedproperty(_params["profile"])
    axis_covariances    = typedproperty(_params["axis_covariances"])
    profile_covariances = typedproperty(_params["profile_covariances"])
    functions           = typedproperty(_params["functions"])
    title               = typedproperty(_params["title"])
    metadata            = typedproperty(_params["metadata"])
    decoration          = typedproperty(_params["decoration"])
    script              = typedproperty(_params["script"])

    description = "Histogram of a distribution, defined by a (possibly weighted) count of observations in each bin of an n-dimensional space."
    validity_rules = ("The *xindex* and *yindex* of each Covariance in *axis_covariances* must be in [0, number of *axis*) and be unique pairs.",
                      "The *xindex* and *yindex* of each Covariance in *profile_covariances* must be in [0, number of *profile*) and be unique pairs.")
    long_description = """
The space is subdivided by an n-dimensional *axis*. As described in <<Collection>>, nesting a histogram within a collection prepends the collection's *axis*. The number of <<Axis>> objects is not necessarily the dimensionality of the space; some binnings, such as <<HexagonalBinning>>, define more than one dimension (though most do not).

The *counts* are separate from the *axis*, though the buffers providing counts must be exactly the right size to fit the n-dimensional binning (including axes inherited from a <<Collection>>).

Histograms with only *axis* and *counts* are pure distributions, histograms in the conventional sense. All other properties provide additional information about the dataset.

Any *profiles* summarize dependent variables (where the *axis* defines independent variables). For instance, a profile can represent mean and standard deviation `y` values for an axis binned in `x`.

The <<Axis>> and <<Profile>> classes internally define summary statistics, such as the mean or median of that axis. However, those <<Statistics>> objects cannot describe correlations among axes. If this information is available, it can be expressed in *axis_covariances* or *profile_covariances*.

Any *functions* associated with the histogram, such as fit results, may be attached directly to the histogram object with names. If an <<EvaluatedFunction>> is included, its binning is derived from the histogram's full *axis* (including any *axis* inherited from a <<Collection>>).

The *title*, *metadata*, *decoration*, and *script* properties have no semantic constraints.

*See also:*

   * <<BinnedEvaluatedFunction>>: for lookup functions that aren't statistical distributions.
"""

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

    def _dump(self, indent, width, end):
        args = ["axis=[" + _dumpeq(_dumplist([x._dump(indent + "    ", width, end) for x in self.axis], indent, width, end), indent, end) + "]", "counts={0}".format(_dumpeq(self.counts._dump(indent + "    ", width, end), indent, end))]
        if len(self.profile) != 0:
            args.append("profile=[{0}]".format(_dumpeq(_dumplist([x._dump(indent + "    ", width, end) for x in self.profile], indent, width, end), indent, end)))
        if len(self.axis_covariances) != 0:
            args.append("axis_covariances=[{0}]".format(_dumpeq(_dumplist([x._dump(indent + "    ", width, end) for x in self.axis_covariances], indent, width, end), indent, end)))
        if len(self.profile_covariances) != 0:
            args.append("profile_covariances=[{0}]".format(_dumpeq(_dumplist([x._dump(indent + "    ", width, end) for x in self.profile_covariances], indent, width, end), indent, end)))
        if len(self.functions) != 0:
            args.append("functions=[{0}]".format(_dumpeq(_dumplist([x._dump(indent + "    ", width, end) for x in self.functions], indent, width, end), indent, end)))
        if self.title is not None:
            args.append("title={0}".format(_dumpstring(self.title)))
        if self.metadata is not None:
            args.append("metadata={0}".format(_dumpeq(self.metadata._dump(indent + "    ", width, end), indent, end)))
        if self.decoration is not None:
            args.append("decoration={0}".format(_dumpeq(self.decoration._dump(indent + "    ", width, end), indent, end)))
        if self.script is not None:
            args.append("script={0}".format(_dumpstring(self.script)))
        return _dumpline(self, args, indent, width, end)

    @property
    def allaxis(self):
        out = list(self.axis)
        node = self
        while hasattr(node, "_parent"):
            node = node._parent
            out = list(node.axis) + out
        return stagg.checktype.Vector(out)

    def _expand_ellipsis(self, where, numdims):
        where2 = [None if isinstance(x, numpy.ndarray) else x for x in where]
        ellipsiscount = where2.count(Ellipsis)
        if ellipsiscount > 1:
            raise IndexError("an index can only have a single ellipsis ('...')")
        elif ellipsiscount == 1:
            ellipsisindex = where2.index(Ellipsis)
            before = where[: ellipsisindex]
            after  = where[ellipsisindex + 1 :]
            num = max(0, numdims - len(before) - len(after))
            where = before + num*(slice(None),) + after

        where = where + max(0, numdims - len(where))*(slice(None),)
        if len(where) != numdims:
            raise IndexError("too many indices for histogram")

        return where

    def _getloc(self, isiloc, where, binnings):
        binnings = binnings + tuple(x.binning for x in self.axis)
        oldshape = sum((x._binshape() for x in binnings), ())
        where = self._expand_ellipsis(where, len(oldshape))

        i = 0
        pairs = []
        for binning in binnings:
            pairs.append(binning._getloc(isiloc, *where[i : i + binning.dimensions]))
            i += binning.dimensions

        out = self.detached(exceptions=("axis", "counts"))
        newaxis = []
        for axis, (newbinning, selfmap) in zip(self.axis, pairs[-len(self.axis):]):
            if newbinning is not None:
                newaxis.append(axis.detached(exceptions=("binning")))
                newaxis[-1].binning = newbinning.detached()
        if len(newaxis) == 0:
            out.axis = [Axis()]
        else:
            out.axis = newaxis

        out.counts = self.counts._rebin(oldshape, pairs)
        return out

    def _add(self, other, pairs, triples, noclobber):
        if not isinstance(other, Histogram):
            raise ValueError("cannot add {0} and {1}".format(self, other))
        if len(self.axis) != len(other.axis):
            raise ValueError("cannot add {0}-dimensional Histogram and {1}-dimensional Histogram".format(len(self.axis), len(other.axis)))

        tmppairs = tuple(Binning._promote(one.binning, two.binning) for one, two in zip(self.axis, other.axis))
        pairs = pairs + tmppairs
        triples = triples + tuple(one._restructure(two) for one, two in tmppairs)

        selfcounts, othercounts = Counts._promote(self.counts, other.counts)

        if not all(isinstance(sm, tuple) and sm[0] is None for binning, sm, om in triples):
            newshape = sum((binning._binshape() for binning, sm, om in triples), ())
            selfcounts = selfcounts._remap(newshape, sum((sm for binning, sm, om in triples), ()))

        if not all(isinstance(om, tuple) and om[0] is None for binning, sm, om in triples):
            newshape = sum((binning._binshape() for binning, sm, om in triples), ())
            othercounts = othercounts._remap(newshape, sum((om for binning, sm, om in triples), ()))

        selfcounts._add(othercounts, noclobber)

        for axis, (binning, sm, om) in zip(self.axis, triples[-len(self.axis):]):
            axis.binning = binning

        if self.counts is not selfcounts:
            self.counts = selfcounts

        if len(self.profile) != 0:
            raise NotImplementedError

        if len(self.axis_covariances) != 0:
            raise NotImplementedError

        if len(self.profile_covariances) != 0:
            raise NotImplementedError

        if len(self.functions) != 0:
            raise NotImplementedError

################################################# Page

class Page(Stagg):
    _params = {
        "buffer": stagg.checktype.CheckClass("Page", "buffer", required=True, type=RawBuffer),
        }

    buffer = typedproperty(_params["buffer"])

    description = ""
    validity_rules = ()
    long_description = """
"""

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

    def _dump(self, indent, width, end):
        args = ["buffer={0}".format(_dumpeq(self.buffer._dump(indent + "    ", width, end), indent, end))]
        return _dumpline(self, args, indent, width, end)

################################################# ColumnChunk

class ColumnChunk(Stagg):
    _params = {
        "pages":        stagg.checktype.CheckVector("ColumnChunk", "pages", required=True, type=Page),
        "page_offsets": stagg.checktype.CheckVector("ColumnChunk", "page_offsets", required=True, type=int, minlen=1),
        "page_min":     stagg.checktype.CheckVector("ColumnChunk", "page_min", required=False, type=Extremes),
        "page_max":     stagg.checktype.CheckVector("ColumnChunk", "page_max", required=False, type=Extremes),
        }

    pages        = typedproperty(_params["pages"])
    page_offsets = typedproperty(_params["page_offsets"])
    page_min  = typedproperty(_params["page_min"])
    page_max  = typedproperty(_params["page_max"])

    description = ""
    validity_rules = ()
    long_description = """
"""

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
        elif not isinstance(pageid, (bool, numpy.bool, numpy.bool_)) and isinstance(pageid, (numbers.Integral, numpy.integer)):
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

    @classmethod
    def _fromflatbuffers(cls, fb):
        out = cls.__new__(cls)
        out._flatbuffers = _MockFlatbuffers()
        out._flatbuffers.Pages = fb.Pages
        out._flatbuffers.PagesLength = fb.PagesLength
        out._flatbuffers.PageOffsets = fb.PageOffsetsAsNumpy
        out._flatbuffers.PageMin = fb.PageMin
        out._flatbuffers.PageMinLength = fb.PageMinLength
        out._flatbuffers.PageMax = fb.PageMax
        out._flatbuffers.PageMaxLength = fb.PageMaxLength
        return out

    def _toflatbuffers(self, builder):
        pages = [x._toflatbuffers(builder) for x in self.pages]
        page_min = None if len(self.page_min) == 0 else [x._toflatbuffers(builder) for x in self.page_min]
        page_max = None if len(self.page_max) == 0 else [x._toflatbuffers(builder) for x in self.page_max]

        stagg.stagg_generated.ColumnChunk.ColumnChunkStartPagesVector(builder, len(pages))
        for x in pages[::-1]:
            builder.PrependUOffsetTRelative(x)
        pages = builder.EndVector(len(pages))

        pageoffsetsbuf = self.page_offsets.tostring()
        stagg.stagg_generated.ColumnChunk.ColumnChunkStartPageOffsetsVector(builder, len(self.page_offsets))
        builder.head = builder.head - len(pageoffsetsbuf)
        builder.Bytes[builder.head : builder.head + len(pageoffsetsbuf)] = pageoffsetsbuf
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

    def _dump(self, indent, width, end):
        args = ["pages=[" + _dumpeq(_dumplist([x._dump(indent + "    ", width, end) for x in self.pages], indent, width, end), indent, end) + "]", "page_offsets={0}".format(_dumparray(self.page_offsets, indent, end))]
        if len(self.page_min) != 0:
            args.append("page_min=[{0}]".format(_dumpeq(_dumplist([x._dump(indent + "    ", width, end) for x in self.page_min], indent, width, end), indent, end)))
        if len(self.page_max) != 0:
            args.append("page_max=[{0}]".format(_dumpeq(_dumplist([x._dump(indent + "    ", width, end) for x in self.page_max], indent, width, end), indent, end)))
        return _dumpline(self, args, indent, width, end)
        
################################################# Chunk

class Chunk(Stagg):
    _params = {
        "column_chunks": stagg.checktype.CheckVector("Chunk", "column_chunks", required=True, type=ColumnChunk),
        "metadata":      stagg.checktype.CheckClass("Chunk", "metadata", required=False, type=Metadata),
        }

    column_chunks = typedproperty(_params["column_chunks"])
    metadata      = typedproperty(_params["metadata"])

    description = ""
    validity_rules = ()
    long_description = """
"""

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

    def _dump(self, indent, width, end):
        args = ["column_chunks=[" + _dumpeq(_dumplist([x._dump(indent + "    ", width, end) for x in self.column_chunks], indent, width, end), indent, end) + "]"]
        if self.metadata is not None:
            args.append("metadata={0}".format(_dumpeq(self.metadata._dump(indent + "    ", width, end), indent, end)))
        return _dumpline(self, args, indent, width, end)

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

    description = ""
    validity_rules = ()
    long_description = """
"""

    def __init__(self, identifier, dtype, endianness=InterpretedBuffer.little_endian, filters=None, postfilter_slice=None, title=None, metadata=None, decoration=None):
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
                builder.PrependUint32(x.value)
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

    def _dump(self, indent, width, end):
        args = ["identifier={0}".format(_dumpstring(self.identifier)), "dtype={0}".format(repr(self.dtype))]
        if self.endianness != InterpretedBuffer.little_endian:
            args.append("endianness={0}".format(repr(self.endianness)))
        if self.dimension_order != InterpretedBuffer.c_order:
            args.append("dimension_order={0}".format(repr(self.dimension_order)))
        if len(self.filters) != 0:
            args.append("filters=[{0}]".format(", ".join(repr(x) for x in self.filters)))
        if self.postfilter_slice is not None:
            args.append("postfilter_slice=slice({0}, {1}, {2})".format(self.postfilter_slice.start if self.postfilter_slice.hasStart else "None",
                                                                       self.postfilter_slice.stop if self.postfilter_slice.hasStop else "None",
                                                                       self.postfilter_slice.step if self.postfilter_slice.hasStep else "None"))
        if self.title is not None:
            args.append("title={0}".format(_dumpstring(self.title)))
        if self.metadata is not None:
            args.append("metadata={0}".format(_dumpeq(self.metadata._dump(indent + "    ", width, end), indent, end)))
        if self.decoration is not None:
            args.append("decoration={0}".format(_dumpeq(self.decoration._dump(indent + "    ", width, end), indent, end)))
        return _dumpline(self, args, indent, width, end)

################################################# NtupleInstance

class NtupleInstance(Stagg):
    _params = {
        "chunks":        stagg.checktype.CheckVector("NtupleInstance", "chunks", required=True, type=Chunk),
        "chunk_offsets": stagg.checktype.CheckVector("NtupleInstance", "chunk_offsets", required=False, type=int),
        }

    chunks              = typedproperty(_params["chunks"])
    chunk_offsets       = typedproperty(_params["chunk_offsets"])

    description = ""
    validity_rules = ()
    long_description = """
"""

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
            elif not isinstance(chunkid, (bool, numpy.bool, numpy.bool_)) and isinstance(chunkid, (numbers.Integral, numpy.integer)):
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

    @classmethod
    def _fromflatbuffers(cls, fb):
        out = cls.__new__(cls)
        out._flatbuffers = _MockFlatbuffers()
        out._flatbuffers.Chunks = fb.Chunks
        out._flatbuffers.ChunksLength = fb.ChunksLength
        out._flatbuffers.ChunkOffsets = lambda: numpy.empty(0, dtype="<i8") if fb.ChunkOffsetsLength() == 0 else fb.ChunkOffsetsAsNumpy()
        return out

    def _toflatbuffers(self, builder):
        chunks = [x._toflatbuffers(builder) for x in self.chunks]

        stagg.stagg_generated.NtupleInstance.NtupleInstanceStartChunksVector(builder, len(chunks))
        for x in chunks[::-1]:
            builder.PrependUOffsetTRelative(x)
        chunks = builder.EndVector(len(chunks))

        if len(self.chunk_offsets) == 0:
            chunk_offsets = None
        else:
            chunkoffsetsbuf = self.chunk_offsets.tostring()
            stagg.stagg_generated.NtupleInstance.NtupleInstanceStartChunkOffsetsVector(builder, len(self.chunk_offsets))
            builder.head = builder.head - len(chunkoffsetsbuf)
            builder.Bytes[builder.head : builder.head + len(chunkoffsetsbuf)] = chunkoffsetsbuf
            chunk_offsets = builder.EndVector(len(self.chunk_offsets))

        stagg.stagg_generated.NtupleInstance.NtupleInstanceStart(builder)
        stagg.stagg_generated.NtupleInstance.NtupleInstanceAddChunks(builder, chunks)
        if chunk_offsets is not None:
            stagg.stagg_generated.NtupleInstance.NtupleInstanceAddChunkOffsets(builder, chunk_offsets)
        return stagg.stagg_generated.NtupleInstance.NtupleInstanceEnd(builder)

    def _dump(self, indent, width, end):
        args = ["chunks=[" + _dumpeq(_dumplist([x._dump(indent + "    ", width, end) for x in self.chunks], indent, width, end), indent, end) + "]"]
        if len(self.chunk_offsets) != 0:
            args.append("chunk_offsets={0}".format(_dumparray(self.chunk_offsets, indent, end)))
        return _dumpline(self, args, indent, width, end)

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

    columns            = typedproperty(_params["columns"])
    instances          = typedproperty(_params["instances"])
    column_statistics  = typedproperty(_params["column_statistics"])
    column_covariances = typedproperty(_params["column_covariances"])
    functions          = typedproperty(_params["functions"])
    title              = typedproperty(_params["title"])
    metadata           = typedproperty(_params["metadata"])
    decoration         = typedproperty(_params["decoration"])
    script             = typedproperty(_params["script"])

    description = ""
    validity_rules = ()
    long_description = """
"""

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

    def _dump(self, indent, width, end):
        args = ["columns=[" + _dumpeq(_dumplist(x._dump(indent + "    ", width, end) for x in self.columns), indent, end) + "]", "instances=[" + _dumpeq(_dumplist(x._dump(indent + "    ", width, end) for x in self.instances), indent, end) + "]"]
        if len(self.column_statistics) != 0:
            args.append("column_statistics=[{0}]".format(_dumpeq(_dumplist([x._dump(indent + "    ", width, end) for x in self.column_statistics], indent, width, end), indent, end)))
        if len(self.column_covariances) != 0:
            args.append("column_covariances=[{0}]".format(_dumpeq(_dumplist([x._dump(indent + "    ", width, end) for x in self.column_covariances], indent, width, end), indent, end)))
        if len(self.functions) != 0:
            args.append("functions=[{0}]".format(_dumpeq(_dumplist([x._dump(indent + "    ", width, end) for x in self.functions], indent, width, end), indent, end)))
        if self.title is not None:
            args.append("title={0}".format(_dumpstring(self.title)))
        if self.metadata is not None:
            args.append("metadata={0}".format(_dumpeq(self.metadata._dump(indent + "    ", width, end), indent, end)))
        if self.decoration is not None:
            args.append("decoration={0}".format(_dumpeq(self.decoration._dump(indent + "    ", width, end), indent, end)))
        if self.script is not None:
            args.append("script={0}".format(_dumpstring(self.script)))
        return _dumpline(self, args, indent, width, end)

    def _add(self, other, pairs, triples, noclobber):
        raise NotImplementedError

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

    description = "Collection of named objects, possibly with one or more common axis."
    validity_rules = ()
    long_description = """
A simple reason for using a collection would be to gather many objects into a convenient package that can be transmitted as a group. For this purpose, *axis* should be empty. Note that objects (such as histograms, functions, and ntuples) do not have names on their own; names are just keys in the *objects* property, used solely for lookup.

Assigning an *axis* to a collection, rather than individually to all objects it contains, is to avoid duplication when defining similarly binned data. As an example of the latter, consider histograms three `h1`, `h2`, `h3` with two sets of cuts applied, `"signal"` and `"control"` (six histograms total).

    Collection({"h1": h1, "h2": h2, "h3": h3},
               axis=[Axis(PredicateBinning("signal"), PredicateBinning("control"))])

This predicate axis (defined by if-then rules when the histograms were filled) is prepended onto the axes defined in each histogram separately. For instance, if `h1` had one regular axis and `h2` had two irregular axes, the `"h1"` in this collection has two axes: predicate, then regular, and the `"h2"` in this collection has three axes: predicate, then irregular, then irregular. This way, hundreds or thousands of histograms with similar binning can be defined in a contiguous block without repetition of axis definition (good for efficiency and avoiding copy-paste errors).

To subdivide one set of objects and not another, or to subdivide two sets of objects differently, put collections inside of collections. In the following example, `h1` and `h2` are subdivided but `h3` is not.

    Collection({"by region":
                    Collection({"h1": h1, "h2": h2},
                    axis=[Axis(PredicateBinning("signal"), PredicateBinning("control"))]),
                "h3": h3})

Similarly, regions can be subdivided into subregions, and other binning types may be used.

The buffers for each object must be the appropriate size to represent all of its axes, including any inherited from collections. (For example, a counts buffer appropriate for a standalone `h1` would not fit an `"h1"` with prepended axes due to being in a collection.)

The *title*, *metadata*, *decoration*, and *script* properties have no semantic constraints.
"""

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

    def _dump(self, indent, width, end):
        args = []
        if len(self.objects) != 0:
            args.append("objects=[{0}]".format(_dumpeq(_dumplist([x._dump(indent + "    ", width, end) for x in self.objects], indent, width, end), indent, end)))
        if len(self.axis) != 0:
            args.append("axis=[{0}]".format(_dumpeq(_dumplist([x._dump(indent + "    ", width, end) for x in self.axis], indent, width, end), indent, end)))
        if self.title is not None:
            args.append("title={0}".format(_dumpstring(self.title)))
        if self.metadata is not None:
            args.append("metadata={0}".format(_dumpeq(self.metadata._dump(indent + "    ", width, end), indent, end)))
        if self.decoration is not None:
            args.append("decoration={0}".format(_dumpeq(self.decoration._dump(indent + "    ", width, end), indent, end)))
        if self.script is not None:
            args.append("script={0}".format(_dumpstring(self.script)))
        return _dumpline(self, args, indent, width, end)

    @staticmethod
    def _pairs_triples(one, two):
        oneaxis = [] if one is None else list(one.axis)
        twoaxis = [] if two is None else list(two.axis)

        if len(oneaxis) != len(twoaxis):
            raise ValueError("cannot add {0}-dimensional Collection and {1}-dimensional Collection".format(len(oneaxis), len(twoaxis)))
        if len(oneaxis) == 0:
            return (), ()

        pairs, triples = Collection._pairs_triples(getattr(one, "_parent", None), getattr(two, "_parent", None))

        tmppairs = tuple(Binning._promote(one.binning, two.binning) for one, two in zip(oneaxis, twoaxis))
        pairs = pairs + tmppairs
        triples = triples + tuple(one._restructure(two) for one, two in tmppairs)

        return pairs, triples

    def _add(self, other, pairs, triples, noclobber):
        if not isinstance(other, Collection):
            raise ValueError("cannot add {0} and {1}".format(self, other))
        if len(self.axis) != len(other.axis):
            raise ValueError("cannot add {0}-dimensional Collection and {1}-dimensional Collection".format(len(self.axis), len(other.axis)))

        tmppairs = tuple(Binning._promote(one.binning, two.binning) for one, two in zip(self.axis, other.axis))
        pairs = pairs + tmppairs
        triples = triples + tuple(one._restructure(two) for one, two in tmppairs)

        if set(self.objects) != set(other.objects):
            newobjects = collections.OrderedDict()
            for n, x in self.objects.items():
                newobjects[n] = x.detached(reclaim=True)
            for n, x in other.objects.items():
                if n in newobjects:
                    newobjects[n]._add(x, pairs, triples, noclobber)
                else:
                    newobjects[n] = x.detached(reclaim=True)
            self.objects = newobjects

        else:
            for n, x in self.objects.items():
                x._add(other.objects[n], pairs, triples, noclobber)

        for axis, (binning, sm, om) in zip(self.axis, triples[-len(self.axis):]):
            axis.binning = binning

_RawBuffer_lookup = {
    stagg.stagg_generated.RawBuffer.RawBuffer.RawInlineBuffer: (RawInlineBuffer, stagg.stagg_generated.RawInlineBuffer.RawInlineBuffer),
    stagg.stagg_generated.RawBuffer.RawBuffer.RawExternalBuffer: (RawExternalBuffer, stagg.stagg_generated.RawExternalBuffer.RawExternalBuffer),
    }
_RawBuffer_invlookup = {x[0]: n for n, x in _RawBuffer_lookup.items()}

_InterpretedBuffer_lookup = {
    stagg.stagg_generated.InterpretedBuffer.InterpretedBuffer.InterpretedInlineBuffer: (InterpretedInlineBuffer, stagg.stagg_generated.InterpretedInlineBuffer.InterpretedInlineBuffer),
    stagg.stagg_generated.InterpretedBuffer.InterpretedBuffer.InterpretedInlineInt64Buffer: (InterpretedInlineInt64Buffer, stagg.stagg_generated.InterpretedInlineInt64Buffer.InterpretedInlineInt64Buffer),
    stagg.stagg_generated.InterpretedBuffer.InterpretedBuffer.InterpretedInlineFloat64Buffer: (InterpretedInlineFloat64Buffer, stagg.stagg_generated.InterpretedInlineFloat64Buffer.InterpretedInlineFloat64Buffer),
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
