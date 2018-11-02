#!/usr/bin/env python

import struct

import numpy
import flatbuffers

import histos.histos_generated.Assignment
import histos.histos_generated.Axis
import histos.histos_generated.BinnedRegion
import histos.histos_generated.Binning
import histos.histos_generated.Buffer
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
import histos.histos_generated.Endianness
import histos.histos_generated.EvaluatedFunction
import histos.histos_generated.ExternalBuffer
import histos.histos_generated.ExternalType
import histos.histos_generated.Extreme
import histos.histos_generated.Filter
import histos.histos_generated.FractionalErrorMethod
import histos.histos_generated.FractionBinning
import histos.histos_generated.FunctionData
import histos.histos_generated.Function
import histos.histos_generated.GenericErrors
import histos.histos_generated.HexagonalBinning
import histos.histos_generated.HexagonalCoordinates
import histos.histos_generated.Histogram
import histos.histos_generated.__init__
import histos.histos_generated.InlineBuffer
import histos.histos_generated.IntegerBinning
import histos.histos_generated.MetadataLanguage
import histos.histos_generated.Metadata
import histos.histos_generated.Moment
import histos.histos_generated.NonRealMapping
import histos.histos_generated.Ntuple
import histos.histos_generated.ObjectData
import histos.histos_generated.Object
import histos.histos_generated.Page
import histos.histos_generated.ParameterizedFunction
import histos.histos_generated.Parameter
import histos.histos_generated.Profile
import histos.histos_generated.Quantile
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
import histos.histos_generated.VariableBinning
import histos.histos_generated.Variation
import histos.histos_generated.WeightedCounts

import histos.checktype

def typedproperty(check):
    @property
    def prop(self):
        private = "_" + check.param
        if not hasattr(self, private):
            setattr(self, private, check.fromflatbuffer(getattr(self._flatbuffers, check.param.capitalize())()))
        return getattr(self, private)

    @prop.setter
    def prop(self, value):
        private = "_" + check.param
        setattr(self, private, check(value))

    return prop

class Histos(object):
    @property
    def isvalid(self):
        try:
            self._valid(None)
        except ValueError:
            return False
        else:
            return True

    def __repr__(self):
        return "<{0} at 0x{1:012x}>".format(type(self).__name__, id(self))

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

    params = {
        "data":     histos.checktype.CheckString("Metadata", "data", required=True),
        "language": histos.checktype.CheckEnum("Metadata", "language", required=True, choices=[unspecified, json]),
        }

    data     = typedproperty(params["data"])
    language = typedproperty(params["language"])

    def __init__(self, data, language=unspecified):
        self.data = data
        self.language = language

################################################# Decoration

class Decoration(Histos):
    unspecified = Enum("unspecified", histos.histos_generated.DecorationLanguage.DecorationLanguage.deco_unspecified)
    css = Enum("css", histos.histos_generated.DecorationLanguage.DecorationLanguage.deco_css)
    vega = Enum("vega", histos.histos_generated.DecorationLanguage.DecorationLanguage.deco_vega)
    root_json = Enum("root_json", histos.histos_generated.DecorationLanguage.DecorationLanguage.deco_root_json)

    params = {
        "data":     histos.checktype.CheckString("Metadata", "data", required=True),
        "language": histos.checktype.CheckEnum("Metadata", "language", required=True, choices=[unspecified, css, vega, root_json]),
        }

    data     = typedproperty(params["data"])
    language = typedproperty(params["language"])

    def __init__(self, data, language=unspecified):
        self.data = data
        self.language = language

################################################# Object

class Object(Histos):
    def __init__(self):
        raise TypeError("{0} is an abstract base class; do not construct".format(type(self).__name__))

################################################# Parameter

class Parameter(Histos):
    params = {
        "identifier": histos.checktype.Check("Parameter", "identifier", required=True),
        "value": histos.checktype.Check("Parameter", "value", required=True),
        }

    identifier = typedproperty(params["identifier"])
    value = typedproperty(params["value"])

    def __init__(self, identifier, value):
        self.identifier = identifier
        self.value = value

################################################# Function

class Function(Object):
    def __init__(self):
        raise TypeError("{0} is an abstract base class; do not construct".format(type(self).__name__))

################################################# ParameterizedFunction

class ParameterizedFunction(Function):
    params = {
        "expression": histos.checktype.Check("ParameterizedFunction", "expression", required=True),
        "parameters": histos.checktype.Check("ParameterizedFunction", "parameters", required=True),
        "contours": histos.checktype.Check("ParameterizedFunction", "contours", required=False),
        }

    expression = typedproperty(params["expression"])
    parameters = typedproperty(params["parameters"])
    contours = typedproperty(params["contours"])

    def __init__(self, identifier, expression, parameters, contours=None, title="", metadata=None, decoration=None):
        self.expression = expression

################################################# EvaluatedFunction

class EvaluatedFunction(Function):
    def __init__(self, identifier, values, derivatives=None, generic_errors=None, title="", metadata=None, decoration=None):
        self.x = x

################################################# Buffer

class Buffer(Histos):
    def __init__(self):
        raise TypeError("{0} is an abstract base class; do not construct".format(type(self).__name__))

################################################# RawInlineBuffer

class RawInlineBuffer(Buffer):
    def __init__(self, buffer, filters=None, postfilter_slice=None):
        self.x = x

################################################# RawExternalBuffer

class RawExternalBuffer(Buffer):
    memory   = Enum("memory", histos.histos_generated.ExternalType.ExternalType.external_memory)
    samefile = Enum("samefile", histos.histos_generated.ExternalType.ExternalType.external_samefile)
    file     = Enum("file", histos.histos_generated.ExternalType.ExternalType.external_file)
    url      = Enum("url", histos.histos_generated.ExternalType.ExternalType.external_url)

    def __init__(self, pointer, numbytes, external_type=memory, filters=None, postfilter_slice=None):
        self.x = x

################################################# BufferInterpretation

class BufferInterpretation(object):
    none    = Enum("none", histos.histos_generated.DType.DType.dtype_none)
    int8    = Enum("int8", histos.histos_generated.DType.DType.dtype_int8)
    uint8   = Enum("uint8", histos.histos_generated.DType.DType.dtype_uint8)
    int16   = Enum("int16", histos.histos_generated.DType.DType.dtype_int16)
    uint16  = Enum("uint16", histos.histos_generated.DType.DType.dtype_uint16)
    int32   = Enum("int32", histos.histos_generated.DType.DType.dtype_int32)
    uint32  = Enum("uint32", histos.histos_generated.DType.DType.dtype_uint32)
    int64   = Enum("int64", histos.histos_generated.DType.DType.dtype_int64)
    uint64  = Enum("uint64", histos.histos_generated.DType.DType.dtype_uint64)
    float32 = Enum("float32", histos.histos_generated.DType.DType.dtype_float32)
    float64 = Enum("float64", histos.histos_generated.DType.DType.dtype_float64)

    little = Enum("little", histos.histos_generated.Endianness.Endianness.little_endian)
    big    = Enum("big", histos.histos_generated.Endianness.Endianness.big_endian)

    c_order       = Enum("c_order", histos.histos_generated.DimensionOrder.DimensionOrder.c_order)
    fortran_order = Enum("fortran", histos.histos_generated.DimensionOrder.DimensionOrder.fortran_order)

################################################# InlineBuffer

class InlineBuffer(RawInlineBuffer, BufferInterpretation):
    def __init__(self, buffer, filters=None, postfilter_slice=None, dtype=BufferInterpretation.none, endianness=BufferInterpretation.little, dimension_order=BufferInterpretation.c_order):
        self.x = x

################################################# ExternalBuffer

class ExternalBuffer(RawExternalBuffer, BufferInterpretation):
    def __init__(self, pointer, numbytes, external_type=RawExternalBuffer.memory, filters=None, postfilter_slice=None, dtype=BufferInterpretation.none, endianness=BufferInterpretation.little, dimension_order=BufferInterpretation.c_order, location=""):
        self.x = x

################################################# Binning

class Binning(Histos):
    def __init__(self):
        raise TypeError("{0} is an abstract base class; do not construct".format(type(self).__name__))

################################################# FractionalBinning

class FractionalBinning(Binning):
    normal = Enum("normal", histos.histos_generated.FractionalErrorMethod.FractionalErrorMethod.frac_normal)
    clopper_pearson  = Enum("clopper_pearson", histos.histos_generated.FractionalErrorMethod.FractionalErrorMethod.frac_clopper_pearson)
    wilson           = Enum("wilson", histos.histos_generated.FractionalErrorMethod.FractionalErrorMethod.frac_wilson)
    agresti_coull    = Enum("agresti_coull", histos.histos_generated.FractionalErrorMethod.FractionalErrorMethod.frac_agresti_coull)
    feldman_cousins  = Enum("feldman_cousins", histos.histos_generated.FractionalErrorMethod.FractionalErrorMethod.frac_feldman_cousins)
    jeffrey          = Enum("jeffrey", histos.histos_generated.FractionalErrorMethod.FractionalErrorMethod.frac_jeffrey)
    bayesian_uniform = Enum("bayesian_uniform", histos.histos_generated.FractionalErrorMethod.FractionalErrorMethod.frac_bayesian_uniform)

    def __init__(self, error_method=normal):
        self.x = x

################################################# IntegerBinning

class IntegerBinning(Binning):
    def __init__(self, min, max, has_underflow=True, has_overflow=True):
        self.x = x

################################################# RealInterval

class RealInterval(Histos):
    def __init__(self, low, high, low_inclusive=True, high_inclusive=False):
        self.x = x

################################################# RealOverflow

class RealOverflow(Histos):
    missing      = Enum("missing", histos.histos_generated.NonRealMapping.NonRealMapping.missing)
    in_underflow = Enum("in_underflow", histos.histos_generated.NonRealMapping.NonRealMapping.in_underflow)
    in_overflow  = Enum("in_overflow", histos.histos_generated.NonRealMapping.NonRealMapping.in_overflow)
    in_nanflow   = Enum("in_nanflow", histos.histos_generated.NonRealMapping.NonRealMapping.in_nanflow)

    def __init__(self, has_underflow=True, has_overflow=True, has_nanflow=True, minf_mapping=in_underflow, pinf_mapping=in_overflow, nan_mapping=in_nanflow):
        self.x = x

################################################# RegularBinning

class RegularBinning(Binning):
    def __init__(self, num, interval, overflow=None, circular=False):
        self.x = x

################################################# TicTacToeOverflowBinning

class TicTacToeOverflowBinning(Binning):
    def __init__(self, numx, numy, x, y, overflow):
        self.x = x

################################################# HexagonalBinning

class HexagonalBinning(Binning):
    offset         = Enum("offset", histos.histos_generated.HexagonalCoordinates.HexagonalCoordinates.hex_offset)
    doubled_offset = Enum("doubled_offset", histos.histos_generated.HexagonalCoordinates.HexagonalCoordinates.hex_doubled_offset)
    cube_xy        = Enum("cube_xy", histos.histos_generated.HexagonalCoordinates.HexagonalCoordinates.hex_cube_xy)
    cube_yz        = Enum("cube_yz", histos.histos_generated.HexagonalCoordinates.HexagonalCoordinates.hex_cube_yz)
    cube_xz        = Enum("cube_xz", histos.histos_generated.HexagonalCoordinates.HexagonalCoordinates.hex_cube_xz)

    def __init__(self, q, r, coordinates=offset, originx=0.0, originy=0.0):
        self.x = x

################################################# VariableBinning

class VariableBinning(Binning):
    def __init__(self, intervals, overflow=None):
        self.x = x

################################################# CategoryBinning

class CategoryBinning(Binning):
    def __init__(self, categories):
        self.x = x

################################################# SparseRegularBinning

class SparseRegularBinning(Binning):
    def __init__(self, bin_width, origin=0.0, has_nanflow=True):
        self.x = x

################################################# Axis

class Axis(Histos):
    def __init__(self, binning=None, expression="", title="", metadata=None, decoration=None):
        self.x = x

################################################# Counts

class Counts(Histos):
    def __init__(self):
        raise TypeError("{0} is an abstract base class; do not construct".format(type(self).__name__))

################################################# UnweightedCounts

class UnweightedCounts(Counts):
    def __init__(self, counts):
        self.x = x

################################################# WeightedCounts

class WeightedCounts(Counts):
    def __init__(self, sumw, sumw2, counts=None):
        self.x = x

################################################# Correlation

class Correlation(Histos):
    def __init__(self, sumwx, sumwxy):
        self.x = x

################################################# Extreme

class Extreme(Histos):
    def __init__(self, min, max, excludes_minf=False, excludes_pinf=False, excludes_nan=True):
        self.x = x

################################################# Moment

class Moment(Histos):
    def __init__(self, n, buffer=None):
        self.x = x

################################################# Quantile

class Quantile(Histos):
    def __init__(self, p, value=None):
        self.x = x

################################################# GenericErrors

class GenericErrors(Histos):
    def __init__(self, error=None, p=0.6826894921370859):
        self.x = x

################################################# DistributionStats

class DistributionStats(Histos):
    def __init__(self, correlation=None, extremes=None, moments=None, quantiles=None, generic_errors=None):
        self.x = x

################################################# Distribution

class Distribution(Histos):
    def __init__(self, counts, stats=None):
        self.x = x

################################################# Profile

class Profile(Histos):
    def __init__(self, expression, title="", metadata=None, decoration=None):
        self.x = x

################################################# Histogram

class Histogram(Object):
    def __init__(self, identifier, axis, distribution, profiles=None, unbinned_stats=None, profile_stats=None, functions=None, title="", metadata=None, decoration=None):
        self.x = x

################################################# Page

class Page(Histos):
    def __init__(self, buffer):
        self.x = x

################################################# ColumnChunk

class ColumnChunk(Histos):
    def __init__(self, pages, page_offsets, page_extremes=None):
        self.x = x

################################################# Chunk

class Chunk(Histos):
    def __init__(self, columns, metadata=None):
        self.x = x

################################################# Column

class Column(Histos, BufferInterpretation):
    def __init__(self, identifier, dtype=BufferInterpretation.none, endianness=BufferInterpretation.little, dimension_order=BufferInterpretation.c_order, filters=None, title="", metadata=None, decoration=None):
        self.x = x

################################################# Ntuple

class Ntuple(Object):
    def __init__(self, identifier, columns, chunks, chunk_offsets, unbinned_stats=None, functions=None, title="", metadata=None, decoration=None):
        self.x = x

################################################# Region

class Region(Histos):
    def __init__(self, expressions):
        self.x = x

################################################# BinnedRegion

class BinnedRegion(Histos):
    def __init__(self, expression, binning):
        self.x = x

################################################# Assignment

class Assignment(Histos):
    def __init__(self, identifier, expression):
        self.x = x

################################################# Variation

class Variation(Histos):
    def __init__(self, assignments, systematic=None, category_systematic=None):
        self.x = x

################################################# Collection

class Collection(Histos):
    def tobuffer(self, internalize=False):
        self._valid(1)
        builder = flatbuffers.Builder(1024)
        builder.Finish(self._toflatbuffers(builder, internalize, None))
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

    def tofile(self, file, internalize=False):
        self._valid(1)

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

    params = {
        "identifier": histos.checktype.CheckString("Collection", "identifier", required=True),
        "title":      histos.checktype.CheckString("Collection", "title", required=False),
        }

    identifier = typedproperty(params["identifier"])
    title      = typedproperty(params["title"])

    def __init__(self, identifier, title=""):
        self.identifier = identifier
        self.title = title

    def _valid(self, multiplicity):
        pass

    def __repr__(self):
        return "<{0} {1} at 0x{2:012x}>".format(type(self).__name__, repr(self.identifier), id(self))

    def _toflatbuffers(self, builder, internalize, file):
        identifier = builder.CreateString(self._identifier)
        if len(self._title) > 0:
            title = builder.CreateString(self._title)
        histos.histos_generated.Collection.CollectionStart(builder)
        histos.histos_generated.Collection.CollectionAddIdentifier(builder, identifier)
        if len(self._title) > 0:
            histos.histos_generated.Collection.CollectionAddTitle(builder, title)
        return histos.histos_generated.Collection.CollectionEnd(builder)
