#!/usr/bin/env python

import struct

import numpy
import flatbuffers

import histos.histos_generated.Assignment
import histos.histos_generated.Axis
import histos.histos_generated.BinnedEvaluatedFunction
import histos.histos_generated.BinnedRegion
import histos.histos_generated.Binning
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
import histos.histos_generated.MetadataLanguage
import histos.histos_generated.Metadata
import histos.histos_generated.Moments
import histos.histos_generated.NonRealMapping
import histos.histos_generated.Ntuple
import histos.histos_generated.ObjectData
import histos.histos_generated.Object
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
    language = [unspecified, json]

    params = {
        "data":     histos.checktype.CheckString("Metadata", "data", required=True),
        "language": histos.checktype.CheckEnum("Metadata", "language", required=True, choices=language),
        }

    data     = typedproperty(params["data"])
    language = typedproperty(params["language"])

    def __init__(self, data, language=unspecified):
        self.data = data
        self.language = language

################################################# Decoration

class Decoration(Histos):
    unspecified = Enum("unspecified", histos.histos_generated.DecorationLanguage.DecorationLanguage.deco_unspecified)
    css         = Enum("css", histos.histos_generated.DecorationLanguage.DecorationLanguage.deco_css)
    vega        = Enum("vega", histos.histos_generated.DecorationLanguage.DecorationLanguage.deco_vega)
    root_json   = Enum("root_json", histos.histos_generated.DecorationLanguage.DecorationLanguage.deco_root_json)
    language = [unspecified, css, vega, root_json]

    params = {
        "data":     histos.checktype.CheckString("Metadata", "data", required=True),
        "language": histos.checktype.CheckEnum("Metadata", "language", required=True, choices=language),
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
        "identifier": histos.checktype.CheckKey("Parameter", "identifier", required=True, type=str),
        "value":      histos.checktype.CheckNumber("Parameter", "value", required=True),
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
        "identifier":     histos.checktype.CheckKey("ParameterizedFunction", "identifier", required=True, type=str),
        "expression": histos.checktype.CheckString("ParameterizedFunction", "expression", required=True),
        "parameters": histos.checktype.CheckVector("ParameterizedFunction", "parameters", required=True, type=Parameter),
        "contours":   histos.checktype.CheckVector("ParameterizedFunction", "contours", required=False, type=float),
        "title":          histos.checktype.CheckString("ParameterizedFunction", "title", required=False),
        "metadata":       histos.checktype.CheckClass("ParameterizedFunction", "metadata", required=False, type=Metadata),
        "decoration":     histos.checktype.CheckClass("ParameterizedFunction", "decoration", required=False, type=Decoration),
        }

    identifier = typedproperty(params["identifier"])
    expression = typedproperty(params["expression"])
    parameters = typedproperty(params["parameters"])
    contours = typedproperty(params["contours"])
    title = typedproperty(params["title"])
    metadata = typedproperty(params["metadata"])
    decoration = typedproperty(params["decoration"])

    def __init__(self, identifier, expression, parameters, contours=None, title="", metadata=None, decoration=None):
        self.expression = expression

################################################# EvaluatedFunction

class EvaluatedFunction(Function):
    params = {
        "identifier":     histos.checktype.CheckKey("EvaluatedFunction", "identifier", required=True, type=str),
        "values":         histos.checktype.CheckVector("EvaluatedFunction", "values", required=True, type=float),
        "derivatives":    histos.checktype.CheckVector("EvaluatedFunction", "derivatives", required=False, type=float),
        "generic_errors": histos.checktype.CheckVector("EvaluatedFunction", "generic_errors", required=False, type=GenericErrors),
        "title":          histos.checktype.CheckString("EvaluatedFunction", "title", required=False),
        "metadata":       histos.checktype.CheckClass("EvaluatedFunction", "metadata", required=False, type=Metadata),
        "decoration":     histos.checktype.CheckClass("EvaluatedFunction", "decoration", required=False, type=Decoration),
        }

    def __init__(self, identifier, values, derivatives=None, generic_errors=None, title="", metadata=None, decoration=None):
        self.identifier = identifier
        self.values = values
        self.derivatives = derivatives
        self.generic_errors = generic_errors
        self.title = title
        self.metadata = metadata
        self.decoration = decoration

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

class InterpretedBuffer(object):
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
    dtypes = [none, int8, uint8, int16, uint16, int32, uint32, int64, uint64, float32, float64]

    little_endian = Enum("little_endian", histos.histos_generated.Endianness.Endianness.little_endian)
    big_endian    = Enum("big_endian", histos.histos_generated.Endianness.Endianness.big_endian)
    endiannesses = [little_endian, big_endian]

    c_order       = Enum("c_order", histos.histos_generated.DimensionOrder.DimensionOrder.c_order)
    fortran_order = Enum("fortran", histos.histos_generated.DimensionOrder.DimensionOrder.fortran_order)
    orders = [c_order, fortran_order]

    def __init__(self):
        raise TypeError("{0} is an abstract base class; do not construct".format(type(self).__name__))

################################################# RawInlineBuffer

class RawInlineBuffer(Buffer, RawBuffer, InlineBuffer):
    params = {
        "buffer":           histos.checktype.CheckBuffer("RawInlineBuffer", "buffer", required=True),
        "filters":          histos.checktype.CheckVector("RawInlineBuffer", "filters", required=False, type=Buffer.filters),
        "postfilter_slice": histos.checktype.CheckSlice("RawInlineBuffer", "postfilter_slice", required=False),
        }

    def __init__(self, buffer, filters=None, postfilter_slice=None):
        self.buffer = buffer
        self.filters = filters
        self.postfilter_slice = postfilter_slice

################################################# RawExternalBuffer

class RawExternalBuffer(Buffer, RawBuffer, ExternalBuffer):
    params = {
        "pointer":          histos.checktype.CheckInteger("RawExternalBuffer", "pointer", required=True, min=0),
        "numbytes":         histos.checktype.CheckInteger("RawExternalBuffer", "numbytes", required=True, min=0),
        "external_type":    histos.checktype.CheckEnum("RawExternalBuffer", "external_type", required=True, choices=ExternalBuffer.types),
        "filters":          histos.checktype.CheckVector("RawExternalBuffer", "filters", required=False, type=Buffer.filters),
        "postfilter_slice": histos.checktype.CheckSlice("RawExternalBuffer", "postfilter_slice", required=False),
        }

    def __init__(self, pointer, numbytes, external_type=memory, filters=None, postfilter_slice=None):
        self.pointer = pointer
        self.numbytes = numbytes
        self.external_type = external_type
        self.filters = filters
        self.postfilter_slice = postfilter_slice

################################################# InlineBuffer

class InterpretedInlineBuffer(Buffer, InterpretedBuffer, InternalBuffer):
    params = {
        "buffer":           histos.checktype.CheckBuffer("InterpretedInlineBuffer", "buffer", required=True),
        "filters":          histos.checktype.CheckVector("InterpretedInlineBuffer", "filters", required=False, type=Buffer.filters),
        "postfilter_slice": histos.checktype.CheckSlice("InterpretedInlineBuffer", "postfilter_slice", required=False),
        "dtype":            histos.checktype.CheckEnum("InterpretedInlineBuffer", "dtype", required=False, choices=InterpretedBuffer.dtypes),
        "endianness":       histos.checktype.CheckEnum("InterpretedInlineBuffer", "endianness", required=False, choices=InterpretedBuffer.endiannesses),
        "dimension_order":  histos.checktype.CheckEnum("InterpretedInlineBuffer", "dimension_order", required=False, choices=InterpretedBuffer.orders),
        }

    def __init__(self, buffer, filters=None, postfilter_slice=None, dtype=InterpretedBuffer.none, endianness=InterpretedBuffer.little_endian, dimension_order=InterpretedBuffer.c_order):
        self.buffer = buffer
        self.filters = filters
        self.postfilter_slice = postfilter_slice
        self.dtype = dtype
        self.endianness = endianness
        self.dimension_order = dimension_order

################################################# ExternalBuffer

class InterpretedExternalBuffer(Buffer, InterpretedBuffer, ExternalBuffer):
    params = {
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

    def __init__(self, pointer, numbytes, external_type=ExternalBuffer.memory, filters=None, postfilter_slice=None, dtype=InterpretedBuffer.none, endianness=InterpretedBuffer.little_endian, dimension_order=InterpretedBuffer.c_order, location=""):
        self.pointer = pointer
        self.numbytes = numbytes
        self.external_type = external_type
        self.filters = filters
        self.postfilter_slice = postfilter_slice
        self.dtype = dtype
        self.endianness = endianness
        self.dimension_order = dimension_order
        self.location = location

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
    error_methods = [normal, clopper_pearson, wilson, agresti_coull, feldman_cousins, jeffrey, bayesian_uniform]

    params = {
        "error_method": histos.checktype.CheckEnum("FractionalBinning", "error_method", required=False, choices=error_methods),
        }

    def __init__(self, error_method=normal):
        self.error_method = error_method

################################################# IntegerBinning

class IntegerBinning(Binning):
    params = {
        "min":           histos.checktype.CheckInteger("IntegerBinning", "min", required=True),
        "max":           histos.checktype.CheckInteger("IntegerBinning", "max", required=True),
        "has_underflow": histos.checktype.CheckBool("IntegerBinning", "has_underflow", required=False),
        "has_overflow":  histos.checktype.CheckBool("IntegerBinning", "has_overflow", required=False),
        }

    def __init__(self, min, max, has_underflow=True, has_overflow=True):
        self.min = min
        self.max = max
        self.has_underflow = has_underflow
        self.has_overflow = has_overflow

    def _valid(self, multiplicity):
        if min >= max:
            raise ValueError("min ({0}) must be strictly less than max ({1})".format(self.min, self.max))

################################################# RealInterval

class RealInterval(Histos):
    params = {
        "low":            histos.checktype.CheckNumber("RealInterval", "low", required=True),
        "high":           histos.checktype.CheckNumber("RealInterval", "high", required=True),
        "low_inclusive":  histos.checktype.CheckBool("RealInterval", "low_inclusive", required=False),
        "high_inclusive": histos.checktype.CheckBool("RealInterval", "high_inclusive", required=False),
        }

    def __init__(self, low, high, low_inclusive=True, high_inclusive=False):
        self.low = low
        self.high = high
        self.low_inclusive = low_inclusive
        self.high_inclusive = high_inclusive

    def _valid(self, multiplicity):
        if self.low >= self.high:
            raise ValueError("low ({0}) must be strictly less than high ({1})".format(self.low, self.high))

################################################# RealOverflow

class RealOverflow(Histos):
    missing      = Enum("missing", histos.histos_generated.NonRealMapping.NonRealMapping.missing)
    in_underflow = Enum("in_underflow", histos.histos_generated.NonRealMapping.NonRealMapping.in_underflow)
    in_overflow  = Enum("in_overflow", histos.histos_generated.NonRealMapping.NonRealMapping.in_overflow)
    in_nanflow   = Enum("in_nanflow", histos.histos_generated.NonRealMapping.NonRealMapping.in_nanflow)
    mappings = [missing, in_underflow, in_overflow, in_nanflow]

    params = {
        "has_underflow": histos.checktype.CheckBool("RealOverflow", "has_underflow", required=False),
        "has_overflow":  histos.checktype.CheckBool("RealOverflow", "has_overflow", required=False),
        "has_nanflow":   histos.checktype.CheckBool("RealOverflow", "has_nanflow", required=False),
        "minf_mapping":  histos.checktype.CheckEnum("RealOverflow", "minf_mapping", required=False, choices=mappings),
        "pinf_mapping":  histos.checktype.CheckEnum("RealOverflow", "pinf_mapping", required=False, choices=mappings),
        "nan_mapping":   histos.checktype.CheckEnum("RealOverflow", "nan_mapping", required=False, choices=mappings),
        }

    def __init__(self, has_underflow=True, has_overflow=True, has_nanflow=True, minf_mapping=in_underflow, pinf_mapping=in_overflow, nan_mapping=in_nanflow):
        self.has_underflow = has_underflow
        self.has_overflow = has_overflow
        self.has_nanflow = has_nanflow
        self.minf_mapping = minf_mapping
        self.pinf_mapping = pinf_mapping
        self.nan_mapping = nan_mapping

################################################# RegularBinning

class RegularBinning(Binning):
    params = {
        "num":      histos.checktype.CheckInteger("RegularBinning", "num", required=True, min=1),
        "interval": histos.checktype.CheckClass("RegularBinning", "interval", required=True, type=RealInterval),
        "overflow": histos.checktype.CheckClass("RegularBinning", "overflow", required=False, type=RealOverflow),
        "circular": histos.checktype.CheckBool("RegularBinning", "circular", required=False),
        }

    def __init__(self, num, interval, overflow=None, circular=False):
        self.num = num
        self.interval = interval
        self.overflow = overflow
        self.circular = circular

################################################# TicTacToeOverflowBinning

class TicTacToeOverflowBinning(Binning):
    params = {
        "xnum":      histos.checktype.CheckInteger("TicTacToeOverflowBinning", "xnum", required=True, min=1),
        "ynum":      histos.checktype.CheckInteger("TicTacToeOverflowBinning", "ynum", required=True, min=1),
        "x":         histos.checktype.CheckClass("TicTacToeOverflowBinning", "x", required=True, type=RealInterval),
        "y":         histos.checktype.CheckClass("TicTacToeOverflowBinning", "y", required=True, type=RealInterval),
        "xoverflow": histos.checktype.CheckClass("TicTacToeOverflowBinning", "xoverflow", required=False, type=RealOverflow),
        "yoverflow": histos.checktype.CheckClass("TicTacToeOverflowBinning", "yoverflow", required=False, type=RealOverflow),
        }

    def __init__(self, xnum, ynum, x, y, xoverflow=None, yoverflow=None):
        self.xnum = xnum
        self.ynum = ynum
        self.x = x
        self.y = y
        self.xoverflow = xoverflow
        self.yoverflow = yoverflow

################################################# HexagonalBinning

class HexagonalBinning(Binning):
    offset         = Enum("offset", histos.histos_generated.HexagonalCoordinates.HexagonalCoordinates.hex_offset)
    doubled_offset = Enum("doubled_offset", histos.histos_generated.HexagonalCoordinates.HexagonalCoordinates.hex_doubled_offset)
    cube_xy        = Enum("cube_xy", histos.histos_generated.HexagonalCoordinates.HexagonalCoordinates.hex_cube_xy)
    cube_yz        = Enum("cube_yz", histos.histos_generated.HexagonalCoordinates.HexagonalCoordinates.hex_cube_yz)
    cube_xz        = Enum("cube_xz", histos.histos_generated.HexagonalCoordinates.HexagonalCoordinates.hex_cube_xz)
    coordinates = [offset, doubled_offset, cube_xy, cube_yz, cube_xz]

    params = {
        "q":           histos.checktype.CheckClass("HexagonalBinning", "q", required=True, type=IntegerBinning),
        "r":           histos.checktype.CheckClass("HexagonalBinning", "r", required=True, type=IntegerBinning),
        "coordinates": histos.checktype.CheckEnum("HexagonalBinning", "coordinates", required=False, choices=coordinates),
        "xorigin":     histos.checktype.CheckNumber("HexagonalBinning", "xorigin", required=False),
        "yorigin":     histos.checktype.CheckNumber("HexagonalBinning", "yorigin", required=False),
        }

    def __init__(self, q, r, coordinates=offset, xorigin=0.0, yorigin=0.0):
        self.q = q
        self.r = r
        self.coordinates = coordinates
        self.originx = originx
        self.originy = originy

################################################# VariableBinning

class VariableBinning(Binning):
    params = {
        "intervals": histos.checktype.CheckVector("VariableBinning", "intervals", required=True, type=RealInterval),
        "overflow":  histos.checktype.CheckClass("VariableBinning", "overflow", required=False, type=RealOverflow),
        }

    def __init__(self, intervals, overflow=None):
        self.intervals = intervals
        self.overflow = overflow

################################################# CategoryBinning

class CategoryBinning(Binning):
    params = {
        "categories":  histos.checktype.CheckVector("CategoryBinning", "categories", required=True, type=str),
        }

    def __init__(self, categories):
        self.categories  = categories 

################################################# SparseRegularBinning

class SparseRegularBinning(Binning):
    params = {
        "bins":        histos.checktype.CheckVector("SparseRegularBinning", "bins", required=True, type=int),
        "bin_width":   histos.checktype.CheckNumber("SparseRegularBinning", "bin_width", required=True, min=0, min_inclusive=False),
        "origin":      histos.checktype.CheckNumber("SparseRegularBinning", "origin", required=False),
        "has_nanflow": histos.checktype.CheckBool("SparseRegularBinning", "has_nanflow", required=False),
        }

    def __init__(self, bins, bin_width, origin=0.0, has_nanflow=True):
        self.bins = bins
        self.bin_width = bin_width
        self.origin = origin
        self.has_nanflow = has_nanflow

################################################# Axis

class Axis(Histos):
    params = {
        "binning":    histos.checktype.CheckClass("Axis", "binning", required=False, type=Binning),
        "expression": histos.checktype.CheckString("Axis", "expression", required=False),
        "title":      histos.checktype.CheckString("Axis", "title", required=False),
        "metadata":   histos.checktype.CheckClass("Axis", "metadata", required=False, type=Metadata),
        "decoration": histos.checktype.CheckClass("Axis", "decoration", required=False, type=Decoration),
        }

    def __init__(self, binning=None, expression="", title="", metadata=None, decoration=None):
        self.binning = binning
        self.expression = expression
        self.title = title
        self.metadata = metadata
        self.decoration = decoration

################################################# Counts

class Counts(Histos):
    def __init__(self):
        raise TypeError("{0} is an abstract base class; do not construct".format(type(self).__name__))

################################################# UnweightedCounts

class UnweightedCounts(Counts):
    params = {
        "counts":  histos.checktype.CheckClass("UnweightedCounts", "counts", required=True, type=InterpretedBuffer),
        }

    def __init__(self, counts):
        self.counts = counts

################################################# WeightedCounts

class WeightedCounts(Counts):
    params = {
        "sumw":   histos.checktype.CheckClass("WeightedCounts", "sumw", required=True, type=InterpretedBuffer),
        "sumw2":  histos.checktype.CheckClass("WeightedCounts", "sumw2", required=True, type=InterpretedBuffer),
        "counts": histos.checktype.CheckClass("WeightedCounts", "counts", required=False, type=UnweightedCounts),
        }

    def __init__(self, sumw, sumw2, counts=None):
        self.sumw = sumw
        self.sumw2 = sumw2
        self.counts = counts

################################################# Correlation

class Correlation(Histos):
    params = {
        "sumwx":   histos.checktype.CheckClass("Correlation", "sumwx", required=True, type=InterpretedBuffer),
        "sumwxy":  histos.checktype.CheckClass("Correlation", "sumwxy", required=True, type=InterpretedBuffer),
        }

    def __init__(self, sumwx, sumwxy):
        self.sumwx = sumwx
        self.sumwxy  = sumwxy 

################################################# Extremes

class Extremes(Histos):
    params = {
        "min":           histos.checktype.CheckClass("Extremes", "min", required=True, type=InterpretedBuffer),
        "max":           histos.checktype.CheckClass("Extremes", "max", required=True, type=InterpretedBuffer),
        "excludes_minf": histos.checktype.CheckBool("Extremes", "excludes_minf", required=False),
        "excludes_pinf": histos.checktype.CheckBool("Extremes", "excludes_pinf", required=False),
        "excludes_nan":  histos.checktype.CheckBool("Extremes", "excludes_nan", required=False),
        }

    def __init__(self, min, max, excludes_minf=False, excludes_pinf=False, excludes_nan=True):
        self.min = min
        self.max = max
        self.excludes_minf = excludes_minf
        self.excludes_pinf = excludes_pinf
        self.excludes_nan = excludes_nan

################################################# Moments

class Moments(Histos):
    params = {
        "sumwn": histos.checktype.CheckClass("Moments", "sumwn", required=True, type=InterpretedBuffer),
        "n":     histos.checktype.CheckInteger("Moments", "n", required=True, min=1),
        }

    def __init__(self, sumwn, n):
        self.sumwn = sumwn
        self.n = n

################################################# Quantiles

class Quantiles(Histos):
    params = {
        "values": histos.checktype.CheckClass("Quantiles", "values", required=True, type=InterpretedBuffer),
        "p":      histos.checktype.CheckNumber("Quantiles", "p", required=True, min=0.0, max=1.0),
        }

    def __init__(self, values, p=0.5):
        self.values = values
        self.p = p

################################################# GenericErrors

class GenericErrors(Histos):
    params = {
        "errors": histos.checktype.CheckClass("GenericErrors", "errors", required=True, type=InterpretedBuffer),
        "p":      histos.checktype.CheckNumber("GenericErrors", "p", required=False, min=0.0, max=1.0),
        }

    def __init__(self, errors, p=0.6826894921370859):
        self.error = error
        self.p = p

################################################# DistributionStats

class DistributionStats(Histos):
    params = {
        "correlation":    histos.checktype.CheckClass("DistributionStats", "correlation", required=False, type=Correlation),
        "extremes":       histos.checktype.CheckClass("DistributionStats", "extremes", required=False, type=Extremes),
        "moments":        histos.checktype.CheckVector("DistributionStats", "moments", required=False, type=Moments),
        "quantiles":      histos.checktype.CheckVector("DistributionStats", "quantiles", required=False, type=Quantiles),
        "generic_errors": histos.checktype.CheckVector("DistributionStats", "generic_errors", required=False, type=GenericErrors),
        }

    def __init__(self, correlation=None, extremes=None, moments=None, quantiles=None, generic_errors=None):
        self.correlation = correlation
        self.extremes = extremes
        self.moments = moments
        self.quantiles = quantiles
        self.generic_errors = generic_errors

################################################# Distribution

class Distribution(Histos):
    params = {
        "counts": histos.checktype.CheckClass("Distribution", "counts", required=True, type=Counts),
        "stats":  histos.checktype.CheckClass("Distribution", "stats", required=False, type=DistributionStats),
        }

    def __init__(self, counts, stats=None):
        self.counts = counts
        self.stats = stats

################################################# Profile

class Profile(Histos):
    params = {
        "expression": histos.checktype.Check("Profile", "expression", required=None),
        "title":      histos.checktype.Check("Profile", "title", required=None),
        "metadata":   histos.checktype.Check("Profile", "metadata", required=None),
        "decoration": histos.checktype.Check("Profile", "decoration", required=None),
        }

    def __init__(self, expression, title="", metadata=None, decoration=None):
        self.expression = expression
        self.title = title
        self.metadata = metadata
        self.decoration = decoration

################################################# Histogram

class Histogram(Object):
    params = {
        "identifier":     histos.checktype.Check("Histogram", "identifier", required=None),
        "axis":           histos.checktype.Check("Histogram", "axis", required=None),
        "distribution":   histos.checktype.Check("Histogram", "distribution", required=None),
        "profiles":       histos.checktype.Check("Histogram", "profiles", required=None),
        "unbinned_stats": histos.checktype.Check("Histogram", "unbinned_stats", required=None),
        "profile_stats":  histos.checktype.Check("Histogram", "profile_stats", required=None),
        "functions":      histos.checktype.Check("Histogram", "functions", required=None),
        "title":          histos.checktype.Check("Histogram", "title", required=None),
        "metadata":       histos.checktype.Check("Histogram", "metadata", required=None),
        "decoration":     histos.checktype.Check("Histogram", "decoration", required=None),
        }

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

################################################# BinnedEvaluatedFunction

class BinnedEvaluatedFunction(Function):
    params = {
        "identifier":     histos.checktype.CheckKey("BinnedEvaluatedFunction", "identifier", required=True, type=str),
        "axis":           histos.checktype.CheckVector("BinnedEvaluatedFunction", "axis", required=True, type=Axis),
        "values":         histos.checktype.CheckVector("BinnedEvaluatedFunction", "values", required=True, type=float),
        "derivatives":    histos.checktype.CheckVector("BinnedEvaluatedFunction", "derivatives", required=False, type=float),
        "generic_errors": histos.checktype.CheckVector("BinnedEvaluatedFunction", "generic_errors", required=False, type=GenericErrors),
        "title":          histos.checktype.CheckString("BinnedEvaluatedFunction", "title", required=False),
        "metadata":       histos.checktype.CheckClass("BinnedEvaluatedFunction", "metadata", required=False, type=Metadata),
        "decoration":     histos.checktype.CheckClass("BinnedEvaluatedFunction", "decoration", required=False, type=Decoration),
        }

    def __init__(self, identifier, axis, values, derivatives=None, generic_errors=None, title="", metadata=None, decoration=None):
        self.identifier = identifier
        self.axis = axis
        self.values = values
        self.derivatives = derivatives
        self.generic_errors = generic_errors
        self.title = title
        self.metadata = metadata
        self.decoration = decoration

################################################# Page

class Page(Histos):
    params = {
        "buffer":  histos.checktype.Check("Page", "buffer", required=None),
        }

    def __init__(self, buffer):
        self.buffer  = buffer 

################################################# ColumnChunk

class ColumnChunk(Histos):
    params = {
        "pages":         histos.checktype.Check("ColumnChunk", "pages", required=None),
        "page_offsets":  histos.checktype.Check("ColumnChunk", "page_offsets", required=None),
        "page_extremes": histos.checktype.Check("ColumnChunk", "page_extremes", required=None),
        }

    def __init__(self, pages, page_offsets, page_extremes=None):
        self.pages = pages
        self.page_offsets = page_offsets
        self.page_extremes = page_extremes

################################################# Chunk

class Chunk(Histos):
    params = {
        "columns":  histos.checktype.Check("Chunk", "columns", required=None),
        "metadata": histos.checktype.Check("Chunk", "metadata", required=None),
        }

    def __init__(self, columns, metadata=None):
        self.columns = columns
        self.metadata = metadata

################################################# Column

class Column(Histos):
    params = {
        "identifier":      histos.checktype.Check("Column", "identifier", required=None),
        "dtype":           histos.checktype.Check("Column", "dtype", required=None),
        "endianness":      histos.checktype.Check("Column", "endianness", required=None),
        "dimension_order": histos.checktype.Check("Column", "dimension_order", required=None),
        "filters":         histos.checktype.Check("Column", "filters", required=None),
        "title":           histos.checktype.Check("Column", "title", required=None),
        "metadata":        histos.checktype.Check("Column", "metadata", required=None),
        "decoration":      histos.checktype.Check("Column", "decoration", required=None),
        }

    def __init__(self, identifier, dtype=InterpretedBuffer.none, endianness=InterpretedBuffer.little_endian, dimension_order=InterpretedBuffer.c_order, filters=None, title="", metadata=None, decoration=None):
        self.identifier = identifier
        self.dtype = dtype
        self.endianness = endianness
        self.dimension_order = dimension_order
        self.filters = filters
        self.title = title
        self.metadata = metadata
        self.decoration = decoration

################################################# Ntuple

class Ntuple(Object):
    params = {
        "identifier":     histos.checktype.Check("Ntuple", "identifier", required=None),
        "columns":        histos.checktype.Check("Ntuple", "columns", required=None),
        "chunks":         histos.checktype.Check("Ntuple", "chunks", required=None),
        "chunk_offsets":  histos.checktype.Check("Ntuple", "chunk_offsets", required=None),
        "unbinned_stats": histos.checktype.Check("Ntuple", "unbinned_stats", required=None),
        "functions":      histos.checktype.Check("Ntuple", "functions", required=None),
        "title":          histos.checktype.Check("Ntuple", "title", required=None),
        "metadata":       histos.checktype.Check("Ntuple", "metadata", required=None),
        "decoration":     histos.checktype.Check("Ntuple", "decoration", required=None),
        }

    def __init__(self, identifier, columns, chunks, chunk_offsets, unbinned_stats=None, functions=None, title="", metadata=None, decoration=None):
        self.identifier = identifier
        self.columns = columns
        self.chunks = chunks
        self.chunk_offsets = chunk_offsets
        self.unbinned_stats = unbinned_stats
        self.functions = functions
        self.title = title
        self.metadata = metadata
        self.decoration = decoration

################################################# Region

class Region(Histos):
    params = {
        "expressions":  histos.checktype.Check("Region", "expressions", required=None),
        }

    def __init__(self, expressions):
        self.expressions  = expressions 

################################################# BinnedRegion

class BinnedRegion(Histos):
    params = {
        "expression": histos.checktype.Check("BinnedRegion", "expression", required=None),
        "binning":    histos.checktype.Check("BinnedRegion", "binning", required=None),
        }

    def __init__(self, expression, binning):
        self.expression = expression
        self.binning  = binning 

################################################# Assignment

class Assignment(Histos):
    params = {
        "identifier": histos.checktype.Check("Assignment", "identifier", required=None),
        "expression": histos.checktype.Check("Assignment", "expression", required=None),
        }

    def __init__(self, identifier, expression):
        self.identifier = identifier
        self.expression  = expression 

################################################# Variation

class Variation(Histos):
    params = {
        "assignments":         histos.checktype.Check("Variation", "assignments", required=None),
        "systematic":          histos.checktype.Check("Variation", "systematic", required=None),
        "category_systematic": histos.checktype.Check("Variation", "category_systematic", required=None),
        }

    def __init__(self, assignments, systematic=None, category_systematic=None):
        self.assignments = assignments
        self.systematic = systematic
        self.category_systematic = category_systematic

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
