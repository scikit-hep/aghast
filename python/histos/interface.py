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
    css         = Enum("css", histos.histos_generated.DecorationLanguage.DecorationLanguage.deco_css)
    vega        = Enum("vega", histos.histos_generated.DecorationLanguage.DecorationLanguage.deco_vega)
    root_json   = Enum("root_json", histos.histos_generated.DecorationLanguage.DecorationLanguage.deco_root_json)

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
        "value":      histos.checktype.Check("Parameter", "value", required=True),
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
        "contours":   histos.checktype.Check("ParameterizedFunction", "contours", required=False),
        }

    expression = typedproperty(params["expression"])
    parameters = typedproperty(params["parameters"])
    contours = typedproperty(params["contours"])

    def __init__(self, identifier, expression, parameters, contours=None, title="", metadata=None, decoration=None):
        self.expression = expression

################################################# EvaluatedFunction

class EvaluatedFunction(Function):
    params = {
        "identifier":     histos.checktype.Check("EvaluatedFunction", "identifier", required=None),
        "values":         histos.checktype.Check("EvaluatedFunction", "values", required=None),
        "derivatives":    histos.checktype.Check("EvaluatedFunction", "derivatives", required=None),
        "generic_errors": histos.checktype.Check("EvaluatedFunction", "generic_errors", required=None),
        "title":          histos.checktype.Check("EvaluatedFunction", "title", required=None),
        "metadata":       histos.checktype.Check("EvaluatedFunction", "metadata", required=None),
        "decoration":     histos.checktype.Check("EvaluatedFunction", "decoration", required=None),
        }

    def __init__(self, identifier, values, derivatives=None, generic_errors=None, title="", metadata=None, decoration=None):
        self.identifier = identifier
        self.values = values
        self.derivatives = derivatives
        self.generic_errors = generic_errors
        self.title = title
        self.metadata = metadata
        self.decoration = decoration

################################################# Buffer

class Buffer(Histos):
    def __init__(self):
        raise TypeError("{0} is an abstract base class; do not construct".format(type(self).__name__))

################################################# RawInlineBuffer

class RawInlineBuffer(Buffer):
    params = {
        "buffer":           histos.checktype.Check("RawInlineBuffer", "buffer", required=None),
        "filters":          histos.checktype.Check("RawInlineBuffer", "filters", required=None),
        "postfilter_slice": histos.checktype.Check("RawInlineBuffer", "postfilter_slice", required=None),
        }

    def __init__(self, buffer, filters=None, postfilter_slice=None):
        self.buffer = buffer
        self.filters = filters
        self.postfilter_slice = postfilter_slice

################################################# RawExternalBuffer

class RawExternalBuffer(Buffer):
    memory   = Enum("memory", histos.histos_generated.ExternalType.ExternalType.external_memory)
    samefile = Enum("samefile", histos.histos_generated.ExternalType.ExternalType.external_samefile)
    file     = Enum("file", histos.histos_generated.ExternalType.ExternalType.external_file)
    url      = Enum("url", histos.histos_generated.ExternalType.ExternalType.external_url)

    params = {
        "pointer":          histos.checktype.Check("     = Enum", "pointer", required=None),
        "numbytes":         histos.checktype.Check("     = Enum", "numbytes", required=None),
        "external_type":    histos.checktype.Check("     = Enum", "external_type", required=None),
        "filters":          histos.checktype.Check("     = Enum", "filters", required=None),
        "postfilter_slice": histos.checktype.Check("     = Enum", "postfilter_slice", required=None),
        }

    def __init__(self, pointer, numbytes, external_type=memory, filters=None, postfilter_slice=None):
        self.pointer = pointer
        self.numbytes = numbytes
        self.external_type = external_type
        self.filters = filters
        self.postfilter_slice = postfilter_slice

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
    params = {
        "buffer":           histos.checktype.Check("InlineBuffer", "buffer", required=None),
        "filters":          histos.checktype.Check("InlineBuffer", "filters", required=None),
        "postfilter_slice": histos.checktype.Check("InlineBuffer", "postfilter_slice", required=None),
        "dtype":            histos.checktype.Check("InlineBuffer", "dtype", required=None),
        "endianness":       histos.checktype.Check("InlineBuffer", "endianness", required=None),
        "dimension_order":  histos.checktype.Check("InlineBuffer", "dimension_order", required=None),
        }

    def __init__(self, buffer, filters=None, postfilter_slice=None, dtype=BufferInterpretation.none, endianness=BufferInterpretation.little, dimension_order=BufferInterpretation.c_order):
        self.buffer = buffer
        self.filters = filters
        self.postfilter_slice = postfilter_slice
        self.dtype = dtype
        self.endianness = endianness
        self.dimension_order = dimension_order

################################################# ExternalBuffer

class ExternalBuffer(RawExternalBuffer, BufferInterpretation):
    params = {
        "pointer":          histos.checktype.Check("ExternalBuffer", "pointer", required=None),
        "numbytes":         histos.checktype.Check("ExternalBuffer", "numbytes", required=None),
        "external_type":    histos.checktype.Check("ExternalBuffer", "external_type", required=None),
        "filters":          histos.checktype.Check("ExternalBuffer", "filters", required=None),
        "postfilter_slice": histos.checktype.Check("ExternalBuffer", "postfilter_slice", required=None),
        "dtype":            histos.checktype.Check("ExternalBuffer", "dtype", required=None),
        "endianness":       histos.checktype.Check("ExternalBuffer", "endianness", required=None),
        "dimension_order":  histos.checktype.Check("ExternalBuffer", "dimension_order", required=None),
        "location":         histos.checktype.Check("ExternalBuffer", "location", required=None),
        }

    def __init__(self, pointer, numbytes, external_type=RawExternalBuffer.memory, filters=None, postfilter_slice=None, dtype=BufferInterpretation.none, endianness=BufferInterpretation.little, dimension_order=BufferInterpretation.c_order, location=""):
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

    params = {
        "error_method": histos.checktype.Check("uniform = Enum", "error_method", required=None),
        }

    def __init__(self, error_method=normal):
        self.error_method = error_method

################################################# IntegerBinning

class IntegerBinning(Binning):
    params = {
        "min":           histos.checktype.Check("IntegerBinning", "min", required=None),
        "max":           histos.checktype.Check("IntegerBinning", "max", required=None),
        "has_underflow": histos.checktype.Check("IntegerBinning", "has_underflow", required=None),
        "has_overflow":  histos.checktype.Check("IntegerBinning", "has_overflow", required=None),
        }

    def __init__(self, min, max, has_underflow=True, has_overflow=True):
        self.min = min
        self.max = max
        self.has_underflow = has_underflow
        self.has_overflow = has_overflow

################################################# RealInterval

class RealInterval(Histos):
    params = {
        "low":            histos.checktype.Check("RealInterval", "low", required=None),
        "high":           histos.checktype.Check("RealInterval", "high", required=None),
        "low_inclusive":  histos.checktype.Check("RealInterval", "low_inclusive", required=None),
        "high_inclusive": histos.checktype.Check("RealInterval", "high_inclusive", required=None),
        }

    def __init__(self, low, high, low_inclusive=True, high_inclusive=False):
        self.low = low
        self.high = high
        self.low_inclusive = low_inclusive
        self.high_inclusive = high_inclusive

################################################# RealOverflow

class RealOverflow(Histos):
    missing      = Enum("missing", histos.histos_generated.NonRealMapping.NonRealMapping.missing)
    in_underflow = Enum("in_underflow", histos.histos_generated.NonRealMapping.NonRealMapping.in_underflow)
    in_overflow  = Enum("in_overflow", histos.histos_generated.NonRealMapping.NonRealMapping.in_overflow)
    in_nanflow   = Enum("in_nanflow", histos.histos_generated.NonRealMapping.NonRealMapping.in_nanflow)

    params = {
        "has_underflow": histos.checktype.Check("nanflow   = Enum", "has_underflow", required=None),
        "has_overflow":  histos.checktype.Check("nanflow   = Enum", "has_overflow", required=None),
        "has_nanflow":   histos.checktype.Check("nanflow   = Enum", "has_nanflow", required=None),
        "minf_mapping":  histos.checktype.Check("nanflow   = Enum", "minf_mapping", required=None),
        "pinf_mapping":  histos.checktype.Check("nanflow   = Enum", "pinf_mapping", required=None),
        "nan_mapping":   histos.checktype.Check("nanflow   = Enum", "nan_mapping", required=None),
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
        "num":      histos.checktype.Check("RegularBinning", "num", required=None),
        "interval": histos.checktype.Check("RegularBinning", "interval", required=None),
        "overflow": histos.checktype.Check("RegularBinning", "overflow", required=None),
        "circular": histos.checktype.Check("RegularBinning", "circular", required=None),
        }

    def __init__(self, num, interval, overflow=None, circular=False):
        self.num = num
        self.interval = interval
        self.overflow = overflow
        self.circular = circular

################################################# TicTacToeOverflowBinning

class TicTacToeOverflowBinning(Binning):
    params = {
        "numx":      histos.checktype.Check("TicTacToeOverflowBinning", "numx", required=None),
        "numy":      histos.checktype.Check("TicTacToeOverflowBinning", "numy", required=None),
        "x":         histos.checktype.Check("TicTacToeOverflowBinning", "x", required=None),
        "y":         histos.checktype.Check("TicTacToeOverflowBinning", "y", required=None),
        "overflow":  histos.checktype.Check("TicTacToeOverflowBinning", "overflow", required=None),
        }

    def __init__(self, numx, numy, x, y, overflow):
        self.numx = numx
        self.numy = numy
        self.x = x
        self.y = y
        self.overflow  = overflow 

################################################# HexagonalBinning

class HexagonalBinning(Binning):
    offset         = Enum("offset", histos.histos_generated.HexagonalCoordinates.HexagonalCoordinates.hex_offset)
    doubled_offset = Enum("doubled_offset", histos.histos_generated.HexagonalCoordinates.HexagonalCoordinates.hex_doubled_offset)
    cube_xy        = Enum("cube_xy", histos.histos_generated.HexagonalCoordinates.HexagonalCoordinates.hex_cube_xy)
    cube_yz        = Enum("cube_yz", histos.histos_generated.HexagonalCoordinates.HexagonalCoordinates.hex_cube_yz)
    cube_xz        = Enum("cube_xz", histos.histos_generated.HexagonalCoordinates.HexagonalCoordinates.hex_cube_xz)

    params = {
        "q":           histos.checktype.Check("xz        = Enum", "q", required=None),
        "r":           histos.checktype.Check("xz        = Enum", "r", required=None),
        "coordinates": histos.checktype.Check("xz        = Enum", "coordinates", required=None),
        "originx":     histos.checktype.Check("xz        = Enum", "originx", required=None),
        "originy":     histos.checktype.Check("xz        = Enum", "originy", required=None),
        }

    def __init__(self, q, r, coordinates=offset, originx=0.0, originy=0.0):
        self.q = q
        self.r = r
        self.coordinates = coordinates
        self.originx = originx
        self.originy = originy

################################################# VariableBinning

class VariableBinning(Binning):
    params = {
        "intervals": histos.checktype.Check("VariableBinning", "intervals", required=None),
        "overflow":  histos.checktype.Check("VariableBinning", "overflow", required=None),
        }

    def __init__(self, intervals, overflow=None):
        self.intervals = intervals
        self.overflow = overflow

################################################# CategoryBinning

class CategoryBinning(Binning):
    params = {
        "categories":  histos.checktype.Check("CategoryBinning", "categories", required=None),
        }

    def __init__(self, categories):
        self.categories  = categories 

################################################# SparseRegularBinning

class SparseRegularBinning(Binning):
    params = {
        "bin_width":   histos.checktype.Check("SparseRegularBinning", "bin_width", required=None),
        "origin":      histos.checktype.Check("SparseRegularBinning", "origin", required=None),
        "has_nanflow": histos.checktype.Check("SparseRegularBinning", "has_nanflow", required=None),
        }

    def __init__(self, bin_width, origin=0.0, has_nanflow=True):
        self.bin_width = bin_width
        self.origin = origin
        self.has_nanflow = has_nanflow

################################################# Axis

class Axis(Histos):
    params = {
        "binning":    histos.checktype.Check("Axis", "binning", required=None),
        "expression": histos.checktype.Check("Axis", "expression", required=None),
        "title":      histos.checktype.Check("Axis", "title", required=None),
        "metadata":   histos.checktype.Check("Axis", "metadata", required=None),
        "decoration": histos.checktype.Check("Axis", "decoration", required=None),
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
        "counts":  histos.checktype.Check("UnweightedCounts", "counts", required=None),
        }

    def __init__(self, counts):
        self.counts  = counts 

################################################# WeightedCounts

class WeightedCounts(Counts):
    params = {
        "sumw":   histos.checktype.Check("WeightedCounts", "sumw", required=None),
        "sumw2":  histos.checktype.Check("WeightedCounts", "sumw2", required=None),
        "counts": histos.checktype.Check("WeightedCounts", "counts", required=None),
        }

    def __init__(self, sumw, sumw2, counts=None):
        self.sumw = sumw
        self.sumw2 = sumw2
        self.counts = counts

################################################# Correlation

class Correlation(Histos):
    params = {
        "sumwx":   histos.checktype.Check("Correlation", "sumwx", required=None),
        "sumwxy":  histos.checktype.Check("Correlation", "sumwxy", required=None),
        }

    def __init__(self, sumwx, sumwxy):
        self.sumwx = sumwx
        self.sumwxy  = sumwxy 

################################################# Extreme

class Extreme(Histos):
    params = {
        "min":           histos.checktype.Check("Extreme", "min", required=None),
        "max":           histos.checktype.Check("Extreme", "max", required=None),
        "excludes_minf": histos.checktype.Check("Extreme", "excludes_minf", required=None),
        "excludes_pinf": histos.checktype.Check("Extreme", "excludes_pinf", required=None),
        "excludes_nan":  histos.checktype.Check("Extreme", "excludes_nan", required=None),
        }

    def __init__(self, min, max, excludes_minf=False, excludes_pinf=False, excludes_nan=True):
        self.min = min
        self.max = max
        self.excludes_minf = excludes_minf
        self.excludes_pinf = excludes_pinf
        self.excludes_nan = excludes_nan

################################################# Moment

class Moment(Histos):
    params = {
        "n":      histos.checktype.Check("Moment", "n", required=None),
        "buffer": histos.checktype.Check("Moment", "buffer", required=None),
        }

    def __init__(self, n, buffer=None):
        self.n = n
        self.buffer = buffer

################################################# Quantile

class Quantile(Histos):
    params = {
        "p":     histos.checktype.Check("Quantile", "p", required=None),
        "value": histos.checktype.Check("Quantile", "value", required=None),
        }

    def __init__(self, p, value=None):
        self.p = p
        self.value = value

################################################# GenericErrors

class GenericErrors(Histos):
    params = {
        "error": histos.checktype.Check("GenericErrors", "error", required=None),
        "p":     histos.checktype.Check("GenericErrors", "p", required=None),
        }

    def __init__(self, error=None, p=0.6826894921370859):
        self.error = error
        self.p = p

################################################# DistributionStats

class DistributionStats(Histos):
    params = {
        "correlation":    histos.checktype.Check("DistributionStats", "correlation", required=None),
        "extremes":       histos.checktype.Check("DistributionStats", "extremes", required=None),
        "moments":        histos.checktype.Check("DistributionStats", "moments", required=None),
        "quantiles":      histos.checktype.Check("DistributionStats", "quantiles", required=None),
        "generic_errors": histos.checktype.Check("DistributionStats", "generic_errors", required=None),
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
        "counts": histos.checktype.Check("Distribution", "counts", required=None),
        "stats":  histos.checktype.Check("Distribution", "stats", required=None),
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

class Column(Histos, BufferInterpretation):
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

    def __init__(self, identifier, dtype=BufferInterpretation.none, endianness=BufferInterpretation.little, dimension_order=BufferInterpretation.c_order, filters=None, title="", metadata=None, decoration=None):
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
