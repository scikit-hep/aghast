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
import histos.histos_generated.Content
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
import histos.histos_generated.Systematic
import histos.histos_generated.TicTacToeOverflowBinning
import histos.histos_generated.VariableBinning
import histos.histos_generated.Variation
import histos.histos_generated.Weights

import histos.checktype

def checkedparameter(check):
    @property
    def prop(self):
        private = "_" + check.param
        if not hasattr(self, private):
            setattr(self, private, getattr(self._flatbuffers, check.param.capitalize())())
        return getattr(self, private)

    @prop.setter
    def prop(self, value):
        private = "_" + check.param
        setattr(self, private, check(value))

    return prop

class Histos(object):
    def _valid(self):
        pass

    @property
    def isvalid(self):
        try:
            self._valid()
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

    def __init__(self, data, language=unspecified):
        self.data = data
        self.language = language

    @property
    def data(self):
        return _get(self, "data", self._flatbuffers.Data)

    @data.setter
    def data(self, value):
        self._data = string("Metadata.data", required("Metadata.data", value))

    @property
    def language(self):
        return _get(self, "language", self._flatbuffers.Language)

    @language.setter
    def language(self, value):
        self._language = enum("Metadata.language", required("Metadata.language", value), [self.unspecified, self.json])

################################################# Decoration

class Decoration(Histos):
    unspecified = Enum("unspecified", histos.histos_generated.DecorationLanguage.DecorationLanguage.deco_unspecified)
    css = Enum("css", histos.histos_generated.DecorationLanguage.DecorationLanguage.deco_css)
    vega = Enum("vega", histos.histos_generated.DecorationLanguage.DecorationLanguage.deco_vega)
    root_json = Enum("root_json", histos.histos_generated.DecorationLanguage.DecorationLanguage.deco_root_json)

    def __init__(self, data, language=unspecified):
        self.data = data
        self.language = language

    @property
    def data(self):
        return _get(self, "data", self._flatbuffers.Data)

    @data.setter
    def data(self, value):
        self._data = string("Decoration.data", required("Decoration.data", value))

    @property
    def language(self):
        return _get(self, "language", self._flatbuffers.Language)

    @language.setter
    def language(self, value):
        self._language = enum("Decoration.language", required("Decoration.language", value), [self.unspecified, self.css, self.vega, self.root_json])

################################################# Object

class Object(Histos):
    def __init__(self):
        raise TypeError("{0} is an abstract base class; do not construct".format(type(self).__name__))

################################################# Parameter

class Parameter(Histos):
    def __init__(self, identifier, value):
        self.identifier = identifier
        self.value = value

    @property
    def identifier(self):
        return _get(self, "identifier", self._flatbuffers.Identifier)

    @identifier.setter
    def identifier(self, value):
        self._identifier = string("Parameter.identifier", required("Parameter.identifier", value))

    @property
    def value(self):
        return _get(self, "value", self._flatbuffers.Value)

    @value.setter
    def value(self, value):
        self._value = number("Parameter.value", required("Parameter.value", value))

################################################# Function

class Function(Histos):
    def __init__(self):
        raise TypeError("{0} is an abstract base class; do not construct".format(type(self).__name__))

################################################# ParameterizedFunction

class ParameterizedFunction(Function):
    def __init__(self, expression, parameters, contours=None):
        self.expression = expression

    @property
    def expression(self):
        return _get(self, "expression", self._flatbuffers.Expression)

    @expression.setter
    def expression(self, value):
        self._expression = string("ParameterizedFunction.expression", required("ParameterizedFunction.expression", value))

################################################# EvaluatedFunction

class EvaluatedFunction(Function):
    def __init__(self, x):
        self.x = x

    @property
    def x(self):
        return _get(self, "x", self._flatbuffers.X)

    @x.setter
    def x(self, value):
        self._x = string("EvaluatedFunction.x", value)

################################################# Buffer

class Buffer(Histos):
    def __init__(self):
        raise TypeError("{0} is an abstract base class; do not construct".format(type(self).__name__))

################################################# RawInlineBuffer

class RawInlineBuffer(Buffer):
    def __init__(self, x):
        self.x = x

    @property
    def x(self):
        return _get(self, "x", self._flatbuffers.X)

    @x.setter
    def x(self, value):
        self._x = string("RawInlineBuffer.x", value)

################################################# RawExternalBuffer

class RawExternalBuffer(Buffer):
    def __init__(self, x):
        self.x = x

    @property
    def x(self):
        return _get(self, "x", self._flatbuffers.X)

    @x.setter
    def x(self, value):
        self._x = string("RawExternalBuffer.x", value)

################################################# InlineBuffer

class InlineBuffer(RawInlineBuffer):
    def __init__(self, x):
        self.x = x

    @property
    def x(self):
        return _get(self, "x", self._flatbuffers.X)

    @x.setter
    def x(self, value):
        self._x = string("InlineBuffer.x", value)

################################################# ExternalBuffer

class ExternalBuffer(RawExternalBuffer):
    def __init__(self, x):
        self.x = x

    @property
    def x(self):
        return _get(self, "x", self._flatbuffers.X)

    @x.setter
    def x(self, value):
        self._x = string("ExternalBuffer.x", value)

################################################# Binning

class Binning(Histos):
    def __init__(self):
        raise TypeError("{0} is an abstract base class; do not construct".format(type(self).__name__))

################################################# FractionalBinning

class FractionalBinning(Binning):
    def __init__(self, x):
        self.x = x

    @property
    def x(self):
        return _get(self, "x", self._flatbuffers.X)

    @x.setter
    def x(self, value):
        self._x = string("FractionalBinning.x", value)

################################################# IntegerBinning

class IntegerBinning(Binning):
    def __init__(self, x):
        self.x = x

    @property
    def x(self):
        return _get(self, "x", self._flatbuffers.X)

    @x.setter
    def x(self, value):
        self._x = string("IntegerBinning.x", value)

################################################# RealInterval

class RealInterval(Histos):
    def __init__(self, x):
        self.x = x

    @property
    def x(self):
        return _get(self, "x", self._flatbuffers.X)

    @x.setter
    def x(self, value):
        self._x = string("RealInterval.x", value)

################################################# RealOverflow

class RealOverflow(Histos):
    def __init__(self, x):
        self.x = x

    @property
    def x(self):
        return _get(self, "x", self._flatbuffers.X)

    @x.setter
    def x(self, value):
        self._x = string("RealOverflow.x", value)

################################################# RegularBinning

class RegularBinning(Binning):
    def __init__(self, x):
        self.x = x

    @property
    def x(self):
        return _get(self, "x", self._flatbuffers.X)

    @x.setter
    def x(self, value):
        self._x = string("RegularBinning.x", value)

################################################# TicTacToeOverflowBinning

class TicTacToeOverflowBinning(Binning):
    def __init__(self, x):
        self.x = x

    @property
    def x(self):
        return _get(self, "x", self._flatbuffers.X)

    @x.setter
    def x(self, value):
        self._x = string("TicTacToeOverflowBinning.x", value)

################################################# HexagonalBinning

class HexagonalBinning(Binning):
    def __init__(self, x):
        self.x = x

    @property
    def x(self):
        return _get(self, "x", self._flatbuffers.X)

    @x.setter
    def x(self, value):
        self._x = string("HexagonalBinning.x", value)

################################################# VariableBinning

class VariableBinning(Binning):
    def __init__(self, x):
        self.x = x

    @property
    def x(self):
        return _get(self, "x", self._flatbuffers.X)

    @x.setter
    def x(self, value):
        self._x = string("VariableBinning.x", value)

################################################# CategoryBinning

class CategoryBinning(Binning):
    def __init__(self, x):
        self.x = x

    @property
    def x(self):
        return _get(self, "x", self._flatbuffers.X)

    @x.setter
    def x(self, value):
        self._x = string("CategoryBinning.x", value)

################################################# SparseRegularBinning

class SparseRegularBinning(Binning):
    def __init__(self, x):
        self.x = x

    @property
    def x(self):
        return _get(self, "x", self._flatbuffers.X)

    @x.setter
    def x(self, value):
        self._x = string("SparseRegularBinning.x", value)

################################################# Axis

class Axis(Histos):
    def __init__(self, x):
        self.x = x

    @property
    def x(self):
        return _get(self, "x", self._flatbuffers.X)

    @x.setter
    def x(self, value):
        self._x = string("Axis.x", value)

################################################# Counts

class Counts(Histos):
    def __init__(self):
        raise TypeError("{0} is an abstract base class; do not construct".format(type(self).__name__))

################################################# UnweightedCounts

class UnweightedCounts(Counts):
    def __init__(self, x):
        self.x = x

    @property
    def x(self):
        return _get(self, "x", self._flatbuffers.X)

    @x.setter
    def x(self, value):
        self._x = string("UnweightedCounts.x", value)

################################################# WeightedCounts

class WeightedCounts(Counts):
    def __init__(self, x):
        self.x = x

    @property
    def x(self):
        return _get(self, "x", self._flatbuffers.X)

    @x.setter
    def x(self, value):
        self._x = string("WeightedCounts.x", value)

################################################# Correlation

class Correlation(Histos):
    def __init__(self, x):
        self.x = x

    @property
    def x(self):
        return _get(self, "x", self._flatbuffers.X)

    @x.setter
    def x(self, value):
        self._x = string("Correlation.x", value)

################################################# Extreme

class Extreme(Histos):
    def __init__(self, x):
        self.x = x

    @property
    def x(self):
        return _get(self, "x", self._flatbuffers.X)

    @x.setter
    def x(self, value):
        self._x = string("Extreme.x", value)

################################################# Moment

class Moment(Histos):
    def __init__(self, x):
        self.x = x

    @property
    def x(self):
        return _get(self, "x", self._flatbuffers.X)

    @x.setter
    def x(self, value):
        self._x = string("Moment.x", value)

################################################# Quantile

class Quantile(Histos):
    def __init__(self, x):
        self.x = x

    @property
    def x(self):
        return _get(self, "x", self._flatbuffers.X)

    @x.setter
    def x(self, value):
        self._x = string("Quantile.x", value)

################################################# GenericErrors

class GenericErrors(Histos):
    def __init__(self, x):
        self.x = x

    @property
    def x(self):
        return _get(self, "x", self._flatbuffers.X)

    @x.setter
    def x(self, value):
        self._x = string("GenericErrors.x", value)

################################################# DistributionStats

class DistributionStats(Histos):
    def __init__(self, x):
        self.x = x

    @property
    def x(self):
        return _get(self, "x", self._flatbuffers.X)

    @x.setter
    def x(self, value):
        self._x = string("DistributionStats.x", value)

################################################# Distribution

class Distribution(Histos):
    def __init__(self, x):
        self.x = x

    @property
    def x(self):
        return _get(self, "x", self._flatbuffers.X)

    @x.setter
    def x(self, value):
        self._x = string("Distribution.x", value)

################################################# Profile

class Profile(Histos):
    def __init__(self, x):
        self.x = x

    @property
    def x(self):
        return _get(self, "x", self._flatbuffers.X)

    @x.setter
    def x(self, value):
        self._x = string("Profile.x", value)

################################################# Histogram

class Histogram(Histos):
    def __init__(self, x):
        self.x = x

    @property
    def x(self):
        return _get(self, "x", self._flatbuffers.X)

    @x.setter
    def x(self, value):
        self._x = string("Histogram.x", value)

################################################# Page

class Page(Histos):
    def __init__(self, x):
        self.x = x

    @property
    def x(self):
        return _get(self, "x", self._flatbuffers.X)

    @x.setter
    def x(self, value):
        self._x = string("Page.x", value)

################################################# ColumnChunk

class ColumnChunk(Histos):
    def __init__(self, x):
        self.x = x

    @property
    def x(self):
        return _get(self, "x", self._flatbuffers.X)

    @x.setter
    def x(self, value):
        self._x = string("ColumnChunk.x", value)

################################################# Chunk

class Chunk(Histos):
    def __init__(self, x):
        self.x = x

    @property
    def x(self):
        return _get(self, "x", self._flatbuffers.X)

    @x.setter
    def x(self, value):
        self._x = string("Chunk.x", value)

################################################# Column

class Column(Histos):
    def __init__(self, x):
        self.x = x

    @property
    def x(self):
        return _get(self, "x", self._flatbuffers.X)

    @x.setter
    def x(self, value):
        self._x = string("Column.x", value)

################################################# Ntuple

class Ntuple(Histos):
    def __init__(self, x):
        self.x = x

    @property
    def x(self):
        return _get(self, "x", self._flatbuffers.X)

    @x.setter
    def x(self, value):
        self._x = string("Ntuple.x", value)

################################################# Region

class Region(Histos):
    def __init__(self, x):
        self.x = x

    @property
    def x(self):
        return _get(self, "x", self._flatbuffers.X)

    @x.setter
    def x(self, value):
        self._x = string("Region.x", value)

################################################# BinnedRegion

class BinnedRegion(Histos):
    def __init__(self, x):
        self.x = x

    @property
    def x(self):
        return _get(self, "x", self._flatbuffers.X)

    @x.setter
    def x(self, value):
        self._x = string("BinnedRegion.x", value)

################################################# Assignment

class Assignment(Histos):
    def __init__(self, x):
        self.x = x

    @property
    def x(self):
        return _get(self, "x", self._flatbuffers.X)

    @x.setter
    def x(self, value):
        self._x = string("Assignment.x", value)

################################################# Systematic

class Systematic(Histos):
    def __init__(self, x):
        self.x = x

    @property
    def x(self):
        return _get(self, "x", self._flatbuffers.X)

    @x.setter
    def x(self, value):
        self._x = string("Systematic.x", value)

################################################# Variation

class Variation(Histos):
    def __init__(self, x):
        self.x = x

    @property
    def x(self):
        return _get(self, "x", self._flatbuffers.X)

    @x.setter
    def x(self, value):
        self._x = string("Variation.x", value)

################################################# Collection

class Collection(Histos):
    def tobuffer(self, internalize=False):
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

    identifier = checkedparameter(params["identifier"])
    title      = checkedparameter(params["title"])

    def __init__(self, identifier, title=""):
        self.identifier = identifier
        self.title = title

    def __repr__(self):
        return "<{0} {1} at 0x{2:012x}>".format(type(self).__name__, repr(self.identifier), id(self))

    def _toflatbuffers(self, builder, internalize, file):
        self._valid()

        identifier = builder.CreateString(self._identifier)
        if len(self._title) > 0:
            title = builder.CreateString(self._title)
        histos.histos_generated.Collection.CollectionStart(builder)
        histos.histos_generated.Collection.CollectionAddIdentifier(builder, identifier)
        if len(self._title) > 0:
            histos.histos_generated.Collection.CollectionAddTitle(builder, title)
        return histos.histos_generated.Collection.CollectionEnd(builder)
