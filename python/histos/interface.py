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

from histos.checktype import *

def _get(obj, field, getter):
    field = "_" + field
    if not hasattr(obj, field):
        setattr(obj, field, getter())
    return getattr(obj, field)

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
        self._data = string("Metadata.data", value)

    @property
    def language(self):
        return _get(self, "language", self._flatbuffers.Language)

    @language.setter
    def language(self, value):
        self._language = enum("Metadata.language", value, [self.unspecified, self.json])

################################################# Decoration

################################################# Object

################################################# Parameter

################################################# Function

################################################# ParameterizedFunction

################################################# EvaluatedFunction

################################################# Buffer

################################################# RawInlineBuffer

################################################# RawExternalBuffer

################################################# InlineBuffer

################################################# ExternalBuffer

################################################# Binning

################################################# FractionalBinning

################################################# IntegerBinning

################################################# RealInterval

################################################# RealOverflow

################################################# RegularBinning

################################################# TicTacToeOverflowBinning

################################################# HexagonalBinning

################################################# VariableBinning

################################################# CategoryBinning

################################################# SparseRegularBinning

################################################# Axis

################################################# Counts

################################################# UnweightedCounts

################################################# WeightedCounts

################################################# Correlation

################################################# Extreme

################################################# Moment

################################################# Quantile

################################################# GenericErrors

################################################# DistributionStats

################################################# Distribution

################################################# Profile

################################################# Histogram

################################################# Page

################################################# ColumnChunk

################################################# Chunk

################################################# Column

################################################# Ntuple

################################################# Region

################################################# BinnedRegion

################################################# Assignment

################################################# Systematic

################################################# Variation

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

    def __init__(self, identifier, title=""):
        self.identifier = identifier
        self.title = title
    
    @property
    def identifier(self):
        return _get(self, "identifier", self._flatbuffers.Identifier)

    @identifier.setter
    def identifier(self, value):
        self._identifier = string("Collection.identifier", value)

    @property
    def title(self):
        return _get(self, "title", self._flatbuffers.Title)

    @title.setter
    def title(self, value):
        self._title = string("Collection.title", value)

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
