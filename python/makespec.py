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

import math
import numbers
try:
    from inspect import signature
except ImportError:
    try:
        from funcsigs import signature
    except ImportError:
        raise ImportError("Install funcsigs package with:\n    pip install funcsigs\nor\n    conda install funcsigs\n(or just use Python >= 3.3).")

import numpy

import stagg
import stagg.checktype
import stagg.interface

prologue = u"""= Stagg Specification

== Introduction

== Basic Types

Stagg is encoded in a Flatbuffers specification (see link:flatbuffers/stagg.fbs[]). Flatbuffers provides a standard suite of types that can be translated into https://google.github.io/flatbuffers/flatbuffers_support.html[many languages]. However, the code the Flatbuffers code generator produces is too low-level even for applications to use as a backend, so we describe the interfaces and types of wrapper classes here.

These class descriptions are sufficiently constrained to fit into any static type system, and while they include heterogeneous lists (lists of an enumerated union type), they can be decomposed into homogeneous lists with an additional level of nesting. In fact, the Flatbuffers specification doesn't allow heterogeneous lists, so it provides an example of this decomposition. (For example, the heterogeneous *objects* list in a <<Collection>>, which can contain <<Histogram>>, <<ParameterizedFunction>>, <<BinnedEvaluatedFunction>>, and <<Ntuple>>, is encoded in Flatbuffers as a homogeneous list of `Object`, which contains a _single_ union of `ObjectData`.)

Basic types, like booleans, integers, floating point numbers, and strings, are passed through without modification (though strings are explicitly encoded in "utf-8"; Flatbuffers strings are not encoding-aware). Integers and floating point numbers may have a constrained range, such as [0, \u221e) for non-negative or (0, \u221e] for positive, including \u221e. (A square bracket includes the endpoint; a round bracket excludes it.) Empty strings and missing strings (null) are distinct.

Lists may contain basic types or class instances, and there is no distinction between empty lists and missing lists (an artifact of Flatbuffers).

In some cases, we want a mapping type, such as str \u2192 X, so that objects are retrievable by name, rather than index. Flatbuffers does not have such a type, so we build it by decomposing the high-level mapping into a low-level pair of lists with equal length. (For example, the *objects* mapping in a <<Collection>> is encoded in Flatbuffers as a list `objects` and a list `lookup`.)

Class objects may be missing (null) if they are not required. Required properties are a Flatbuffers feature: it doesn't generate code that would allow the serialized object to be missing. The class schemas can evolve to include more properties (with full forward and backward compatibility), but properties cannot be removed and required properties cannot become non-required.

Any properties that are not required have a default value (usually null).





"""

epilogue = u"""
"""

classes = [
    stagg.Collection,
    stagg.Histogram,
    stagg.Axis,
    stagg.IntegerBinning,
    stagg.RegularBinning,
    stagg.RealInterval,
    stagg.RealOverflow,
    stagg.HexagonalBinning,
    stagg.EdgesBinning,
    stagg.IrregularBinning,
    stagg.CategoryBinning,
    stagg.SparseRegularBinning,
    stagg.FractionBinning,
    stagg.PredicateBinning,
    stagg.VariationBinning,
    stagg.Variation,
    stagg.Assignment,
    stagg.UnweightedCounts,
    stagg.WeightedCounts,
    stagg.InterpretedInlineBuffer,
    stagg.InterpretedInlineInt64Buffer,
    stagg.InterpretedInlineFloat64Buffer,
    stagg.InterpretedExternalBuffer,
    stagg.Profile,
    stagg.Statistics,
    stagg.Moments,
    stagg.Quantiles,
    stagg.Modes,
    stagg.Extremes,
    stagg.StatisticFilter,
    stagg.Covariance,
    stagg.ParameterizedFunction,
    stagg.Parameter,
    stagg.EvaluatedFunction,
    stagg.BinnedEvaluatedFunction,
    stagg.Ntuple,
    stagg.Column,
    stagg.NtupleInstance,
    stagg.Chunk,
    stagg.ColumnChunk,
    stagg.Page,
    stagg.RawInlineBuffer,
    stagg.RawExternalBuffer,
    stagg.Metadata,
    stagg.Decoration,
    ]

unions = {
    stagg.interface.RawBuffer: [stagg.RawInlineBuffer, stagg.RawExternalBuffer],
    stagg.interface.InterpretedBuffer: [stagg.InterpretedInlineBuffer, stagg.InterpretedInlineInt64Buffer, stagg.InterpretedInlineFloat64Buffer, stagg.InterpretedExternalBuffer],
    stagg.interface.Binning: [stagg.IntegerBinning, stagg.RegularBinning, stagg.HexagonalBinning, stagg.EdgesBinning, stagg.IrregularBinning, stagg.CategoryBinning, stagg.SparseRegularBinning, stagg.FractionBinning, stagg.PredicateBinning, stagg.VariationBinning],
    stagg.interface.Counts: [stagg.UnweightedCounts, stagg.WeightedCounts],
    stagg.interface.Function: [stagg.ParameterizedFunction, stagg.EvaluatedFunction],
    stagg.interface.FunctionObject: [stagg.ParameterizedFunction, stagg.BinnedEvaluatedFunction],
    stagg.interface.Object: [stagg.Histogram, stagg.Ntuple, stagg.ParameterizedFunction, stagg.BinnedEvaluatedFunction, stagg.Collection],
    }

def num(x):
    if not isinstance(x, (bool, numpy.bool, numpy.bool)) and isinstance(x, (numbers.Real, numpy.integer, numpy.floating)):
        if x == float("-inf"):
            return u"\u2012\u221e"
        elif x == float("inf"):
            return u"\u221e"
        elif x == -0.5*math.pi:
            return u"\u2012\u03c0/2"
        elif x == 0.5*math.pi:
            return u"\u03c0/2"
        elif x == stagg.interface.MININT64:
            return u"\u20122\u2076\u00b3"  # -2**63
        elif x == stagg.interface.MAXINT64:
            return u"2\u2076\u00b3 \u2012 1"  # 2**63 - 1
        elif x == 0.5:
            return "1/2"
        elif x < 0:
            return u"\u2012" + repr(abs(x))
        else:
            return repr(abs(x))
    elif x is None:
        return "null"
    elif x is True:
        return "true"
    elif x is False:
        return "false"
    elif x == []:
        return "null/empty"
    else:
        return "`+{0}+`".format(repr(x))

def formatted(cls, end="\n"):
    out = [end + end + "== {0}{1}".format(cls.__name__, end), "*" + cls.description.strip() + "*"]

    out.append(end + "[%hardbreaks]")
    for name, param in signature(cls.__init__).parameters.items():
        if name != "self":
            check = cls._params[name]
            hasdefault = param.default is not param.empty

            islist = False
            if isinstance(check, stagg.checktype.CheckBool):
                typestring = "bool"
            elif isinstance(check, stagg.checktype.CheckString):
                typestring = "str"
            elif isinstance(check, stagg.checktype.CheckNumber):
                typestring = "float in {0}{1}, {2}{3}".format(
                    "[" if check.min_inclusive else "(",
                    num(check.min),
                    num(check.max),
                    "]" if check.max_inclusive else ")")
            elif isinstance(check, stagg.checktype.CheckInteger):
                typestring = "int in {0}{1}, {2}{3}".format(
                    "(" if check.min == float("-inf") else "[",
                    num(check.min),
                    num(check.max),
                    ")" if check.max == float("inf") else "]")
            elif isinstance(check, stagg.checktype.CheckEnum):
                typestring = "one of {" + ", ".join("`+" + str(x) + "+`" for x in check.choices) + "}"
            elif isinstance(check, stagg.checktype.CheckClass):
                if check.type in unions:
                    typestring = " or ".join("<<{0}>>".format(x.__name__) for x in unions[check.type])
                else:
                    typestring = "<<{0}>>".format(check.type.__name__)
            elif isinstance(check, stagg.checktype.CheckKey) and check.type is str:
                typestring = "unique str"
            elif isinstance(check, (stagg.checktype.CheckVector, stagg.checktype.CheckLookup)):
                islist = True
                if check.type is str:
                    subtype = "str"
                elif check.type is int:
                    subtype = "int"
                elif check.type is float:
                    subtype = "float"
                elif isinstance(check.type, list):
                    subtype = "{" + ", ".join("`+" + str(x) + "+`" for x in check.type) + "}"
                else:
                    if check.type in unions:
                        subtype = " or ".join("<<{0}>>".format(x.__name__) for x in unions[check.type])
                    else:
                        subtype = "<<{0}>>".format(check.type.__name__)
                if check.minlen != 0 or check.maxlen != float("inf"):
                    withlength = " with length in [{0}, {1}{2}".format(
                        num(check.minlen),
                        num(check.maxlen),
                        ")" if check.maxlen == float("inf") else "]")
                else:
                    withlength = ""
                if isinstance(check, stagg.checktype.CheckVector):
                    typestring = "list of {0}{1}".format(subtype, withlength)
                else:
                    typestring = u"str \u2192 {0}{1}".format(subtype, withlength)
            elif isinstance(check, stagg.checktype.CheckBuffer):
                typestring = "buffer"
            elif isinstance(check, stagg.checktype.CheckSlice):
                typestring = "slice (start:stop:step)"
            else:
                raise AssertionError(type(check))

            linebreak = " +" + end if len(name) + len(typestring) > 75 else " "
            if check.required:
                defaultstring = "(required)"
            elif hasdefault:
                defaultstring = "(default: {0})".format(num([] if islist and param.default is None else param.default))
            else:
                raise AssertionError

            out.append(u"\u2022{nbsp}" + " *{0}*: {1}{2}{3}".format(name, typestring, linebreak, defaultstring))

    if len(cls.validity_rules) != 0:
        out.append((" +" + end).join(cls.validity_rules))

    if cls.long_description is not None:
        out.append(end + "*Details:*" + end)
        out.append(cls.long_description.strip())

    return end.join(out)

if __name__ == "__main__":
    with open("../specification.adoc", "w") as file:
        file.write(prologue)

        for cls in classes:
            file.write(formatted(cls))

        file.write(epilogue)
