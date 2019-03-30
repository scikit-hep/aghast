![](docs/source/logo-300px.png)

# aghast

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/diana-hep/aghast/binder-3?filepath=binder%2Fexamples.ipynb)

Aghast is a histogramming library that does not fill histograms and does not plot them. Its role is behind the scenes, to provide better communication between histogramming libraries.

Specifically, it is a structured representation of **ag**gregated, **h**istogram-like **st**atistics as sharable "ghasts." It has all of the "bells and whistles" often associated with plain histograms, such as number of entries, unbinned mean and standard deviation, bin errors, associated fit functions, profile plots, and even simple ntuples (needed for unbinned fits or machine learning applications). [ROOT](https://root.cern.ch/root/htmldoc/guides/users-guide/Histograms.html) has all of these features; [Numpy](https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram.html) has none of them.

The purpose of aghast is to be an intermediate when converting ROOT histograms into Numpy, or vice-versa, or both of these into [Boost.Histogram](https://github.com/boostorg/histogram), [Physt](https://physt.readthedocs.io/en/latest/index.html), [Pandas](https://pandas.pydata.org), etc. Without an intermediate representation, converting between _N_ libraries (to get the advantages of all) would equire _N(N  ‒ 1)/2_ conversion routines; with an intermediate representation, we only need _N_, and the mapping of feature to feature can be made explicit in terms of a common language.

Furthermore, aghast is a [Flatbuffers](http://google.github.io/flatbuffers/) schema, so it can be deciphered in [many languages](https://google.github.io/flatbuffers/flatbuffers_support.html), with [lazy, random-access](https://github.com/mzaks/FlatBuffersSwift/wiki/FlatBuffers-Explained), and uses a [small amount of memory](http://google.github.io/flatbuffers/md__benchmarks.html). A collection of histograms, functions, and ntuples can be shared among processes as shared memory, used in remote procedure calls, processed incrementally in a memory-mapped file, or saved in files with future-proof [schema evolution](https://google.github.io/flatbuffers/md__schemas.html).

## Installation from packages

Not on PyPI or conda yet. Refer to manual installation.

## Manual installation

After you git-clone this GitHub repository and ensure that `numpy` is installed, somehow:

```bash
pip install "flatbuffers>=1.8.0"  # for the flatbuffers Python runtime with Numpy interface
cd python                         # only implementation so far is in Python
python setup.py install           # if you want to use it outside of this directory
```

Now you should be able to `import aghast` or `from aghast import *` in Python.

If you need to change `flatbuffers/aghast.fbs`, you'll need to additionally:

   1. Get `flatc` to generate Python sources from `flatbuffers/aghast.fbs`. I use `conda install -c conda-forge flatbuffers`. (The `flatc` executable is _not_ included in the pip `flatbuffers` package, and the Python runtime is _not_ included in the conda `flatbuffers` package. They're disjoint.)
   2. In the `python` directory, run `./generate_flatbuffers.py` (which calls `flatc` and does some post-processing).

Every time you change `flatbuffers/aghast.fbs`, re-run `./generate_flatbuffers.py`.

## Documentation

Suite of examples as a Jupyter notebook:

   * [in GitHub](binder/examples.ipynb)
   * [on Binder](https://mybinder.org/v2/gh/diana-hep/aghast/binder-3?filepath=binder%2Fexamples.ipynb)

Full specification:

   * [Introduction](specification.adoc#introduction)
   * [Data types](specification.adoc#data-types)
      * [Collection](specification.adoc#collection)
      * [Histogram](specification.adoc#histogram)
      * [Axis](specification.adoc#axis)
      * [IntegerBinning](specification.adoc#integerbinning)
      * [RegularBinning](specification.adoc#regularbinning)
      * [RealInterval](specification.adoc#realinterval)
      * [RealOverflow](specification.adoc#realoverflow)
      * [HexagonalBinning](specification.adoc#hexagonalbinning)
      * [EdgesBinning](specification.adoc#edgesbinning)
      * [IrregularBinning](specification.adoc#irregularbinning)
      * [CategoryBinning](specification.adoc#categorybinning)
      * [SparseRegularBinning](specification.adoc#sparseregularbinning)
      * [FractionBinning](specification.adoc#fractionbinning)
      * [PredicateBinning](specification.adoc#predicatebinning)
      * [VariationBinning](specification.adoc#variationbinning)
      * [Variation](specification.adoc#variation)
      * [Assignment](specification.adoc#assignment)
      * [UnweightedCounts](specification.adoc#unweightedcounts)
      * [WeightedCounts](specification.adoc#weightedcounts)
      * [InterpretedInlineBuffer](specification.adoc#interpretedinlinebuffer)
      * [InterpretedInlineInt64Buffer](specification.adoc#interpretedinlineint64buffer)
      * [InterpretedInlineFloat64Buffer](specification.adoc#interpretedinlinefloat64buffer)
      * [InterpretedExternalBuffer](specification.adoc#interpretedexternalbuffer)
      * [Profile](specification.adoc#profile)
      * [Statistics](specification.adoc#statistics)
      * [Moments](specification.adoc#moments)
      * [Quantiles](specification.adoc#quantiles)
      * [Modes](specification.adoc#modes)
      * [Extremes](specification.adoc#extremes)
      * [StatisticFilter](specification.adoc#statisticfilter)
      * [Covariance](specification.adoc#covariance)
      * [ParameterizedFunction](specification.adoc#parameterizedfunction)
      * [Parameter](specification.adoc#parameter)
      * [EvaluatedFunction](specification.adoc#evaluatedfunction)
      * [BinnedEvaluatedFunction](specification.adoc#binnedevaluatedfunction)
      * [Ntuple](specification.adoc#ntuple)
      * [Column](specification.adoc#column)
      * [NtupleInstance](specification.adoc#ntupleinstance)
      * [Chunk](specification.adoc#chunk)
      * [ColumnChunk](specification.adoc#columnchunk)
      * [Page](specification.adoc#page)
      * [RawInlineBuffer](specification.adoc#rawinlinebuffer)
      * [RawExternalBuffer](specification.adoc#rawexternalbuffer)
      * [Metadata](specification.adoc#metadata)
      * [Decoration](specification.adoc#decoration)
