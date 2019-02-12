# Stagg

Stagg is a histogramming library that does not fill histograms and does not plot them. Its role is behind the scenes, to provide better communication between histogramming libraries.

Specifically, it is a structured representation of histograms and **st**atistical **agg**regates in general. It has all of the "bells and whistles" often associated with plain histograms, such as number of entries, unbinned mean and standard deviation, bin errors, associated fit functions, profile plots, and even simple ntuples (needed for unbinned fits or machine learning applications). [ROOT](https://root.cern.ch/root/htmldoc/guides/users-guide/Histograms.html) has all of these features; [Numpy](https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram.html) has none of them.

The purpose of Stagg is to be an intermediate when converting ROOT histograms into Numpy, or vice-versa, or both of these into [Boost.Histogram](https://hdembinski.github.io/histogram/doc/html/), [Physt](https://physt.readthedocs.io/en/latest/index.html), [Pandas](https://pandas.pydata.org), etc. Without an intermediate representation, converting between _N_ libraries (to get the advantages of all) would equire _N(N  ‒ 1)/2_ conversion routines; with an intermediate representation, we only need _N_, and the mapping can be made explicit in terms of a common language.

Furthermore, Stagg is a [Flatbuffers](http://google.github.io/flatbuffers/) schema, so it can be deciphered in [many languages](https://google.github.io/flatbuffers/flatbuffers_support.html), with [lazy, random-access](https://github.com/mzaks/FlatBuffersSwift/wiki/FlatBuffers-Explained), and uses a [small amount of memory](http://google.github.io/flatbuffers/md__benchmarks.html). A collection of histograms, functions, and ntuples can be shared among processes as shared memory, used in remote procedure calls, processed incrementally in a memory-mapped file, or saved in files with future-proof [schema evolution](https://google.github.io/flatbuffers/md__schemas.html).

# Installation

Not on PyPI yet. For now, you have to clone this GitHub repository.

# Documentation

Suite of examples as a Jupyter notebook:

   * [in GitHub](https://github.com/diana-hep/stagg/blob/master/binder/examples.ipynb)
   * [on Binder](HERE)

Full specification:

   * [in GitHub](https://github.com/diana-hep/stagg/blob/master/binder/examples.ipynb)
