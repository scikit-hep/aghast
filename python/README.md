![](https://github.com/scikit-hep/aghast/raw/master/docs/source/logo-300px.png)

# aghast

[![Build Status](https://travis-ci.org/scikit-hep/aghast.svg?branch=master)](https://travis-ci.org/scikit-hep/aghast)
[![PyPI version](https://badge.fury.io/py/aghast.svg)](https://badge.fury.io/py/aghast)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/aghast.svg)](https://pypi.org/project/aghast/)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/scikit-hep/aghast/master?urlpath=lab/tree/binder%2Ftutorial.ipynb)

Aghast is a histogramming library that does not fill histograms and does not plot them. Its role is behind the scenes, to provide better communication between histogramming libraries.

Specifically, it is a structured representation of **ag**gregated, **h**istogram-like **st**atistics as sharable "ghasts." It has all of the "bells and whistles" often associated with plain histograms, such as number of entries, unbinned mean and standard deviation, bin errors, associated fit functions, profile plots, and even simple ntuples (needed for unbinned fits or machine learning applications). [ROOT](https://root.cern.ch/root/htmldoc/guides/users-guide/Histograms.html) has all of these features; [Numpy](https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram.html) has none of them.

The purpose of aghast is to be an intermediate when converting ROOT histograms into Numpy, or vice-versa, or both of these into [Boost.Histogram](https://github.com/boostorg/histogram), [Physt](https://physt.readthedocs.io/en/latest/index.html), [Pandas](https://pandas.pydata.org), etc. Without an intermediate representation, converting between _N_ libraries (to get the advantages of all) would equire _N(N  ‒ 1)/2_ conversion routines; with an intermediate representation, we only need _N_, and the mapping of feature to feature can be made explicit in terms of a common language.

Furthermore, aghast is a [Flatbuffers](http://google.github.io/flatbuffers/) schema, so it can be deciphered in [many languages](https://google.github.io/flatbuffers/flatbuffers_support.html), with [lazy, random-access](https://github.com/mzaks/FlatBuffersSwift/wiki/FlatBuffers-Explained), and uses a [small amount of memory](http://google.github.io/flatbuffers/md__benchmarks.html). A collection of histograms, functions, and ntuples can be shared among processes as shared memory, used in remote procedure calls, processed incrementally in a memory-mapped file, or saved in files with future-proof [schema evolution](https://google.github.io/flatbuffers/md__schemas.html).

## Installation from packages

Install aghast like any other Python package:

```bash
pip install aghast                        # maybe with sudo or --user, or in virtualenv
```

<!-- or install with [conda](https://conda.io/en/latest/miniconda.html): -->

<!-- ```bash -->
<!-- conda config --add channels conda-forge   # if you haven't added conda-forge already -->
<!-- conda install uproot -->
<!-- ``` -->

_(Not on conda yet.)_

## Manual installation

After you git-clone this GitHub repository and ensure that `numpy` is installed, somehow:

```bash
pip install "flatbuffers>=1.8.0"          # for the flatbuffers runtime (with Numpy)
cd python                                 # only implementation so far is in Python
python setup.py install                   # to use it outside of this directory, Python2 is not supported right now
```

Now you should be able to `import aghast` or `from aghast import *` in Python.

If you need to change `flatbuffers/aghast.fbs`, you'll need to additionally:

   1. Get `flatc` to generate Python sources from `flatbuffers/aghast.fbs`. I use `conda install -c conda-forge flatbuffers`. (The `flatc` executable is _not_ included in the pip `flatbuffers` package, and the Python runtime is _not_ included in the conda `flatbuffers` package. They're disjoint.)
   2. In the `python` directory, run `./generate_flatbuffers.py` (which calls `flatc` and does some post-processing).

Every time you change `flatbuffers/aghast.fbs`, re-run `./generate_flatbuffers.py`.

If you want to use some specific packages on Anaconda channels, the recommended way is:

```bash
cd python                                           # only implementation so far is in Python
# add the packages you need to "environment-test.yml" or "requirements-test.txt"
conda env create -f environment-test.yml -n aghast  # create (or update) your aghast conda environment
conda activate aghast                               # activate your aghast environment
python setup.py install                             # to use it outside of this directory, Python2 is not supported right now
python -m ipykernel install --name aghast           # install your jupyter kernel "aghast"
```

Now you should be able to `import aghast` or `from aghast import *` in your notebooks with the kernel "aghast".

## Documentation

Full specification:

   * [Introduction](https://github.com/scikit-hep/aghast/blob/master/specification.adoc#introduction)
   * [Data types](https://github.com/scikit-hep/aghast/blob/master/specification.adoc#data-types)
      * [Collection](https://github.com/scikit-hep/aghast/blob/master/specification.adoc#collection)
      * [Histogram](https://github.com/scikit-hep/aghast/blob/master/specification.adoc#histogram)
      * [Axis](https://github.com/scikit-hep/aghast/blob/master/specification.adoc#axis)
      * [IntegerBinning](https://github.com/scikit-hep/aghast/blob/master/specification.adoc#integerbinning)
      * [RegularBinning](https://github.com/scikit-hep/aghast/blob/master/specification.adoc#regularbinning)
      * [RealInterval](https://github.com/scikit-hep/aghast/blob/master/specification.adoc#realinterval)
      * [RealOverflow](https://github.com/scikit-hep/aghast/blob/master/specification.adoc#realoverflow)
      * [HexagonalBinning](https://github.com/scikit-hep/aghast/blob/master/specification.adoc#hexagonalbinning)
      * [EdgesBinning](https://github.com/scikit-hep/aghast/blob/master/specification.adoc#edgesbinning)
      * [IrregularBinning](https://github.com/scikit-hep/aghast/blob/master/specification.adoc#irregularbinning)
      * [CategoryBinning](https://github.com/scikit-hep/aghast/blob/master/specification.adoc#categorybinning)
      * [SparseRegularBinning](https://github.com/scikit-hep/aghast/blob/master/specification.adoc#sparseregularbinning)
      * [FractionBinning](https://github.com/scikit-hep/aghast/blob/master/specification.adoc#fractionbinning)
      * [PredicateBinning](https://github.com/scikit-hep/aghast/blob/master/specification.adoc#predicatebinning)
      * [VariationBinning](https://github.com/scikit-hep/aghast/blob/master/specification.adoc#variationbinning)
      * [Variation](https://github.com/scikit-hep/aghast/blob/master/specification.adoc#variation)
      * [Assignment](https://github.com/scikit-hep/aghast/blob/master/specification.adoc#assignment)
      * [UnweightedCounts](https://github.com/scikit-hep/aghast/blob/master/specification.adoc#unweightedcounts)
      * [WeightedCounts](https://github.com/scikit-hep/aghast/blob/master/specification.adoc#weightedcounts)
      * [InterpretedInlineBuffer](https://github.com/scikit-hep/aghast/blob/master/specification.adoc#interpretedinlinebuffer)
      * [InterpretedInlineInt64Buffer](https://github.com/scikit-hep/aghast/blob/master/specification.adoc#interpretedinlineint64buffer)
      * [InterpretedInlineFloat64Buffer](https://github.com/scikit-hep/aghast/blob/master/specification.adoc#interpretedinlinefloat64buffer)
      * [InterpretedExternalBuffer](https://github.com/scikit-hep/aghast/blob/master/specification.adoc#interpretedexternalbuffer)
      * [Profile](https://github.com/scikit-hep/aghast/blob/master/specification.adoc#profile)
      * [Statistics](https://github.com/scikit-hep/aghast/blob/master/specification.adoc#statistics)
      * [Moments](https://github.com/scikit-hep/aghast/blob/master/specification.adoc#moments)
      * [Quantiles](https://github.com/scikit-hep/aghast/blob/master/specification.adoc#quantiles)
      * [Modes](https://github.com/scikit-hep/aghast/blob/master/specification.adoc#modes)
      * [Extremes](https://github.com/scikit-hep/aghast/blob/master/specification.adoc#extremes)
      * [StatisticFilter](https://github.com/scikit-hep/aghast/blob/master/specification.adoc#statisticfilter)
      * [Covariance](https://github.com/scikit-hep/aghast/blob/master/specification.adoc#covariance)
      * [ParameterizedFunction](https://github.com/scikit-hep/aghast/blob/master/specification.adoc#parameterizedfunction)
      * [Parameter](https://github.com/scikit-hep/aghast/blob/master/specification.adoc#parameter)
      * [EvaluatedFunction](https://github.com/scikit-hep/aghast/blob/master/specification.adoc#evaluatedfunction)
      * [BinnedEvaluatedFunction](https://github.com/scikit-hep/aghast/blob/master/specification.adoc#binnedevaluatedfunction)
      * [Ntuple](https://github.com/scikit-hep/aghast/blob/master/specification.adoc#ntuple)
      * [Column](https://github.com/scikit-hep/aghast/blob/master/specification.adoc#column)
      * [NtupleInstance](https://github.com/scikit-hep/aghast/blob/master/specification.adoc#ntupleinstance)
      * [Chunk](https://github.com/scikit-hep/aghast/blob/master/specification.adoc#chunk)
      * [ColumnChunk](https://github.com/scikit-hep/aghast/blob/master/specification.adoc#columnchunk)
      * [Page](https://github.com/scikit-hep/aghast/blob/master/specification.adoc#page)
      * [RawInlineBuffer](https://github.com/scikit-hep/aghast/blob/master/specification.adoc#rawinlinebuffer)
      * [RawExternalBuffer](https://github.com/scikit-hep/aghast/blob/master/specification.adoc#rawexternalbuffer)
      * [Metadata](https://github.com/scikit-hep/aghast/blob/master/specification.adoc#metadata)
      * [Decoration](https://github.com/scikit-hep/aghast/blob/master/specification.adoc#decoration)

## Tutorial examples

[Run this tutorial on Binder](https://mybinder.org/v2/gh/scikit-hep/aghast/master?urlpath=lab/tree/binder%2Ftutorial.ipynb).

### Conversions

The main purpose of aghast is to move aggregated, histogram-like statistics (called "ghasts") from one framework to the next. This requires a conversion of high-level domain concepts.

Consider the following example: in Numpy, a histogram is simply a 2-tuple of arrays with special meaning—bin contents, then bin edges.


```python
import numpy

numpy_hist = numpy.histogram(numpy.random.normal(0, 1, int(10e6)), bins=80, range=(-5, 5))
numpy_hist
```




    (array([     2,      5,      9,     15,     29,     49,     80,    104,
               237,    352,    555,    867,   1447,   2046,   3037,   4562,
              6805,   9540,  13529,  18584,  25593,  35000,  46024,  59103,
             76492,  96441, 119873, 146159, 177533, 210628, 246316, 283292,
            321377, 359314, 393857, 426446, 453031, 474806, 489846, 496646,
            497922, 490499, 473200, 453527, 425650, 393297, 358537, 321099,
            282519, 246469, 211181, 177550, 147417, 120322,  96592,  76665,
             59587,  45776,  34459,  25900,  18876,  13576,   9571,   6662,
              4629,   3161,   2069,   1334,    878,    581,    332,    220,
               135,     65,     39,     26,     19,     15,      4,      4]),
     array([-5.   , -4.875, -4.75 , -4.625, -4.5  , -4.375, -4.25 , -4.125,
            -4.   , -3.875, -3.75 , -3.625, -3.5  , -3.375, -3.25 , -3.125,
            -3.   , -2.875, -2.75 , -2.625, -2.5  , -2.375, -2.25 , -2.125,
            -2.   , -1.875, -1.75 , -1.625, -1.5  , -1.375, -1.25 , -1.125,
            -1.   , -0.875, -0.75 , -0.625, -0.5  , -0.375, -0.25 , -0.125,
             0.   ,  0.125,  0.25 ,  0.375,  0.5  ,  0.625,  0.75 ,  0.875,
             1.   ,  1.125,  1.25 ,  1.375,  1.5  ,  1.625,  1.75 ,  1.875,
             2.   ,  2.125,  2.25 ,  2.375,  2.5  ,  2.625,  2.75 ,  2.875,
             3.   ,  3.125,  3.25 ,  3.375,  3.5  ,  3.625,  3.75 ,  3.875,
             4.   ,  4.125,  4.25 ,  4.375,  4.5  ,  4.625,  4.75 ,  4.875,
             5.   ]))



We convert that into the aghast equivalent (a "ghast") with a connector (two functions: `from_numpy` and `to_numpy`).


```python
import aghast

ghastly_hist = aghast.from_numpy(numpy_hist)
ghastly_hist
```




    <Histogram at 0x7f0dc88a9b38>



This object is instantiated from a class structure built from simple pieces.


```python
ghastly_hist.dump()
```

    Histogram(
      axis=[
        Axis(binning=RegularBinning(num=80, interval=RealInterval(low=-5.0, high=5.0)))
      ],
      counts=
        UnweightedCounts(
          counts=
            InterpretedInlineInt64Buffer(
              buffer=
                  [     2      5      9     15     29     49     80    104    237    352
                      555    867   1447   2046   3037   4562   6805   9540  13529  18584
                    25593  35000  46024  59103  76492  96441 119873 146159 177533 210628
                   246316 283292 321377 359314 393857 426446 453031 474806 489846 496646
                   497922 490499 473200 453527 425650 393297 358537 321099 282519 246469
                   211181 177550 147417 120322  96592  76665  59587  45776  34459  25900
                    18876  13576   9571   6662   4629   3161   2069   1334    878    581
                      332    220    135     65     39     26     19     15      4      4])))


Now it can be converted to a ROOT histogram with another connector.


```python
root_hist = aghast.to_root(ghastly_hist, "root_hist")
root_hist
```

    <ROOT.TH1D object ("root_hist") at 0x55555e208ef0>

```python
import ROOT
canvas = ROOT.TCanvas()
root_hist.Draw()
canvas.Draw()
```


![png](docs/tutorial-9_0.png)


And Pandas with yet another connector.


```python
pandas_hist = aghast.to_pandas(ghastly_hist)
pandas_hist
```




<div>
<table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>unweighted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>[-5.0, -4.875)</th>
      <td>2</td>
    </tr>
    <tr>
      <th>[-4.875, -4.75)</th>
      <td>5</td>
    </tr>
    <tr>
      <th>[-4.75, -4.625)</th>
      <td>9</td>
    </tr>
    <tr>
      <th>[-4.625, -4.5)</th>
      <td>15</td>
    </tr>
    <tr>
      <th>[-4.5, -4.375)</th>
      <td>29</td>
    </tr>
    <tr>
      <th>[-4.375, -4.25)</th>
      <td>49</td>
    </tr>
    <tr>
      <th>[-4.25, -4.125)</th>
      <td>80</td>
    </tr>
    <tr>
      <th>[-4.125, -4.0)</th>
      <td>104</td>
    </tr>
    <tr>
      <th>[-4.0, -3.875)</th>
      <td>237</td>
    </tr>
    <tr>
      <th>[-3.875, -3.75)</th>
      <td>352</td>
    </tr>
    <tr>
      <th>[-3.75, -3.625)</th>
      <td>555</td>
    </tr>
    <tr>
      <th>[-3.625, -3.5)</th>
      <td>867</td>
    </tr>
    <tr>
      <th>[-3.5, -3.375)</th>
      <td>1447</td>
    </tr>
    <tr>
      <th>[-3.375, -3.25)</th>
      <td>2046</td>
    </tr>
    <tr>
      <th>[-3.25, -3.125)</th>
      <td>3037</td>
    </tr>
    <tr>
      <th>[-3.125, -3.0)</th>
      <td>4562</td>
    </tr>
    <tr>
      <th>[-3.0, -2.875)</th>
      <td>6805</td>
    </tr>
    <tr>
      <th>[-2.875, -2.75)</th>
      <td>9540</td>
    </tr>
    <tr>
      <th>[-2.75, -2.625)</th>
      <td>13529</td>
    </tr>
    <tr>
      <th>[-2.625, -2.5)</th>
      <td>18584</td>
    </tr>
    <tr>
      <th>[-2.5, -2.375)</th>
      <td>25593</td>
    </tr>
    <tr>
      <th>[-2.375, -2.25)</th>
      <td>35000</td>
    </tr>
    <tr>
      <th>[-2.25, -2.125)</th>
      <td>46024</td>
    </tr>
    <tr>
      <th>[-2.125, -2.0)</th>
      <td>59103</td>
    </tr>
    <tr>
      <th>[-2.0, -1.875)</th>
      <td>76492</td>
    </tr>
    <tr>
      <th>[-1.875, -1.75)</th>
      <td>96441</td>
    </tr>
    <tr>
      <th>[-1.75, -1.625)</th>
      <td>119873</td>
    </tr>
    <tr>
      <th>[-1.625, -1.5)</th>
      <td>146159</td>
    </tr>
    <tr>
      <th>[-1.5, -1.375)</th>
      <td>177533</td>
    </tr>
    <tr>
      <th>[-1.375, -1.25)</th>
      <td>210628</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>[1.25, 1.375)</th>
      <td>211181</td>
    </tr>
    <tr>
      <th>[1.375, 1.5)</th>
      <td>177550</td>
    </tr>
    <tr>
      <th>[1.5, 1.625)</th>
      <td>147417</td>
    </tr>
    <tr>
      <th>[1.625, 1.75)</th>
      <td>120322</td>
    </tr>
    <tr>
      <th>[1.75, 1.875)</th>
      <td>96592</td>
    </tr>
    <tr>
      <th>[1.875, 2.0)</th>
      <td>76665</td>
    </tr>
    <tr>
      <th>[2.0, 2.125)</th>
      <td>59587</td>
    </tr>
    <tr>
      <th>[2.125, 2.25)</th>
      <td>45776</td>
    </tr>
    <tr>
      <th>[2.25, 2.375)</th>
      <td>34459</td>
    </tr>
    <tr>
      <th>[2.375, 2.5)</th>
      <td>25900</td>
    </tr>
    <tr>
      <th>[2.5, 2.625)</th>
      <td>18876</td>
    </tr>
    <tr>
      <th>[2.625, 2.75)</th>
      <td>13576</td>
    </tr>
    <tr>
      <th>[2.75, 2.875)</th>
      <td>9571</td>
    </tr>
    <tr>
      <th>[2.875, 3.0)</th>
      <td>6662</td>
    </tr>
    <tr>
      <th>[3.0, 3.125)</th>
      <td>4629</td>
    </tr>
    <tr>
      <th>[3.125, 3.25)</th>
      <td>3161</td>
    </tr>
    <tr>
      <th>[3.25, 3.375)</th>
      <td>2069</td>
    </tr>
    <tr>
      <th>[3.375, 3.5)</th>
      <td>1334</td>
    </tr>
    <tr>
      <th>[3.5, 3.625)</th>
      <td>878</td>
    </tr>
    <tr>
      <th>[3.625, 3.75)</th>
      <td>581</td>
    </tr>
    <tr>
      <th>[3.75, 3.875)</th>
      <td>332</td>
    </tr>
    <tr>
      <th>[3.875, 4.0)</th>
      <td>220</td>
    </tr>
    <tr>
      <th>[4.0, 4.125)</th>
      <td>135</td>
    </tr>
    <tr>
      <th>[4.125, 4.25)</th>
      <td>65</td>
    </tr>
    <tr>
      <th>[4.25, 4.375)</th>
      <td>39</td>
    </tr>
    <tr>
      <th>[4.375, 4.5)</th>
      <td>26</td>
    </tr>
    <tr>
      <th>[4.5, 4.625)</th>
      <td>19</td>
    </tr>
    <tr>
      <th>[4.625, 4.75)</th>
      <td>15</td>
    </tr>
    <tr>
      <th>[4.75, 4.875)</th>
      <td>4</td>
    </tr>
    <tr>
      <th>[4.875, 5.0)</th>
      <td>4</td>
    </tr>
  </tbody>
</table>
<p>80 rows × 1 columns</p>
</div>



### Serialization

A ghast is also a [Flatbuffers](http://google.github.io/flatbuffers/) object, which has a [multi-lingual](https://google.github.io/flatbuffers/flatbuffers_support.html), [random-access](https://github.com/mzaks/FlatBuffersSwift/wiki/FlatBuffers-Explained), [small-footprint](http://google.github.io/flatbuffers/md__benchmarks.html) serialization:


```python
ghastly_hist.tobuffer()
```

    bytearray("\x04\x00\x00\x00\x90\xff\xff\xff\x10\x00\x00\x00\x00\x01\n\x00\x10\x00\x0c\x00\x0b\x00\x04
               \x00\n\x00\x00\x00`\x00\x00\x00\x00\x00\x00\x01\x04\x00\x00\x00\x01\x00\x00\x00\x0c\x00\x00
               \x00\x08\x00\x0c\x00\x0b\x00\x04\x00\x08\x00\x00\x00\x10\x00\x00\x00\x00\x00\x00\x02\x08\x00
               (\x00\x1c\x00\x04\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x14\xc0\x00\x00\x00\x00\x00
               \x00\x14@\x01\x00\x00\x00\x00\x00\x00\x00P\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x08
               \x00\n\x00\t\x00\x04\x00\x08\x00\x00\x00\x0c\x00\x00\x00\x00\x02\x06\x00\x08\x00\x04\x00\x06
               \x00\x00\x00\x04\x00\x00\x00\x80\x02\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x05\x00\x00\x00
               \x00\x00\x00\x00\t\x00\x00\x00\x00\x00\x00\x00\x0f\x00\x00\x00\x00\x00\x00\x00\x1d\x00\x00
               \x00\x00\x00\x00\x001\x00\x00\x00\x00\x00\x00\x00P\x00\x00\x00\x00\x00\x00\x00h\x00\x00\x00
               \x00\x00\x00\x00\xed\x00\x00\x00\x00\x00\x00\x00`\x01\x00\x00\x00\x00\x00\x00+\x02\x00\x00
               \x00\x00\x00\x00c\x03\x00\x00\x00\x00\x00\x00\xa7\x05\x00\x00\x00\x00\x00\x00\xfe\x07\x00
               \x00\x00\x00\x00\x00\xdd\x0b\x00\x00\x00\x00\x00\x00\xd2\x11\x00\x00\x00\x00\x00\x00\x95\x1a
               \x00\x00\x00\x00\x00\x00D%\x00\x00\x00\x00\x00\x00\xd94\x00\x00\x00\x00\x00\x00\x98H\x00\x00
               \x00\x00\x00\x00\xf9c\x00\x00\x00\x00\x00\x00\xb8\x88\x00\x00\x00\x00\x00\x00\xc8\xb3\x00\x00
               \x00\x00\x00\x00\xdf\xe6\x00\x00\x00\x00\x00\x00\xcc*\x01\x00\x00\x00\x00\x00\xb9x\x01\x00
               \x00\x00\x00\x00A\xd4\x01\x00\x00\x00\x00\x00\xef:\x02\x00\x00\x00\x00\x00}\xb5\x02\x00\x00
               \x00\x00\x00\xc46\x03\x00\x00\x00\x00\x00,\xc2\x03\x00\x00\x00\x00\x00\x9cR\x04\x00\x00\x00
               \x00\x00a\xe7\x04\x00\x00\x00\x00\x00\x92{\x05\x00\x00\x00\x00\x00\x81\x02\x06\x00\x00\x00
               \x00\x00\xce\x81\x06\x00\x00\x00\x00\x00\xa7\xe9\x06\x00\x00\x00\x00\x00\xb6>\x07\x00\x00
               \x00\x00\x00vy\x07\x00\x00\x00\x00\x00\x06\x94\x07\x00\x00\x00\x00\x00\x02\x99\x07\x00\x00
               \x00\x00\x00\x03|\x07\x00\x00\x00\x00\x00p8\x07\x00\x00\x00\x00\x00\x97\xeb\x06\x00\x00\x00
               \x00\x00\xb2~\x06\x00\x00\x00\x00\x00Q\x00\x06\x00\x00\x00\x00\x00\x89x\x05\x00\x00\x00\x00
               \x00K\xe6\x04\x00\x00\x00\x00\x00\x97O\x04\x00\x00\x00\x00\x00\xc5\xc2\x03\x00\x00\x00\x00
               \x00\xed8\x03\x00\x00\x00\x00\x00\x8e\xb5\x02\x00\x00\x00\x00\x00\xd9?\x02\x00\x00\x00\x00
               \x00\x02\xd6\x01\x00\x00\x00\x00\x00Py\x01\x00\x00\x00\x00\x00y+\x01\x00\x00\x00\x00\x00\xc3
               \xe8\x00\x00\x00\x00\x00\x00\xd0\xb2\x00\x00\x00\x00\x00\x00\x9b\x86\x00\x00\x00\x00\x00\x00
               ,e\x00\x00\x00\x00\x00\x00\xbcI\x00\x00\x00\x00\x00\x00\x085\x00\x00\x00\x00\x00\x00c%\x00
               \x00\x00\x00\x00\x00\x06\x1a\x00\x00\x00\x00\x00\x00\x15\x12\x00\x00\x00\x00\x00\x00Y\x0c
               \x00\x00\x00\x00\x00\x00\x15\x08\x00\x00\x00\x00\x00\x006\x05\x00\x00\x00\x00\x00\x00n\x03
               \x00\x00\x00\x00\x00\x00E\x02\x00\x00\x00\x00\x00\x00L\x01\x00\x00\x00\x00\x00\x00\xdc\x00
               \x00\x00\x00\x00\x00\x00\x87\x00\x00\x00\x00\x00\x00\x00A\x00\x00\x00\x00\x00\x00\x00\'\x00
               \x00\x00\x00\x00\x00\x00\x1a\x00\x00\x00\x00\x00\x00\x00\x13\x00\x00\x00\x00\x00\x00\x00\x0f
               \x00\x00\x00\x00\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00")

```python
print("Numpy size: ", numpy_hist[0].nbytes + numpy_hist[1].nbytes)

tmessage = ROOT.TMessage()
tmessage.WriteObject(root_hist)
print("ROOT size:  ", tmessage.Length())

import pickle
print("Pandas size:", len(pickle.dumps(pandas_hist)))

print("Aghast size: ", len(ghastly_hist.tobuffer()))
```

    Numpy size:  1288
    ROOT size:   1962
    Pandas size: 2984
    Aghast size:  792


Aghast is generally forseen as a memory format, like [Apache Arrow](https://arrow.apache.org), but for statistical aggregations. Like Arrow, it reduces the need to implement $N(N - 1)/2$ conversion functions among $N$ statistical libraries to just $N$ conversion functions. (See the figure on Arrow's website.)

### Translation of conventions

Aghast also intends to be as close to zero-copy as possible. This means that it must make graceful translations among conventions. Different histogramming libraries handle overflow bins in different ways:


```python
fromroot = aghast.from_root(root_hist)
fromroot.axis[0].binning.dump()
print("Bin contents length:", len(fromroot.counts.array))
```

    RegularBinning(
      num=80,
      interval=RealInterval(low=-5.0, high=5.0),
      overflow=RealOverflow(loc_underflow=BinLocation.below1, loc_overflow=BinLocation.above1))
    Bin contents length: 82



```python
ghastly_hist.axis[0].binning.dump()
print("Bin contents length:", len(ghastly_hist.counts.array))
```

    RegularBinning(num=80, interval=RealInterval(low=-5.0, high=5.0))
    Bin contents length: 80


And yet we want to be able to manipulate them as though these differences did not exist.


```python
sum_hist = fromroot + ghastly_hist
```


```python
sum_hist.axis[0].binning.dump()
print("Bin contents length:", len(sum_hist.counts.array))
```

    RegularBinning(
      num=80,
      interval=RealInterval(low=-5.0, high=5.0),
      overflow=RealOverflow(loc_underflow=BinLocation.above1, loc_overflow=BinLocation.above2))
    Bin contents length: 82


The binning structure keeps track of the existence of underflow/overflow bins and where they are located.

   * ROOT's convention is to put underflow before the normal bins (`below1`) and overflow after (`above1`), so that the normal bins are effectively 1-indexed.
   * Boost.Histogram's convention is to put overflow after the normal bins (`above1`) and underflow after that (`above2`), so that underflow is accessed via `myhist[-1]` in Numpy.
   * Numpy histograms don't have underflow/overflow bins.
   * Pandas could have `Intervals` that extend to infinity.

Aghast accepts all of these, so that it doesn't have to manipulate the bin contents buffer it receives, but knows how to deal with them if it has to combine histograms that follow different conventions.

### Binning types

All the different axis types have an equivalent in aghast (and not all are single-dimensional).


```python
aghast.IntegerBinning(5, 10).dump()
aghast.RegularBinning(100, aghast.RealInterval(-5, 5)).dump()
aghast.HexagonalBinning(0, 100, 0, 100, aghast.HexagonalBinning.cube_xy).dump()
aghast.EdgesBinning([0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100]).dump()
aghast.IrregularBinning([aghast.RealInterval(0, 5),
                         aghast.RealInterval(10, 100),
                         aghast.RealInterval(-10, 10)],
                       overlapping_fill=aghast.IrregularBinning.all).dump()
aghast.CategoryBinning(["one", "two", "three"]).dump()
aghast.SparseRegularBinning([5, 3, -2, 8, -100], 10).dump()
aghast.FractionBinning(error_method=aghast.FractionBinning.clopper_pearson).dump()
aghast.PredicateBinning(["signal region", "control region"]).dump()
aghast.VariationBinning([aghast.Variation([aghast.Assignment("x", "nominal")]),
                         aghast.Variation([aghast.Assignment("x", "nominal + sigma")]),
                         aghast.Variation([aghast.Assignment("x", "nominal - sigma")])]).dump()
```

    IntegerBinning(min=5, max=10)
    RegularBinning(num=100, interval=RealInterval(low=-5.0, high=5.0))
    HexagonalBinning(qmin=0, qmax=100, rmin=0, rmax=100, coordinates=HexagonalBinning.cube_xy)
    EdgesBinning(edges=[0.01 0.05 0.1 0.5 1 5 10 50 100])
    IrregularBinning(
      intervals=[
        RealInterval(low=0.0, high=5.0),
        RealInterval(low=10.0, high=100.0),
        RealInterval(low=-10.0, high=10.0)
      ],
      overlapping_fill=IrregularBinning.all)
    CategoryBinning(categories=['one', 'two', 'three'])
    SparseRegularBinning(bins=[5 3 -2 8 -100], bin_width=10.0)
    FractionBinning(error_method=FractionBinning.clopper_pearson)
    PredicateBinning(predicates=['signal region', 'control region'])
    VariationBinning(
      variations=[
        Variation(assignments=[
            Assignment(identifier='x', expression='nominal')
          ]),
        Variation(
          assignments=[
            Assignment(identifier='x', expression='nominal + sigma')
          ]),
        Variation(
          assignments=[
            Assignment(identifier='x', expression='nominal - sigma')
          ])
      ])


The meanings of these binning classes are given in [the specification](https://github.com/scikit-hep/aghast/blob/master/specification.adoc#integerbinning), but many of them can be converted into one another, and converting to `CategoryBinning` (strings) often makes the intent clear.


```python
aghast.IntegerBinning(5, 10).toCategoryBinning().dump()
aghast.RegularBinning(10, aghast.RealInterval(-5, 5)).toCategoryBinning().dump()
aghast.EdgesBinning([0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100]).toCategoryBinning().dump()
aghast.IrregularBinning([aghast.RealInterval(0, 5),
                         aghast.RealInterval(10, 100),
                         aghast.RealInterval(-10, 10)],
                       overlapping_fill=aghast.IrregularBinning.all).toCategoryBinning().dump()
aghast.SparseRegularBinning([5, 3, -2, 8, -100], 10).toCategoryBinning().dump()
aghast.FractionBinning(error_method=aghast.FractionBinning.clopper_pearson).toCategoryBinning().dump()
aghast.PredicateBinning(["signal region", "control region"]).toCategoryBinning().dump()
aghast.VariationBinning([aghast.Variation([aghast.Assignment("x", "nominal")]),
                         aghast.Variation([aghast.Assignment("x", "nominal + sigma")]),
                         aghast.Variation([aghast.Assignment("x", "nominal - sigma")])]
                        ).toCategoryBinning().dump()
```

    CategoryBinning(categories=['5', '6', '7', '8', '9', '10'])
    CategoryBinning(
      categories=['[-5, -4)', '[-4, -3)', '[-3, -2)', '[-2, -1)', '[-1, 0)', '[0, 1)', '[1, 2)', '[2, 3)',
                  '[3, 4)', '[4, 5)'])
    CategoryBinning(
      categories=['[0.01, 0.05)', '[0.05, 0.1)', '[0.1, 0.5)', '[0.5, 1)', '[1, 5)', '[5, 10)', '[10, 50)',
                  '[50, 100)'])
    CategoryBinning(categories=['[0, 5)', '[10, 100)', '[-10, 10)'])
    CategoryBinning(categories=['[50, 60)', '[30, 40)', '[-20, -10)', '[80, 90)', '[-1000, -990)'])
    CategoryBinning(categories=['pass', 'all'])
    CategoryBinning(categories=['signal region', 'control region'])
    CategoryBinning(categories=['x := nominal', 'x := nominal + sigma', 'x := nominal - sigma'])


This technique can also clear up confusion about overflow bins.


```python
aghast.RegularBinning(5, aghast.RealInterval(-5, 5), aghast.RealOverflow(
    loc_underflow=aghast.BinLocation.above2,
    loc_overflow=aghast.BinLocation.above1,
    loc_nanflow=aghast.BinLocation.below1
    )).toCategoryBinning().dump()
```

    CategoryBinning(
      categories=['{nan}', '[-5, -3)', '[-3, -1)', '[-1, 1)', '[1, 3)', '[3, 5)', '[5, +inf]',
                  '[-inf, -5)'])


## Fancy binning types

You might also be wondering about `FractionBinning`, `PredicateBinning`, and `VariationBinning`.

`FractionBinning` is an axis of two bins: #passing and #total, #failing and #total, or #passing and #failing. Adding it to another axis effectively makes an "efficiency plot."


```python
h = aghast.Histogram([aghast.Axis(aghast.FractionBinning()),
                      aghast.Axis(aghast.RegularBinning(10, aghast.RealInterval(-5, 5)))],
                    aghast.UnweightedCounts(
                        aghast.InterpretedInlineBuffer.fromarray(
                            numpy.array([[  9,  25,  29,  35,  54,  67,  60,  84,  80,  94],
                                         [ 99, 119, 109, 109,  95, 104, 102, 106, 112, 122]]))))
df = aghast.to_pandas(h)
df
```




<div>
<table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>unweighted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="10" valign="top">pass</th>
      <th>[-5.0, -4.0)</th>
      <td>9</td>
    </tr>
    <tr>
      <th>[-4.0, -3.0)</th>
      <td>25</td>
    </tr>
    <tr>
      <th>[-3.0, -2.0)</th>
      <td>29</td>
    </tr>
    <tr>
      <th>[-2.0, -1.0)</th>
      <td>35</td>
    </tr>
    <tr>
      <th>[-1.0, 0.0)</th>
      <td>54</td>
    </tr>
    <tr>
      <th>[0.0, 1.0)</th>
      <td>67</td>
    </tr>
    <tr>
      <th>[1.0, 2.0)</th>
      <td>60</td>
    </tr>
    <tr>
      <th>[2.0, 3.0)</th>
      <td>84</td>
    </tr>
    <tr>
      <th>[3.0, 4.0)</th>
      <td>80</td>
    </tr>
    <tr>
      <th>[4.0, 5.0)</th>
      <td>94</td>
    </tr>
    <tr>
      <th rowspan="10" valign="top">all</th>
      <th>[-5.0, -4.0)</th>
      <td>99</td>
    </tr>
    <tr>
      <th>[-4.0, -3.0)</th>
      <td>119</td>
    </tr>
    <tr>
      <th>[-3.0, -2.0)</th>
      <td>109</td>
    </tr>
    <tr>
      <th>[-2.0, -1.0)</th>
      <td>109</td>
    </tr>
    <tr>
      <th>[-1.0, 0.0)</th>
      <td>95</td>
    </tr>
    <tr>
      <th>[0.0, 1.0)</th>
      <td>104</td>
    </tr>
    <tr>
      <th>[1.0, 2.0)</th>
      <td>102</td>
    </tr>
    <tr>
      <th>[2.0, 3.0)</th>
      <td>106</td>
    </tr>
    <tr>
      <th>[3.0, 4.0)</th>
      <td>112</td>
    </tr>
    <tr>
      <th>[4.0, 5.0)</th>
      <td>122</td>
    </tr>
  </tbody>
</table>
</div>




```python
df = df.unstack(level=0)
df
```




<div>
<table class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="2" halign="left">unweighted</th>
    </tr>
    <tr>
      <th></th>
      <th>all</th>
      <th>pass</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>[-5.0, -4.0)</th>
      <td>99</td>
      <td>9</td>
    </tr>
    <tr>
      <th>[-4.0, -3.0)</th>
      <td>119</td>
      <td>25</td>
    </tr>
    <tr>
      <th>[-3.0, -2.0)</th>
      <td>109</td>
      <td>29</td>
    </tr>
    <tr>
      <th>[-2.0, -1.0)</th>
      <td>109</td>
      <td>35</td>
    </tr>
    <tr>
      <th>[-1.0, 0.0)</th>
      <td>95</td>
      <td>54</td>
    </tr>
    <tr>
      <th>[0.0, 1.0)</th>
      <td>104</td>
      <td>67</td>
    </tr>
    <tr>
      <th>[1.0, 2.0)</th>
      <td>102</td>
      <td>60</td>
    </tr>
    <tr>
      <th>[2.0, 3.0)</th>
      <td>106</td>
      <td>84</td>
    </tr>
    <tr>
      <th>[3.0, 4.0)</th>
      <td>112</td>
      <td>80</td>
    </tr>
    <tr>
      <th>[4.0, 5.0)</th>
      <td>122</td>
      <td>94</td>
    </tr>
  </tbody>
</table>
</div>




```python
df["unweighted", "pass"] / df["unweighted", "all"]
```




    [-5.0, -4.0)    0.090909
    [-4.0, -3.0)    0.210084
    [-3.0, -2.0)    0.266055
    [-2.0, -1.0)    0.321101
    [-1.0, 0.0)     0.568421
    [0.0, 1.0)      0.644231
    [1.0, 2.0)      0.588235
    [2.0, 3.0)      0.792453
    [3.0, 4.0)      0.714286
    [4.0, 5.0)      0.770492
    dtype: float64



`PredicateBinning` means that each bin represents a predicate (if-then rule) in the filling procedure. Aghast doesn't _have_ a filling procedure, but filling-libraries can use this to encode relationships among histograms that a fitting-library can take advantage of, for combined signal-control region fits, for instance. It's possible for those regions to overlap: an input datum might satisfy more than one predicate, and `overlapping_fill` determines which bin(s) were chosen: `first`, `last`, or `all`.

`VariationBinning` means that each bin represents a variation of one of the paramters used to calculate the fill-variables. This is used to determine sensitivity to systematic effects, by varying them and re-filling. In this kind of binning, the same input datum enters every bin.


```python
xdata = numpy.random.normal(0, 1, int(1e6))
sigma = numpy.random.uniform(-0.1, 0.8, int(1e6))

h = aghast.Histogram([aghast.Axis(aghast.VariationBinning([
                         aghast.Variation([aghast.Assignment("x", "nominal")]),
                         aghast.Variation([aghast.Assignment("x", "nominal + sigma")])])),
                     aghast.Axis(aghast.RegularBinning(10, aghast.RealInterval(-5, 5)))],
                    aghast.UnweightedCounts(
                        aghast.InterpretedInlineBuffer.fromarray(
                            numpy.concatenate([
                                numpy.histogram(xdata, bins=10, range=(-5, 5))[0],
                                numpy.histogram(xdata + sigma, bins=10, range=(-5, 5))[0]]))))
df = aghast.to_pandas(h)
df
```




<div>
<table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>unweighted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="10" valign="top">x := nominal</th>
      <th>[-5.0, -4.0)</th>
      <td>31</td>
    </tr>
    <tr>
      <th>[-4.0, -3.0)</th>
      <td>1309</td>
    </tr>
    <tr>
      <th>[-3.0, -2.0)</th>
      <td>21624</td>
    </tr>
    <tr>
      <th>[-2.0, -1.0)</th>
      <td>135279</td>
    </tr>
    <tr>
      <th>[-1.0, 0.0)</th>
      <td>341683</td>
    </tr>
    <tr>
      <th>[0.0, 1.0)</th>
      <td>341761</td>
    </tr>
    <tr>
      <th>[1.0, 2.0)</th>
      <td>135675</td>
    </tr>
    <tr>
      <th>[2.0, 3.0)</th>
      <td>21334</td>
    </tr>
    <tr>
      <th>[3.0, 4.0)</th>
      <td>1273</td>
    </tr>
    <tr>
      <th>[4.0, 5.0)</th>
      <td>31</td>
    </tr>
    <tr>
      <th rowspan="10" valign="top">x := nominal + sigma</th>
      <th>[-5.0, -4.0)</th>
      <td>14</td>
    </tr>
    <tr>
      <th>[-4.0, -3.0)</th>
      <td>559</td>
    </tr>
    <tr>
      <th>[-3.0, -2.0)</th>
      <td>10814</td>
    </tr>
    <tr>
      <th>[-2.0, -1.0)</th>
      <td>84176</td>
    </tr>
    <tr>
      <th>[-1.0, 0.0)</th>
      <td>271999</td>
    </tr>
    <tr>
      <th>[0.0, 1.0)</th>
      <td>367950</td>
    </tr>
    <tr>
      <th>[1.0, 2.0)</th>
      <td>209479</td>
    </tr>
    <tr>
      <th>[2.0, 3.0)</th>
      <td>49997</td>
    </tr>
    <tr>
      <th>[3.0, 4.0)</th>
      <td>4815</td>
    </tr>
    <tr>
      <th>[4.0, 5.0)</th>
      <td>193</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.unstack(level=0)
```




<div>
<table class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="2" halign="left">unweighted</th>
    </tr>
    <tr>
      <th></th>
      <th>x := nominal</th>
      <th>x := nominal + sigma</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>[-5.0, -4.0)</th>
      <td>31</td>
      <td>14</td>
    </tr>
    <tr>
      <th>[-4.0, -3.0)</th>
      <td>1309</td>
      <td>559</td>
    </tr>
    <tr>
      <th>[-3.0, -2.0)</th>
      <td>21624</td>
      <td>10814</td>
    </tr>
    <tr>
      <th>[-2.0, -1.0)</th>
      <td>135279</td>
      <td>84176</td>
    </tr>
    <tr>
      <th>[-1.0, 0.0)</th>
      <td>341683</td>
      <td>271999</td>
    </tr>
    <tr>
      <th>[0.0, 1.0)</th>
      <td>341761</td>
      <td>367950</td>
    </tr>
    <tr>
      <th>[1.0, 2.0)</th>
      <td>135675</td>
      <td>209479</td>
    </tr>
    <tr>
      <th>[2.0, 3.0)</th>
      <td>21334</td>
      <td>49997</td>
    </tr>
    <tr>
      <th>[3.0, 4.0)</th>
      <td>1273</td>
      <td>4815</td>
    </tr>
    <tr>
      <th>[4.0, 5.0)</th>
      <td>31</td>
      <td>193</td>
    </tr>
  </tbody>
</table>
</div>



### Collections

You can gather many objects (histograms, functions, ntuples) into a `Collection`, partly for convenience of encapsulating all of them in one object.


```python
aghast.Collection({"one": fromroot, "two": ghastly_hist}).dump()
```

    Collection(
      objects={
        'one': Histogram(
          axis=[
            Axis(
              binning=
                RegularBinning(
                  num=80,
                  interval=RealInterval(low=-5.0, high=5.0),
                  overflow=RealOverflow(loc_underflow=BinLocation.below1, loc_overflow=BinLocation.above1)),
              statistics=[
                Statistics(
                  moments=[
                    Moments(sumwxn=InterpretedInlineInt64Buffer(buffer=[1e+07]), n=0),
                    Moments(sumwxn=InterpretedInlineFloat64Buffer(buffer=[1e+07]), n=0, weightpower=1),
                    Moments(sumwxn=InterpretedInlineFloat64Buffer(buffer=[1e+07]), n=0, weightpower=2),
                    Moments(sumwxn=InterpretedInlineFloat64Buffer(buffer=[2468.31]), n=1, weightpower=1),
                    Moments(
                      sumwxn=InterpretedInlineFloat64Buffer(buffer=[1.00118e+07]),
                      n=2,
                      weightpower=1)
                  ])
              ])
          ],
          counts=
            UnweightedCounts(
              counts=
                InterpretedInlineFloat64Buffer(
                  buffer=
                      [0.00000e+00 2.00000e+00 5.00000e+00 9.00000e+00 1.50000e+01 2.90000e+01
                       4.90000e+01 8.00000e+01 1.04000e+02 2.37000e+02 3.52000e+02 5.55000e+02
                       8.67000e+02 1.44700e+03 2.04600e+03 3.03700e+03 4.56200e+03 6.80500e+03
                       9.54000e+03 1.35290e+04 1.85840e+04 2.55930e+04 3.50000e+04 4.60240e+04
                       5.91030e+04 7.64920e+04 9.64410e+04 1.19873e+05 1.46159e+05 1.77533e+05
                       2.10628e+05 2.46316e+05 2.83292e+05 3.21377e+05 3.59314e+05 3.93857e+05
                       4.26446e+05 4.53031e+05 4.74806e+05 4.89846e+05 4.96646e+05 4.97922e+05
                       4.90499e+05 4.73200e+05 4.53527e+05 4.25650e+05 3.93297e+05 3.58537e+05
                       3.21099e+05 2.82519e+05 2.46469e+05 2.11181e+05 1.77550e+05 1.47417e+05
                       1.20322e+05 9.65920e+04 7.66650e+04 5.95870e+04 4.57760e+04 3.44590e+04
                       2.59000e+04 1.88760e+04 1.35760e+04 9.57100e+03 6.66200e+03 4.62900e+03
                       3.16100e+03 2.06900e+03 1.33400e+03 8.78000e+02 5.81000e+02 3.32000e+02
                       2.20000e+02 1.35000e+02 6.50000e+01 3.90000e+01 2.60000e+01 1.90000e+01
                       1.50000e+01 4.00000e+00 4.00000e+00 0.00000e+00]))),
        'two': Histogram(
          axis=[
            Axis(binning=RegularBinning(num=80, interval=RealInterval(low=-5.0, high=5.0)))
          ],
          counts=
            UnweightedCounts(
              counts=
                InterpretedInlineInt64Buffer(
                  buffer=
                      [     2      5      9     15     29     49     80    104    237    352
                          555    867   1447   2046   3037   4562   6805   9540  13529  18584
                        25593  35000  46024  59103  76492  96441 119873 146159 177533 210628
                       246316 283292 321377 359314 393857 426446 453031 474806 489846 496646
                       497922 490499 473200 453527 425650 393297 358537 321099 282519 246469
                       211181 177550 147417 120322  96592  76665  59587  45776  34459  25900
                        18876  13576   9571   6662   4629   3161   2069   1334    878    581
                          332    220    135     65     39     26     19     15      4      4])))
      })


Not only for convenience: [you can also define](https://github.com/scikit-hep/aghast/blob/master/specification.adoc#Collection) an `Axis` in the `Collection` to subdivide all contents by that `Axis`. For instance, you can make a collection of qualitatively different histograms all have a signal and control region with `PredicateBinning`, or all have systematic variations with `VariationBinning`.

It is not necessary to rely on naming conventions to communicate this information from filler to fitter.

### Histogram → histogram conversions

I said in the introduction that aghast does not fill histograms and does not plot histograms—the two things data analysts are expecting to do. These would be done by user-facing libraries.

Aghast does, however, transform histograms into other histograms, and not just among formats. You can combine histograms with `+`. In addition to adding histogram counts, it combines auxiliary statistics appropriately (if possible).


```python
h1 = aghast.Histogram([
    aghast.Axis(aghast.RegularBinning(10, aghast.RealInterval(-5, 5)),
        statistics=[aghast.Statistics(
            moments=[
                aghast.Moments(aghast.InterpretedInlineBuffer.fromarray(numpy.array([10])), n=1),
                aghast.Moments(aghast.InterpretedInlineBuffer.fromarray(numpy.array([20])), n=2)],
            quantiles=[
                aghast.Quantiles(aghast.InterpretedInlineBuffer.fromarray(numpy.array([30])), p=0.5)],
            mode=aghast.Modes(aghast.InterpretedInlineBuffer.fromarray(numpy.array([40]))),
            min=aghast.Extremes(aghast.InterpretedInlineBuffer.fromarray(numpy.array([50]))),
            max=aghast.Extremes(aghast.InterpretedInlineBuffer.fromarray(numpy.array([60]))))])],
    aghast.UnweightedCounts(aghast.InterpretedInlineBuffer.fromarray(numpy.arange(10))))
h2 = aghast.Histogram([
    aghast.Axis(aghast.RegularBinning(10, aghast.RealInterval(-5, 5)),
        statistics=[aghast.Statistics(
            moments=[
                aghast.Moments(aghast.InterpretedInlineBuffer.fromarray(numpy.array([100])), n=1),
                aghast.Moments(aghast.InterpretedInlineBuffer.fromarray(numpy.array([200])), n=2)],
            quantiles=[
                aghast.Quantiles(aghast.InterpretedInlineBuffer.fromarray(numpy.array([300])), p=0.5)],
            mode=aghast.Modes(aghast.InterpretedInlineBuffer.fromarray(numpy.array([400]))),
            min=aghast.Extremes(aghast.InterpretedInlineBuffer.fromarray(numpy.array([500]))),
            max=aghast.Extremes(aghast.InterpretedInlineBuffer.fromarray(numpy.array([600]))))])],
    aghast.UnweightedCounts(aghast.InterpretedInlineBuffer.fromarray(numpy.arange(100, 200, 10))))
```


```python
(h1 + h2).dump()
```

    Histogram(
      axis=[
        Axis(
          binning=RegularBinning(num=10, interval=RealInterval(low=-5.0, high=5.0)),
          statistics=[
            Statistics(
              moments=[
                Moments(sumwxn=InterpretedInlineInt64Buffer(buffer=[110]), n=1),
                Moments(sumwxn=InterpretedInlineInt64Buffer(buffer=[220]), n=2)
              ],
              min=Extremes(values=InterpretedInlineInt64Buffer(buffer=[50])),
              max=Extremes(values=InterpretedInlineInt64Buffer(buffer=[600])))
          ])
      ],
      counts=
        UnweightedCounts(
          counts=InterpretedInlineInt64Buffer(buffer=[100 111 122 133 144 155 166 177 188 199])))


The corresponding moments of `h1` and `h2` were matched and added, quantiles and modes were dropped (no way to combine them), and the correct minimum and maximum were picked; the histogram contents were added as well.

Another important histogram → histogram conversion is axis-reduction, which can take three forms:

   * slicing an axis, either dropping the eliminated bins or adding them to underflow/overflow (if possible, depends on binning type);
   * rebinning by combining neighboring bins;
   * projecting out an axis, removing it entirely, summing over all existing bins.

All of these operations use a Pandas-inspired `loc`/`iloc` syntax.


```python
h = aghast.Histogram(
    [aghast.Axis(aghast.RegularBinning(10, aghast.RealInterval(-5, 5)))],
    aghast.UnweightedCounts(
        aghast.InterpretedInlineBuffer.fromarray(numpy.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90]))))
```

`loc` slices in the data's coordinate system. `1.5` rounds up to bin index `6`. The first five bins get combined into an overflow bin: `150 = 10 + 20 + 30 + 40 + 50`.


```python
h.loc[1.5:].dump()
```

    Histogram(
      axis=[
        Axis(
          binning=
            RegularBinning(
              num=4,
              interval=RealInterval(low=1.0, high=5.0),
              overflow=
                RealOverflow(
                  loc_underflow=BinLocation.above1,
                  minf_mapping=RealOverflow.missing,
                  pinf_mapping=RealOverflow.missing,
                  nan_mapping=RealOverflow.missing)))
      ],
      counts=UnweightedCounts(counts=InterpretedInlineInt64Buffer(buffer=[60 70 80 90 150])))


`iloc` slices by bin index number.


```python
h.iloc[6:].dump()
```

    Histogram(
      axis=[
        Axis(
          binning=
            RegularBinning(
              num=4,
              interval=RealInterval(low=1.0, high=5.0),
              overflow=
                RealOverflow(
                  loc_underflow=BinLocation.above1,
                  minf_mapping=RealOverflow.missing,
                  pinf_mapping=RealOverflow.missing,
                  nan_mapping=RealOverflow.missing)))
      ],
      counts=UnweightedCounts(counts=InterpretedInlineInt64Buffer(buffer=[60 70 80 90 150])))


Slices have a `start`, `stop`, and `step` (`start:stop:step`). The `step` parameter rebins:


```python
h.iloc[::2].dump()
```

    Histogram(
      axis=[
        Axis(binning=RegularBinning(num=5, interval=RealInterval(low=-5.0, high=5.0)))
      ],
      counts=UnweightedCounts(counts=InterpretedInlineInt64Buffer(buffer=[10 50 90 130 170])))


Thus, you can slice and rebin as part of the same operation.

Projecting uses the same mechanism, except that `None` passed as an axis's slice projects it.


```python
h2 = aghast.Histogram(
    [aghast.Axis(aghast.RegularBinning(10, aghast.RealInterval(-5, 5))),
     aghast.Axis(aghast.RegularBinning(10, aghast.RealInterval(-5, 5)))],
    aghast.UnweightedCounts(
        aghast.InterpretedInlineBuffer.fromarray(numpy.arange(100))))

h2.iloc[:, None].dump()
```

    Histogram(
      axis=[
        Axis(binning=RegularBinning(num=10, interval=RealInterval(low=-5.0, high=5.0)))
      ],
      counts=
        UnweightedCounts(
          counts=InterpretedInlineInt64Buffer(buffer=[45 145 245 345 445 545 645 745 845 945])))


Thus, all three axis reduction operations can be performed in a single syntax.

In general, an n-dimensional ghastly histogram can be sliced like an n-dimensional Numpy array. This includes integer and boolean indexing (though that necessarily changes the binning to `IrregularBinning`).


```python
h.iloc[[4, 3, 6, 7, 1]].dump()
```

    Histogram(
      axis=[
        Axis(
          binning=
            IrregularBinning(
              intervals=[
                RealInterval(low=-1.0, high=0.0),
                RealInterval(low=-2.0, high=-1.0),
                RealInterval(low=1.0, high=2.0),
                RealInterval(low=2.0, high=3.0),
                RealInterval(low=-4.0, high=-3.0)
              ]))
      ],
      counts=UnweightedCounts(counts=InterpretedInlineInt64Buffer(buffer=[40 30 60 70 10])))



```python
h.iloc[[True, False, True, False, True, False, True, False, True, False]].dump()
```

    Histogram(
      axis=[
        Axis(
          binning=
            IrregularBinning(
              intervals=[
                RealInterval(low=-5.0, high=-4.0),
                RealInterval(low=-3.0, high=-2.0),
                RealInterval(low=-1.0, high=0.0),
                RealInterval(low=1.0, high=2.0),
                RealInterval(low=3.0, high=4.0)
              ]))
      ],
      counts=UnweightedCounts(counts=InterpretedInlineInt64Buffer(buffer=[0 20 40 60 80])))


`loc` for numerical binnings accepts

   * a real number
   * a real-valued slice
   * `None` for projection
   * ellipsis (`...`)

`loc` for categorical binnings accepts

   * a string
   * an iterable of strings
   * an _empty_ slice
   * `None` for projection
   * ellipsis (`...`)

`iloc` accepts

   * an integer
   * an integer-valued slice
   * `None` for projection
   * integer-valued array-like
   * boolean-valued array-like
   * ellipsis (`...`)

### Bin counts → Numpy

Frequently, one wants to extract bin counts from a histogram. The `loc`/`iloc` syntax above creates _histograms_ from _histograms_, not bin counts.

A histogram's `counts` property has a slice syntax.


```python
allcounts = numpy.arange(12) * numpy.arange(12)[:, None]   # multiplication table
allcounts[10, :] = -999   # underflows
allcounts[11, :] = 999    # overflows
allcounts[:, 0]  = -999   # underflows
allcounts[:, 1]  = 999    # overflows
print(allcounts)
```

    [[-999  999    0    0    0    0    0    0    0    0    0    0]
     [-999  999    2    3    4    5    6    7    8    9   10   11]
     [-999  999    4    6    8   10   12   14   16   18   20   22]
     [-999  999    6    9   12   15   18   21   24   27   30   33]
     [-999  999    8   12   16   20   24   28   32   36   40   44]
     [-999  999   10   15   20   25   30   35   40   45   50   55]
     [-999  999   12   18   24   30   36   42   48   54   60   66]
     [-999  999   14   21   28   35   42   49   56   63   70   77]
     [-999  999   16   24   32   40   48   56   64   72   80   88]
     [-999  999   18   27   36   45   54   63   72   81   90   99]
     [-999  999 -999 -999 -999 -999 -999 -999 -999 -999 -999 -999]
     [-999  999  999  999  999  999  999  999  999  999  999  999]]



```python
h2 = aghast.Histogram(
    [aghast.Axis(aghast.RegularBinning(10, aghast.RealInterval(-5, 5),
                     aghast.RealOverflow(loc_underflow=aghast.RealOverflow.above1,
                                       loc_overflow=aghast.RealOverflow.above2))),
     aghast.Axis(aghast.RegularBinning(10, aghast.RealInterval(-5, 5),
                     aghast.RealOverflow(loc_underflow=aghast.RealOverflow.below2,
                                       loc_overflow=aghast.RealOverflow.below1)))],
    aghast.UnweightedCounts(
        aghast.InterpretedInlineBuffer.fromarray(allcounts)))
```


```python
print(h2.counts[:, :])
```

    [[ 0  0  0  0  0  0  0  0  0  0]
     [ 2  3  4  5  6  7  8  9 10 11]
     [ 4  6  8 10 12 14 16 18 20 22]
     [ 6  9 12 15 18 21 24 27 30 33]
     [ 8 12 16 20 24 28 32 36 40 44]
     [10 15 20 25 30 35 40 45 50 55]
     [12 18 24 30 36 42 48 54 60 66]
     [14 21 28 35 42 49 56 63 70 77]
     [16 24 32 40 48 56 64 72 80 88]
     [18 27 36 45 54 63 72 81 90 99]]


To get the underflows and overflows, set the slice extremes to `-inf` and `+inf`.


```python
print(h2.counts[-numpy.inf:numpy.inf, :])
```

    [[-999 -999 -999 -999 -999 -999 -999 -999 -999 -999]
     [   0    0    0    0    0    0    0    0    0    0]
     [   2    3    4    5    6    7    8    9   10   11]
     [   4    6    8   10   12   14   16   18   20   22]
     [   6    9   12   15   18   21   24   27   30   33]
     [   8   12   16   20   24   28   32   36   40   44]
     [  10   15   20   25   30   35   40   45   50   55]
     [  12   18   24   30   36   42   48   54   60   66]
     [  14   21   28   35   42   49   56   63   70   77]
     [  16   24   32   40   48   56   64   72   80   88]
     [  18   27   36   45   54   63   72   81   90   99]
     [ 999  999  999  999  999  999  999  999  999  999]]



```python
print(h2.counts[:, -numpy.inf:numpy.inf])
```

    [[-999    0    0    0    0    0    0    0    0    0    0  999]
     [-999    2    3    4    5    6    7    8    9   10   11  999]
     [-999    4    6    8   10   12   14   16   18   20   22  999]
     [-999    6    9   12   15   18   21   24   27   30   33  999]
     [-999    8   12   16   20   24   28   32   36   40   44  999]
     [-999   10   15   20   25   30   35   40   45   50   55  999]
     [-999   12   18   24   30   36   42   48   54   60   66  999]
     [-999   14   21   28   35   42   49   56   63   70   77  999]
     [-999   16   24   32   40   48   56   64   72   80   88  999]
     [-999   18   27   36   45   54   63   72   81   90   99  999]]


Also note that the underflows are now all below the normal bins and overflows are now all above the normal bins, regardless of how they were arranged in the ghast. This allows analysis code to be independent of histogram source.

## Other types

Aghast can attach fit functions to histograms, can store standalone functions, such as lookup tables, and can store ntuples for unweighted fits or machine learning.

# Acknowledgements

Support for this work was provided by NSF cooperative agreement OAC-1836650 (IRIS-HEP), grant OAC-1450377 (DIANA/HEP) and PHY-1520942 (US-CMS LHC Ops).

Thanks especially to the gracious help of [aghast contributors](https://github.com/scikit-hep/aghast/graphs/contributors)!
