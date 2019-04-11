#!/usr/bin/env python

# BSD 3-Clause License; see https://github.com/scikit-hep/aghast/blob/master/LICENSE

import unittest

import numpy

import aghast
from aghast import *

class Test(unittest.TestCase):
    def runTest(self):
        pass

    def test_numpy1d(self):
        before = numpy.histogram(numpy.random.normal(0, 1, int(1e6)), bins=100, range=(-5, 5))
        after = aghast.tonumpy(aghast.fromnumpy(before))
        assert numpy.array_equal(before[0], after[0])
        assert numpy.array_equal(before[1], after[1])

    def test_numpy2d(self):
        before = numpy.histogram2d(x=numpy.random.normal(0, 1, int(1e6)), y=numpy.random.normal(0, 1, int(1e6)), bins=(10, 10), range=((-5, 5), (-5, 5)))
        after = aghast.tonumpy(aghast.fromnumpy(before))
        assert numpy.array_equal(before[0], after[0])
        assert numpy.array_equal(before[1], after[1])
        assert numpy.array_equal(before[2], after[2])

    def test_numpydd(self):
        before = numpy.histogramdd((numpy.random.normal(0, 1, int(1e6)), numpy.random.normal(0, 1, int(1e6)), numpy.random.normal(0, 1, int(1e6))), bins=(5, 5, 5), range=((-5, 5), (-5, 5), (-5, 5)))
        after = aghast.tonumpy(aghast.fromnumpy(before))
        assert numpy.array_equal(before[0], after[0])
        assert numpy.array_equal(before[1][0], after[1][0])
        assert numpy.array_equal(before[1][1], after[1][1])
        assert numpy.array_equal(before[1][2], after[1][2])
