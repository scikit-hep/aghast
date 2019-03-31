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

import unittest

import pytest
import numpy as np

import aghast
from aghast import *

hist = pytest.importorskip("fnal_column_analysis_tools.hist")
connect = pytest.importorskip("aghast._connect._fnalhist")

class Test(unittest.TestCase):
    def runTest(self):
        pass

    def test_fnalhist(self):
        h_mascots = hist.Hist("fermi mascot showdown",
                              hist.Cat("animal", "type of animal"),
                              hist.Cat("vocalization", "onomatopoiea is that how you spell it?"),
                              hist.Bin("height", "height [m]", 10, 0, 5),
                              # weight is a reserved keyword
                              hist.Bin("mass", "weight (g=9.81m/s**2) [kg]", np.power(10., np.arange(5)-1)),
                             )
        adult_bison_h = np.random.normal(loc=2.5, scale=0.2, size=40)
        adult_bison_w = np.random.normal(loc=700, scale=100, size=40)
        h_mascots.fill(animal="bison", vocalization="huff", height=adult_bison_h, mass=adult_bison_w)
        goose_h = np.random.normal(loc=0.4, scale=0.05, size=1000)
        goose_w = np.random.normal(loc=7, scale=1, size=1000)
        h_mascots.fill(animal="goose", vocalization="honk", height=goose_h, mass=goose_w)
        crane_h = np.random.normal(loc=1, scale=0.05, size=4)
        crane_w = np.random.normal(loc=10, scale=1, size=4)
        h_mascots.fill(animal="crane", vocalization="none", height=crane_h, mass=crane_w)
        baby_bison_h = np.random.normal(loc=.5, scale=0.1, size=20)
        baby_bison_w = np.random.normal(loc=200, scale=10, size=20)
        baby_bison_cutefactor = 2.5*np.ones_like(baby_bison_w)
        h_mascots.fill(animal="bison", vocalization="baa", height=baby_bison_h, mass=baby_bison_w, weight=baby_bison_cutefactor)
        h_mascots.fill(animal="fox", vocalization="none", height=1., mass=30.)
        h_mascots.fill(animal="unicorn", vocalization="", height=np.nan, mass=np.nan)

        h_converted = aghast.tofnalhist(aghast.fromfnalhist(h_mascots))
        for k in h_mascots.values().keys():
            assert k in h_converted.values().keys()
            assert np.all(h_mascots.values(overflow='allnan')[k] == h_converted.values(overflow='allnan')[k]), "mismatch for %r" % k
