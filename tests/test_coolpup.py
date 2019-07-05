#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 13:24:55 2019

@author: s1529682
"""

from coolpuppy import *
import numpy as np
import pandas as pd
from scipy import sparse
import cooler
import pytest


amap = np.loadtxt('tests/Scc1-control.10000-10.0K_over_CH12_loops_Rao_10-shifts_dist_100000-inf_unbalanced_covnorm.np.txt')

def test_cornerCV():
    assert np.isclose(cornerCV(amap), 0.07768632150324585)
    assert np.isclose(cornerCV(amap, 3), 0.06738199521452154)

def test_get_enrichment():
    assert np.isclose(get_enrichment(amap, 1), 1.6003706696232056)
    assert np.isclose(get_enrichment(amap, 3), 1.4221401510110887)

bed = pd.read_csv('tests/test.bed', sep='\t', names=['chr', 'start', 'end'])

def test_filter_bed():
    assert filter_bed(bed, 1000, 2000, ['chr1']).shape == (1, 3)

bedpe = pd.read_csv('tests/test.bedpe', sep='\t',
                    names=['chr1', 'start1', 'end1', 'chr2', 'start2', 'end2'])

def test_filter_bedpe():
    assert filter_bedpe(bedpe, 100000, 1000000, ['chr3']).shape == (1, 6)


def test_auto_read_bed():
    assert np.all(auto_read_bed('tests/test.bed') == bed)
    assert np.all(auto_read_bed('tests/test.bedpe') == bedpe)

