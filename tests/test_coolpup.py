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
amapTAD = np.loadtxt('tests/Scc1-control.10000-10.0K_over_CH12_TADs_Rao_10-shifts_local_rescaled_unbalanced_covnorm.np.txt')

def test_cornerCV():
    assert np.isclose(cornerCV(amap), 0.0752115185866688)
    assert np.isclose(cornerCV(amap, 3), 0.07433758192350227)

def test_get_enrichment():
    assert np.isclose(get_enrichment(amap, 1), 1.6307401942981528)
    assert np.isclose(get_enrichment(amap, 3), 1.4057840415780747)

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

def test_get_mids():
    bed_mids = get_mids(bed, 1000, bed=True)
    assert np.all(bed_mids['Bin'] == [1, 12, 2, 1])
    assert np.all(bed_mids['Pad'] == [500, 10000, 1000, 500])

    bedpe_mids = get_mids(bedpe, 1000, bed=False)
    assert np.all(bedpe_mids['Bin1'] == [1, 21, 1, 101])
    assert np.all(bedpe_mids['Bin2'] == [3, 51, 1, 201])
    assert np.all(bedpe_mids['Pad1'] == [100, 1000, 150, 1000])
    assert np.all(bedpe_mids['Pad2'] == [100, 1000, 50, 1000])


def test_pileupsWithControl():
    np.random.seed(0)
    loops = auto_read_bed('tests/CH12_loops_Rao.bed')
    loopmids = get_mids(loops, resolution=10000, bed=False)
    pup = pileupsWithControl(loopmids, 'tests/Scc1-control.10000.cool',
                             mindist=100000, balance=False, cov_norm=True,
                             bed=False, pad=10)
    assert np.allclose(pup, amap)