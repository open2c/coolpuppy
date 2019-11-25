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
import subprocess


amap = np.loadtxt('tests/Scc1-control.10000-10.0K_over_CH12_loops_Rao_10-shifts_dist_210000-inf_unbalanced_covnorm.np.txt')
amapTAD = np.loadtxt('tests/Scc1-control.10000-10.0K_over_CH12_TADs_Rao_10-shifts_local_rescaled_unbalanced_covnorm.np.txt')

def test_cornerCV():
#    print(cornerCV(amap), cornerCV(amap, 3))
    assert np.isclose(cornerCV(amap), 0.1458657084504096)
    assert np.isclose(cornerCV(amap, 3), 0.15836641893081616)

def test_get_enrichment():
#    print(get_enrichment(amap, 1), get_enrichment(amap, 3))
    assert np.isclose(get_enrichment(amap, 1), 2.0709878918199576)
    assert np.isclose(get_enrichment(amap, 3), 1.732941070459152)

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
    bed_mids = get_mids(bed, 1000, kind='bed')
    assert np.all(bed_mids['Bin'] == [1, 12, 2, 1])
    assert np.all(bed_mids['Pad'] == [500, 10000, 1000, 500])

    bedpe_mids = get_mids(bedpe, 1000, kind='bedpe')
    assert np.all(bedpe_mids['Bin1'] == [1, 21, 1, 101])
    assert np.all(bedpe_mids['Bin2'] == [3, 51, 1, 201])
    assert np.all(bedpe_mids['Pad1'] == [100, 1000, 150, 1000])
    assert np.all(bedpe_mids['Pad2'] == [100, 1000, 50, 1000])

amapbed2 = np.loadtxt('tests/Scc1-control.10000-10.0K_over_Bonev_CTCF+_vs_Bonev_CTCF-_10-shifts_dist_210000-inf_unbalanced_covnorm_subset-1000.np.txt')


def test___main__():
    # Loops
    subprocess.run("""coolpup.py tests/Scc1-control.10000.cool
                      tests/CH12_loops_Rao.bed --mindist 210000
                      --unbalanced --coverage_norm --outdir tests
                      --outname test_loop.txt --n_proc 2
                      --seed 0""".split())
    assert np.allclose(amap, np.loadtxt('tests/test_loop.txt'))

    # Bed2
    subprocess.run("""coolpup.py tests/Scc1-control.10000.cool
                      tests/Bonev_CTCF+.bed --bed2 tests/Bonev_CTCF-.bed
                      --mindist 210000 --subset 1000
                      --unbalanced --coverage_norm --outdir tests
                      --outname test_bed2.txt --n_proc 2
                      --seed 0""".split())
    assert np.allclose(amapbed2, np.loadtxt('tests/test_bed2.txt'))

    # TADs
    subprocess.run("""coolpup.py tests/Scc1-control.10000.cool
                      tests/CH12_TADs_Rao.bed --local --rescale
                      --unbalanced --coverage_norm --outdir tests
                      --outname test_tad.txt --n_proc 2
                      --seed 0""".split())
    assert np.allclose(amapTAD, np.loadtxt('tests/test_tad.txt'))

    # Numeric chroms
    subprocess.run("""coolpup.py tests/Scc1-control.10000.numeric_chroms.cool
                      tests/CH12_loops_Rao_numeric_chroms.bed
                      --mindist 210000
                      --unbalanced --coverage_norm --outdir tests
                      --outname test_loop_numeric.txt --n_proc 2
                      --seed 0""".split())
    assert np.allclose(amap, np.loadtxt('tests/test_loop_numeric.txt'))

#def test_pileupsWithControl():
#    loops = auto_read_bed('tests/CH12_loops_Rao.bed')
#    loopmids = get_mids(loops, resolution=10000, kind='bedpe')
#    pup = pileupsWithControl(loopmids, 'tests/Scc1-control.10000.cool',
#                             mindist=210000, balance=False, cov_norm=True,
#                             kind='bedpe', pad=10, seed=0)
#    assert np.allclose(pup, amap)
#
#    np.random.seed(0)
#    ctcf_left = auto_read_bed('tests/Bonev_CTCF+.bed').sample(1000)
#    ctcf_right = auto_read_bed('tests/Bonev_CTCF-.bed').sample(1000)
#
#    leftmids = get_mids(ctcf_left, resolution=10000, kind='bed')
#    rightmids = get_mids(ctcf_right, resolution=10000, kind='bed')
#
#    pup = pileupsWithControl(leftmids, 'tests/Scc1-control.10000.cool',
#                             mids2=rightmids, ordered_mids=True,
#                             mindist=210000, balance=False, cov_norm=True,
#                             kind='bed', pad=10, seed=0)
#    assert np.allclose(pup, amapbed2)
#
#    np.random.seed(0)
#    loops = auto_read_bed('tests/CH12_loops_Rao_numeric_chroms.bed')
#    loopmids = get_mids(loops, resolution=10000, kind='bedpe')
#    pup = pileupsWithControl(loopmids, 'tests/Scc1-control.10000.numeric_chroms.cool',
#                             mindist=210000, balance=False, cov_norm=True,
#                             kind='bedpe', pad=10, seed=0)
#    assert np.allclose(pup, amap)
