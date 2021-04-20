#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 13:24:55 2019

@author: Ilya Flyamer
"""

from coolpuppy import PileUpper, CoordCreator
import pandas as pd
import numpy as np
import cooler
import bioframe as bf
import os.path as op

def test_bystrand_pileups_with_expected(request):
    """
    Test the snipping on matrix:
    """
    # Read cool file and create regions out of it:
    clr = cooler.Cooler(op.join(request.fspath.dirname, "data/CN.mm9.1000kb.cool"))
    exp = pd.read_table(op.join(request.fspath.dirname, "data/CN.mm9.toy_expected.tsv"))
    regions = bf.read_table(
        op.join(request.fspath.dirname, "data/CN.mm9.toy_regions.bed"), schema="bed4"
    )
    features = bf.read_table(op.join(request.fspath.dirname, 'data/toy_features.bed'), schema='bed')
    cc = CoordCreator(features, 1_000_000, basetype='bed', local=False, pad=2_000_000, mindist=0)
    # Test with ooe=True
    pu = PileUpper(clr, cc, expected=exp, regions=regions, ooe=True)
    pup = pu.pileupsByStrandWithControl()
    assert np.all(pup.sort_values('orientation')['n'] == [1, 3, 1, 1])
    # Test with ooe=False
    pu = PileUpper(clr, cc, expected=exp, regions=regions, ooe=False)
    pup = pu.pileupsByStrandWithControl()
    assert np.all(pup.sort_values('orientation')['n'] == [1, 3, 1, 1])
    # No regions provided without expected
    pu = PileUpper(clr, cc, expected=False, ooe=False)
    pup = pu.pileupsByStrandWithControl()
    assert np.all(pup.sort_values('orientation')['n'] == [1, 3, 1, 1])

def test_bystrand_pileups_with_controls(request):
    """
    Test the snipping on matrix:
    """
    # Read cool file and create regions out of it:
    clr = cooler.Cooler(op.join(request.fspath.dirname, "data/CN.mm9.1000kb.cool"))
    regions = bf.read_table(
        op.join(request.fspath.dirname, "data/CN.mm9.toy_regions.bed"), schema="bed4"
    )
    features = bf.read_table(op.join(request.fspath.dirname, 'data/toy_features.bed'), schema='bed')
    cc = CoordCreator(features, 1_000_000, basetype='bed', local=False, pad=2_000_000, mindist=0)
    pu = PileUpper(clr, cc, expected=False, regions=regions, control=True)
    pup = pu.pileupsByStrandWithControl()
    assert np.all(pup.sort_values('orientation')['n'] == [1, 3, 1, 1])
    
def test_bystrand_bydistance_pileups_with_controls(request):
    """
    Test the snipping on matrix:
    """
    # Read cool file and create regions out of it:
    clr = cooler.Cooler(op.join(request.fspath.dirname, "data/CN.mm9.1000kb.cool"))
    regions = bf.read_table(
        op.join(request.fspath.dirname, "data/CN.mm9.toy_regions.bed"), schema="bed4"
    )
    features = bf.read_table(op.join(request.fspath.dirname, 'data/toy_features.bed'), schema='bed')
    cc = CoordCreator(features, 1_000_000, basetype='bed', local=False, pad=2_000_000, mindist=0)
    pu = PileUpper(clr, cc, expected=False, regions=regions, control=True)
    pup = pu.pileupsByStrandByDistanceWithControl()
    assert np.all(pup.sort_values(['orientation', 'distance_band'])['n'] == [1, 2, 1, 1, 1])