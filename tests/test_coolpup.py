#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 13:24:55 2019

@author: Ilya Flyamer
"""

from matplotlib.pyplot import ioff
from coolpuppy.coolpup import PileUpper, CoordCreator, pileup
from cooltools.lib import io, common
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
    regions = io.read_viewframe_from_file(
        op.join(request.fspath.dirname, "data/CN.mm9.toy_regions.bed"),
        verify_cooler=clr,
    )
    exp = io.read_expected_from_file(
        op.join(request.fspath.dirname, "data/CN.mm9.toy_expected.tsv"),
        expected_value_cols=["balanced.avg"],
        verify_view=regions,
        verify_cooler=clr,
    )

    features = bf.read_table(
        op.join(request.fspath.dirname, "data/toy_features.bed"), schema="bed"
    )
    cc = CoordCreator(
        features,
        1_000_000,
        features_format="bed",
        local=False,
        flank=2_000_000,
        mindist=0,
    )
    # Test with ooe=True
    pu = PileUpper(clr, cc, expected=exp, view_df=regions, ooe=True)
    pup = pu.pileupsByStrandWithControl()
    assert np.all(pup.sort_values("orientation")["n"] == [1, 3, 1, 1, 6])
    # Test with ooe=False
    pu = PileUpper(clr, cc, expected=exp, view_df=regions, ooe=False)
    pup = pu.pileupsByStrandWithControl()
    assert np.all(pup.sort_values("orientation")["n"] == [1, 3, 1, 1, 6])
    # No regions provided without expected
    pu = PileUpper(clr, cc, expected=False, ooe=False)
    pup = pu.pileupsByStrandWithControl()
    assert np.all(pup.sort_values("orientation")["n"] == [1, 3, 1, 1, 6])
    # Unbalanced
    pu = PileUpper(
        clr, cc, expected=False, ooe=False, clr_weight_name=None, coverage_norm=True
    )
    pup = pu.pileupsByStrandWithControl()
    assert np.all(pup.sort_values("orientation")["n"] == [1, 3, 1, 1, 6])
    # Test ignore_group_order
    pu = PileUpper(
        clr, cc, expected=False, ooe=False, control=False
    )
    pup = pu.pileupsByStrandWithControl(ignore_group_order=True)
    assert not pup[pup["orientation"] == "+-"].empty
    assert pup[pup["orientation"] == "-+"].empty
    assert np.all(pup.sort_values("orientation")["n"] == [1, 4, 1, 6])


def test_bystrand_pileups_with_controls(request):
    """
    Test the snipping on matrix:
    """
    # Read cool file and create regions out of it:
    clr = cooler.Cooler(op.join(request.fspath.dirname, "data/CN.mm9.1000kb.cool"))
    regions = bf.read_table(
        op.join(request.fspath.dirname, "data/CN.mm9.toy_regions.bed"), schema="bed4"
    )
    features = bf.read_table(
        op.join(request.fspath.dirname, "data/toy_features.bed"), schema="bed"
    )
    cc = CoordCreator(
        features,
        1_000_000,
        features_format="bed",
        local=False,
        flank=2_000_000,
        mindist=0,
    )
    pu = PileUpper(clr, cc, expected=False, view_df=regions, control=True)
    pup = pu.pileupsByStrandWithControl()
    assert np.all(pup.sort_values("orientation")["n"] == [1, 3, 1, 1, 6])


def test_bystrand_bydistance_pileups_with_controls(request):
    """
    Test the snipping on matrix:
    """
    # Read cool file and create regions out of it:
    clr = cooler.Cooler(op.join(request.fspath.dirname, "data/CN.mm9.1000kb.cool"))
    regions = bf.read_table(
        op.join(request.fspath.dirname, "data/CN.mm9.toy_regions.bed"), schema="bed4"
    )
    features = bf.read_table(
        op.join(request.fspath.dirname, "data/toy_features.bed"), schema="bed"
    )

    pup = pileup(
        clr=clr,
        features=features,
        features_format="bed",
        view_df=regions,
        mindist=0,
        flank=2_000_000,
        nshifts=1,
        by_strand=True,
        by_distance=True,
    )
    assert np.all(
        pup.sort_values(["orientation", "distance_band"])["n"] == [1, 2, 1, 1, 1, 6]
    )

    distance_bins = np.append([0], 50000 * 2 ** np.arange(30))
    pup = pileup(
        clr=clr,
        features=features,
        features_format="bed",
        view_df=regions,
        mindist=0,
        flank=2_000_000,
        nshifts=1,
        by_strand=True,
        by_distance=distance_bins,
    )
    assert np.all(
        pup.sort_values(["orientation", "distance_band"])["n"] == [1, 2, 1, 1, 1, 6]
    )
def test_pileups_with_stripes(request):
    """
    Test the snipping on matrix:
    """
    # Read cool file and create regions out of it:
    clr = cooler.Cooler(op.join(request.fspath.dirname, "data/CN.mm9.1000kb.cool"))
    regions = bf.read_table(
        op.join(request.fspath.dirname, "data/CN.mm9.toy_regions.bed"), schema="bed4"
    )
    features = bf.read_table(
        op.join(request.fspath.dirname, "data/toy_features.bed"), schema="bed"
    )
    cc = CoordCreator(
        features,
        1_000_000,
        features_format="bed",
        local=False,
        flank=2_000_000,
        mindist=0,
    )
    #Generate pileup with stripes
    pu = PileUpper(clr, cc, expected=False, view_df=regions, control=False, 
                   store_stripes=True, clr_weight_name=None, ignore_diags=0)
    pup = pu.pileupsWithControl()
    assert np.all(
        pup["coordinates"][0][0] == ['chr1', '102000000', '102500000', 'chr1', '105000000', '105500000']
    )
    assert np.all(
        pup["vertical_stripe"][0][0] == np.array([22015, 11287, 10852, 11376,  9998])
    )
