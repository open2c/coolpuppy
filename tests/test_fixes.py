# -*- coding: utf-8 -*-
"""Regression tests for bugs fixed after v1.1.0."""
import io as _io
import os.path as op
import shutil
import warnings
from functools import partial

import bioframe as bf
import cooler
import h5py
import matplotlib
import numpy as np
import pandas as pd
import pytest
from cooltools.lib import io as ctio

matplotlib.use("Agg")

from coolpuppy.coolpup import CoordCreator, PileUpper, pileup  # noqa: E402
from coolpuppy.lib import io, numutils, puputils  # noqa: E402
from coolpuppy.lib.puputils import accumulate_values, sum_pups  # noqa: E402

DATA = op.join(op.dirname(__file__), "data")


@pytest.fixture
def clr():
    return cooler.Cooler(op.join(DATA, "CN.mm9.1000kb.cool"))


@pytest.fixture
def regions(clr):
    return ctio.read_viewframe_from_file(
        op.join(DATA, "CN.mm9.toy_regions.bed"), verify_cooler=clr
    )


@pytest.fixture
def expected(clr, regions):
    return ctio.read_expected_from_file(
        op.join(DATA, "CN.mm9.toy_expected.tsv"),
        expected_value_cols=["balanced.avg"],
        verify_view=regions,
        verify_cooler=clr,
    )


@pytest.fixture
def features():
    return bf.read_table(op.join(DATA, "toy_features.bed"), schema="bed")


def random_features(n=300, seed=0, chroms=("chr1", "chr2", "chr3")):
    rng = np.random.default_rng(seed)
    chrom = rng.choice(list(chroms), n)
    start = rng.integers(5_000_000, 140_000_000, n)
    return (
        pd.DataFrame({"chrom": chrom, "start": start, "end": start + 50_000})
        .sort_values(["chrom", "start"])
        .reset_index(drop=True)
    )


def _pup(value, **extra):
    pup = {
        "data": np.full((3, 3), float(value)),
        "cov_start": np.zeros(3),
        "cov_end": np.zeros(3),
        "n": 1,
        "num": np.ones((3, 3), dtype=int),
        "horizontal_stripe": [],
        "vertical_stripe": [],
        "coordinates": [],
    }
    pup.update(extra)
    return pup


# --- sum_pups / accumulate_values (extra_sum_funcs, issue #146) ---


def test_sum_pups_with_extra_funcs_keeps_data():
    p1, p2 = _pup(1, score=[0.5]), _pup(2, score=[0.7])
    funcs = {"score": partial(accumulate_values, key="score")}
    out = sum_pups(p1, p2, extra_funcs=funcs)
    assert np.all(out["data"] == 3)
    assert out["n"] == 2
    assert out["score"] == [0.5, 0.7]
    # inputs are not modified
    assert p1["score"] == [0.5] and np.all(p1["data"] == 1)


def test_accumulate_values_tolerates_missing_key():
    # e.g. a region where no snippet passed filtering returns an empty pileup
    funcs = {"score": partial(accumulate_values, key="score")}
    out = sum_pups(_pup(1, score=[0.5]), _pup(0), extra_funcs=funcs)
    assert out["score"] == [0.5]
    out = sum_pups(_pup(0), _pup(1, score=[0.5]), extra_funcs=funcs)
    assert out["score"] == [0.5]


def _add_center_value(snip):
    snip["center_val"] = float(np.nan_to_num(snip["data"][1, 1]))
    return snip


def _center_val_pileup(clr, regions, feats):
    cc = CoordCreator(feats, clr.binsize, features_format="bed", local=True,
                      flank=1_000_000)
    pu = PileUpper(clr, cc, view_df=regions)
    pup = pu.pileupsWithControl(
        postprocess_func=_add_center_value,
        extra_sum_funcs={"center_val": partial(accumulate_values, key="center_val")},
    )
    return pup.loc[pup["group"] == "all"].iloc[0]


def test_extra_sum_funcs_across_regions(clr, regions):
    """Per-snippet values (as in the TAD score tutorial) are accumulated over all
    regions; previously only one region's pileup survived."""
    feats = random_features(80, seed=1, chroms=("chr1", "chr2"))
    feats = feats[(feats.start > 105_000_000) & (feats.end < 145_000_000)]
    assert feats.chrom.nunique() == 2
    row = _center_val_pileup(clr, regions, feats)
    assert row["n"] == len(feats)
    assert len(row["center_val"]) == len(feats)


def test_extra_sum_funcs_with_empty_region(clr, regions):
    """A region without any snippets used to fail with 'center_val not in dict2'
    (issue #146)."""
    feats = random_features(80, seed=1, chroms=("chr1", "chr2"))
    in_foo = (feats.chrom == "chr1") & (feats.start > 105_000_000) & (
        feats.end < 145_000_000
    )
    outside_bar = (feats.chrom == "chr2") & (feats.end < 90_000_000)
    feats = feats[in_foo | outside_bar]
    assert outside_bar.sum() > 0
    row = _center_val_pileup(clr, regions, feats)
    assert row["n"] == in_foo.sum()
    assert len(row["center_val"]) == in_foo.sum()


# --- domain score with rescale_flank != 1 (issue #161) ---


def _domain_map(size, flank, central=4.0, side=1.0):
    fp = int(round(size * flank / (2 * flank + 1)))
    amap = np.full((size, size), side)
    amap[fp:-fp, fp:-fp] = central
    return amap


@pytest.mark.parametrize("flank", [0.25, 0.5, 1, 2])
def test_domain_score_fractional_flank(flank):
    amap = _domain_map(99, flank)
    assert numutils.get_domain_score(amap, flank) == pytest.approx(4.0)
    assert numutils.get_local_enrichment(amap, flank) == pytest.approx(4.0)


def test_domain_score_flank1_unchanged():
    rng = np.random.default_rng(0)
    amap = rng.random((99, 99)) + 1
    c = 33
    old = np.nansum(amap[c:-c, c:-c]) / (
        np.nansum(amap[:c, c:-c]) + np.nansum(amap[c:-c, -c:])
    ) * 2
    assert numutils.get_domain_score(amap, 1) == pytest.approx(old)


# --- pair generation ---


@pytest.mark.parametrize(
    "mindist,maxdist", [(0, None), (2_000_000, 20_000_000), ("auto", 10_000_000)]
)
def test_cis_combinations_match_bruteforce(clr, mindist, maxdist):
    feats = random_features(300)
    cc = CoordCreator(
        feats, clr.binsize, features_format="bed", flank=1_000_000,
        mindist=mindist, maxdist=maxdist, nshifts=0,
    )
    got = set()
    for chrom in cc.final_chroms:
        for row in cc.get_combinations(cc.filter_func_chrom(chrom), chunksize=500):
            got.add((row["chrom1"], row["start1"], row["chrom2"], row["start2"]))
    expected_pairs = set()
    lo = cc.mindist
    hi = np.inf if maxdist is None else maxdist
    for chrom, g in cc.intervals.groupby("chrom"):
        g = g.reset_index(drop=True)
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                d = abs(g.loc[j, "center"] - g.loc[i, "center"])
                if lo <= d <= hi:
                    expected_pairs.add((chrom, g.loc[i, "start"], chrom, g.loc[j, "start"]))
    assert got == expected_pairs
    assert len(got) > 0


def test_trans_combinations_count(clr):
    feats = random_features(40, chroms=("chr1", "chr2"))
    cc = CoordCreator(feats, clr.binsize, features_format="bed", flank=1_000_000,
                      trans=True, nshifts=0)
    view = cc.intervals
    r1 = ("chr1", 0, int(clr.chromsizes["chr1"]))
    r2 = ("chr2", 0, int(clr.chromsizes["chr2"]))
    rows = list(cc.get_combinations(cc.filter_func_region(r1), cc.filter_func_region(r2),
                                    chunksize=7))
    n1, n2 = (view.chrom == "chr1").sum(), (view.chrom == "chr2").sum()
    assert len(rows) == n1 * n2


def test_bedpe2bed_center():
    df = pd.DataFrame({"chrom1": ["chr1"], "start1": [100], "end1": [200],
                       "chrom2": ["chr1"], "start2": [1000], "end2": [1100]})
    cc = CoordCreator(df, 100, features_format="bedpe", nshifts=0, mindist=0, flank=100)
    out = cc.bedpe2bed(df, ends=False, how="center")
    assert out.iloc[0].tolist() == ["chr1", 150, 1050]


# --- pileup() arguments ---


def test_by_distance_accepts_list(clr, regions, features):
    kw = dict(clr=clr, features=features, features_format="bed", view_df=regions,
              mindist=0, flank=2_000_000)
    edges = [0, 1_000_000, 3_000_000, 10_000_000]
    a = pileup(by_distance=edges, **kw)
    b = pileup(by_distance=np.array(edges), **kw)
    assert a["n"].tolist() == b["n"].tolist()


def test_nproc_zero_means_all_cores(clr, regions, features):
    pup = pileup(clr, features, features_format="bed", view_df=regions, mindist=0,
                 flank=2_000_000, nproc=0)
    assert pup.loc[pup["group"] == "all", "n"].iloc[0] > 0


def test_seeded_controls_reproducible_across_nproc(clr):
    feats = random_features(120, chroms=("chr1", "chr2", "chr3"))
    kw = dict(features_format="bed", flank=2_000_000, maxdist=30_000_000, nshifts=3,
              seed=42)
    a = pileup(clr, feats, nproc=1, **kw)["data"].iloc[0]
    b = pileup(clr, feats, nproc=1, **kw)["data"].iloc[0]
    c = pileup(clr, feats, nproc=2, **kw)["data"].iloc[0]
    np.testing.assert_allclose(a, b, equal_nan=True)
    np.testing.assert_allclose(a, c, equal_nan=True)


def test_coverage_norm_computes_coverage(clr, tmp_path):
    path = tmp_path / "copy.cool"
    shutil.copy(clr.filename, path)
    with h5py.File(path, "a") as f:  # the test file ships with coverage stored
        for col in ["cov_tot_raw", "cov_cis_raw"]:
            if col in f["bins"]:
                del f["bins"][col]
    c = cooler.Cooler(str(path))
    assert "cov_tot_raw" not in c.bins().columns
    feats = random_features(50)
    pup = pileup(c, feats, features_format="bed", flank=2_000_000, maxdist=20_000_000,
                 clr_weight_name=None, coverage_norm=True)
    assert "cov_tot_raw" in cooler.Cooler(str(path)).bins().columns
    assert np.isfinite(pup["data"].iloc[0]).any()


def test_expected_not_ooe_masks_invalid_pixels(clr, regions, expected, features):
    cc = CoordCreator(features, clr.binsize, features_format="bed", local=True,
                      flank=3_000_000)
    pu = PileUpper(clr, cc, expected=expected, view_df=regions, ooe=False, ignore_diags=2)
    region = pu.view_df.index[0]
    ff = cc.filter_func_region(region=pu.view_df.loc[region])
    snips = list(pu._stream_snips(cc.pos_stream(ff), region))
    assert len(snips) > 0
    for roi, exp in zip(snips[::2], snips[1::2]):
        assert roi["kind"] == "ROI" and exp["kind"] == "control"
        assert np.isnan(roi["data"]).any()  # diagonals are ignored
        np.testing.assert_array_equal(np.isnan(roi["data"]), np.isnan(exp["data"]))


def test_store_stripes_with_control(clr, regions, features):
    # Stripes are divided by the control's central stripes
    pup = pileup(clr, features, features_format="bed", view_df=regions, mindist=0,
                 flank=2_000_000, store_stripes=True, nshifts=2, seed=0)
    ctrl = pileup(clr, features, features_format="bed", view_df=regions, mindist=0,
                  flank=2_000_000, store_stripes=True)
    assert len(pup["horizontal_stripe"].iloc[0]) == len(ctrl["horizontal_stripe"].iloc[0])
    assert np.isfinite(np.concatenate(list(pup["horizontal_stripe"].iloc[0]))).any()


# --- divide_pups (issue #163) ---


def test_divide_pups_multiple_groups(clr, regions, features):
    kw = dict(features_format="bed", view_df=regions, mindist=0, flank=2_000_000,
              by_strand=True, store_stripes=True)
    p1 = pileup(clr, features, **kw)
    p2 = pileup(clr, features, **kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        div = puputils.divide_pups(p1, p2)
    assert len(div) == len(p1)
    for arr in div["data"]:
        finite = np.isfinite(arr)
        np.testing.assert_allclose(arr[finite], 1)
    assert "vertical_stripe" in div.columns



def test_divide_pups_matches_by_separation(clr, regions):
    # ++ and -- pileups split by distance: their "group" values differ, so the rows
    # have to be matched by distance band
    feats = random_features(n=200, seed=3, chroms=("chr1", "chr2"))
    feats["strand"] = np.random.default_rng(3).choice(["+", "-"], len(feats))
    pup = pileup(
        clr, feats, features_format="bed", view_df=regions, flank=2_000_000,
        by_strand=True, by_distance=[0, 5_000_000, 10_000_000], mindist=0, nshifts=0,
    )
    drop = ["strand1", "strand2", "orientation"]
    plus = pup.loc[pup["orientation"] == "++"].drop(columns=drop)
    minus = pup.loc[pup["orientation"] == "--"].drop(columns=drop)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        div = puputils.divide_pups(plus, minus)
    bands = sorted(set(plus["separation"]) & set(minus["separation"]))
    assert len(bands) >= 2
    assert sorted(div["separation"]) == bands
    for band in bands:
        expect = (
            plus.loc[plus["separation"] == band, "data"].iloc[0]
            / minus.loc[minus["separation"] == band, "data"].iloc[0]
        )
        got = div.loc[div["separation"] == band, "data"].iloc[0]
        np.testing.assert_allclose(got, expect, equal_nan=True)

def test_divide_pups_single_row_clrs(clr, regions, features):
    kw = dict(features_format="bed", view_df=regions, mindist=0, flank=2_000_000)
    p = pileup(clr, features, **kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        div = puputils.divide_pups(p, p)
    assert "\n" not in div["clrs"].iloc[0]


# --- IO (issue #166) ---


def test_io_roundtrip_long_chrom_names(clr, regions, features, tmp_path):
    pup = pileup(clr, features, features_format="bed", view_df=regions, mindist=0,
                 flank=2_000_000, store_stripes=True)
    pup["coordinates"] = [
        np.where(c == "chr1", "chrUn_KI270742v1", c) for c in pup["coordinates"]
    ]
    out = tmp_path / "pup.clpy"
    io.save_pileup_df(str(out), pup)
    assert [p.name for p in tmp_path.iterdir()] == ["pup.clpy"]  # no temp file left
    loaded = io.load_pileup_df(str(out))
    assert loaded["coordinates"].iloc[0][0][0] == "chrUn_KI270742v1"
    np.testing.assert_allclose(loaded["data"].iloc[0], pup["data"].iloc[0], equal_nan=True)


def test_io_load_without_attrs_warns(clr, regions, features, tmp_path, caplog):
    pup = pileup(clr, features, features_format="bed", view_df=regions, mindist=0,
                 flank=2_000_000)
    out = str(tmp_path / "pup.clpy")
    io.save_pileup_df(out, pup)
    with h5py.File(out, "a") as f:
        del f["attrs"]
    with caplog.at_level("WARNING", logger="coolpuppy"):
        loaded = io.load_pileup_df(out)
    assert "no metadata" in caplog.text
    assert len(loaded) == len(pup)


# --- plotting and CLI ---


def test_plot_default_score(clr, regions, features):
    from coolpuppy import plotpup

    pup = pileup(clr, features, features_format="bed", view_df=regions, mindist=0,
                 flank=4_000_000)
    fg = plotpup.plot(pup)  # default score="score" used to raise KeyError
    assert fg is not None


def test_plotpup_cli_args():
    from coolpuppy.plotpuppy_CLI import parse_args_plotpuppy

    args = parse_args_plotpuppy().parse_args(
        ["--input_pups", "a.clpy", "--col_order", "a,b c", "--query", "n > 1"]
    )
    assert args.col_order == ["a", "b", "c"]
    assert args.query == ["n > 1"]
    args = parse_args_plotpuppy().parse_args(["--input_pups", "a.clpy"])
    assert args.query is None


@pytest.mark.parametrize(
    "ext,content,expected_format",
    [
        (".bed", "chr1\t1\t2\n", "bed"),
        (".bedpe", "chr1\t1\t2\tchr1\t5\t6\n", "bedpe"),
        (".tsv", "chr1\t1\t2\tchr1\t5\t6\n", "bedpe"),
        (".txt", "chr1\t1\t2\tname\t0\t+\n", "bed"),
    ],
)
def test_cli_guess_features_format(ext, content, expected_format):
    from coolpuppy.CLI import _guess_features_format

    buf = _io.StringIO(content)
    assert _guess_features_format(ext, buf, None, len(content.split("\t"))) == expected_format
    assert buf.tell() == 0


# --- out-of-bounds regions are reported once per pileup ---


def test_skipped_regions_single_warning(clr, regions, caplog):
    # Two features per region lie within the flank of the region edges, so they
    # are skipped; one in the middle of each region is kept
    features = pd.DataFrame(
        {
            "chrom": ["chr1"] * 3 + ["chr2"] * 3,
            "start": [100_500_000, 125_000_000, 149_000_000] * 2,
        }
    )
    features["end"] = features["start"] + 10_000
    with caplog.at_level("INFO", logger="coolpuppy"):
        pup = pileup(
            clr, features, features_format="bed", view_df=regions, local=True,
            flank=5_000_000, nshifts=0, nproc=1,
        )
    skipped = [r for r in caplog.records if "extend beyond" in r.getMessage()]
    assert len(skipped) == 1
    assert skipped[0].getMessage().startswith("4 regions of interest (66.7%)")
    # A large fraction is skipped here, so it's a warning
    assert skipped[0].levelname == "WARNING"
    assert pup.loc[0, "n"] == 2
    assert not any(str(c).startswith("_") for c in pup.columns)


def test_few_skipped_regions_only_info(clr, regions, caplog):
    # One feature at a region edge among many in the middle: not worth a warning
    starts = [100_500_000] + list(range(110_000_000, 140_000_000, 1_000_000))
    features = pd.DataFrame({"chrom": "chr1", "start": starts})
    features["end"] = features["start"] + 10_000
    with caplog.at_level("INFO", logger="coolpuppy"):
        pileup(
            clr, features, features_format="bed", view_df=regions, local=True,
            flank=5_000_000, nshifts=0, nproc=1,
        )
    skipped = [r for r in caplog.records if "extend beyond" in r.getMessage()]
    assert len(skipped) == 1
    assert skipped[0].getMessage().startswith("1 regions of interest")
    assert skipped[0].levelname == "INFO"
