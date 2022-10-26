# -*- coding: utf-8 -*-
import warnings

import os
from multiprocessing import Pool
from functools import partial, reduce
import logging
import itertools

from natsort import natsorted
from more_itertools import collapse

import numpy as np
import pandas as pd
import bioframe

import cooler
from cooltools import numutils as ctutils
from cooltools.lib import common, checks
from cooltools.api import snipping, coverage

from .lib import numutils
from .lib.puputils import _add_snip, group_by_region, norm_coverage, sum_pups

logger = logging.getLogger("coolpuppy")


def bin_distance_intervals(intervals, band_edges="default"):
    """

    Parameters
    ----------
    intervals : pd.DataFrame
        Dataframe containing intervals with any annotations.
        Has to have a 'distance' column
    band_edges : list or array-like, or "default", optional
        Edges of distance bands used to split the intervals into groups.
        Default is np.append([0], 50000 * 2 ** np.arange(30))

    Returns
    -------
    snip : pd.DataFrame
        The same dataframe with added ['distance_band'] annotation.

    """
    if band_edges == "default":
        band_edges = np.append([0], 50000 * 2 ** np.arange(30))
    edge_ids = np.searchsorted(band_edges, intervals["distance"], side="right")
    bands = [tuple(band_edges[i - 1 : i + 1]) for i in edge_ids]
    intervals["distance_band"] = bands
    return intervals


def assign_groups(intervals, groupby=[]):
    """Assign groups to rows based on a list of columns

    Parameters
    ----------
    intervals : pd.DataFrame
        Dataframe containing intervals with any annotations.
    groupby : list, optional
        List of columns to use to assign a group. The default is [].

    Returns
    -------
    intervals : pd.DataFrame
        Adds a "group" column with the annotation based on `groupby`. If groupby is
        empty, assigns "all" to all rows.

    """
    if not groupby:
        intervals["group"] = "all"
    else:
        intervals["group"] = list(intervals[groupby].values)
    return intervals


def expand(intervals, flank, resolution, rescale_flank=None):
    intervals = intervals.copy()
    if rescale_flank is None:
        intervals["exp_start"] = (
            np.floor(intervals["center"] / resolution) * resolution - flank
        )
        intervals["exp_end"] = (
            np.floor(intervals["center"] / resolution + 1) * resolution + flank
        )
    else:
        intervals[["exp_start", "exp_end"]] = bioframe.expand(
            intervals, scale=2 * rescale_flank + 1
        )[["start", "end"]]
    return intervals


def expand2D(intervals, flank, resolution, rescale_flank=None):
    if rescale_flank is None:
        intervals["exp_start1"] = (
            np.floor(intervals["center1"] // resolution) * resolution - flank
        )
        intervals["exp_end1"] = (
            np.floor(intervals["center1"] / resolution + 1) * resolution + flank
        )
        intervals["exp_start2"] = (
            np.floor(intervals["center2"] // resolution) * resolution - flank
        )
        intervals["exp_end2"] = (
            np.floor(intervals["center2"] / resolution + 1) * resolution + flank
        )
    else:
        intervals[["exp_start1", "exp_end1"]] = bioframe.expand(
            intervals, scale=2 * rescale_flank + 1, cols=["chrom1", "start1", "end1"]
        )[["start1", "end1"]]
        intervals[["exp_start2", "exp_end2"]] = bioframe.expand(
            intervals, scale=2 * rescale_flank + 1, cols=["chrom2", "start2", "end2"]
        )[["start2", "end2"]]
    return intervals


class CoordCreator:
    def __init__(
        self,
        features,
        resolution,
        *,
        features_format="auto",
        flank=100000,
        rescale_flank=None,
        chroms="all",
        minshift=10**5,
        maxshift=10**6,
        nshifts=10,
        mindist="auto",
        maxdist=None,
        local=False,
        subset=0,
        trans=False,
        seed=None,
    ):
        """Generator of coordinate pairs for pileups.

        Parameters
        ----------
        features : DataFrame
            A bed- or bedpe-style file with coordinates.
        resolution : int, optional
            Data resolution.
        features_format : str, optional
            Format of the features. Options:
                bed: chrom, start, end
                bedpe: chrom1, start1, end1, chrom2, start2, end2
                auto (default): determined from the columns in the DataFrame
        flank : int, optional
            Padding around the central bin, in bp. For example, with 5000 bp resolution
            and 100000 flank, final pileup is 205000×205000 bp.
            The default is 100000.
        rescale_flank : float, optional
            Fraction of ROI size added on each end when extracting snippets, if rescale.
            The default is None. If specified, overrides flank.
        chroms : str or list, optional
            Which chromosomes to use for pileups. Has to be in a list even for a
            single chromosome, e.g. ['chr1'].
            The default is "all"
        minshift : int, optional
            Minimal shift applied when generating random controls, in bp.
            The default is 10 ** 5.
        maxshift : int, optional
            Maximal shift applied when generating random controls, in bp.
            The default is 10 ** 6.
        nshifts : int, optional
            How many shifts to generate per region of interest. Does not take chromosome
            boundaries into account
            The default is 10.
        mindist : int, optional
            Shortest interactions to consider. Uses midpoints of regions of interest.
            "auto" selects it to avoid the two shortest diagonals of the matrix, i.e.
            2 * flank + 2 * resolution
            The default is "auto".
        maxdist : int, optional
            Longest interactions to consider.
            The default is None.
        local : bool, optional
            Whether to generate local coordinates, i.e. on-diagonal.
            The default is False.
        subset : int, optional
            What subset of the coordinate files to use. 0 or negative to use all.
            The default is 0.
        seed : int, optional
            Seed for np.random to make it reproducible.
            The default is None.
        trans : bool, optional
            Whether to generate inter-chromosomal (trans) pileups.
            The default is False

        Returns
        -------
        Object that generates coordinates for pileups required for PileUpper.

        """
        self.intervals = features.copy()
        self.resolution = resolution
        self.features_format = features_format
        self.flank = flank
        self.rescale_flank = rescale_flank
        self.chroms = chroms
        self.minshift = minshift
        self.maxshift = maxshift
        self.nshifts = nshifts
        self.trans = trans
        if mindist == "auto":
            self.mindist = 2 * self.flank + 2 * self.resolution
        else:
            self.mindist = mindist
            if self.trans:
                warnings.warn("Ignoring mindist when using trans", stacklevel=2)
        if maxdist is None:
            self.maxdist = np.inf
        else:
            self.maxdist = maxdist
            if self.trans:
                warnings.warn("Ignoring maxdist when using trans", stacklevel=2)
        self.local = local
        self.subset = subset
        self.seed = seed
        self.process()

    def process(self):
        if self.features_format is None or self.features_format == "auto":
            if all(
                [
                    name in self.intervals.columns
                    for name in ["chrom1", "start1", "end1", "chrom2", "start2", "end2"]
                ]
            ):
                self.kind = "bedpe"
            elif all(
                [name in self.intervals.columns for name in ["chrom", "start", "end"]]
            ):
                self.kind = "bed"
            else:
                raise ValueError(
                    "Can't determine kind of input, please specify and/or name columns correctly:"
                    "'chrom1', 'start1', 'end1', 'chrom2', 'start2', 'end2' for bedpe kind"
                    "'chrom', 'start', 'end' for bed kind"
                )
        else:
            self.kind = self.features_format

        if self.subset > 0:
            self.intervals = self._subset(self.intervals)

        if self.kind == "bed":
            assert all(
                [name in self.intervals.columns for name in ["chrom", "start", "end"]]
            ), "Column names must include chrom, start, and end"
            self.intervals["chrom"] = self.intervals["chrom"].astype(str)
            self.intervals["center"] = (
                self.intervals["start"] + self.intervals["end"]
            ) / 2
            self.intervals = expand(
                self.intervals, self.flank, self.resolution, self.rescale_flank
            )
        else:
            assert all(
                [
                    name in self.intervals.columns
                    for name in ["chrom1", "start1", "end1", "chrom2", "start2", "end2"]
                ]
            ), "Column names must include chrom1, start1, end1, chrom2, start2, and end2"
            self.intervals[["chrom1", "chrom2"]] = self.intervals[
                ["chrom1", "chrom2"]
            ].astype(str)
            self.intervals["center1"] = (
                self.intervals["start1"] + self.intervals["end1"]
            ) / 2
            self.intervals["center2"] = (
                self.intervals["start2"] + self.intervals["end2"]
            ) / 2
            self.intervals["distance"] = (
                self.intervals["center2"] - self.intervals["center1"]
            )
            self.intervals = self.intervals[
                (self.mindist <= self.intervals["distance"].abs())
                & (self.intervals["distance"].abs() <= self.maxdist)
            ]
            self.intervals = self.intervals.reset_index(drop=True)
            self.intervals = expand2D(
                self.intervals, self.flank, self.resolution, self.rescale_flank
            )

        if self.intervals.shape[0] == 0:
            warnings.warn(
                "No regions in features (maybe all below mindist?),"
                " returning empty output",
                stacklevel=2,
            )
            self.pos_stream = self.empty_stream
            self.final_chroms = []
            return

        if self.nshifts > 0 and self.kind == "bedpe":
            self.intervals = self._control_regions(self.intervals)

        if self.kind == "bed":
            basechroms = set(self.intervals["chrom"])
        else:
            if self.local:
                raise ValueError("Can't make local with both sides of loops defined")
            if self.trans:
                basechroms = set(
                    self.intervals["chrom1"].unique().tolist()
                    + self.intervals["chrom2"].unique().tolist()
                )
            else:
                basechroms = set(self.intervals["chrom1"]).intersection(
                    set(self.intervals["chrom2"])
                )
        self.basechroms = natsorted(list(basechroms))
        if self.chroms == "all":
            self.final_chroms = natsorted(list(basechroms))
        else:
            self.final_chroms = natsorted(
                list(set(self.chroms).intersection(set(self.basechroms)))
            )

        if len(self.final_chroms) == 0:
            raise ValueError(
                """No chromosomes are in common between the coordinate
                   file and the cooler file. Are they in the same
                   format, e.g. starting with "chr"?
                   """
            )

        self.intervals = self._binnify(self.intervals)

        if self.kind == "bed":
            dups = self.intervals.duplicated(subset=["stBin", "endBin"])
        else:
            dups = self.intervals.duplicated(
                subset=["stBin1", "endBin1", "stBin2", "endBin2"]
            )
        if dups.any():
            logger.debug(
                f"{'{:.2f}'.format(dups.mean()*100)}% of intervals fall within the same bin as another interval. These are all included in the pileup."
            )

        if self.trans & self.local:
            raise ValueError("Cannot do local with trans=True")

        if self.kind == "bed":
            self.pos_stream = self.get_combinations
        else:
            self.pos_stream = self.get_intervals_stream

    def _control_regions(self, intervals2d, nshifts=0):
        if nshifts > 0:
            control_intervals = pd.concat([intervals2d] * nshifts).reset_index(
                drop=True
            )
            shift = np.random.randint(
                self.minshift, self.maxshift, control_intervals.shape[0]
            )
            sign = np.random.choice([-1, 1], control_intervals.shape[0])
            shift *= sign
            if (
                self.trans
            ):  # The two trans coordinates can shift in different directions
                shift2 = np.random.randint(
                    self.minshift, self.maxshift, control_intervals.shape[0]
                )
                sign2 = np.random.choice([-1, 1], control_intervals.shape[0])
                shift2 = shift2 * sign2
                control_intervals[["exp_start1", "exp_end1", "center1"]] = (
                    control_intervals[
                        [
                            "exp_start1",
                            "exp_end1",
                            "center1",
                        ]
                    ]
                    + shift[:, np.newaxis]
                )
                control_intervals[["exp_start2", "exp_end2", "center2"]] = (
                    control_intervals[["exp_start2", "exp_end2", "center2"]]
                    + shift2[:, np.newaxis]
                )
            else:
                control_intervals[
                    [
                        "exp_start1",
                        "exp_end1",
                        "center1",
                        "exp_start2",
                        "exp_end2",
                        "center2",
                    ]
                ] = (
                    control_intervals[
                        [
                            "exp_start1",
                            "exp_end1",
                            "center1",
                            "exp_start2",
                            "exp_end2",
                            "center2",
                        ]
                    ]
                    + shift[:, np.newaxis]
                )
            control_intervals[
                ["stBin1", "endBin1", "stBin2", "endBin2"]
            ] = control_intervals[
                ["stBin1", "endBin1", "stBin2", "endBin2"]
            ] + np.round(
                shift[:, np.newaxis] / self.resolution
            ).astype(
                int
            )
            intervals2d["kind"] = "ROI"
            control_intervals["kind"] = "control"
            intervals2d = pd.concat([intervals2d, control_intervals]).reset_index(
                drop=True
            )
        else:
            intervals2d["kind"] = "ROI"
        return intervals2d

    def _subset(self, df):
        if self.seed is not None:
            np.random.seed(self.seed)
        if self.subset > 0 and self.subset < len(df):
            return df.sample(self.subset)
        else:
            return df

    def bedpe2bed(self, df, ends=True, how="center"):
        # If ends==True, will combine all coordinates from both loop ends into one bed-style df
        # Else, will convert to long intervals covering the whole loop, for e.g. rescaled averaging
        # how: center - take center of loop bases; inner or outer
        if ends:
            df1 = df[["chrom1", "start1", "end1"]]
            df1.columns = ["chrom", "start", "end"]
            df2 = df[["chrom2", "start2", "end2"]]
            df2.columns = ["chrom", "start", "end"]
            return (
                pd.concat([df1, df2])
                .sort_values(["chrom", "start", "end"])
                .reset_index(drop=True)
            )

        if how == "center":
            df["start"] = np.mean(df["start1"], df["end1"], axis=0)
            df["end"] = np.mean(df["start2"], df["end2"], axis=0)
        elif how == "outer":
            df = df[["chrom1", "start1", "end2"]]
            df.columns = ["chrom", "start", "end"]
        elif how == "inner":
            df = df[["chrom1", "end1", "start2"]]
            df.columns = ["chrom", "start", "end"]
        return df

    def _binnify(self, intervals):
        if self.kind == "bed":
            intervals = intervals.sort_values(["chrom", "start"])
            intervals["stBin"] = np.floor(
                intervals["exp_start"] / self.resolution
            ).astype(int)
            intervals["endBin"] = np.ceil(
                intervals["exp_end"] / self.resolution
            ).astype(int)
            intervals[["exp_start", "exp_end"]] = (
                intervals[["stBin", "endBin"]] * self.resolution
            )
        elif self.kind == "bedpe":
            intervals = intervals.sort_values(["chrom1", "chrom2", "start1", "start2"])
            intervals["stBin1"] = np.floor(
                intervals["exp_start1"] / self.resolution
            ).astype(int)
            intervals["endBin1"] = np.ceil(
                intervals["exp_end1"] / self.resolution
            ).astype(int)
            intervals["stBin2"] = np.floor(
                intervals["exp_start2"] / self.resolution
            ).astype(int)
            intervals["endBin2"] = np.ceil(
                intervals["exp_end2"] / self.resolution
            ).astype(int)
            intervals[["exp_start1", "exp_end1"]] = (
                intervals[["stBin1", "endBin1"]] * self.resolution
            )
            intervals[["exp_start2", "exp_end2"]] = (
                intervals[["stBin2", "endBin2"]] * self.resolution
            )
        else:
            raise ValueError(
                """
                kind can only be "bed" or "bedpe"
                """
            )
        return intervals

    def filter_func_all(self, intervals):
        return intervals

    def _filter_func_chrom(self, intervals, chrom):
        return intervals[intervals["chrom"] == chrom].reset_index(drop=True)

    def _filter_func_pairs_chrom(self, intervals, chrom):
        return intervals[
            (intervals["chrom1"] == chrom) & (intervals["chrom2"] == chrom)
        ].reset_index(drop=True)

    def filter_func_chrom(self, chrom):
        if self.kind == "bed":
            return partial(self._filter_func_chrom, chrom=chrom)
        else:
            return partial(self._filter_func_pairs_chrom, chrom=chrom)

    def _filter_func_region(self, intervals, region):
        chrom, start, end = region
        return intervals[
            (intervals["chrom"] == chrom)
            & (intervals["start"] >= start)
            & (intervals["end"] < end)
        ].reset_index(drop=True)

    def _filter_func_pairs_region(self, intervals, region):
        chrom, start, end = region
        return intervals[
            (intervals["chrom1"] == chrom)
            & (intervals["chrom2"] == chrom)
            & (intervals["start1"] >= start)
            & (intervals["end1"] < end)
            & (intervals["start2"] >= start)
            & (intervals["end2"] < end)
        ].reset_index(drop=True)

    def _filter_func_trans_pairs(self, intervals, region1, region2):
        chrom1, start1, end1 = region1
        chrom2, start2, end2 = region2
        return pd.concat(
            [
                intervals[
                    (intervals["chrom1"] == chrom1)
                    & (intervals["chrom2"] == chrom2)
                    & (intervals["start1"] >= start1)
                    & (intervals["end1"] < end1)
                    & (intervals["start2"] >= start2)
                    & (intervals["end2"] < end2)
                ].reset_index(drop=True),
                intervals[
                    (intervals["chrom2"] == chrom1)
                    & (intervals["chrom1"] == chrom2)
                    & (intervals["start2"] >= start1)
                    & (intervals["end2"] < end1)
                    & (intervals["start1"] >= start2)
                    & (intervals["end1"] < end2)
                ].reset_index(drop=True),
            ]
        )

    def filter_func_trans_pairs(self, region1, region2):
        return partial(self._filter_func_trans_pairs, region1=region1, region2=region2)

    def filter_func_region(self, region):
        if self.kind == "bed":
            return partial(self._filter_func_region, region=region)
        else:
            return partial(self._filter_func_pairs_region, region=region)

    def get_combinations(
        self,
        filter_func1,
        filter_func2=None,
        intervals=None,
        control=False,
        groupby=[],
        modify_2Dintervals_func=None,
    ):

        if intervals is None:
            intervals = self.intervals
        if not len(intervals) >= 1:
            logger.debug("Empty selection")
            yield None

        intervals_left = filter_func1(intervals)
        if filter_func2 is None:
            intervals_right = intervals_left
        else:
            intervals_right = filter_func2(intervals)

        if self.local:
            merged = pd.merge(
                intervals_left,
                intervals_left,
                left_index=True,
                right_index=True,
                suffixes=["1", "2"],
            )
            merged["coordinates"] = merged.apply(
                lambda x: ".".join(
                    x[["chrom1", "start1", "end1", "chrom2", "start2", "end2"]].astype(
                        str
                    )
                ),
                axis=1,
            )
            merged = self._control_regions(merged, self.nshifts * control)
            if modify_2Dintervals_func is not None:
                merged = modify_2Dintervals_func(merged)
            merged = assign_groups(merged, groupby=groupby)
            merged = merged.reindex(
                columns=list(merged.columns)
                + [
                    "data",
                    "cov_start",
                    "cov_end",
                    "horizontal_stripe",
                    "vertical_stripe",
                ]
            )
            for row in merged.to_dict(orient="records"):
                yield row

        else:
            intervals_left = intervals_left.rename(
                columns=lambda x: x + "1"
            ).reset_index(drop=True)
            intervals_right = intervals_right.rename(
                columns=lambda x: x + "2"
            ).reset_index(drop=True)

            if self.trans:
                for x, y in itertools.product(
                    intervals_left.index, intervals_right.index
                ):
                    combinations = pd.concat(
                        [
                            intervals_left.iloc[[x]].reset_index(drop=True),
                            intervals_right.iloc[[y]].reset_index(drop=True),
                        ],
                        axis=1,
                    )
                    combinations = self._control_regions(
                        combinations, self.nshifts * control
                    )
                    if not combinations.empty:
                        combinations["coordinates"] = combinations.apply(
                            lambda x: ".".join(
                                x[
                                    [
                                        "chrom1",
                                        "start1",
                                        "end1",
                                        "chrom2",
                                        "start2",
                                        "end2",
                                    ]
                                ].astype(str)
                            ),
                            axis=1,
                        )
                    if modify_2Dintervals_func is not None:
                        combinations = modify_2Dintervals_func(combinations)
                    combinations = assign_groups(combinations, groupby=groupby)
                    combinations = combinations.reindex(
                        columns=list(combinations.columns)
                        + [
                            "data",
                            "horizontal_stripe",
                            "vertical_stripe",
                            "cov_start",
                            "cov_end",
                        ]
                    )
                    for row in combinations.to_dict(orient="records"):
                        yield row
            else:
                for i in range(1, intervals.shape[0]):
                    combinations = pd.concat(
                        [
                            intervals_left.iloc[:-i].reset_index(drop=True),
                            intervals_right.iloc[i:].reset_index(drop=True),
                        ],
                        axis=1,
                    )
                    combinations["distance"] = (
                        combinations["center2"] - combinations["center1"]
                    )
                    combinations = combinations[
                        (self.mindist <= combinations["distance"].abs())
                        & (combinations["distance"].abs() <= self.maxdist)
                    ]
                    combinations = self._control_regions(
                        combinations, self.nshifts * control
                    )
                    if not combinations.empty:
                        combinations["coordinates"] = combinations.apply(
                            lambda x: ".".join(
                                x[
                                    [
                                        "chrom1",
                                        "start1",
                                        "end1",
                                        "chrom2",
                                        "start2",
                                        "end2",
                                    ]
                                ].astype(str)
                            ),
                            axis=1,
                        )
                    if modify_2Dintervals_func is not None:
                        combinations = modify_2Dintervals_func(combinations)
                    combinations = assign_groups(combinations, groupby=groupby)
                    combinations = combinations.reindex(
                        columns=list(combinations.columns)
                        + [
                            "data",
                            "horizontal_stripe",
                            "vertical_stripe",
                            "cov_start",
                            "cov_end",
                        ]
                    )
                    for row in combinations.to_dict(orient="records"):
                        yield row

    def get_intervals_stream(
        self,
        filter_func1,
        filter_func2=None,
        intervals=None,
        control=False,
        groupby=[],
        modify_2Dintervals_func=None,
    ):

        if intervals is None:
            intervals = self.intervals
        intervals = filter_func1(intervals)
        intervals = self._control_regions(intervals, self.nshifts * control)
        if not intervals.empty:
            intervals["coordinates"] = intervals.apply(
                lambda x: ".".join(
                    x[["chrom1", "start1", "end1", "chrom2", "start2", "end2"]].astype(
                        str
                    )
                ),
                axis=1,
            )
        if modify_2Dintervals_func is not None:
            intervals = modify_2Dintervals_func(intervals)
        intervals = assign_groups(intervals, groupby)
        intervals = intervals.reindex(
            columns=list(intervals.columns)
            + [
                "data",
                "cov_start",
                "cov_end",
                "horizontal_stripe",
                "vertical_stripe",
            ]
        )
        if not len(intervals) >= 1:
            logger.debug("Empty selection")
            yield None
        for interval in intervals.to_dict(orient="records"):
            yield interval

    def empty_stream(self, *args, **kwargs):
        yield from ()


class PileUpper:
    def __init__(
        self,
        clr,
        CC,
        *,
        view_df=None,
        clr_weight_name="weight",
        expected=False,
        expected_value_col="balanced.avg",
        ooe=True,
        control=False,
        coverage_norm=False,
        rescale=False,
        rescale_size=99,
        flip_negative_strand=False,
        ignore_diags=2,
        store_stripes=False,
        nproc=1,
    ):
        """Creates pileups


        Parameters
        ----------
        clr : cool
            Cool file with Hi-C data.
        CC : CoordCreator
            CoordCreator object with correct settings.
        clr_weight_name : bool or str, optional
            Whether to use balanced data, and which column to use as weights.
            The default is "weight". Provide False to use raw data.
        expected : DataFrame, optional
            If using expected, pandas DataFrame with by-distance expected.
            The default is False.
        view_df : DataFrame
            A dataframe with region coordinates used in expected (see bioframe
            documentation for details). Can be ommited if no expected is provided, or
            expected is for whole chromosomes.
        ooe : bool, optional
            Whether to normalize each snip by expected value. If False, all snips are
            accumulated, all expected values are accumulated, and then the former
            divided by the latter - like with randomly shifted controls. Only has effect
            when expected is provided.
        control : bool, optional
            Whether to use randomly shifted controls.
            The default is False.
        coverage_norm : bool or str, optional
            Whether to normalize final the final pileup by accumulated coverage as an
            alternative to balancing. Useful for single-cell Hi-C data. Can be either
            boolean, or string: "cis" or "total" to use "cov_cis_raw" or "cov_tot_raw"
            columns in the cooler bin table, respectively. If True, will attempt to use
            "cov_tot_raw" if available, otherwise will compute and store coverage in the
            cooler with default column names, and use "cov_tot_raw". Alternatively, if
            a different string is provided, will attempt to use a column with the that
            name in the cooler bin table, and will raise a ValueError if it does not exist.
            Only allowed when clr_weight_name is False.
            The default is False.
        rescale : bool, optional
            Whether to rescale the pileups.
            The default is False
        rescale_size : int, optional
            Final shape of rescaled pileups. E.g. if 99, pileups will be squares of
            99×99 pixels.
            The default is 99.
        flip_negative_strand : bool, optional
            Flip snippets so the positive strand always points to bottom-right.
            Requires strands to be annotated for each feature (or two strands for
            bedpe format features)
        ignore_diags : int, optional
            How many diagonals to ignore to avoid short-distance artefacts.
            The default is 2.
        store_stripes: bool, optional
            Whether to store horizontal and vertical stripes and coordinates in the output
            The default is False
        nproc : int, optional
            Number of processes to use. The default is 1.

        Returns
        -------
        Object that generates pileups.

        """
        self.clr = clr
        self.resolution = self.clr.binsize
        self.CC = CC
        assert self.resolution == self.CC.resolution
        self.__dict__.update(self.CC.__dict__)
        self.clr_weight_name = clr_weight_name
        self.expected = expected
        self.expected_value_col = expected_value_col
        self.ooe = ooe
        self.control = control
        self.pad_bins = self.CC.flank // self.resolution
        self.coverage_norm = coverage_norm
        self.rescale = rescale
        self.rescale_size = rescale_size
        self.flip_negative_strand = flip_negative_strand
        self.ignore_diags = ignore_diags
        self.store_stripes = store_stripes
        self.nproc = nproc

        if view_df is None:
            # Generate viewframe from clr.chromsizes:
            self.view_df = common.make_cooler_view(clr)
        else:
            self.view_df = bioframe.make_viewframe(view_df, check_bounds=clr.chromsizes)
        if self.expected is not None and self.expected is not False:
            # subset expected if some regions not mentioned in view
            self.expected = self.expected[
                (self.expected["region1"].isin(self.view_df["name"]))
                & (self.expected["region2"].isin(self.view_df["name"]))
            ].reset_index(drop=True)
            if self.control:
                warnings.warn(
                    "Can't do both expected and control shifts; defaulting to expected",
                    stacklevel=2,
                )
                self.control = False
            if self.trans:
                try:
                    _ = checks.is_valid_expected(
                        self.expected,
                        "trans",
                        self.view_df,
                        verify_cooler=clr,
                        expected_value_cols=[
                            self.expected_value_col,
                        ],
                        raise_errors=True,
                    )
                except Exception as e:
                    raise ValueError("provided expected is not valid") from e

                self.expected_df = self.expected
                self.expected = True
            else:
                self.expected = self.expected[
                    self.expected["region1"] == self.expected["region2"]
                ].reset_index(drop=True)
                try:
                    _ = checks.is_valid_expected(
                        self.expected,
                        "cis",
                        self.view_df,
                        verify_cooler=clr,
                        expected_value_cols=[
                            self.expected_value_col,
                        ],
                        raise_errors=True,
                    )
                except Exception as e:
                    raise ValueError("provided expected is not valid") from e
                self.ExpSnipper = snipping.ExpectedSnipper(
                    self.clr, self.expected, view_df=self.view_df
                )
                self.expected_selections = {
                    region_name: self.ExpSnipper.select(region_name, region_name)
                    for region_name in self.view_df["name"]
                }
                self.expected_df = self.expected
                self.expected = True
        self.view_df = self.view_df.set_index("name")
        self.view_df_extents = {}

        for region_name, region in self.view_df.iterrows():
            lo, hi = self.clr.extent(region)
            chroffset = self.clr.offset(region[0])
            self.view_df_extents[region_name] = lo - chroffset, hi - chroffset

        self.chroms = natsorted(
            list(set(self.CC.final_chroms) & set(self.clr.chromnames))
        )
        self.view_df = self.view_df[self.view_df["chrom"].isin(self.chroms)]

        if self.view_df["chrom"].unique().shape[0] == 0:
            raise ValueError(
                """No chromosomes are in common between the coordinate
                   file and the cooler file. Are they in the same
                   format, e.g. starting with "chr"?
                   """
            )

        if self.trans:
            if self.view_df["chrom"].unique().shape[0] < 2:
                raise ValueError("Trying to do trans with fewer than two chromosomes")

        if self.coverage_norm is True:
            self.coverage_norm = "cov_tot_raw"
        elif self.coverage_norm == "cis":
            self.coverage_norm = "cov_cis_raw"
        elif self.coverage_norm == "total":
            self.coverage_norm = "cov_tot_raw"
        elif self.coverage_norm and self.coverage_norm not in self.clr.bins().columns:
            raise ValueError(
                f"coverage_norm {self.coverage_norm} not found in cooler bins"
            )

        if (
            self.coverage_norm in ["cov_cis_raw", "cov_tot_raw"]
            and self.coverage_norm not in self.clr.bins().columns
        ):
            with Pool(self.nproc) as pool:
                _ = coverage.coverage(
                    self.clr, map=pool.map, store=True, ignore_diags=self.ignore_diags
                )
                del _

        if self.coverage_norm and self.clr_weight_name:
            raise ValueError(
                "Can't do coverage normalization when clr_weight_name is provided"
            )

        if self.rescale:
            if self.rescale_flank is None:
                raise ValueError("Cannot use rescale without setting rescale_flank")
            elif self.rescale_size % 2 == 0:
                raise ValueError("Please provide an odd rescale_size")
            else:
                logger.info(
                    "Rescaling with rescale_flank = "
                    + str(self.rescale_flank)
                    + " to "
                    + str(self.rescale_size)
                    + "x"
                    + str(self.rescale_size)
                    + " pixels"
                )

        self.empty_outmap = self.make_outmap()

        self.empty_pup = {
            "data": self.empty_outmap,
            "horizontal_stripe": [],
            "vertical_stripe": [],
            "n": 0,
            "num": self.empty_outmap,
            "cov_start": np.zeros((self.empty_outmap.shape[0])),
            "cov_end": np.zeros((self.empty_outmap.shape[1])),
            "coordinates": [],
        }

    def get_expected_trans(self, region1, region2):
        exp_value = self.expected_df.loc[
            (self.expected_df["region1"] == region1)
            & (self.expected_df["region2"] == region2),
            self.expected_value_col,
        ].item()
        return exp_value

    def make_outmap(
        self,
    ):
        """Generate zero-filled array of the right shape

        Returns
        -------
        outmap: array
            Array of zeros of correct shape.

        """
        if self.rescale:
            outmap = np.zeros((self.rescale_size, self.rescale_size))
        else:
            outmap = np.zeros((2 * self.pad_bins + 1, 2 * self.pad_bins + 1))
        return outmap

    def get_data(self, region1, region2=None):
        """Get sparse data for a region

        Parameters
        ----------
        region1 : tuple or str
            Region for which to load the data. Either tuple of (chr, start, end), or
            string with region name.
        region2 : tuple or str, optional
            Second region for between which and the first region to load the data. Either tuple of (chr, start, end), or
            string with region name.
            Default is None

        Returns
        -------
        data : csr
            Sparse csr matrix for the corresponding region.

        """
        logger.debug("Loading data")

        assert isinstance(region1, str)
        region1 = self.view_df.loc[region1]

        if region2 is None:
            region2 = region1
        else:
            region2 = self.view_df.loc[region2]

        data = self.clr.matrix(sparse=True, balance=self.clr_weight_name).fetch(
            region1, region2
        )
        # data = sparse.triu(data)
        return data.tocsr()

    def _stream_snips(self, intervals, region1, region2=None):
        mymap = self.make_outmap()
        cov_start = np.zeros(mymap.shape[0])
        cov_end = np.zeros(mymap.shape[1])

        try:
            row1 = next(intervals)
        except StopIteration:
            # logger.info(f"Nothing to sum up between regions {region1} & {region2}")
            return
        if row1 is None:
            # logger.info(f"Nothing to sum up between region {region1} & {region2}")
            return

        intervals = itertools.chain([row1], intervals)

        if region2 is None:
            region2 = region1

        min_left1, max_right1 = self.view_df_extents[region1]
        min_left2, max_right2 = self.view_df_extents[region2]

        bigdata = self.get_data(region1=region1, region2=region2)

        region1_coords = self.view_df.loc[region1]
        region2_coords = self.view_df.loc[region2]
        if self.clr_weight_name:
            isnan1 = np.isnan(
                self.clr.bins()[self.clr_weight_name].fetch(region1_coords).values
            )
            isnan2 = np.isnan(
                self.clr.bins()[self.clr_weight_name].fetch(region2_coords).values
            )
        else:
            isnan1 = isnan = np.zeros_like(
                self.clr.bins()["start"].fetch(region1_coords).values
            ).astype(bool)
            isnan2 = isnan = np.zeros_like(
                self.clr.bins()["start"].fetch(region2_coords).values
            ).astype(bool)

        if self.coverage_norm:
            coverage1 = self.clr.bins()[self.coverage_norm].fetch(region1_coords).values
            coverage2 = self.clr.bins()[self.coverage_norm].fetch(region2_coords).values

        ar = np.arange(max_right1 - min_left1, dtype=np.int32)

        diag_indicator = ctutils.LazyToeplitz(-ar, ar)

        for snip in intervals:
            snip["stBin1"], snip["endBin1"], snip["stBin2"], snip["endBin2"] = (
                snip["stBin1"] - min_left1,
                snip["endBin1"] - min_left1,
                snip["stBin2"] - min_left2,
                snip["endBin2"] - min_left2,
            )
            if (snip["stBin1"] < 0 or snip["endBin1"] > (max_right1 - min_left1)) or (
                snip["stBin2"] < 0 or snip["endBin2"] > (max_right2 - min_left2)
            ):
                continue
            data = (
                bigdata[
                    snip["stBin1"] : snip["endBin1"], snip["stBin2"] : snip["endBin2"]
                ]
                .toarray()
                .astype(float)
            )
            data[isnan1[snip["stBin1"] : snip["endBin1"]], :] = np.nan
            data[:, isnan2[snip["stBin2"] : snip["endBin2"]]] = np.nan

            if self.expected:
                if self.trans:
                    exp_value = self.get_expected_trans(region1, region2)
                    exp_data = np.full((data.shape), exp_value)
                else:
                    exp_data = self.expected_selections[region1][
                        snip["stBin1"] : snip["endBin1"],
                        snip["stBin2"] : snip["endBin2"],
                    ]

                if not self.ooe:
                    exp_snip = snip.copy()
                    exp_snip["kind"] = "control"
                    exp_snip["data"] = exp_data

            if not self.trans:
                D = (
                    diag_indicator[
                        snip["stBin1"] : snip["endBin1"],
                        snip["stBin2"] : snip["endBin2"],
                    ]
                    < self.ignore_diags
                )
                data[D] = np.nan

            if self.coverage_norm:
                snip["cov_start"] = coverage1[snip["stBin1"] : snip["endBin1"]]
                snip["cov_end"] = coverage2[snip["stBin2"] : snip["endBin2"]]
            if self.expected and self.ooe:
                with np.errstate(divide="ignore", invalid="ignore"):
                    data = data / exp_data
            snip["data"] = data

            if self.rescale:
                snip = self._rescale_snip(snip)
                if self.expected and not self.ooe:
                    exp_snip = self._rescale_snip(exp_snip)

            if (
                self.flip_negative_strand
                and "strand1" in snip
                and "strand2" in snip
                and snip["strand1"] == "-"
            ):
                snip["data"] = np.rot90(np.flipud(snip["data"]))
                if self.expected and not self.ooe:
                    exp_data = np.rot90(np.flipud(exp_data))

            if self.store_stripes:
                cntr = int(np.floor(snip["data"].shape[0] / 2))
                snip["horizontal_stripe"] = np.array(snip["data"][cntr, :], dtype=float)
                snip["vertical_stripe"] = np.array(
                    snip["data"][:, cntr][::-1], dtype=float
                )
            else:
                snip["horizontal_stripe"] = []
                snip["vertical_stripe"] = []
                snip["coordinates"] = []

            yield snip

            if self.expected and not self.ooe:
                yield exp_snip

    def _rescale_snip(self, snip):
        """

        Parameters
        ----------
        snip : pd.Series
            Containing at least:
                ['data'] - the snippet as a 2D array,
                ['cov_start'] and ['cov_end'] as 1D arrays (can be all 0)
                ['stBin1'], ['endBin1'] - coordinates of the left side of the pileup
                ['stBin2'], ['endBin2'] - coordinates of the left side of the pileup
            And any other annotations

        Yields
        ------
        snip : pd.Series
            Outputs same snip as came in, but appropriately rescaled

        """
        if snip["data"].size == 0 or np.all(np.isnan(snip["data"])):
            snip["data"] = np.zeros((self.rescale_size, self.rescale_size))
        else:
            if self.local:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    snip["data"] = np.nanmean(
                        np.dstack((snip["data"], snip["data"].T)), 2
                    )
            nans = np.isnan(snip["data"]) * 1
            snip["data"] = np.nan_to_num(snip["data"])
            snip["data"] = ctutils.zoom_array(
                snip["data"], (self.rescale_size, self.rescale_size)
            )
            nanzoom = ctutils.zoom_array(nans, (self.rescale_size, self.rescale_size))
            snip["data"][np.floor(nanzoom).astype(bool)] = np.nan
            snip["data"] = snip["data"] * (1 / np.isfinite(nanzoom))
        if self.coverage_norm:
            snip["cov_start"] = ctutils.zoom_array(
                snip["cov_start"], (self.rescale_size,)
            )
            snip["cov_end"] = ctutils.zoom_array(snip["cov_end"], (self.rescale_size,))
        return snip

    def accumulate_stream(self, snip_stream, postprocess_func=None, extra_funcs=None):
        """

        Parameters
        ----------
        snip_stream : generator
            Generator of pd.Series, each one containing at least:
                a snippet as a 2D array in ['data'],
                ['cov_start'] and ['cov_end'] as 1D arrays (can be all 0)
            And any other annotations
        postprocess_func : function, optional
            Any additional postprocessing of each snip needed, in one function.
            Can be used to modify the data in un-standard way, or create groups when
            it can't be done before snipping, or to assign each snippet to multiple
            groups. Example: `lib.puputils.group_by_region`.
        extra_funcs : dict, optional
            Any additional functions to be applied every time a snip is added to a
            pileup or two pileups are summed up - see `_add_snip` and `sum_pups`.

        Returns
        -------
        outdict : dict
            Dictionary of accumulated snips (each as a Series) for each group.
            Always includes "all"
        """
        if postprocess_func is not None:
            snip_stream = map(postprocess_func, snip_stream)
        outdict = {"ROI": {}, "control": {}}
        for snip in collapse(snip_stream, base_type=dict):
            kind = snip["kind"]
            key = snip["group"]
            if isinstance(key, str):
                _add_snip(outdict[kind], key, snip, extra_funcs=extra_funcs)
            else:
                _add_snip(outdict[kind], tuple(key), snip, extra_funcs=extra_funcs)
        sum_func = partial(sum_pups, extra_funcs=extra_funcs)
        if "all" not in outdict["ROI"]:
            outdict["ROI"]["all"] = reduce(
                sum_func, outdict["ROI"].values(), self.empty_pup
            )
        if self.control or (self.expected and not self.ooe):
            if "all" not in outdict["control"]:
                outdict["control"]["all"] = reduce(
                    sum_func,
                    outdict["control"].values(),
                    self.empty_pup,
                )
        return outdict

    def pileup_region(
        self,
        region1,
        region2=None,
        groupby=[],
        modify_2Dintervals_func=None,
        postprocess_func=None,
        extra_sum_funcs=None,
    ):
        """

        Parameters
        ----------
        region1 : str
            Region name.
        region2 : str, optional
            Region name.
        groupby : list of str, optional
            Which attributes of each snip to assign a group to it
        modify_2Dintervals_func : function, optional
            A function to apply to a dataframe of genomic intervals used for pileups.
            If possible, much preferable to `postprocess_func` for better speed.
            Good example is the `bin_distance_intervals` function above.
        postprocess_func : function, optional
            Additional function to apply to each snippet before grouping.
            Good example is the `lib.puputils.bin_distance` function, but using
            bin_distance_intervals as modify_2Dintervals_func is much faster.
        extra_sum_funcs : dict, optional
            Any additional functions to be applied every time a snip is added to a
            pileup or two pileups are summed up - see `_add_snip` and `sum_pups`.

        Returns
        -------
        pileup : dict
            accumulated snips as a dict
        """

        region1_coords = self.view_df.loc[region1]

        if region2 is None:
            region2 = region1
            region2_coords = region1_coords
        else:
            region2_coords = self.view_df.loc[region2]

        if (self.kind == "bedpe") and (self.trans):
            filter_func1 = self.CC.filter_func_trans_pairs(
                region1=region1_coords, region2=region2_coords
            )
            filter_func2 = None
        else:
            filter_func1 = self.CC.filter_func_region(region=region1_coords)
            if region2 == region1:
                filter_func2 = None
            else:
                filter_func2 = self.CC.filter_func_region(region=region2_coords)

        intervals = self.CC.pos_stream(
            filter_func1,
            filter_func2,
            control=self.control,
            groupby=groupby,
            modify_2Dintervals_func=modify_2Dintervals_func,
        )

        final = self.accumulate_stream(
            self._stream_snips(intervals=intervals, region1=region1, region2=region2),
            postprocess_func=postprocess_func,
            extra_funcs=extra_sum_funcs,
        )
        if final["ROI"]["all"]["n"] > 0:
            logger.info(f"{region1, region2}: {final['ROI']['all']['n']}")

        return final

    def pileupsWithControl(
        self,
        nproc=None,
        groupby=[],
        modify_2Dintervals_func=None,
        postprocess_func=None,
        extra_sum_funcs=None,
    ):
        """Perform pileups across all chromosomes and applies required
        normalization

        Parameters
        ----------
        nproc : int, optional
            How many cores to use. Sends a whole chromosome per process.
            The default is None, which uses the same number as nproc set at creation of
            the object.
        groupby : list of str, optional
            Which attributes of each snip to assign a group to it
        modify_2Dintervals_func : function, optional
            Function to apply to the DataFrames of coordinates before fetching snippets
            based on them. Preferable to using the `postprocess_func`, since at the
            earlier stage it can be vectorized and much more efficient.
        postprocess_func : function, optional
            Additional function to apply to each snippet before grouping.
            Good example is the `lib.puputils.bin_distance` function.
        extra_sum_funcs : dict, optional
            Any additional functions to be applied every time a snip is added to a
            pileup or two pileups are summed up - see `_add_snip` and `sum_pups`.

        Returns
        -------
        pileup_df : 2D array
            Normalized pileups in a pandas DataFrame, with columns `data` and `num`.
            `data` contains the normalized pileups, and `num` - how many snippets were
            combined (the regions of interest, not control regions). Each condition
            from `groupby` is a row, plus an additional row `all` is created with all
            data.

        """

        if nproc is None:
            nproc = self.nproc
        if len(self.chroms) == 0:
            return self.make_outmap(), 0
        sum_func = partial(sum_pups, extra_funcs=extra_sum_funcs)
        # Generate all combinations of chromosomes
        regions1 = []
        regions2 = []
        if self.trans:
            for region1, region2 in itertools.combinations(self.view_df.index, 2):
                if (
                    self.view_df.loc[region1, "chrom"]
                    != self.view_df.loc[region2, "chrom"]
                ):
                    regions1.append(region1)
                    regions2.append(region2)
        else:
            regions1 = self.view_df.index
            regions2 = regions1
        f = partial(
            self.pileup_region,
            groupby=groupby,
            modify_2Dintervals_func=modify_2Dintervals_func,
            postprocess_func=postprocess_func,
            extra_sum_funcs=extra_sum_funcs,
        )
        if nproc > 1:
            from multiprocessing_logging import install_mp_handler, uninstall_mp_handler

            install_mp_handler()
            with Pool(nproc) as p:
                pileups = list(p.starmap(f, zip(regions1, regions2)))
            uninstall_mp_handler()
        else:
            pileups = list(map(f, regions1, regions2))
        roi = (
            pd.DataFrame(
                [
                    {k: pd.Series(v) for k, v in pileup["ROI"].items()}
                    for pileup in pileups
                ]
            )
            .apply(lambda x: reduce(sum_func, x.dropna()))
            .T
        )
        if self.control or (self.expected and not self.ooe):
            ctrl = (
                pd.DataFrame(
                    [
                        {k: pd.Series(v) for k, v in pileup["control"].items()}
                        for pileup in pileups
                    ]
                )
                .apply(lambda x: reduce(sum_func, x.dropna()))
                .T
            )

        if self.coverage_norm:
            roi = roi.apply(norm_coverage, axis=1)
            if self.control:
                ctrl = ctrl.apply(norm_coverage, axis=1)
            elif self.expected:
                warnings.warn(
                    "Expected can not be normalized to coverage", stacklevel=2
                )
        normalized_roi = pd.DataFrame(roi["data"] / roi["num"], columns=["data"])
        if self.control or (self.expected and not self.ooe):
            normalized_control = pd.DataFrame(
                ctrl["data"] / ctrl["num"], columns=["data"]
            )
            normalized_roi = normalized_roi / normalized_control
            normalized_roi["control_n"] = ctrl["n"]
            normalized_roi["control_num"] = ctrl["num"]

        normalized_roi["data"] = normalized_roi["data"].apply(
            lambda x: np.where(x == np.inf, np.nan, x)
        )

        normalized_roi["n"] = roi["n"]
        normalized_roi["num"] = roi["num"]
        if self.store_stripes:
            normalized_roi["coordinates"] = roi["coordinates"]
            normalized_roi["coordinates"] = [
                [x.split(".") for x in y] for y in normalized_roi["coordinates"]
            ]
            normalized_roi["horizontal_stripe"] = roi["horizontal_stripe"]
            normalized_roi["vertical_stripe"] = roi["vertical_stripe"]

            if self.control or (self.expected and not self.ooe):
                # Generate stripes of normalized control arrays
                cntr = int(np.floor(normalized_control["data"]["all"].shape[0] / 2))
                control_horizontalstripe = np.array(
                    normalized_control["data"]["all"][cntr, :], dtype=float
                )
                control_verticalstripe = np.array(
                    normalized_control["data"]["all"][:, cntr][::-1], dtype=float
                )
                normalized_roi["horizontal_stripe"] = normalized_roi.apply(
                    lambda row: np.divide(
                        row["horizontal_stripe"], control_horizontalstripe
                    ),
                    axis=1,
                )
                normalized_roi["vertical_stripe"] = normalized_roi.apply(
                    lambda row: np.divide(
                        row["vertical_stripe"], control_verticalstripe
                    ),
                    axis=1,
                )
            normalized_roi["vertical_stripe"] = normalized_roi["vertical_stripe"].apply(
                np.vstack
            )
            normalized_roi["horizontal_stripe"] = normalized_roi[
                "horizontal_stripe"
            ].apply(np.vstack)
            normalized_roi["coordinates"] = normalized_roi["coordinates"].apply(
                np.vstack
            )
            if self.local:
                normalized_roi["vertical_stripe"] = normalized_roi[
                    "vertical_stripe"
                ].apply(lambda x: numutils._copy_array_halves(x))
                normalized_roi["horizontal_stripe"] = normalized_roi[
                    "horizontal_stripe"
                ].apply(lambda x: numutils._copy_array_halves(x))

        if self.local:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                normalized_roi["data"] = normalized_roi["data"].apply(
                    lambda x: np.nanmean(np.dstack((x, x.T)), 2)
                )
        n = normalized_roi.loc["all", "n"]
        normalized_roi = normalized_roi.reset_index().rename(columns={"index": "group"})
        if groupby:
            normalized_roi[groupby] = pd.DataFrame(
                [
                    ("all",) * len(groupby) if i == "all" else i
                    for i in normalized_roi["group"].to_list()
                ],
                columns=groupby,
            )
            normalized_roi.assign(**dict(zip(groupby, zip(*normalized_roi["group"]))))
            normalized_roi = normalized_roi.drop(columns="group")
            for i, val in enumerate(groupby):
                normalized_roi.insert(0, val, normalized_roi.pop(val))
        if extra_sum_funcs:
            for key in extra_sum_funcs:
                normalized_roi[key] = roi[key].values
                if self.control:
                    normalized_roi[f"control_{key}"] = ctrl[key]
        logger.info(f"Total number of piled up windows: {int(n)}")

        # Store attributes
        exclude_attributes = [
            "CC",
            "intervals",
            "features_format",
            "kind",
            "basechroms",
            "final_chroms",
            "pos_stream",
            "view_df",
            "ExpSnipper",
            "expected_selections",
            "expected_df",
            "view_df_extents",
            "regions",
            "empty_outmap",
            "empty_pup",
        ]

        for name, attr in self.__dict__.items():
            if name not in exclude_attributes:
                if type(attr) == list:
                    attr = str(attr)
                if type(attr) == cooler.api.Cooler:
                    attr = os.path.abspath(attr.filename)
                normalized_roi[name] = attr
        return normalized_roi

    def pileupsByStrandWithControl(self, nproc=None, groupby=[]):
        """Perform by-strand pileups across all chromosomes and applies required
        normalization. Simple wrapper around pileupsWithControl.
        Assumes the features in CoordCreator file has a "strand" column.

        Parameters
        ----------
        nproc : int, optional
            How many cores to use. Sends a whole chromosome per process.
            The default is None, which uses the same number as nproc set at creation of
            the object.
        groupby : list of str, optional
            Which attributes of each snip to assign a group to it

        Returns
        -------
        pileup_df : 2D array
            Normalized pileups in a pandas DataFrame, with columns `data` and `num`.
            `data` contains the normalized pileups, and `num` - how many snippets were
            combined (the regions of interest, not control regions).
            Each distance band is a row, annotated in columns `separation`
        """
        if nproc is None:
            nproc = self.nproc
        normalized_pileups = self.pileupsWithControl(
            nproc=nproc,
            groupby=["strand1", "strand2"] + groupby,
        )
        normalized_pileups.insert(
            0,
            "orientation",
            (normalized_pileups["strand1"] + normalized_pileups["strand2"]).replace(
                {"allall": "all"}
            ),
        )
        return normalized_pileups

    def pileupsByWindowWithControl(
        self,
        nproc=None,
    ):
        """Perform by-window (i.e. for each region) pileups across all chromosomes and applies required
        normalization. Simple wrapper around pileupsWithControl

        Parameters
        ----------
        nproc : int, optional
            How many cores to use. Sends a whole chromosome per process.
            The default is None, which uses the same number as nproc set at creation of
            the object.

        Returns
        -------
        pileup_df : 2D array
            Normalized pileups in a pandas DataFrame, with columns `data` and `num`.
            `data` contains the normalized pileups, and `num` - how many snippets were
            combined (the regions of interest, not control regions). Each window is a
            row (coordinates are recorded in columns ['chrom', 'start', 'end']), plus
            an additional row is created with all data (with "all" in the "chrom" column
            and -1 in start and end).
        """
        if nproc is None:
            nproc = self.nproc
        if self.local:
            raise ValueError("Cannot do by-window pileups for local")

        normalized_pileups = self.pileupsWithControl(
            nproc=nproc, postprocess_func=group_by_region
        )
        normalized_pileups = pd.concat(
            [
                pd.DataFrame(
                    normalized_pileups["group"].to_list(),
                    index=normalized_pileups.index,
                    columns=["chrom", "start", "end"],
                ),
                normalized_pileups,
            ],
            axis=1,
        )
        normalized_pileups.loc[
            normalized_pileups["group"] == "all", ["chrom", "start", "end"]
        ] = ["all", -1, -1]
        normalized_pileups[["start", "end"]] = normalized_pileups[
            ["start", "end"]
        ].astype(int)
        normalized_pileups = normalized_pileups.drop(columns="group")
        # Sorting places "all" at the end! So no need to drop and append
        normalized_pileups = bioframe.sort_bedframe(
            normalized_pileups, view_df=self.view_df.reset_index()
        ).reset_index(drop=True)
        return normalized_pileups

    def pileupsByDistanceWithControl(
        self, nproc=None, distance_edges="default", groupby=[]
    ):
        """Perform by-distance pileups across all chromosomes and applies required
        normalization. Simple wrapper around pileupsWithControl

        Parameters
        ----------
        nproc : int, optional
            How many cores to use. Sends a whole chromosome per process.
            The default is None, which uses the same number as nproc set at creation of
            the object.
        distance_edges : list/array of int
            How to group snips by distance (based on their centres).
            Default uses separations [0, 50_000, 100_000, 200_000, ...]
        groupby : list of str, optional
            Which attributes of each snip to assign a group to it

        Returns
        -------
        pileup_df : 2D array
            Normalized pileups in a pandas DataFrame, with columns `data` and `num`.
            `data` contains the normalized pileups, and `num` - how many snippets were
            combined (the regions of interest, not control regions).
            Each distance band is a row, annotated in column `distance_band`
        """
        if nproc is None:
            nproc = self.nproc
        if self.trans:
            raise ValueError("Cannot do by-distance pileups for trans")
        elif self.local:
            raise ValueError("Cannot do by-distance pileups for local")
        if distance_edges != "default":
            if not all(isinstance(n, int) for n in distance_edges):
                raise ValueError("Distance edges must be integers")
            distance_edges = list(np.sort(distance_edges))
            for n in range(len(distance_edges)):
                if np.min(distance_edges) < self.mindist:
                    distance_edges[np.argmin(distance_edges)] = self.mindist
                else:
                    break
        bin_func = partial(bin_distance_intervals, band_edges=distance_edges)
        normalized_pileups = self.pileupsWithControl(
            nproc=nproc,
            modify_2Dintervals_func=bin_func,
            groupby=["distance_band"] + groupby,
        )
        normalized_pileups = normalized_pileups.loc[
            normalized_pileups["distance_band"] != (), :
        ].reset_index(drop=True)
        # Create a nicely formatted "distance_band" columns (e.g. for plotting)
        normalized_pileups.insert(
            0,
            "separation",
            normalized_pileups["distance_band"].apply(
                lambda x: x
                if x == "all"
                else f"{x[0]/1000000}Mb-\n{x[1]/1000000}Mb"
                if len(x) == 2
                else f"{x[0]/1000000}Mb+"
            ),
        )
        # Move "all" to the bottom while sorting the distances
        i = np.where(normalized_pileups["separation"] == "all")[0]
        normalized_pileups = pd.concat(
            [
                normalized_pileups.drop(i).sort_values("distance_band"),
                normalized_pileups.iloc[i, :],
            ],
            ignore_index=True,
        ).reset_index(drop=True)
        return normalized_pileups

    def pileupsByStrandByDistanceWithControl(
        self, nproc=None, distance_edges="default", groupby=[]
    ):
        """Perform by-strand by-distance pileups across all chromosomes and applies
        required normalization. Simple wrapper around pileupsWithControl.
        Assumes the features in CoordCreator file has a "strand" column.

        Parameters
        ----------
        nproc : int, optional
            How many cores to use. Sends a whole chromosome per process.
            The default is None, which uses the same number as nproc set at creation of
            the object.
        distance_edges : list/array of int
            How to group snips by distance (based on their centres).
            Default uses separations [0, 50_000, 100_000, 200_000, ...]
        groupby : list of str, optional
            Which attributes of each snip to assign a group to it


        Returns
        -------
        pileup_df : 2D array
            Normalized pileups in a pandas DataFrame, with columns `data` and `num`.
            `data` contains the normalized pileups, and `num` - how many snippets were
            combined (the regions of interest, not control regions).
            Each distance band is a row, annotated in columns `separation`
        """
        if nproc is None:
            nproc = self.nproc
        if self.trans:
            raise ValueError("Cannot do by-distance pileups for trans")
        if distance_edges != "default":
            if not all(isinstance(n, int) for n in distance_edges):
                raise ValueError("Distance edges must be integers")
            distance_edges = list(np.sort(distance_edges))
            for n in range(len(distance_edges)):
                if np.min(distance_edges) < self.mindist:
                    distance_edges[np.argmin(distance_edges)] = self.mindist
                else:
                    break
        bin_func = partial(bin_distance_intervals, band_edges=distance_edges)
        normalized_pileups = self.pileupsWithControl(
            nproc=nproc,
            modify_2Dintervals_func=bin_func,
            groupby=["strand1", "strand2", "distance_band"] + groupby,
        )
        normalized_pileups.insert(
            0,
            "orientation",
            (normalized_pileups["strand1"] + normalized_pileups["strand2"]).replace(
                {"allall": "all"}
            ),
        )
        normalized_pileups = normalized_pileups.loc[
            normalized_pileups["distance_band"] != (), :
        ].reset_index(drop=True)
        normalized_pileups.insert(
            0,
            "separation",
            normalized_pileups["distance_band"].apply(
                lambda x: x
                if x == "all"
                else f"{x[0]/1000000}Mb-\n{x[1]/1000000}Mb"
                if len(x) == 2
                else f"{x[0]/1000000}Mb+"
            ),
        )
        # Move "all" to the bottom while sorting the distances
        i = np.where(normalized_pileups["separation"] == "all")[0]
        normalized_pileups = pd.concat(
            [
                normalized_pileups.drop(i).sort_values(
                    ["orientation", "distance_band"]
                ),
                normalized_pileups.iloc[i, :],
            ],
            ignore_index=True,
        ).reset_index(drop=True)
        return normalized_pileups


def pileup(
    clr,
    features,
    features_format="bed",
    view_df=None,
    expected_df=None,
    expected_value_col="balanced.avg",
    clr_weight_name="weight",
    flank=100000,
    minshift=10**5,
    maxshift=10**6,
    nshifts=0,
    ooe=True,
    mindist="auto",
    maxdist=None,
    min_diag=2,
    subset=0,
    by_window=False,
    by_strand=False,
    by_distance=False,
    groupby=[],
    flip_negative_strand=False,
    local=False,
    coverage_norm=False,
    trans=False,
    rescale=False,
    rescale_flank=1,
    rescale_size=99,
    store_stripes=False,
    nproc=1,
    seed=None,
):
    """Create pileups

    Parameters
    ----------
    clr : cool
        Cool file with Hi-C data.
    features : DataFrame
        A bed- or bedpe-style file with coordinates.
    features_format : str, optional
        Format of the features. Options:
            bed: chrom, start, end
            bedpe: chrom1, start1, end1, chrom2, start2, end2
            auto (default): determined from the columns in the DataFrame
    view_df : DataFrame
        A dataframe with region coordinates used in expected (see bioframe
        documentation for details). Can be ommited if no expected is provided, or
        expected is for whole chromosomes.
    expected_df : DataFrame, optional
        If using expected, pandas DataFrame with by-distance expected.
        The default is False.
    expected_value_col : str, optional
        Which column in the expected_df contains values to use for normalization
    clr_weight_name : bool or str, optional
        Whether to use balanced data, and which column to use as weights.
        The default is "weight". Provide False to use raw data.
    flank : int, optional
        Padding around the central bin, in bp. For example, with 5000 bp resolution
        and 100000 flank, final pileup is 205000×205000 bp.
        The default is 100000.
    minshift : int, optional
        Minimal shift applied when generating random controls, in bp.
        The default is 10 ** 5.
    maxshift : int, optional
        Maximal shift applied when generating random controls, in bp.
        The default is 10 ** 6.
    nshifts : int, optional
        How many shifts to generate per region of interest. Does not take chromosome
        boundaries into account
        The default is 10.
    ooe : bool, optional
        Whether to normalize each snip by expected value. If False, all snips are
        accumulated, all expected values are accumulated, and then the former
        divided by the latter - like with randomly shifted controls. Only has effect
        when expected is provided.
        Default is True.
    mindist : int, optional
        Shortest interactions to consider. Uses midpoints of regions of interest.
        "auto" selects it to avoid the two shortest diagonals of the matrix, i.e.
        2 * flank + 2 * resolution
        The default is "auto".
    maxdist : int, optional
        Longest interactions to consider.
        The default is None.
    min_diag : int, optional
        How many diagonals to ignore to avoid short-distance artefacts.
        The default is 2.
    subset : int, optional
        What subset of the coordinate files to use. 0 or negative to use all.
        The default is 0.
    by_window : bool, optional
        Whether to create a separate pileup for each feature by accumulating all of its
        interactions with other features. Produces as many pileups, as there are
        features.
        The default is False.
    by_strand : bool, optional
        Whether to create a separate pileup for each combination of "strand1", "strand2"
        in features. If features_format=='bed', first creates pairwise combinations of
        features, and the original features need to have a column "strand". If
        features_format=='bedpe', they need to have "strand1" and "strand2" columns.
        The default is False.
    by_distance : bool or list, optional
        Whether to create a separate pileup for different distance separations. If
        features_format=='bed', internally creates pairwise combinations of features.
        If True, splits all separations using edges defined like this:
            band_edges = np.append([0], 50000 * 2 ** np.arange(30))
        Alternatively, a list of integer values can be given with custom distance edges.
        The default is False.
    groupby: list of str, optional
        Additional columns of features to use for groupby. If feature_format=='bed',
        each columns should be specified twice with suffixes "1" and "2", i.e. if
        features have a column "group", specify ["group1", "group2"].
        The default is [].
    flip_negative_strand : bool, optional
        Flip snippets so the positive strand always points to bottom-right.
        Requires strands to be annotated for each feature (or two strands for
        bedpe format features)
    local : bool, optional
        Whether to generate local coordinates, i.e. on-diagonal.
        The default is False.
    coverage_norm : bool or str, optional
        Whether to normalize final the final pileup by accumulated coverage as an
        alternative to balancing. Useful for single-cell Hi-C data. Can be either
        boolean, or string: "cis" or "total" to use "cov_cis_raw" or "cov_tot_raw"
        columns in the cooler bin table, respectively. If True, will attempt to use
        "cov_tot_raw" if available, otherwise will compute and store coverage in the
        cooler with default column names, and use "cov_tot_raw". Alternatively, if
        a different string is provided, will attempt to use a column with the that
        name in the cooler bin table, and will raise a ValueError if it does not exist.
        Only allowed when clr_weight_name is False.
        The default is False.
    trans : bool, optional
        Whether to generate inter-chromosomal (trans) pileups.
        The default is False
    rescale : bool, optional
        Whether to rescale the pileups.
        The default is False
    rescale_flank : float, optional
        Fraction of ROI size added on each end when extracting snippets, if rescale.
        The default is None. If specified, overrides flank.
    rescale_size : int, optional
        Final shape of rescaled pileups. E.g. if 99, pileups will be squares of
        99×99 pixels.
        The default is 99.
    store_stripes: bool, optional
        Whether to store horizontal and vertical stripes and coordinates in the output
        The default is False
    nproc : int, optional
        Number of processes to use. The default is 1.
    seed : int, optional
        Seed for np.random to make it reproducible.
        The default is None.

    Returns
    -------
    pileup_df - pandas DataFrame containing the pileups and their grouping information,
    if any, all possible annotations from the arguments of this function.
    """
    if by_distance:
        if by_distance is True or by_distance == "default":
            distance_edges = "default"
            by_distance = True
        elif len(by_distance) > 0:
            distance_edges = by_distance
            by_distance = True
        else:
            raise ValueError(
                "Invalid by_distance value, should be either 'default' or a list of integers"
            )
        if local:
            raise ValueError(
                "Can't do local pileups by distance, please specify only one of those arguments"
            )
    else:
        by_distance = False

    if not rescale:
        rescale_flank = None

    if seed is not None:
        np.random.seed(seed)

    if nproc == 0:
        nproc = -1
    else:
        nproc = nproc

    if view_df is None:
        # full chromosome case
        view_df = common.make_cooler_view(clr)
    else:
        # verify view is compatible with cooler
        try:
            _ = checks.is_compatible_viewframe(
                view_df,
                clr,
                check_sorting=True,
                raise_errors=True,
            )
        except Exception as e:
            raise ValueError("view_df is not a valid viewframe or incompatible") from e
    if nshifts > 0:
        control = True
    else:
        control = False

    if expected_df is None:
        expected = None
        expected_df = None
        expected_value_col = None
    else:
        expected = True
        if trans:
            try:
                _ = checks.is_valid_expected(
                    expected_df,
                    "trans",
                    view_df,
                    verify_cooler=clr,
                    expected_value_cols=[
                        expected_value_col,
                    ],
                    raise_errors=True,
                )
            except Exception as e:
                raise ValueError("provided expected is not valid") from e
        else:
            try:
                _ = checks.is_valid_expected(
                    expected_df,
                    "cis",
                    view_df,
                    verify_cooler=clr,
                    expected_value_cols=[
                        expected_value_col,
                    ],
                    raise_errors=True,
                )
            except Exception as e:
                raise ValueError("provided expected is not valid") from e

    if mindist is None:
        mindist = "auto"
    else:
        mindist = mindist

    if maxdist is None:
        maxdist = np.inf
    else:
        maxdist = maxdist

    if rescale and rescale_size % 2 == 0:
        raise ValueError("Please provide an odd rescale_size")

    chroms = list(view_df["chrom"].unique())

    if by_window:
        if features_format != "bed":
            raise ValueError("Can't make by-window pileups without making combinations")
        if local:
            raise ValueError("Can't make local by-window pileups")

    CC = CoordCreator(
        features=features,
        resolution=clr.binsize,
        features_format=features_format,
        flank=flank,
        rescale_flank=rescale_flank,
        chroms=chroms,
        minshift=minshift,
        maxshift=maxshift,
        nshifts=nshifts,
        mindist=mindist,
        maxdist=maxdist,
        local=local,
        subset=subset,
        seed=seed,
        trans=trans,
    )

    PU = PileUpper(
        clr=clr,
        CC=CC,
        view_df=view_df,
        clr_weight_name=clr_weight_name,
        expected=expected_df,
        ooe=ooe,
        control=control,
        coverage_norm=coverage_norm,
        rescale=rescale,
        rescale_size=rescale_size,
        flip_negative_strand=flip_negative_strand,
        ignore_diags=min_diag,
        store_stripes=store_stripes,
        nproc=nproc,
    )

    if by_window:
        pups = PU.pileupsByWindowWithControl()
        pups["by_window"] = True
        pups["by_strand"] = False
        pups["by_distance"] = False
    elif by_strand and by_distance:
        pups = PU.pileupsByStrandByDistanceWithControl(
            nproc=nproc, distance_edges=distance_edges, groupby=groupby
        )
        pups["by_window"] = False
        pups["by_strand"] = True
        pups["by_distance"] = True
    elif by_strand:
        pups = PU.pileupsByStrandWithControl(groupby=groupby)
        pups["by_window"] = False
        pups["by_strand"] = True
        pups["by_distance"] = False
    elif by_distance:
        pups = PU.pileupsByDistanceWithControl(
            nproc=nproc, distance_edges=distance_edges, groupby=groupby
        )
        pups["by_window"] = False
        pups["by_strand"] = False
        pups["by_distance"] = True
    else:
        pups = PU.pileupsWithControl(groupby=groupby)
        pups["by_window"] = False
        pups["by_strand"] = False
        pups["by_distance"] = False
    pups["groupby"] = [groupby] * pups.shape[0]
    pups["expected"] = pups["expected"].fillna(False)
    coolname = os.path.splitext(os.path.basename(clr.filename))[0]
    pups["cooler"] = coolname
    return pups
