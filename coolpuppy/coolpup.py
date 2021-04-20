# -*- coding: utf-8 -*-
import numpy as np
import warnings
import pandas as pd
import itertools
from multiprocessing import Pool
from functools import partial, reduce
import logging
from natsort import natsorted
from scipy import sparse
from cooltools import numutils, snipping
import yaml
import io
from more_itertools import collapse
import flammkuchen as fl


def save_pileup_df(filename, df, metadata=None):
    """
    Saves a dataframe with metadata into a binary HDF5 file using `flammkuchen`

    Parameters
    ----------
    filename : str
        File to save to.
    df : pd.DataFrame
        DataFrame to save into binary hdf5 file.
    metadata : dict, optional
        Dictionary with meatadata.

    Returns
    -------
    None.

    """
    if metadata is None:
        metadata = {}
    tosave = {"metadata": metadata, "data": df}
    fl.save(filename, tosave)


def load_pileup_df(filename):
    """
    Loads a dataframe saved using `save_pileup_df`

    Parameters
    ----------
    filename : str
        File to load from.

    Returns
    -------
    metadata : dict
    data : pd.DataFrame

    """
    loaded = fl.load(filename)
    try:
        assert "metadata" in loaded
        assert "data" in loaded
    except:
        raise ValueError("The specified file doesn't contain metadata or data")
    metadata = loaded["metadata"]
    data = loaded["data"]
    return metadata, data


def save_array_with_header(array, header, filename):
    """Save a numpy array with a YAML header generated from a dictionary

    Parameters
    ----------
    array : np.array
        Array to save.
    header : dict
        Dictionaty to save into the header.
    filename : string
        Name of file to save array and metadata into.

    """
    header = yaml.dump(header).strip()
    np.savetxt(filename, array, header=header)


def load_array_with_header(filename):
    """Load array from files generated using `save_array_with_header`.
    They are simple txt files with an optional header in the first lines, commented
    using "# ". If uncommented, the header is in YAML.

    Parameters
    ----------
    filename : string
        File to load from.

    Returns
    -------
    data : dict
        Dictionary with information from the header. Access the associated data in an
        array using data['data'].

    """
    with open(filename) as f:
        read_data = f.read()

    lines = read_data.split("\n")
    header = "\n".join([line[2:] for line in lines if line.startswith("# ")])
    if len(header) > 0:
        metadata = yaml.load(header, Loader=yaml.FullLoader)
    else:
        metadata = {}
    data = "\n".join([line for line in lines if not line.startswith("# ")])
    with io.StringIO(data) as f:
        metadata["data"] = np.loadtxt(f)
    return metadata


def corner_cv(amap, i=4):
    """Get coefficient of variation for upper left and lower right corners of a pileup
    to estimate how noisy it is

    Parameters
    ----------
    amap : 2D array
        Pileup.
    i : int, optional
        How many bins to use from each upper left and lower right corner: final corner
        shape is i^2.
        The default is 4.

    Returns
    -------
    CV : float
        Coefficient of variation for the corner pixels.

    """
    corners = np.concatenate((amap[0:i, 0:i], amap[-i:, -i:]))
    corners = corners[np.isfinite(corners)]
    return np.std(corners) / np.mean(corners)


def norm_cis(amap, i=3):
    """Normalize the pileup by mean of pixels from upper left and lower right corners

    Parameters
    ----------
    amap : 2D array
        Pileup.
    i : int, optional
        How many bins to use from each upper left and lower right corner: final corner
        shape is i^2. 0 will not normalize.
        The default is 3.

    Returns
    -------
    amap : 2D array
        Normalized pileup.

    """
    if i > 0:
        return amap / np.nanmean((amap[0:i, 0:i] + amap[-i:, -i:])) * 2
    else:
        return amap


def get_enrichment(amap, n):
    """Get values from the center of a pileup for a square with side *n*

    Parameters
    ----------
    amap : 2D array
        Pileup.
    n : int
        Side of the central square to use.

    Returns
    -------
    enrichment : float
        Mean of the pixels in the central square.

    """
    c = int(np.floor(amap.shape[0] / 2))
    return np.nanmean(amap[c - n // 2 : c + n // 2 + 1, c - n // 2 : c + n // 2 + 1])


def get_local_enrichment(amap, pad=1):
    """Get values from the central part of a pileup for a square, ignoring padding

    Parameters
    ----------
    amap : 2D array
        Pileup.
    pad : int
        Relative padding used, i.e. if 1 the central third is used, if 2 the central
        fifth is used.
        The default is 1.

    Returns
    -------
    enrichment : float
        Mean of the pixels in the central square.

    """
    c = amap.shape[0] / (pad * 2 + 1)
    assert int(c) == c
    c = int(c)
    return np.nanmean(amap[c:-c, c:-c])


def get_insulation_strength(amap, ignore_central=0, ignore_diags=2):
    """Divide values in upper left and lower right corners over upper right and lower
    left, ignoring the central bins.

    Parameters
    ----------
    amap : 2D array
        Pileup.
    ignore_central : int, optional
        How many central bins to ignore. Has to be odd or 0. The default is 0.

    Returns
    -------
    float
        Insulation strength.

    """
    for d in range(ignore_diags):
        amap = numutils.fill_diag(amap, np.nan, d)
        if d != 0:
            amap = numutils.fill_diag(amap, np.nan, -d)
    if ignore_central != 0 and ignore_central % 2 != 1:
        raise ValueError(f"ignore_central has to be odd (or 0), got {ignore_central}")
    i = (amap.shape[0] - ignore_central) // 2
    intra = np.nanmean(np.concatenate([amap[:i, :i].ravel(), amap[-i:, -i:].ravel()]))
    inter = np.nanmean(np.concatenate([amap[:i, -i:].ravel(), amap[-i:, :i].ravel()]))
    return intra / inter


def prepare_single(item):
    """Generate enrichment and corner CV, reformat into a list

    Parameters
    ----------
    item : tuple
        Key, (n, pileup).

    Returns
    -------
    list
        Concatenated list of key, n, enrichment1, enrichment3, cv3, cv5.

    """
    key, (n, amap) = item
    enr1 = get_enrichment(amap, 1)
    enr3 = get_enrichment(amap, 3)
    cv3 = corner_cv(amap, 3)
    cv5 = corner_cv(amap, 5)
    return list(key) + [n, enr1, enr3, cv3, cv5]


def bin_distance_intervals(
    intervals, band_edges=np.append([0], 50000 * 2 ** np.arange(30))
):
    edge_ids = np.searchsorted(band_edges, intervals["distance"], side="right")
    bands = [tuple(band_edges[i - 1 : i + 1]) for i in edge_ids]
    intervals["distance_band"] = bands
    return intervals


def bin_distance(snip, band_edges=50000 * 2 ** np.arange(30)):
    i = np.searchsorted(band_edges, snip["distance"])
    snip["distance_band"] = tuple(band_edges[i - 1 : i + 1])
    return snip


def group_by_region(snip):
    snip1 = snip.copy()
    snip1["group"] = tuple(snip1[["chrom1", "start1", "end1"]])
    snip2 = snip.copy()
    snip2["group"] = tuple(snip2[["chrom2", "start2", "end2"]])
    yield from (snip1, snip2)


def assign_groups(intervals, groupby=[]):
    if not groupby:
        intervals["group"] = "all"
    else:
        intervals["group"] = list(intervals[groupby].values)
    return intervals


def expand(intervals, pad, resolution, fraction_pad=None):
    if fraction_pad is None:
        intervals["exp_start"] = np.floor(intervals["center"] - pad)
        intervals["exp_end"] = np.ceil(intervals["center"] + pad + 0.5)
    else:
        intervals["exp_start"] = np.floor(
            intervals["center"]
            - (intervals["center"] - intervals["start"]) * fraction_pad
        )
        intervals["exp_end"] = np.ceil(
            intervals["center"]
            + (intervals["end"] - intervals["center"]) * fraction_pad
            + 0.5
        )
    intervals[["exp_start", "exp_end"]] = intervals[["exp_start", "exp_end"]].astype(
        int
    )
    return intervals


def expand2D(intervals, pad, fraction_pad=None):
    if fraction_pad is None:
        intervals["exp_start1"] = np.floor(intervals["center1"] - pad)
        intervals["exp_end1"] = np.ceil(intervals["center1"] + pad + 0.5)
        intervals["exp_start2"] = np.floor(intervals["center2"] - pad)
        intervals["exp_end2"] = np.ceil(intervals["center2"] + pad + 0.5)
    else:
        intervals["exp_start1"] = np.floor(
            intervals["start1"]
            - (intervals["center1"] - intervals["start1"]) * fraction_pad
        )
        intervals["exp_end1"] = np.ceil(
            intervals["end1"]
            + (intervals["end1"] - intervals["center1"]) * fraction_pad
            + 0.5
        )
        intervals["exp_start2"] = np.floor(
            intervals["start2"]
            - (intervals["center2"] - intervals["start2"]) * fraction_pad
        )
        intervals["exp_end2"] = np.ceil(
            intervals["end2"]
            + (intervals["end2"] - intervals["center2"]) * fraction_pad
            + 0.5
        )
    intervals[["exp_start1", "exp_end1", "exp_start2", "exp_end2"]] = intervals[
        ["exp_start1", "exp_end1", "exp_start2", "exp_end2"]
    ].astype(int)
    return intervals


def combine_rows(row1, row2, normalize_order=True):
    d = row2.center - row1.center
    if d < 0 and normalize_order:
        row1, row2 = row2, row1
        d = -d
    # row1.index = [i+'1' for i in row1.index]
    # row2.index = [i+'2' for i in row2.index]
    double_row = pd.Series(
        index=[i + "1" for i in row1.index]
        + [i + "2" for i in row2.index]
        + ["distance"],
        data=np.concatenate([row1.values, row2.values, [d]]),
    )
    # double_row['distance'] = d
    return double_row


def _add_snip(outdict, key, snip):
    if key not in outdict:
        outdict[key] = snip[["data", "cov_start", "cov_end"]]
        outdict[key]["num"] = np.isfinite(snip["data"]).astype(int)
        outdict[key]["n"] = 1
    else:
        outdict[key]["data"] = np.nansum([outdict[key]["data"], snip["data"]], axis=0)
        outdict[key]["num"] += np.isfinite(snip["data"]).astype(int)
        outdict[key]["cov_start"] = np.nansum(
            [outdict[key]["cov_start"], snip["cov_start"]], axis=0
        )
        outdict[key]["cov_end"] = np.nansum(
            [outdict[key]["cov_end"], snip["cov_end"]], axis=0
        )
        outdict[key]["n"] += 1


def sum_pups(pup1, pup2):
    """
    Only preserves data, cov_start, cov_end, n and num
    Assumes n=1 if not present, and calculates num if not present
    """
    pup = {
        "data": pup1["data"] + pup2["data"],
        "cov_start": pup1["cov_start"] + pup2["cov_start"],
        "cov_end": pup1["cov_end"] + pup2["cov_end"],
        "n": pup1.get("n", 1) + pup2.get("n", 1),
        "num": pup1.get("num", np.isfinite(pup1["data"]).astype(int))
        + pup2.get("num", np.isfinite(pup2["data"]).astype(int)),
    }
    return pd.Series(pup)


def norm_coverage(snip):
    """Normalize a pileup by coverage arrays

    Parameters
    ----------
    loop : 2D array
        Pileup.
    cov_start : 1D array
        Accumulated coverage of the left side of the pileup.
    cov_end : 1D array
        Accumulated coverage of the bottom side of the pileup.

    Returns
    -------
    loop : 2D array
        Normalized pileup.

    """
    coverage = np.outer(snip["cov_start"], snip["cov_end"])
    coverage = coverage / np.nanmean(coverage)
    snip["data"] /= coverage
    snip["data"][np.isnan(snip["data"])] = 0
    return snip


class CoordCreator:
    def __init__(
        self,
        baselist,
        resolution,
        *,
        basetype="auto",
        anchor=False,
        pad=100000,
        fraction_pad=None,
        chroms="all",
        minshift=10 ** 5,
        maxshift=10 ** 6,
        nshifts=10,
        mindist="auto",
        maxdist=None,
        local=False,
        subset=0,
        seed=None,
    ):
        """Generator of coordinate pairs for pileups.

        Parameters
        ----------
        baselist : DataFrame
            A bed- or bedpe-style file with coordinates.
        resolution : int, optional
            Data resolution.
        basetype : str, optional
            Format of the baselist. Options:
                bed: chrom, start, end
                bedpe: chrom1, start1, end1, chrom2, start2, end2
                auto (default): determined from the columns in the DataFrame
        anchor : tuple of (str, int, int), optional
            Coordinates (chr, start, end) of an anchor region used to create
            interactions with baselist (in bp). Anchor is on the left of the final pileup.
            The default is False.
        pad : int, optional
            Padding around the central bin, in bp. For example, with 5000 bp resolution
            and 100000 pad, final pileup is 205000×205000 bp.
            The default is 100000.
        fraction_pad : float, optional
            Fraction of ROI size added on each end when extracting snippets, if rescale.
            The default is None. If specified, overrides pad.
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
            2 * pad + 2 * resolution
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

        Returns
        -------
        Object that generates coordinates for pileups required for PileUpper.

        """
        self.intervals = baselist
        # self.stdin = self.intervals == sys.stdin
        self.resolution = resolution
        self.basetype = basetype
        self.anchor = anchor
        self.pad = pad
        # self.pad_bins = pad // self.resolution
        self.fraction_pad = fraction_pad
        self.chroms = chroms
        self.minshift = minshift
        self.maxshift = maxshift
        self.nshifts = nshifts
        if mindist == "auto":
            self.mindist = 2 * self.pad + 2 * self.resolution
        else:
            self.mindist = mindist
        if maxdist is None:
            self.maxdist = np.inf
        else:
            self.maxdist = maxdist
        self.local = local
        self.subset = subset
        self.seed = seed
        self.process()

    def process(self):
        if self.basetype is None or self.basetype == "auto":
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
            self.kind = self.basetype

        if self.intervals.shape[0] == 0:
            warnings.warn("No regions in baselist, returning empty output")
            self.pos_stream = self.empty_stream
            self.final_chroms = []
            return

        if self.subset > 0:
            self.intervals = self._subset(self.intervals)

        if self.anchor:
            assert self.kind == "bed"
            self.intervals = pd.DataFrame(
                {
                    "chrom": self.anchor[0],
                    "start": self.anchor[1],
                    "end": self.anchor[2],
                }
            )

        if self.kind == "bed":
            assert all(
                [name in self.intervals.columns for name in ["chrom", "start", "end"]]
            )
            self.intervals["center"] = (
                self.intervals["start"] + self.intervals["end"]
            ) / 2
            self.intervals = expand(self.intervals, self.pad, self.fraction_pad)
        else:
            assert all(
                [
                    name in self.intervals.columns
                    for name in ["chrom1", "start1", "end1", "chrom2", "start2", "end2"]
                ]
            )
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
            self.intervals = expand2D(self.intervals, self.pad, self.fraction_pad)

        if self.nshifts > 0 and self.kind == "bedpe":
            self.intervals = self._control_regions(self.intervals)

        if self.kind == "bed":
            basechroms = set(self.intervals["chrom"])
            # if self.anchor:
            #     if self.anchor[0] not in basechroms:
            #         raise ValueError(
            #             """The anchor chromosome is not found in the baselist.
            #                Are they in the same format, e.g. starting with "chrom"?
            #                Alternatively, all regions in that chromosome might have
            #                been filtered by some filters."""
            #         )
            #     else:
            #         basechroms = [self.anchor[0]]
        else:
            if self.anchor:
                raise ValueError("Can't use anchor with both sides of loops defined")
            elif self.local:
                raise ValueError("Can't make local with both sides of loops defined")
            basechroms = set(self.intervals["chrom1"]) & set(self.intervals["chrom2"])
        self.basechroms = natsorted(list(basechroms))
        if self.chroms == "all":
            self.final_chroms = natsorted(list(basechroms))
        else:
            self.final_chroms = natsorted(list(set(self.chroms) & set(self.basechroms)))
        if len(self.final_chroms) == 0:
            raise ValueError(
                """No chromosomes are in common between the coordinate
                   file/anchor and the cooler file. Are they in the same
                   format, e.g. starting with "chr"?
                   """
            )

        self.intervals = self._binnify(self.intervals)

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

    # def filter_bed(self, df):
    #     length = df["end"] - df["start"]
    #     df = df[(length >= self.minsize) & (length <= self.maxsize)]
    #     if self.chroms != "all":
    #         df = df[df["chrom"].isin(self.chroms)]
    #     if not np.all(df["end"] >= df["start"]):
    #         raise ValueError("Some ends in the file are smaller than starts")
    #     return df

    def filter_bedpe(self, df):
        mid1 = np.mean(df[["start1", "end1"]], axis=1)
        mid2 = np.mean(df[["start2", "end2"]], axis=1)
        length = mid2 - mid1
        df = df[(length >= self.mindist) & (length <= self.maxdist)]
        if self.chroms != "all":
            df = df[(df["chrom1"].isin(self.chroms)) & (df["chrom2"].isin(self.chroms))]
        return df

    # def auto_read_bed(
    #     self, file, kind="auto",
    # ):
    #     row1 = None
    #     if kind == "auto":  # Guessing whether it's bed or bedpe style file
    #         if self.stdin:
    #             row1 = file.__next__().split("\t")
    #         else:
    #             with open(file, "r") as fobject:
    #                 row1 = fobject.readline().split("\t")
    #         if len(row1) == 6:
    #             try:
    #                 row1 = [
    #                     row1[0],
    #                     int(row1[1]),
    #                     int(row1[2]),
    #                     row1[3],
    #                     int(row1[4]),
    #                     int(row1[5]),
    #                 ]
    #                 kind = "bedpe"
    #             except:
    #                 raise ValueError(
    #                     "Can't determine the type of baselist file,"
    #                     "please specify bed or bedpe"
    #                 )
    #         elif len(row1) == 3:
    #             try:
    #                 row1 = [row1[0], int(row1[1]), int(row1[2])]
    #                 kind = "bed"
    #             except:
    #                 raise ValueError(
    #                     "Can't determine the type of baselist file,"
    #                     "please specify bed or bedpe"
    #                 )
    #         else:
    #             raise ValueError(
    #                 f"""Input bed(pe) file has unexpected number of
    #                     columns: got {len(row1)}, expect 3 (bed) or 6 (bedpe)
    #                     """
    #             )

    #     if kind == "bed":
    #         filter_func = self.filter_bed
    #         names = ["chrom", "start", "end"]
    #         dtype = {"chrom": "str", "start": "int", "end": "int"}
    #         if row1 is not None:
    #             row1 = filter_func(
    #                 pd.DataFrame([row1], columns=names).astype(dtype=dtype)
    #             )
    #     elif kind == "bedpe":  # bedpe
    #         filter_func = self.filter_bedpe
    #         names = ["chrom1", "start1", "end1", "chrom2", "start2", "end2"]
    #         dtype = {
    #             "chrom1": "str",
    #             "start1": "int",
    #             "end1": "int",
    #             "chrom2": "str",
    #             "start2": "int",
    #             "end2": "int",
    #         }
    #         if row1 is not None:
    #             row1 = filter_func(
    #                 pd.DataFrame([row1], columns=names).astype(dtype=dtype)
    #             )
    #     else:
    #         raise ValueError(
    #             f"""Unsupported input kind: {kind}.
    #                          Expect auto, bed or bedpe"""
    #         )
    #     bases = []

    #     if self.stdin:
    #         if row1 is not None:
    #             if row1.shape[0] == 1:
    #                 bases.append(row1)

    #     for chunk in pd.read_csv(
    #         file,
    #         sep="\t",
    #         names=names,
    #         index_col=False,
    #         chunksize=10 ** 4,
    #         dtype=dtype,
    #     ):
    #         bases.append(filter_func(chunk))
    #     bases = pd.concat(bases)
    #     if (
    #         not self.stdin
    #     ):  # Would mean we read it twice when checking and in the first chunk
    #         bases = bases.iloc[1:]

    #         # bases["chrom"] = bases["chrom"].astype(str)
    #         # bases[["start", "end"]] = bases[["start", "end"]].astype(np.uint64)

    #         # bases[["chrom1", "chrom2"]] = bases[["chrom1", "chrom2"]].astype(str)
    #         # bases[["start1", "end1", "start2", "end2"]] = bases[
    #         # ["start1", "end1", "start2", "end2"]
    #         # ].astype(np.uint64)
    #     return bases, kind

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

    def _binnify_region(self, region):
        start_bin = np.floor(region[1] / self.resolution).astype(int)
        end_bin = np.ceil(region[2] / self.resolution).astype(int)
        return region[0], start_bin, end_bin

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
            # intervals = intervals.drop_duplicates(
            #     ["chrom", "stBin", 'endBin']
            # )
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
            # intervals = intervals.drop_duplicates(
            #     ["chrom", "stBin1", 'endBin1', 'stBin2', 'endBin2']
            # )
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

    def filter_func_region(self, region):
        if self.kind == "bed":
            return partial(self._filter_func_region, region=region)
        else:
            return partial(self._filter_func_pairs_region, region=region)

    def _get_combinations(
        self,
        filter_func,
        intervals=None,
        control=False,
        groupby=[],
        modify_2Dintervals_func=None,
    ):
        if intervals is None:
            intervals = self.intervals
        intervals = filter_func(self.intervals)
        if not len(intervals) >= 1:
            logging.debug("Empty selection")
            yield None

        if self.local:
            merged = pd.merge(
                intervals,
                intervals,
                left_index=True,
                right_index=True,
                suffixes=["1", "2"],
            )
            merged = self._control_regions(merged, self.nshifts * control)
            if modify_2Dintervals_func is not None:
                merged = modify_2Dintervals_func(merged)
            merged = assign_groups(merged, groupby=groupby)
            merged = merged.reindex(
                columns=list(merged.columns) + ["data", "cov_start", "cov_end"]
            )
            for _, row in merged.iterrows():
                yield row
        else:  # all combinations
            intervals_left = intervals.rename(columns=lambda x: x + "1")
            intervals_right = intervals.rename(columns=lambda x: x + "2")
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
                if modify_2Dintervals_func is not None:
                    combinations = modify_2Dintervals_func(combinations)
                combinations = assign_groups(combinations, groupby=groupby)
                combinations = combinations.reindex(
                    columns=list(combinations.columns)
                    + ["data", "cov_start", "cov_end"]
                )
                for _, row in combinations.iterrows():
                    yield row

    def get_combinations(
        self,
        filter_func,
        intervals=None,
        control=False,
        groupby=[],
        modify_2Dintervals_func=None,
    ):
        stream = self._get_combinations(
            filter_func,
            intervals,
            control=control,
            groupby=groupby,
            modify_2Dintervals_func=modify_2Dintervals_func,
        )
        # if not self.local:
        #     stream = self.filter_pos_stream_distance(stream)
        return stream

    def get_intervals_stream(
        self,
        filter_func,
        intervals=None,
        control=False,
        groupby=[],
        modify_2Dintervals_func=None,
    ):
        if intervals is None:
            intervals = self.intervals
        intervals = self._control_regions(intervals, self.nshifts * control)
        if modify_2Dintervals_func is not None:
            intervals = modify_2Dintervals_func(filter_func(intervals))
        intervals = assign_groups(intervals, groupby)
        intervals = intervals.reindex(
            columns=list(intervals.columns) + ["data", "cov_start", "cov_end"]
        )
        if not len(intervals) >= 1:
            logging.debug("Empty selection")
            yield None
        for i, interval in intervals.iterrows():
            yield interval

    # def _get_position_pairs_stream(self, filter_func, intervals=None):
    #     if intervals is None:
    #         intervals = self.intervals
    #     intervals = filter_func(intervals)
    #     if not len(intervals) >= 1:
    #         logging.debug("Empty selection")
    #         yield None
    #     for i, row in itertools.combinations(intervals.index, intervals.index):
    #         yield intervals.iloc[i], intervals.iloc[j]

    # def get_position_pairs_stream(self, filter_func, intervals=None):
    #     stream = self.get_intervals_stream(filter_func, intervals)
    #     if not self.local:
    #         stream = self.filter_pos_stream_distance(stream)
    #     return stream

    # def control_regions(self, filter_func, pos_pairs=None):
    #     if self.seed is not None:
    #         np.random.seed(self.seed)
    #     if pos_pairs is None:
    #         source = self.pos_stream(filter_func)
    #     else:
    #         source = pos_pairs
    #     # try:
    #     #     row1 = next(source)
    #     # except StopIteration:
    #     #     logging.debug("Empty selection")
    #     #     raise StopIteration
    #     # else:
    #     #     source = itertools.chain([row1], source)
    #     for orig_row in source:
    #         shift = np.random.randint(self.minshift, self.maxshift, self.nshifts)
    #         sign = np.random.choice([-1, 1], self.nshifts)
    #         shift *= sign
    #         controls = pd.DataFrame([orig_row]*self.nshifts)
    #         controls[['start1', 'start2', 'end1', 'end2']] = controls[['start1', 'start2', 'end1', 'end2']] + shift[:, np.newaxis]
    #         controls[['stBin1', 'endBin1', 'stBin2', 'endBin2']] = controls[['stBin1', 'endBin1', 'stBin2', 'endBin2']] + (shift // self.resolution)[:, np.newaxis]
    #         for i, row in controls.iterrows():
    #             yield row

    # def get_combinations_by_window(self, chrom, ctrl=False, intervals=None):
    #     assert self.kind == "bed"
    #     if intervals is None:
    #         chrmids = self.filter_func_chrom(chrom)(self.intervals)
    #     else:
    #         chrmids = self.filter_func_chrom(chrom)(intervals)
    #     for i, (b, m, p) in chrmids[["Bin", "Mids", "Pad"]].astype(int).iterrows():
    #         out_stream = self.get_combinations(
    #             self.filter_func_all, intervals=chrmids, anchor=(chrom, m, m)
    #         )
    #         if ctrl:
    #             out_stream = self.control_regions(self.filter_func_all, out_stream)
    #         yield (m - p, m + p), out_stream

    # def filter_pos_stream_distance(self, stream):
    #     for row in stream:
    #         if self.mindist <= abs(row['distance']) <= self.maxdist:
    #             yield row

    def empty_stream(self, *args, **kwargs):
        yield from ()

    # def _chrom_mids(self, chroms, intervals):
    #     for chrom in chroms:
    #         if self.kind == "bed":
    #             yield chrom, intervals[intervals["chrom"] == chrom]
    #         else:
    #             yield chrom, intervals[
    #                 (intervals["chrom1"] == chrom) & (intervals["chrom2"] == chrom)
    #             ]

    # def chrom_mids(self):
    #     chrommids = self._chrom_mids(self.final_chroms, self.intervals)
    #     if self.intervals2 is not None:
    #         chrommids2 = self._chrom_mids(self.final_chroms, self.intervals2)
    #     else:
    #         chrommids2 = zip(itertools.cycle([None]), itertools.cycle([None]))
    #     chrommids = zip(chrommids, chrommids2)
    #     for i in chrommids:
    #         yield i


class PileUpper:
    def __init__(
        self,
        clr,
        CC,
        *,
        regions=None,
        balance="weight",
        expected=False,
        ooe=True,
        control=False,
        coverage_norm=False,
        rescale_size=99,
        ignore_diags=2,
    ):
        """Creates pileups


        Parameters
        ----------
        clr : cool
            Cool file with Hi-C data.
        CC : CoordCreator
            CoordCreator object with correct settings.
        balance : bool or str, optional
            Whether to use balanced data, and which column to use as weights.
            The default is "weight".
        expected : DataFrame, optional
            If using expected, pandas DataFrame with chromosome-wide expected.
            The default is False.
        regions : DataFrame
            A datafrome with region coordinates used in expected
        ooe : bool, optional
            Whether to normalize each snip by expected value. If False, all snips are
            accumulated, all expected values are accumulated, and then the former
            divided by the latter - like with randomly shifted controls. Only has effect
            when expected is provided.
        control : bool, optional
            Whether to use randomly shifted controls.
            The default is False.
        coverage_norm : bool, optional
            Whether to normalize final the final pileup by accumulated coverage as an
            alternative to balancing. Useful for single-cell Hi-C data.
            The default is False.
        rescale_size : int, optional
            Final shape of rescaled pileups. E.g. if 99, pileups will be squares of
            99×99 pixels.
            The default is 99.
        ignore_diags : int, optional
            How many diagonals to ignore to avoid short-distance artefacts.
            The default is 2.

        Returns
        -------
        Object that generates pileups.

        """
        self.clr = clr
        self.resolution = self.clr.binsize
        self.CC = CC
        assert self.resolution == self.CC.resolution
        self.__dict__.update(self.CC.__dict__)
        self.balance = balance
        self.expected = expected
        if regions is None:
            if set(self.expected["region"]).issubset(clr.chromnames):
                regions = pd.DataFrame(
                    [(chrom, 0, l, chrom) for chrom, l in clr.chromsizes.items()],
                    columns=["chrom", "start", "end", "name"],
                )
            else:
                raise ValueError(
                    "Please provide the regions table, if region names"
                    "are not simply chromosome names."
                )
        self.regions = regions.set_index("name")
        self.region_extents = {}
        for region_name, region in self.regions.iterrows():
            lo, hi = self.clr.extent(region)
            chroffset = self.clr.offset(region[0])
            self.region_extents[region_name] = lo - chroffset, hi - chroffset

        if self.expected is not False:
            for region_name, group in self.expected.groupby("region"):
                n_diags = group.shape[0]
                region = self.regions.loc[region_name]
                lo, hi = self.region_extents[region_name]
                if n_diags != (hi - lo):
                    raise ValueError(
                        "Region shape mismatch between expected and cooler. "
                        "Are they using the same resolution?"
                    )

        self.ooe = ooe
        self.control = control
        self.pad_bins = self.CC.pad // self.resolution
        self.coverage_norm = coverage_norm
        self.rescale = self.CC.fraction_pad is not None
        self.rescale_size = rescale_size
        self.ignore_diags = ignore_diags
        # self.CoolSnipper = snipping.CoolerSnipper(
        #     self.clr, cooler_opts=dict(balance=self.balance)
        # )

        self.chroms = natsorted(
            list(set(self.CC.final_chroms) & set(self.clr.chromnames))
        )
        self.regions = self.regions[self.regions["chrom"].isin(self.chroms)]
        # self.regions = {
        #     chrom: (chrom, 0, self.clr.chromsizes[chrom])
        #     for chrom in self.chroms  # cooler.util.parse_region_string(chrom) for chrom in self.chroms
        # }
        if self.expected is not False:
            if self.control:
                warnings.warn(
                    "Can't do both expected and control shifts; defaulting to expected"
                )
                self.control = False
            assert isinstance(self.expected, pd.DataFrame)
            self.expected = self.expected[
                self.expected["region"].isin(self.regions.index)
            ]
            self.ExpSnipper = snipping.ExpectedSnipper(
                self.clr, self.expected, regions=self.regions.reset_index()
            )
            self.expected_selections = {
                region_name: self.ExpSnipper.select(region_name, region_name)
                for region_name in self.regions.index
            }
            self.expected = True

    # def get_matrix(self, matrix, chrom, left_interval, right_interval):
    #     lo_left, hi_left = left_interval
    #     lo_right, hi_right = right_interval
    #     lo_left *= self.resolution
    #     hi_left *= self.resolution
    #     lo_right *= self.resolution
    #     hi_right *= self.resolution
    #     matrix = self.CoolSnipper.snip(
    #         matrix,
    #         self.regions[chrom],
    #         self.regions[chrom],
    #         (lo_left, hi_left, lo_right, hi_right),
    #     )
    #     return matrix

    def get_expected_matrix(self, region, left_interval, right_interval):
        """Generate expected matrix for a region

        Parameters
        ----------
        region : str
            Region name.
        left_interval : tuple
            Tuple of (chrom, start, end) of the snip on the left side
        right_interval : tuple
            Tuple of (chrom, start, end) of the snip on the right side

        Returns
        -------
        exp_matrix : array
            Array of expected values for the selected coordinates.

        """
        lo_left, hi_left = left_interval
        lo_right, hi_right = right_interval
        exp_matrix = self.ExpSnipper.snip(
            self.expected_selections[region],
            region,
            region,
            (lo_left, hi_left, lo_right, hi_right),
        )
        return exp_matrix

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

    def get_data(self, region):
        """Get sparse data for a region

        Parameters
        ----------
        region : tuple or str
            Region for which to load the data. Either tuple of (chr, start, end), or
            string with region name.

        Returns
        -------
        data : csr
            Sparse csr matrix for the corresponding region.

        """
        logging.debug("Loading data")
        if isinstance(region, str):
            region = self.regions.loc[region]
        data = self.clr.matrix(sparse=True, balance=self.balance).fetch(region)
        data = sparse.triu(data)
        return data.tocsr()

    def get_coverage(self, data):
        """Get total coverage profile for upper triangular data

        Parameters
        ----------
        data : array_like
            2D array with upper triangular data.

        Returns
        -------
        coverage : array
            1D array of coverage.

        """
        coverage = np.nan_to_num(np.ravel(data.sum(axis=0))) + np.nan_to_num(
            np.ravel(data.sum(axis=1))
        )
        return coverage

    def _stream_snips(
        self,
        intervals,
        region,
    ):
        mymap = self.make_outmap()
        cov_start = np.zeros(mymap.shape[0])
        cov_end = np.zeros(mymap.shape[1])
        try:
            row1 = next(intervals)
        except StopIteration:
            logging.info(f"Nothing to sum up in region {region}")
            return mymap, mymap, cov_start, cov_end, 0
        intervals = itertools.chain([row1], intervals)

        bigdata = self.get_data(
            region
        )  # self.CoolSnipper.select(self.regions[chrom], self.regions[chrom])
        min_left, max_right = self.region_extents[region]
        if self.coverage_norm:
            coverage = self.get_coverage(bigdata)

        ar = np.arange(max_right - min_left, dtype=np.int32)
        diag_indicator = numutils.LazyToeplitz(-ar, ar)

        for snip in intervals:
            snip[["stBin1", "endBin1", "stBin2", "endBin2"]] -= min_left
            if snip["stBin1"] < 0 or snip["endBin2"] > (max_right - min_left):
                continue
            data = (
                bigdata[
                    snip["stBin1"] : snip["endBin1"], snip["stBin2"] : snip["endBin2"]
                ]
                .toarray()
                .astype(float)
            )
            if self.expected:
                exp_data = self.get_expected_matrix(
                    region,
                    (snip["exp_start1"], snip["exp_end1"]),
                    (snip["exp_start2"], snip["exp_end2"]),
                )
                if not self.ooe:
                    exp_snip = snip.copy()
                    exp_snip["kind"] = "control"
                    exp_snip["data"] = exp_data
            D = (
                diag_indicator[
                    snip["stBin1"] : snip["endBin1"], snip["stBin2"] : snip["endBin2"]
                ]
                < self.ignore_diags
            )
            data[D] = np.nan
            if self.local:
                data = np.nansum(np.dstack((data, data.T)), 2)
            if self.coverage_norm:
                cov_start = coverage[snip["stBin1"] : snip["endBin1"]]
                cov_end = coverage[snip["stBin2"] : snip["endBin2"]]
                snip["cov_start"] = cov_start
                snip["cov_end"] = cov_end
            if self.expected and self.ooe:
                data = data / exp_data
            snip["data"] = data

            if self.rescale:
                snip = self._rescale_snip(snip)
                if self.expected and not self.ooe:
                    exp_snip = self._rescale_snip(exp_snip)

            yield snip

            if self.expected and not self.ooe:
                exp_snip["data"] = exp_data
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
            snip["data"] = numutils.zoom_array(
                snip["data"], (self.rescale_size, self.rescale_size)
            )
        snip["cov_start"] = numutils.zoom_array(snip["cov_start"], (self.rescale_size,))
        snip["cov_end"] = numutils.zoom_array(snip["cov_end"], (self.rescale_size,))
        return snip

    def accumulate_stream(self, snip_stream, postprocess_func=None):
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
            Can be used to modify the data in un-standard way, or creqate groups when
            it can't be done before snipping, or to assign each snippet to multiple
            groups. Example: `group_by_region`.

        Returns
        -------
        outdict : dict
            Dictionary of accumulated snips (each as a Series) for each group.
            Always includes "all"
        """

        if postprocess_func is not None:
            snip_stream = map(postprocess_func, snip_stream)
        outdict = {"ROI": {}, "control": {}}
        for snip in collapse(snip_stream, base_type=pd.Series):
            kind = snip["kind"]
            key = snip["group"]
            if isinstance(key, str):
                _add_snip(outdict[kind], key, snip)
            else:
                _add_snip(outdict[kind], tuple(key), snip)
        if "all" not in outdict["ROI"]:
            outdict["ROI"]["all"] = reduce(sum_pups, outdict["ROI"].values())
        if self.control or (self.expected and not self.ooe):
            if "all" not in outdict["control"]:
                outdict["control"]["all"] = reduce(
                    sum_pups, outdict["control"].values()
                )
        return outdict

    def pileup_region(
        self,
        region,
        groupby=[],
        modify_2Dintervals_func=None,
        postprocess_func=None,
    ):
        """

        Parameters
        ----------
        region : str
            Region name.
        groupby : str or list of str, optional
            Which attributes of each snip to assign a group to it
        modify_2Dintervals_func : function, optional
            A function to apply to a dataframe of genomic intervals used for pileups.
            If possible, much preferable to `postprocess_func` for better speed.
            Good example is the `bin_distance_intervals` function above.
        postprocess_func : function, optional
            Additional function to apply to each snippet before grouping.
            Good example is the `bin_distance` function above, but using
            bin_distance_intervals as modify_2Dintervals_func is much faster.

        Returns
        -------
        pileup : dict
            accumulated snips as a dict
        """
        region_coords = self.regions.loc[region]

        # mymap = self.make_outmap()
        # cov_start = np.zeros(mymap.shape[0])
        # cov_end = np.zeros(mymap.shape[1])

        if self.anchor:
            assert region_coords[0] == self.anchor[0]
            logging.info(
                f"Anchor: {region_coords[0]}:{self.anchor[1]}-{self.anchor[2]}"
            )

        filter_func = self.CC.filter_func_region(region=region_coords)

        intervals = self.CC.pos_stream(
            filter_func,
            control=self.control,
            groupby=groupby,
            modify_2Dintervals_func=modify_2Dintervals_func,
        )
        final = self.accumulate_stream(
            self._stream_snips(
                intervals=intervals,
                region=region,
            ),
            postprocess_func=postprocess_func,
        )
        logging.info(f"{region}: {final['ROI']['all']['n']}")
        return final

    def pileupsWithControl(
        self, nproc=1, groupby=[], modify_2Dintervals_func=None, postprocess_func=None
    ):
        """Perform pileups across all chromosomes and applies required
        normalization

        Parameters
        ----------
        nproc : int, optional
            How many cores to use. Sends a whole chromosome per process.
            The default is 1.
        groupby : str or list of str, optional
            Which attributes of each snip to assign a group to it
        modify_2Dintervals_func : function, optional
            Function to apply to the DataFrames of coordinates before fetching snippets
            based on them. Preferable to using the `postprocess_func`, since at the
            earlier stage it can be vectorized and much more efficient.
        postprocess_func : function, optional
            Additional function to apply to each snippet before grouping.
            Good example is the `bin_distance` function above.

        Returns
        -------
        pileup_df : 2D array
            Normalized pileups in a pandas DataFrame, with columns `data` and `num`.
            `data` contains the normalized pileups, and `num` - how many snippets were
            combined (the regions of interest, not control regions). Each condition
            from `groupby` is a row, plus an additional row `all` is created with all
            data.

        """
        if len(self.chroms) == 0:
            return self.make_outmap(), 0

        if nproc > 1:
            p = Pool(nproc)
            mymap = p.map
        else:
            mymap = map
        # Loops
        f = partial(
            self.pileup_region,
            groupby=groupby,
            modify_2Dintervals_func=modify_2Dintervals_func,
            postprocess_func=postprocess_func,
        )
        pileups = list(mymap(f, self.regions.index))
        roi = (
            pd.DataFrame([pileup["ROI"] for pileup in pileups])
            .apply(lambda x: reduce(sum_pups, x.dropna()))
            .T
        )
        if self.control or (self.expected and not self.ooe):
            ctrl = (
                pd.DataFrame([pileup["control"] for pileup in pileups])
                .apply(lambda x: reduce(sum_pups, x.dropna()))
                .T
            )
        if self.coverage_norm:
            roi = roi.apply(norm_coverage, axis=1)
            if self.control:
                ctrl = ctrl.apply(norm_coverage, axis=1)
            elif self.expected:
                warnings.warn("Expected can not be normalized to coverage")
        normalized_roi = pd.DataFrame(roi["data"] / roi["num"], columns=["data"])
        if self.control or (self.expected and not self.ooe):
            normalized_control = pd.DataFrame(
                ctrl["data"] / ctrl["num"], columns=["data"]
            )
            normalized_roi = normalized_roi / normalized_control
        normalized_roi["n"] = roi["n"]
        if nproc > 1:
            p.close()
        # pileup[~np.isfinite(pileup)] = 0

        if groupby:
            normalized_roi = normalized_roi.reset_index()
            normalized_roi[groupby] = pd.DataFrame(
                [
                    ("all",) * len(groupby) if i == "all" else i
                    for i in normalized_roi["index"].to_list()
                ],
                columns=groupby,
            )
            normalized_roi = normalized_roi.drop(columns="index")
            normalized_roi = normalized_roi.set_index(groupby)
            n = normalized_roi.loc["all", "n"].values[0]
        else:
            n = normalized_roi.loc["all", "n"]
        logging.info(f"Total number of piled up windows: {n}")
        return normalized_roi

    def pileupsByWindowWithControl(
        self,
        nproc=1,
    ):
        """Perform by-window pileups across all chromosomes and applies required
        normalization. Simple wrapper around pileupsWithControl

        Parameters
        ----------
        nproc : int, optional
            How many cores to use. Sends a whole chromosome per process.
            The default is 1.

        Returns
        -------
        pileup_df : 2D array
            Normalized pileups in a pandas DataFrame, with columns `data` and `num`.
            `data` contains the normalized pileups, and `num` - how many snippets were
            combined (the regions of interest, not control regions). Each window is a
            row, plus an additional row `all` is created with all data.
        """
        normalized_pileups = self.pileupsWithControl(
            nproc=nproc, postprocess_func=group_by_region
        )
        normalized_pileups = normalized_pileups.drop(index="all").reset_index()
        normalized_pileups[["chrom", "start", "end"]] = pd.DataFrame(
            normalized_pileups["index"].to_list(), index=normalized_pileups.index
        )
        normalized_pileups.drop(columns="index")
        return normalized_pileups

    def pileupsByDistanceWithControl(
        self, nproc=1, distance_edges=np.append([0], 50000 * 2 ** np.arange(30))
    ):
        """Perform by-distance pileups across all chromosomes and applies required
        normalization. Simple wrapper around pileupsWithControl

        Parameters
        ----------
        nproc : int, optional
            How many cores to use. Sends a whole chromosome per process.
            The default is 1.
        distance_edges : list/array of int
            How to group snips by distance (based on their centres).
            Default uses separations [0, 50_000, 100_000, 200_000, ...]

        Returns
        -------
        pileup_df : 2D array
            Normalized pileups in a pandas DataFrame, with columns `data` and `num`.
            `data` contains the normalized pileups, and `num` - how many snippets were
            combined (the regions of interest, not control regions).
            Each distance band is a row, annotated in column `distance_band`
        """
        bin_func = partial(bin_distance_intervals, band_edges=distance_edges)
        normalized_pileups = self.pileupsWithControl(
            nproc=nproc, modify_2Dintervals_func=bin_func, groupby=["distance_band"]
        )
        normalized_pileups = normalized_pileups.drop(index="all").reset_index()
        return normalized_pileups

    def pileupsByStrandByDistanceWithControl(
        self, nproc=1, distance_edges=np.append([0], 50000 * 2 ** np.arange(30))
    ):
        """Perform by-strand by-distance pileups across all chromosomes and applies
        required normalization. Simple wrapper around pileupsWithControl.
        Assumes the baselist in CoordCreator file has a "strand" column.

        Parameters
        ----------
        nproc : int, optional
            How many cores to use. Sends a whole chromosome per process.
            The default is 1.
        distance_edges : list/array of int
            How to group snips by distance (based on their centres).
            Default uses separations [0, 50_000, 100_000, 200_000, ...]

        Returns
        -------
        pileup_df : 2D array
            Normalized pileups in a pandas DataFrame, with columns `data` and `num`.
            `data` contains the normalized pileups, and `num` - how many snippets were
            combined (the regions of interest, not control regions).
            Each distance band is a row, annotated in columns `separation`
        """
        bin_func = partial(bin_distance_intervals, band_edges=distance_edges)
        normalized_pileups = self.pileupsWithControl(
            nproc=nproc,
            modify_2Dintervals_func=bin_func,
            groupby=["strand1", "strand2", "distance_band"],
        )
        normalized_pileups = normalized_pileups.drop(index="all").reset_index()
        normalized_pileups["orientation"] = (
            normalized_pileups["strand1"] + normalized_pileups["strand2"]
        )
        normalized_pileups = normalized_pileups[
            ["orientation", "distance_band", "data", "n"]
        ]
        return normalized_pileups

    def pileupsByStrandWithControl(self, nproc=1):
        """Perform by-strand pileups across all chromosomes and applies required
        normalization. Simple wrapper around pileupsWithControl.
        Assumes the baselist in CoordCreator file has a "strand" column.

        Parameters
        ----------
        nproc : int, optional
            How many cores to use. Sends a whole chromosome per process.
            The default is 1.
        distance_edges : list/array of int
            How to group snips by distance (based on their centres).
            Default uses separations [0, 50_000, 100_000, 200_000, ...]

        Returns
        -------
        pileup_df : 2D array
            Normalized pileups in a pandas DataFrame, with columns `data` and `num`.
            `data` contains the normalized pileups, and `num` - how many snippets were
            combined (the regions of interest, not control regions).
            Each distance band is a row, annotated in columns `separation`
        """
        normalized_pileups = self.pileupsWithControl(
            nproc=nproc,
            groupby=["strand1", "strand2"],
        )
        normalized_pileups = normalized_pileups.drop(index="all").reset_index()
        normalized_pileups["orientation"] = (
            normalized_pileups["strand1"] + normalized_pileups["strand2"]
        )
        normalized_pileups = normalized_pileups[["orientation", "data", "n"]]
        return normalized_pileups
