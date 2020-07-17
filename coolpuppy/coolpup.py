# -*- coding: utf-8 -*-
import numpy as np
import cooler
import sys
import warnings
import pandas as pd
import itertools
from multiprocessing import Pool
from functools import partial
import logging
from natsort import index_natsorted, order_by_index, natsorted
from scipy import sparse
from scipy.linalg import toeplitz
from cooltools import numutils, snipping
import yaml
import io


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

def get_insulation_strength(amap, ignore_central=0):
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
    if not ignore_central==0 or ignore_central%2==1:
        raise ValueError(f'ignore_central has to be odd (or 0), got {ignore_central}')
    i = (amap.shape[0] - ignore_central)//2
    intra = np.nansum(amap[  :i,   :i]) + np.nansum(amap[-i:, -i:])
    inter = np.nansum(amap[:i, -i: ]) + np.nansum(amap[-i:,   :i])
    return intra/inter

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


def norm_coverage(loop, cov_start, cov_end):
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
    coverage = np.outer(cov_start, cov_end)
    coverage /= np.nanmean(coverage)
    loop /= coverage
    loop[np.isnan(loop)] = 0
    return loop


class CoordCreator:
    def __init__(
        self,
        baselist,
        resolution,
        bed2=None,
        bed2_ordered=True,
        anchor=False,
        pad=100000,
        chroms="all",
        minshift=10 ** 5,
        maxshift=10 ** 6,
        nshifts=10,
        mindist="auto",
        maxdist=None,
        minsize=0,
        maxsize=None,
        local=False,
        subset=0,
        seed=None,
    ):
        """Generator of coordinate pairs for pileups.

        Parameters
        ----------
        baselist : str
            Path to a bed- or bedpe-style file with coordinates.
        resolution : int
            Data resolution.
        bed2 : str, optional
            Path to a second bed-style file with coordinates. If specified,
            interactions between baselist and bed2 are used.
            The default is None.
        bed2_ordered : bool, optional
            Whether interactions always have coordinates from baselist on the left and
            from bed2 on the bottom. If False, all interactions will be considered
            irrespective of order.
            The default is True.
        anchor : tuple of (str, int, int), optional
            Coordinates (chr, start, end) of an anchor region used to create
            interactions with baselist (in bp). Anchor is on the left of the final pileup.
            The default is False.
        pad : int, optional
            Paddin around the central bin, in bp. For example, with 5000 bp resolution
            and 100000 pad, final pileup is 205000×205000 bp.
            The default is 100000.
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
        minsize : int, optional
            Shortest regions to consider. Only applies to bed files.
            The default is 0.
        maxsize : int, optional
            Longest regions to consider. Only applies to bed files.
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
        self.baselist = baselist
        self.stdin = self.baselist == sys.stdin
        self.resolution = resolution
        self.bed2 = bed2
        self.bed2_ordered = bed2_ordered
        self.anchor = anchor
        self.pad = pad
        self.pad_bins = pad // self.resolution
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
        self.minsize = minsize
        if maxsize is None:
            self.maxsize = np.inf
        else:
            self.maxsize = maxsize
        self.local = local
        self.subset = subset
        self.seed = seed

    def filter_bed(self, df):
        length = df["end"] - df["start"]
        df = df[(length >= self.minsize) & (length <= self.maxsize)]
        if self.chroms != "all":
            df = df[df["chr"].isin(self.chroms)]
        if not np.all(df["end"] >= df["start"]):
            raise ValueError("Some ends in the file are smaller than starts")
        return df

    def filter_bedpe(self, df):
        mid1 = np.mean(df[["start1", "end1"]], axis=1)
        mid2 = np.mean(df[["start2", "end2"]], axis=1)
        length = mid2 - mid1
        df = df[(length >= self.mindist) & (length <= self.maxdist)]
        if self.chroms != "all":
            df = df[(df["chr1"].isin(self.chroms)) & (df["chr2"].isin(self.chroms))]
        return df

    def auto_read_bed(
        self, file, kind="auto",
    ):
        if kind == "auto":  # Guessing whether it's bed or bedpe style file
            if self.stdin:
                row1 = file.__next__().split("\t")
            else:
                with open(file, "r") as fobject:
                    row1 = fobject.readline().split("\t")
            if len(row1) == 6:
                filetype = "bedpe"
                row1 = [
                    row1[0],
                    int(row1[1]),
                    int(row1[2]),
                    row1[3],
                    int(row1[4]),
                    int(row1[5]),
                ]
            elif len(row1) == 3:
                filetype = "bed"
                row1 = [row1[0], int(row1[1]), int(row1[2])]
            else:
                raise ValueError(
                    f"""Input bed(pe) file has unexpected number of
                        columns: got {len(row1)}, expect 3 (bed) or 6 (bedpe)
                        """
                )

        if filetype == "bed" or kind == "bed":
            filter_func = self.filter_bed
            names = ["chr", "start", "end"]
            dtype = {"chr": "str", "start": "int", "end": "int"}
            row1 = filter_func(pd.DataFrame([row1], columns=names).astype(dtype=dtype))
        elif filetype == "bedpe" or kind == "bedpe":  # bedpe
            filter_func = self.filter_bedpe
            names = ["chr1", "start1", "end1", "chr2", "start2", "end2"]
            dtype = {
                "chr1": "str",
                "start1": "int",
                "end1": "int",
                "chr2": "str",
                "start2": "int",
                "end2": "int",
            }
            row1 = filter_func(pd.DataFrame([row1], columns=names).astype(dtype=dtype))
        else:
            raise ValueError(
                f"""Unsupported input kind: {kind}.
                             Expect bed or bedpe"""
            )
        bases = []

        appended = False
        if kind == "auto":
            if row1.shape[0] == 1:
                bases.append(row1)
                appended = True

        for chunk in pd.read_csv(
            file,
            sep="\t",
            names=names,
            index_col=False,
            chunksize=10 ** 4,
            dtype=dtype,
        ):
            bases.append(filter_func(chunk))
        bases = pd.concat(bases)
        if appended:  # Would mean we read it twice when checking and in the first chunk
            bases = bases.iloc[1:]
        if filetype == "bed" or kind == "bed":
            kind = "bed"
            # bases["chr"] = bases["chr"].astype(str)
            # bases[["start", "end"]] = bases[["start", "end"]].astype(np.uint64)
        if filetype == "bedpe" or kind == "bedpe":
            kind = "bedpe"
            # bases[["chr1", "chr2"]] = bases[["chr1", "chr2"]].astype(str)
            # bases[["start1", "end1", "start2", "end2"]] = bases[
            # ["start1", "end1", "start2", "end2"]
            # ].astype(np.uint64)
        return bases, kind

    def subset(self, df):
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
            df1 = df[["chr1", "start1", "end1"]]
            df1.columns = ["chr", "start", "end"]
            df2 = df[["chr2", "start2", "end2"]]
            df2.columns = ["chr", "start", "end"]
            return (
                pd.concat([df1, df2])
                .sort_values(["chr", "start", "end"])
                .reset_index(drop=True)
            )

        if how == "center":
            df["start"] = np.mean(df["start1"], df["end1"], axis=0)
            df["end"] = np.mean(df["start2"], df["end2"], axis=0)
        elif how == "outer":
            df = df[["chr1", "start1", "end2"]]
            df.columns = ["chr", "start", "end"]
        elif how == "inner":
            df = df[["chr1", "end1", "start2"]]
            df.columns = ["chr", "start", "end"]
        return df

    def _get_mids(self, intervals):
        if self.kind == "bed":
            intervals = intervals.sort_values(["chr", "start"])
            mids = np.round((intervals["end"] + intervals["start"]) / 2).astype(int)
            widths = np.round((intervals["end"] - intervals["start"])).astype(int)
            mids = pd.DataFrame(
                {
                    "chr": intervals["chr"],
                    "Mids": mids,
                    "Bin": mids // self.resolution,
                    "Pad": widths / 2,
                }
            ).drop_duplicates(
                ["chr", "Bin"]
            )  # .drop('Bin', axis=1)
        elif self.kind == "bedpe":
            intervals = intervals.sort_values(["chr1", "chr2", "start1", "start2"])
            mids1 = np.round((intervals["end1"] + intervals["start1"]) / 2).astype(int)
            widths1 = np.round((intervals["end1"] - intervals["start1"])).astype(int)
            mids2 = np.round((intervals["end2"] + intervals["start2"]) / 2).astype(int)
            widths2 = np.round((intervals["end2"] - intervals["start2"])).astype(int)
            mids = pd.DataFrame(
                {
                    "chr1": intervals["chr1"],
                    "Mids1": mids1,
                    "Bin1": mids1 // self.resolution,
                    "Pad1": widths1 / 2,
                    "chr2": intervals["chr2"],
                    "Mids2": mids2,
                    "Bin2": mids2 // self.resolution,
                    "Pad2": widths2 / 2,
                },
            ).drop_duplicates(
                ["chr1", "chr2", "Bin1", "Bin2"]
            )  # .drop(['Bin1', 'Bin2'], axis=1)
        else:
            raise ValueError(
                """
                kind can only be "bed" or "bedpe"
                """
            )
        return mids

    def filter_func_all(self, mids):
        return mids

    def _filter_func_chrom(self, mids, chrom):
        return mids[mids["chr"] == chrom]

    def _filter_func_pairs_chrom(self, mids, chrom):
        return mids[(mids["chr1"] == chrom) & (mids["chr2"] == chrom)]

    def filter_func_chrom(self, chrom):
        if self.kind == "bed":
            return partial(self._filter_func_chrom, chrom=chrom)
        else:
            return partial(self._filter_func_pairs_chrom, chrom=chrom)

    def _filter_func_region(self, mids, region):
        chrom, start, end = region
        start /= self.resolution
        end /= self.resolution
        return mids[
            (mids["chr"] == chrom) & (mids["Bin"] >= start) & (mids["Bin"] < end)
        ]

    def _filter_func_pairs_region(self, mids, region):
        chrom, start, end = region
        start /= self.resolution
        end /= self.resolution
        return mids[
            (mids["chr1"] == chrom)
            & (mids["chr2"] == chrom)
            & (mids["Bin1"] >= start)
            & (mids["Bin1"] < end)
            & (mids["Bin2"] >= start)
            & (mids["Bin2"] < end)
        ]

    def filter_func_region(self, region):
        if self.kind == "bed":
            return partial(self._filter_func_region, region)
        else:
            return partial(self._filter_func_pairs_region, region)

    def _get_combinations(self, filter_func, mids=None, mids2=None, anchor=None):
        if anchor is None:
            anchor = self.anchor

        if (self.local and anchor) or (self.local and (self.mids2 is not None)):
            raise ValueError(
                """Can't have a local pileup with an anchor or with two bed files"""
            )
        if mids is None:
            mids = self.mids
        mids = filter_func(self.mids)
        if not len(mids) > 1:
            logging.debug("Empty selection")
            yield None, None, None, None
        m = mids["Bin"].values.astype(int)
        p = (mids["Pad"] // self.resolution).values.astype(int)

        if mids2 is None:
            mids2 = self.mids2
        if mids2 is not None:
            mids2 = filter_func(mids2)
            m2 = mids2["Bin"].values.astype(int)
            p2 = (mids2["Pad"] // self.resolution).values.astype(int)
        if self.local:
            for i, pi in zip(m, p):
                yield i, i, pi, pi
        elif anchor:
            anchor_bin = int((anchor[1] + anchor[2]) / 2 // self.resolution)
            anchor_pad = int(round((anchor[2] - anchor[1]) / 2)) // self.resolution
            for i, pi in zip(m, p):
                yield anchor_bin, i, anchor_pad, pi
        elif mids2 is None:
            for i, j in zip(itertools.combinations(m, 2), itertools.combinations(p, 2)):
                yield list(i) + list(j)
        elif (mids2 is not None) and self.bed2_ordered:
            for i, j in zip(itertools.product(m, m2), itertools.product(p, p2)):
                if i[1] > i[0]:
                    yield list(i) + list(j)
        elif (mids2 is not None) and (not self.bed2_ordered):
            for i, j in itertools.chain(
                zip(itertools.product(m, m2), itertools.product(p, p2)),
                zip(itertools.product(m2, m), itertools.product(p2, p)),
            ):
                yield list(i) + list(j)

    def get_combinations(self, filter_func, mids=None, mids2=None, anchor=None):
        stream = self._get_combinations(filter_func, mids, mids2, anchor)
        if not self.local:
            stream = self.filter_pos_stream_distance(stream)
        return stream
    

    def get_positions_stream(self, filter_func, mids=None):
        if mids is None:
            mids = self.mids
        mids = filter_func(mids)
        if not len(mids) >= 1:
            logging.debug("Empty selection")
            yield None, None
        m = mids["Bin"].astype(int).values
        p = (mids["Pad"] // self.resolution).astype(int).values
        for posdata in zip(m, p):
            yield posdata

    def _get_position_pairs_stream(self, filter_func, mids=None):
        if mids is None:
            mids = self.mids
        mids = filter_func(mids)
        if not len(mids) >= 1:
            logging.debug("Empty selection")
            yield None, None, None, None
        m1 = mids["Bin1"].astype(int).values
        m2 = mids["Bin2"].astype(int).values
        p1 = (mids["Pad1"] // self.resolution).astype(int).values
        p2 = (mids["Pad2"] // self.resolution).astype(int).values
        for posdata in zip(m1, m2, p1, p2):
            yield posdata
            
    def get_position_pairs_stream(self, filter_func, mids=None):
        stream = self._get_position_pairs_stream(filter_func, mids)
        if not self.local:
            stream = self.filter_pos_stream_distance(stream)
        return stream

    def control_regions(self, filter_func, pos_pairs=None):
        if self.seed is not None:
            np.random.seed(self.seed)
        minbin = self.minshift // self.resolution
        maxbin = self.maxshift // self.resolution
        if pos_pairs is None:
            source = self.pos_stream(filter_func)
        else:
            source = map(lambda x: x[1:], pos_pairs.itertuples())
        # try:
        #     row1 = next(source)
        # except StopIteration:
        #     logging.debug("Empty selection")
        #     raise StopIteration
        # else:
        #     source = itertools.chain([row1], source)
        for start, end, p1, p2 in source:
            for i in range(self.nshifts):
                shift = np.random.randint(minbin, maxbin)
                sign = np.sign(np.random.random() - 0.5).astype(int)
                shift *= sign
                yield start + shift, end + shift, p1, p2

    def get_combinations_by_window(self, chrom, ctrl=False, mids=None):
        assert self.kind == "bed"
        if mids is None:
            chrmids = self.filter_func_chrom(chrom)(self.mids)
        else:
            chrmids = self.filter_func_chrom(chrom)(mids)
        for i, (b, m, p) in chrmids[["Bin", "Mids", "Pad"]].astype(int).iterrows():
            out_stream = self.get_combinations(
                self.filter_func_all, mids=chrmids, anchor=(chrom, m, m)
            )
            if ctrl:
                out_stream = self.CC.control_regions(self.filter_func_all, out_stream)
            yield (m - p, m + p), out_stream
            
            
    def filter_pos_stream_distance(self, stream):
        for (m1, m2, p1, p2) in stream:
            if self.mindist < abs(m2 - m1) * self.resolution < self.maxdist:
                yield (m1, m2, p1, p2)

    def process(self):
        self.bases, self.kind = self.auto_read_bed(self.baselist)
        if self.bed2 is not None:
            self.bed2, self.bed2kind = self.auto_read_bed(self.bed2)
            if self.kind != "bed":
                raise ValueError(
                    """Please provide two BED files; baselist doesn't seem to be one"""
                )
            elif self.bed2kind != "bed":
                raise ValueError(
                    """Please provide two BED files; bed2 doesn't seem to be one"""
                )
        if self.kind == "bed":
            basechroms = set(self.bases["chr"])
            if self.anchor:
                if self.anchor[0] not in basechroms:
                    raise ValueError(
                        """The anchor chromosome is not found in the baselist.
                           Are they in the same format, e.g. starting with "chr"?
                           Alternatively, all regions in that chromosome might have
                           been filtered by some filters."""
                    )
                else:
                    basechroms = [self.anchor[0]]
        else:
            if self.anchor:
                raise ValueError("Can't use anchor with both sides of loops defined")
            elif self.local:
                raise ValueError("Can't make local with both sides of loops defined")
            basechroms = set(self.bases["chr1"]) & set(self.bases["chr2"])
        if self.bed2 is not None:
            bed2chroms = set(self.bases["chr"])
            basechroms = basechroms & bed2chroms
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
                   Alternatively, all regions might have been filtered
                   by distance/size filters."""
            )

        self.mids = self._get_mids(self.bases)
        if self.bed2 is not None:
            self.mids2 = self._get_mids(self.bed2)
        else:
            self.mids2 = None
        if self.subset > 0:
            self.mids = self.mids.sample(self.subset)
            if self.mids2 is not None:
                self.mids2 = self.mids2.sample(self.subset)

        if self.kind == "bed":
            self.pos_stream = self.get_combinations
        else:
            self.pos_stream = self.get_position_pairs_stream
        

    def _chrom_mids(self, chroms, mids):
        for chrom in chroms:
            if self.kind == "bed":
                yield chrom, mids[mids["chr"] == chrom]
            else:
                yield chrom, mids[(mids["chr1"] == chrom) & (mids["chr2"] == chrom)]

    def chrom_mids(self):
        chrommids = self._chrom_mids(self.final_chroms, self.mids)
        if self.mids2 is not None:
            chrommids2 = self._chrom_mids(self.final_chroms, self.mids2)
        else:
            chrommids2 = zip(itertools.cycle([None]), itertools.cycle([None]))
        chrommids = zip(chrommids, chrommids2)
        for i in chrommids:
            yield i


class PileUpper:
    def __init__(
        self,
        clr,
        CC,
        balance="weight",
        expected=False,
        control=False,
        coverage_norm=False,
        rescale=False,
        rescale_pad=1,
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
        control : bool, optional
            Whether to use randomly shifted controls.
            The default is False.
        coverage_norm : bool, optional
            Whether to normalize final the final pileup by accumulated coverage as an
            alternative to balancing. Useful for single-cell hi-C data.
            The default is False.
        rescale : bool, optional
            Whether to use real sizes of all ROIs and rescale them to the same shape
            The default is False.
        rescale_pad : float, optional
            Fraction of ROI size added on each end when extracting snippets, if rescale.
            The default is 1.
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
        self.control = control
        self.pad_bins = self.pad // self.resolution
        self.coverage_norm = coverage_norm
        self.rescale = rescale
        self.rescale_pad = rescale_pad
        self.rescale_size = rescale_size
        self.ignore_diags = ignore_diags
        # self.CoolSnipper = snipping.CoolerSnipper(
        #     self.clr, cooler_opts=dict(balance=self.balance)
        # )
        self.matsizes = np.ceil(self.clr.chromsizes / self.resolution).astype(int)

        self.chroms = natsorted(
            list(set(self.CC.final_chroms) & set(self.clr.chromnames))
        )
        self.regions = {
            chrom: cooler.util.parse_region_string(chrom) for chrom in self.chroms
        }
        if self.expected is not False:
            if self.control:
                warnings.warn(
                    "Can't do both expected and control shifts; defaulting to expected"
                )
                self.control = False
            assert isinstance(self.expected, pd.DataFrame)
            self.ExpSnipper = snipping.ExpectedSnipper(self.clr, self.expected)
            self.expected_selections = {
                chrom: self.ExpSnipper.select(region, region)
                for chrom, region in self.regions.items()
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

    def get_expected_matrix(self, chrom, left_interval, right_interval):
        """Generate expected matrix for a region

        Parameters
        ----------
        chrom : str
            Chromosome name.
        left_interval : tuple
            Tuple of (lo_left, hi_left) bin IDs in the chromosome.
        right_interval : tuple
            Tuple of (lo_right, hi_right) bin IDs in the chromosome.

        Returns
        -------
        exp_matrix : array
            Array of expected values for the selected coordinates.

        """
        lo_left, hi_left = left_interval
        lo_right, hi_right = right_interval
        lo_left *= self.resolution
        hi_left *= self.resolution
        lo_right *= self.resolution
        hi_right *= self.resolution
        exp_matrix = self.ExpSnipper.snip(
            self.expected_selections[chrom],
            self.regions[chrom],
            self.regions[chrom],
            (lo_left, hi_left, lo_right, hi_right),
        )
        return exp_matrix

    def make_outmap(self,):
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
            string with chromosome name.

        Returns
        -------
        data : csr
            Sparse csr matrix for the corresponding region.

        """
        logging.debug("Loading data")
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
        coverage = np.nan_to_num(np.ravel(np.sum(data, axis=0))) + np.nan_to_num(
            np.ravel(np.sum(data, axis=1))
        )
        return coverage

    def _do_pileups(
        self, mids, chrom, expected=False,
    ):
        mymap = self.make_outmap()
        cov_start = np.zeros(mymap.shape[0])
        cov_end = np.zeros(mymap.shape[1])
        num = np.zeros_like(mymap)
        n = 0
        try:
            mids_row1 = next(mids)
        except StopIteration:
            logging.info(f"Nothing to sum up in chromosome {chrom}")
            return mymap, mymap, cov_start, cov_end, n
        else:
            mids = itertools.chain([mids_row1], mids)
        
        if expected:
            data = None
            logging.debug("Doing expected")
        else:
            data = self.get_data(
                chrom
            )  # self.CoolSnipper.select(self.regions[chrom], self.regions[chrom])
        max_right = self.matsizes[chrom]

        if self.coverage_norm:
            coverage = self.get_coverage(data)

        for stBin, endBin, stPad, endPad in mids:
            rot_flip = False
            rot = False
            if stBin > endBin:
                stBin, stPad, endBin, endPad = endBin, endPad, stBin, stPad
                if self.anchor is None:
                    rot_flip = True
                else:
                    rot = True
            if self.rescale:
                stPad = stPad + int(round(self.rescale_pad * 2 * stPad))
                endPad = endPad + int(round(self.rescale_pad * 2 * endPad))
            else:
                stPad, endPad = self.pad_bins, self.pad_bins
            lo_left = stBin - stPad
            hi_left = stBin + stPad + 1
            lo_right = endBin - endPad
            hi_right = endBin + endPad + 1
            if lo_left < 0 or hi_right > max_right:
                continue
            diag = hi_left - lo_right
            if not expected:
                try:
                    newmap = data[lo_left:hi_left, lo_right:hi_right].toarray()
                except (IndexError, ValueError):
                    continue
            else:
                newmap = self.get_expected_matrix(
                    chrom, (lo_left, hi_left), (lo_right, hi_right)
                )
            #                if (
            #                    newmap.shape != mymap.shape and not self.rescale
            #                ):  # AFAIK only happens at ends of chroms
            #                    height, width = newmap.shape
            #                    h, w = mymap.shape
            #                    x = w - width
            #                    y = h - height
            #                    newmap = np.pad(
            #                        newmap, [(y, 0), (0, x)], "constant"
            #                    )  # Padding to adjust to the right shape
            newmap = newmap.astype(float)
            if not self.local:
                ignore_indices = np.tril_indices_from(
                    newmap, diag - (stPad * 2 + 1) - 1 + self.ignore_diags
                )
                newmap[ignore_indices] = np.nan
            else:
                newmap = np.triu(newmap, self.ignore_diags)
                newmap += np.triu(newmap, 1).T
            if self.rescale:
                if newmap.size == 0 or np.all(np.isnan(newmap)):
                    newmap = np.zeros((self.rescale_size, self.rescale_size))
                else:
                    newmap = numutils.zoom_array(
                        newmap, (self.rescale_size, self.rescale_size)
                    )
            if rot_flip:
                newmap = np.rot90(np.flipud(newmap), 1)
            elif rot:
                newmap = np.rot90(newmap, -1)

            mymap = np.nansum([mymap, newmap], axis=0)
            if self.coverage_norm and not expected and (self.balance is False):
                new_cov_start = coverage[lo_left:hi_left]
                new_cov_end = coverage[lo_right:hi_right]
                if self.rescale:
                    if len(new_cov_start) == 0:
                        new_cov_start = np.zeros(self.rescale_size)
                    if len(new_cov_end) == 0:
                        new_cov_end = np.zeros(self.rescale_size)
                    new_cov_start = numutils.zoom_array(
                        new_cov_start, (self.rescale_size,)
                    )
                    new_cov_end = numutils.zoom_array(new_cov_end, (self.rescale_size,))
                else:
                    l = len(new_cov_start)
                    r = len(new_cov_end)
                    new_cov_start = np.pad(
                        new_cov_start, (mymap.shape[0] - l, 0), "constant"
                    )
                    new_cov_end = np.pad(
                        new_cov_end, (0, mymap.shape[1] - r), "constant"
                    )
                cov_start += np.nan_to_num(new_cov_start)
                cov_end += +np.nan_to_num(new_cov_end)
            num += np.isfinite(newmap).astype(int)
            n += 1
        return mymap, num, cov_start, cov_end, n

    def pileup_chrom(
        self, chrom, expected=False, ctrl=False,
    ):
        """

        Parameters
        ----------
        chrom : str
            Chromosome name.
        expected : bool, optional
            Whether to create pileup of expected values. The default is False.
        ctrl : bool, optional
            Whether to pileup randomly shifted control regions. The default is False.


        Returns
        -------
        pileup : 2D array
            Pileup for the specified chromosome.
        n : int
            How many ROIs were piled up.
        cov_start : 1D array
            Accumulated coverage of the left side of the pileup.
        cov_end : 1D array
            Accumulated coverage of the bottom side of the pileup.

        """

        mymap = self.make_outmap()
        cov_start = np.zeros(mymap.shape[0])
        cov_end = np.zeros(mymap.shape[1])

        if self.anchor:
            assert chrom == self.anchor[0]
            logging.info(f"Anchor: {chrom}:{self.anchor[1]}-{self.anchor[2]}")

        filter_func = self.CC.filter_func_chrom(chrom=chrom)

        if ctrl:
            mids = self.CC.control_regions(filter_func)
        else:
            mids = self.CC.pos_stream(filter_func)
        mymap, num, cov_start, cov_end, n = self._do_pileups(
            mids=mids, chrom=chrom, expected=expected,
        )
        logging.info(f"{chrom}: {n}")
        return mymap, num, cov_start, cov_end, n

    def pileupsWithControl(self, nproc=1):
        """Perform pileups across all chromosomes and applies required
        normalization

        Parameters
        ----------
        nproc : int, optional
            How many cores to use. Sends a whole chromosome per process.
            The default is 1.

        Returns
        -------
        loop : 2D array
            Normalized pileup.

        """

        if nproc > 1:
            p = Pool(nproc)
            mymap = p.map
        else:
            mymap = map
        # Loops
        f = partial(self.pileup_chrom, ctrl=False, expected=False,)
        loops, nums, cov_starts, cov_ends, ns = list(zip(*mymap(f, self.chroms)))
        loop = np.sum(loops, axis=0)
        n = np.sum(ns)
        n_return = n
        num = np.sum(nums, axis=0)
        if self.coverage_norm:
            cov_start = np.sum(cov_starts, axis=0)
            cov_end = np.sum(cov_ends, axis=0)
            loop = norm_coverage(loop, cov_start, cov_end)
        loop /= num
        logging.info(f"Total number of piled up windows: {n}")
        # Controls
        if self.expected is not False:
            f = partial(self.pileup_chrom, ctrl=False, expected=True,)
            exps, nums, cov_starts, cov_ends, ns = list(zip(*mymap(f, self.chroms)))
            exp = np.sum(exps, axis=0)
            num = np.sum(nums, axis=0)
            exp /= num
            loop /= exp
        elif self.control:
            f = partial(self.pileup_chrom, ctrl=True, expected=False,)
            ctrls, nums, cov_starts, cov_ends, ns = list(zip(*mymap(f, self.chroms)))
            ctrl = np.sum(ctrls, axis=0)
            num = np.sum(nums, axis=0)
            n = np.sum(ns)
            if self.coverage_norm:
                cov_start = np.sum(cov_starts, axis=0)
                cov_end = np.sum(cov_ends, axis=0)
                ctrl = norm_coverage(ctrl, cov_start, cov_end)
            ctrl /= num
            logging.info(f"Total number of piled up control windows: {n}")
            loop /= ctrl
        if nproc > 1:
            p.close()
        loop[~np.isfinite(loop)] = 0
        return loop, n_return

    def pileupsByWindow(
        self, chrom, expected=False, ctrl=False,
    ):
        """Creates pileups for each window against the rest for a chromosome

        Parameters
        ----------
        chrom : str
            Chromosome name.
        expected : bool, optional
            Whether to create pileup of expected values. The default is False.
        ctrl : bool, optional
            Whether to pileup randomly shifted control regions. The default is False.


        Returns
        -------
        pileups : dict
            Keys are tuples of (start, end) coordinates.
            Values are tuples of (n, pileup)
            n : int
            How many ROIs were piled up.
            pileup : 2D array
            Pileup for the region
        """
        pileups = dict()
        for (start, end), stream in self.CC.get_combinations_by_window(chrom, ctrl):
            pileup, nums, cov_starts, cov_ends, ns = self._do_pileups(
                mids=stream, chrom=chrom, expected=expected,
            )
            n = np.sum(ns)
            num = np.sum(nums, axis=0)
            if n > 0:
                pileup = pileup / num
            else:
                pileup = self.make_outmap()
            pileups[(start, end)] = n, pileup
        n_pileups = len(pileups)
        if expected:
            kind = "expected"
        elif ctrl:
            kind = "control"
        else:
            kind = ""
        logging.info(f"{chrom}: {n_pileups} {kind} by-window pileups")
        return pileups

    def pileupsByWindowWithControl(
        self, nproc=1,
    ):
        """Perform by-window pileups across all chromosomes and applies required
        normalization

        Parameters
        ----------
        nproc : int, optional
            How many cores to use. Sends a whole chromosome per process.
            The default is 1.

        Returns
        -------
        finloops : dict
            Keys are tuples of (chrom, start, end) coordinates.
            Values are tuples of (n, pileup)
            n : int
            How many ROIs were piled up.
            pileup : 2D array
            Pileup for the region
        """
        if nproc > 1:
            p = Pool(nproc)
            mymap = p.map
        else:
            mymap = map
        # Loops
        f = partial(self.pileupsByWindow, ctrl=False, expected=False,)
        loops = {chrom: lps for chrom, lps in zip(self.chroms, mymap(f, self.chroms))}
        # Controls
        if self.expected is not False:
            f = partial(self.pileupsByWindow, ctrl=False, expected=True,)
            ctrls = {
                chrom: lps for chrom, lps in zip(self.chroms, mymap(f, self.chroms))
            }
        elif self.control:
            f = partial(self.pileupsByWindow, ctrl=True, expected=False,)
            ctrls = {
                chrom: lps for chrom, lps in zip(self.chroms, mymap(f, self.chroms))
            }
        if nproc > 1:
            p.close()

        finloops = {}
        for chrom in loops.keys():
            for pos, lp in loops[chrom].items():
                if self.expected is not False or self.control:
                    loop = lp[1] / ctrls[chrom][pos][1]
                loop[~np.isfinite(loop)] = 0
                finloops[(chrom, pos[0], pos[1])] = lp[0], loop
        return finloops
