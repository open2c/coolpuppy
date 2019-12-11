# -*- coding: utf-8 -*-
import numpy as np
import cooler
import sys
import pandas as pd
import itertools
from multiprocessing import Pool
from functools import partial
import logging
from natsort import index_natsorted, order_by_index, natsorted
from scipy import sparse
from scipy.linalg import toeplitz
from cooltools import numutils
from cooltools import snipping


def cornerCV(amap, i=4):
    corners = np.concatenate((amap[0:i, 0:i], amap[-i:, -i:]))
    corners = corners[np.isfinite(corners)]
    return np.std(corners) / np.mean(corners)


def normCis(amap, i=3):
    return amap / np.nanmean((amap[0:i, 0:i] + amap[-i:, -i:])) * 2


def get_enrichment(amap, n):
    c = int(np.floor(amap.shape[0] / 2))
    return np.nanmean(amap[c - n // 2 : c + n // 2 + 1, c - n // 2 : c + n // 2 + 1])


def prepare_single(item):
    key, (n, amap) = item
    enr1 = get_enrichment(amap, 1)
    enr3 = get_enrichment(amap, 3)
    cv3 = cornerCV(amap, 3)
    cv5 = cornerCV(amap, 5)
    return list(key) + [n, enr1, enr3, cv3, cv5]


def norm_coverage(loop, cov_start, cov_end):
    coverage = np.outer(cov_start, cov_end)
    coverage /= np.nanmean(coverage)
    loop /= coverage
    loop[np.isnan(loop)] = 0
    return loop


class BaselistCreator:
    def __init__(
        self,
        baselist,
        resolution,
        bed2=None,
        bed2_ordered=False,
        anchor=False,
        pad=100000,
        chroms="all",
        minshift=10 ** 5,
        maxshift=10 ** 6,
        nshifts=10,
        mindist=0,
        maxdist=np.inf,
        minsize=0,
        maxsize=np.inf,
        local=False,
        subset=0,
        seed=None,
    ):
        self.baselist = baselist
        self.stdin = self.baselist == sys.stdin
        self.resolution = resolution
        self.bed2 = bed2
        self.bed2_ordered = bed2_ordered
        self.anchor = anchor
        self.chroms = chroms
        self.minshift = minshift
        self.maxshift = maxshift
        self.nshifts = nshifts
        self.mindist = mindist
        self.maxdist = maxdist
        self.minsize = minsize
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
                    """Input bed(pe) file has unexpected number of
                                 columns: got {}, expect 3 (bed) or 6 (bedpe)
                                 """.format(
                        len(row1)
                    )
                )

        if filetype == "bed" or kind == "bed":
            filter_func = self.filter_bed
            names = ["chr", "start", "end"]
            row1 = filter_func(pd.DataFrame([row1], columns=names))
        elif filetype == "bedpe" or kind == "bedpe":  # bedpe
            filter_func = self.filter_bedpe
            names = ["chr1", "start1", "end1", "chr2", "start2", "end2"]
            row1 = filter_func(pd.DataFrame([row1], columns=names))
        else:
            raise ValueError(
                """Unsupported input kind: {}.
                             Expect {} or {}""".format(
                    kind, "bed", "bedpe"
                )
            )
        bases = []

        appended = False
        if kind == "auto":
            if row1.shape[0] == 1:
                bases.append(row1)
                appended = True

        for chunk in pd.read_csv(
            file, sep="\t", names=names, index_col=False, chunksize=10 ** 4
        ):
            bases.append(filter_func(chunk))
        bases = pd.concat(bases)
        if appended:  # Would mean we read it twice when checking and in the first chunk
            bases = bases.iloc[1:]
        if filetype == "bed" or kind == "bed":
            kind = "bed"
            bases["chr"] = bases["chr"].astype(str)
            bases[["start", "end"]] = bases[["start", "end"]].astype(np.uint64)
        if filetype == "bedpe" or kind == "bedpe":
            kind = "bedpe"
            bases[["chr1", "chr2"]] = bases[["chr1", "chr2"]].astype(str)
            bases[["start1", "end1", "start2", "end2"]] = bases[
                ["start1", "end1", "start2", "end2"]
            ].astype(np.uint64)
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
            return partial(self._filter_func_region, region=region)
        else:
            return partial(self._filter_func_pairs_region, region)

    def get_combinations(self, filter_func, mids=None, mids2=None, anchor=None):
        if anchor is None:
            anchor = self.anchor
        if (self.local and anchor) or (self.local and (self.mids2 is not None)):
            raise ValueError(
                """Can't have a local pileup with an anchor or with
                                two bed files"""
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
        if self.mids2 is not None:
            mids2 = filter_func(self.mids2)
            m2 = mids2["Bin"].values.astype(int)
            p2 = (mids2["Pad"] // self.resolution).values.astype(int)

        if self.local:
            for i, pi in zip(m, p):
                yield i, i, pi, pi
        elif anchor:
            anchor_bin = int((anchor[1] + anchor[2]) / 2 // self.resolution)
            anchor_pad = int(round((anchor[2] - anchor[1]) / 2))
            for i, pi in zip(m, p):
                yield anchor_bin, i, anchor_pad, pi
        elif self.mids2 is None:
            for i, j in zip(itertools.combinations(m, 2), itertools.combinations(p, 2)):
                yield list(i) + list(j)
        elif (self.mids2 is not None) and self.bed2_ordered:
            for i, j in zip(itertools.product(m, m2), itertools.product(p, p2)):
                if i[1] > i[0]:
                    yield list(i) + list(j)
        elif (self.mids2 is not None) and (not self.bed2_ordered):
            for i, j in itertools.chain(
                zip(itertools.product(m, m2), itertools.product(p, p2)),
                zip(itertools.product(m2, m), itertools.product(p2, p)),
            ):
                yield list(i) + list(j)

    def get_positions_stream(self, filter_func, mids=None):
        if mids is None:
            mids = self.mids
        mids = filter_func(mids)
        if not len(mids) > 1:
            logging.debug("Empty selection")
            yield None, None
        m = mids["Bin"].astype(int).values
        p = (mids["Pad"] // self.resolution).astype(int).values
        for posdata in zip(m, p):
            yield posdata

    def get_position_pairs_stream(self, filter_func, mids=None):
        if mids is None:
            mids = self.mids
        mids = filter_func(mids)
        if not len(mids) > 1:
            logging.debug("Empty selection")
            yield None, None, None, None
        m1 = mids["Bin1"].astype(int).values
        m2 = mids["Bin2"].astype(int).values
        p1 = (mids["Pad1"] // self.resolution).astype(int).values
        p2 = (mids["Pad2"] // self.resolution).astype(int).values
        for posdata in zip(m1, m2, p1, p2):
            yield posdata

    def control_regions(self, filter_func, pos_pairs=None):
        if self.seed is not None:
            np.random.seed(self.seed)
        minbin = self.minshift // self.resolution
        maxbin = self.maxshift // self.resolution
        if pos_pairs is None:
            source = self.pos_stream(filter_func)
        else:
            source = map(lambda x: x[1:], pos_pairs.itertuples())
        row1 = source.__next__()
        if row1[0] is None:  # Checking if empty selection
            logging.debug("Empty selection")
            yield row1
        else:
            source = itertools.chain([row1], source)
        for start, end, p1, p2 in source:
            for i in range(self.nshifts):
                shift = np.random.randint(minbin, maxbin)
                sign = np.sign(np.random.random() - 0.5).astype(int)
                shift *= sign
                yield start + shift, end + shift, p1, p2

    def get_combinations_by_window(self, chrom, ctrl=False, mids=None):
        assert self.kind == "bed"
        if mids is None:
            chrmids = self.filter_func_chrom(self.mids, chrom)
        else:
            chrmids = self.filter_func_chrom(mids, chrom)
        for i, (b, m, p) in chrmids[["Bin", "Mids", "Pad"]].astype(int).iterrows():
            out_stream = self.BC.get_combinations(
                self.filter_func_all, mids=chrmids, anchor=(chrom, m, m)
            )
            if ctrl:
                out_stream = self.BC.control_regions(self.filter_func_all, out_stream)
            yield (m - p, m + p), out_stream

    def process(self):
        self.bases, self.kind = self.auto_read_bed(self.baselist)
        if self.bed2 is not None:
            self.bed2, self.bed2kind = self.auto_read_bed(self.bed2)
            if self.kind != "bed":
                raise ValueError(
                    """Please provide two BED files; baselist doesn't
                             seem to be one"""
                )
            elif self.bed2kind != "bed":
                raise ValueError(
                    """Please provide two BED files; bed2 doesn't
                             seem to be one"""
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
        if self.bed2:
            self.mids2 = self.subset(self._get_mids(self.bed2))
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

    def _chrom_mids(self, bed2=False):
        if bed2:
            mids = self.mids2
        else:
            mids = self.mids
        for chrom in self.final_chroms:
            if self.kind == "bed":
                yield chrom, self.mids[mids["chr"] == chrom]
            else:
                yield chrom, mids[mids["chr1"] == chrom]

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
        BC,
        balance="weight",
        expected=None,
        pad=100000,
        anchor=None,
        coverage_norm=False,
        rescale=False,
        rescale_pad=1,
        rescale_size=99,
    ):
        self.clr = clr
        self.resolution = self.clr.binsize
        self.BC = BC
        self.__dict__.update(self.BC.__dict__)
        self.balance = balance
        self.expected = expected
        self.pad = pad * 1000
        self.pad_bins = self.pad // self.resolution
        self.anchor = anchor
        self.coverage_norm = coverage_norm
        self.rescale = rescale
        self.rescale_pad = rescale_pad
        self.rescale_size = rescale_size

        self.CoolSnipper = snipping.CoolerSnipper(
            self.clr, cooler_opts=dict(balance=self.balance)
        )

        if self.expected:
            self.ExpSnipper = snipping.ExpectedSnipper(self.clr, self.expected)

        self.chroms = natsorted(
            list(set(self.BC.final_chroms) & set(self.clr.chromnames))
        )

    def get_expected_matrix(self, exp_selection, left_interval, right_interval):
        lo_left, hi_left = left_interval
        lo_right, hi_right = right_interval
        exp_selection_region, exp_selection = exp_selection
        #    exp_lo = lo_right - hi_left + 1
        #    exp_hi = hi_right - lo_left
        #    if exp_lo < 0:
        #        exp_subset = expected[0:exp_hi]
        #        #        if local:
        #        exp_subset = np.pad(exp_subset, (-exp_lo, 0), mode="reflect")
        #        #        else:
        #        #            exp_subset = np.pad(exp_subset, (-exp_lo, 0), mode='constant')
        #        i = len(exp_subset) // 2
        #        exp_matrix = toeplitz(exp_subset[i::-1], exp_subset[i:])
        #    else:
        #        exp_subset = expected[exp_lo:exp_hi]
        #        i = len(exp_subset) // 2
        #        exp_matrix = toeplitz(exp_subset[i::-1], exp_subset[i:])
        exp_matrix = self.ExpSnipper.snip(
            exp_selection, exp_selection_region, (lo_left, hi_left, lo_right, hi_right)
        )
        return exp_matrix

    def make_outmap(self,):
        if self.rescale:
            return np.zeros((self.rescale_size, self.rescale_size))
        else:
            return np.zeros((2 * self.pad_bins + 1, 2 * self.pad_bins + 1))

    def get_data(self, region):
        logging.debug("Loading data")
        data = self.clr.matrix(sparse=True, balance=self.balance).fetch(region)
        #    if local:
        data = data.tocsr()
        #    else:
        #        data = sparse.triu(data, 2).tocsr()
        return data

    def get_coverage(self, data):
        coverage = np.nan_to_num(np.ravel(np.sum(data, axis=0))) + np.nan_to_num(
            np.ravel(np.sum(data, axis=1))
        )
        return coverage

    def _do_pileups(
        self, mids, chrom, expected=False,
    ):

        if expected:
            data = None
            r = cooler.util.parse_region_string(chrom)
            exp_selection = self.ExpSnipper.select(r, r)
            logging.info("Doing expected")
        else:
            data = self.get_data(chrom)
            exp_selection = None
        if data is None:
            assert exp_selection is not None
        else:
            assert exp_selection is None
        max_right = data.shape[0]
        mymap = self.make_outmap()
        if self.coverage_norm:
            coverage = self.get_coverage(data)
        cov_start = np.zeros(mymap.shape[0])
        cov_end = np.zeros(mymap.shape[1])
        n = 0
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
            if (
                self.mindist <= abs(endBin - stBin) * self.resolution < self.maxdist
                or self.BC.local
            ):
                if not exp_selection:
                    try:
                        newmap = data[lo_left:hi_left, lo_right:hi_right].toarray()
                    except (IndexError, ValueError) as e:
                        continue
                else:
                    newmap = self.get_expected_matrix(
                        exp_selection, (lo_left, hi_left), (lo_right, hi_right)
                    )
                if (
                    newmap.shape != mymap.shape and not self.rescale
                ):  # AFAIK only happens at ends of chroms
                    height, width = newmap.shape
                    h, w = mymap.shape
                    x = w - width
                    y = h - height
                    newmap = np.pad(
                        newmap, [(y, 0), (0, x)], "constant"
                    )  # Padding to adjust to the right shape
                if self.rescale:
                    if newmap.size == 0:
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
                if (
                    self.coverage_norm
                    and (exp_selection is None)
                    and (self.balance is False)
                ):
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
                        new_cov_end = numutils.zoom_array(
                            new_cov_end, (self.rescale_size,)
                        )
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
                n += 1
        if self.BC.local:
            mymap = np.triu(mymap, 0)
            mymap += np.rot90(np.fliplr(np.triu(mymap, 1)))
        return mymap, n, cov_start, cov_end

    def pileup_chrom(
        self, chrom, expected=False, ctrl=False,
    ):

        mymap = self.make_outmap()
        cov_start = np.zeros(mymap.shape[0])
        cov_end = np.zeros(mymap.shape[1])

        if self.anchor:
            assert chrom == self.anchor[0]
            logging.info("Anchor: %s:%s-%s" % self.anchor)

        filter_func = self.BC.filter_func_chrom(chrom=chrom)

        if ctrl:
            mids = self.BC.control_regions(filter_func)
        else:
            mids = self.BC.pos_stream(filter_func)
        mids_row1 = mids.__next__()
        if mids_row1[0] is None:  # Checking if empty selection
            logging.info("Nothing to sum up in chromosome %s" % chrom)
            return self.make_outmap(), 0, cov_start, cov_end
        else:
            mids = itertools.chain([mids_row1], mids)
        mymap, n, cov_start, cov_end = self._do_pileups(
            mids=mids, chrom=chrom, expected=expected
        )
        logging.info("%s: %s" % (chrom, n))
        return mymap, n, cov_start, cov_end

    def pileupsWithControl(self, nproc=1):

        if nproc > 1:
            p = Pool(nproc)
            mymap = p.map
        else:
            mymap = map
        # Loops
        f = partial(self.pileup_chrom, ctrl=False, expected=False,)
        loops, ns, cov_starts, cov_ends = list(zip(*mymap(f, self.chroms)))
        loop = np.sum(loops, axis=0)
        n = np.sum(ns)
        if self.coverage_norm:
            cov_start = np.sum(cov_starts, axis=0)
            cov_end = np.sum(cov_starts, axis=0)
            loop = norm_coverage(loop, cov_start, cov_end)
        loop /= n
        logging.info("Total number of piled up windows: %s" % n)
        # Controls
        if self.nshifts > 0:
            f = partial(self.pileup_chrom, ctrl=True, expected=False,)
            ctrls, ns, cov_starts, cov_ends = list(zip(*mymap(f, self.chroms)))
            ctrl = np.sum(ctrls, axis=0)
            n = np.sum(ns)
            if self.coverage_norm:
                cov_start = np.sum(cov_starts, axis=0)
                cov_end = np.sum(cov_starts, axis=0)
                ctrl = norm_coverage(ctrl, cov_start, cov_end)
            ctrl /= n
            logging.info("Total number of piled up control windows: %s" % n)
            loop /= ctrl
        elif self.expected is not False:
            f = partial(self.pileup_chrom, ctrl=False, expected=self.expected,)
            exps, ns, cov_starts, cov_ends = list(zip(*map(f, self.chroms)))
            exp = np.sum(exps, axis=0)
            n = np.sum(ns)
            exp /= n
            loop /= exp
        if nproc > 1:
            p.close()
        loop[~np.isfinite(loop)] = 0
        return loop

    def pileupsByWindow(
        self, chrom, ctrl=False, expected=False,
    ):
        mymaps = dict()
        for (start, end), stream in self.BC.get_combinations_by_window(chrom, ctrl):
            mymap, n, cov_starts, cov_ends = self._do_pileups(
                mids=stream, chrom=chrom, expected=expected,
            )
            if n > 0:
                mymap = mymap / n
            else:
                mymap = self.make_outmap()
            mymaps[(start, end)] = n, mymap
        return mymaps

    def pileupsByWindowWithControl(
        self, mids, nshifts=10, expected=None, nproc=1,
    ):
        p = Pool(nproc)
        # Loops
        f = partial(self.pileupsByWindow, ctrl=False, expected=False,)
        loops = {chrom: lps for chrom, lps in zip(self.chroms, p.map(f, self.chroms))}
        # Controls
        if expected is not False:
            f = partial(self.pileupsByWindow, ctrl=False, expected=expected,)
            ctrls = {
                chrom: lps for chrom, lps in zip(self.chroms, p.map(f, self.chroms))
            }
        elif nshifts > 0:
            f = partial(self.pileupsByWindow, ctrl=True, expected=False,)
            ctrls = {
                chrom: lps for chrom, lps in zip(self.chroms, p.map(f, self.chroms))
            }
        p.close()

        finloops = {}
        for chrom in loops.keys():
            for pos, lp in loops[chrom].items():
                loop = lp[1] / ctrls[chrom][pos][1]
                loop[~np.isfinite(loop)] = 0
                finloops[(chrom, pos[0], pos[1])] = lp[0], loop
        return finloops
