#!/usr/bin/env python

# -*- coding: utf-8 -*-

# Takes a cooler file and a bed file with coordinates of features, i.e. ChIP-seq
# peaks, finds all cis intersections of the features and makes a pileup for them
# using sparse-whole chromosome matrices (in parallel). Can also use paired bed
# intervals, i.e. called loops. Based on Max's approach with shifted windows to
# normalize for scaling.

# Comes with a battery included - has a simple qsub launch script for an SGE
# cluster.

import numpy as np
import cooler
import pandas as pd
import itertools
from multiprocessing import Pool
from functools import partial
import os
from natsort import index_natsorted, order_by_index
from scipy import sparse
from scipy.linalg import toeplitz
from cooltools import numutils

def cornerCV(amap, i=4):
    corners = np.concatenate((amap[0:i, 0:i], amap[-i:, -i:]))
    corners = corners[np.isfinite(corners)]
    return np.std(corners)/np.mean(corners)

#def normCis(amap, i=3):
#    return amap/np.nanmean((amap[0:i, 0:i]+amap[-i:, -i:]))*2

def get_enrichment(amap, n):
    c = int(np.floor(amap.shape[0]/2))
    return np.nanmean(amap[c-n//2:c+n//2+1, c-n//2:c+n//2+1])

def get_mids(intervals, resolution, combinations=True):
    if combinations:
        intervals = intervals.sort_values(['chr', 'start'])
        mids = np.round((intervals['end']+intervals['start'])/2).astype(int)
        widths = np.round((intervals['end']-intervals['start'])).astype(int)
        mids = pd.DataFrame({'chr':intervals['chr'],
                             'Mids':mids,
                             'Bin':mids//resolution,
                             'Pad':widths/2}).drop_duplicates(['chr', 'Bin']).drop('Bin', axis=1)
    else:
        intervals = intervals.sort_values(['chr1', 'chr2',
                                           'start1', 'start2'])
        mids1 = np.round((intervals['end1']+intervals['start1'])/2).astype(int)
        widths1 = np.round((intervals['end1']-intervals['start1'])).astype(int)
        mids2 = np.round((intervals['end2']+intervals['start2'])/2).astype(int)
        widths2 = np.round((intervals['end2']-intervals['start2'])).astype(int)
        mids = pd.DataFrame({'chr1':intervals['chr1'],
                             'Mids1':mids1,
                             'Bin1':mids1//resolution,
                             'Pad1':widths1/2,
                             'chr2':intervals['chr2'],
                             'Mids2':mids2,
                             'Bin2':mids2//resolution,
                             'Pad2':widths2/2},
                            ).drop_duplicates(['chr1', 'Mids1',
                                      'chr2', 'Mids2']).drop_duplicates(['chr',
                                                 'Bin1', 'Bin2']).drop(['Bin1',
                                                         '      Bin2'], axis=1)
    return mids

def get_combinations(mids, res, local=False, anchor=None):
    if local and anchor:
        raise ValueError("Can't have a local pileup with an anchor")
    m = (mids['Mids']//res).values.astype(int)
    p = (mids['Pad']//res).values.astype(int)
    if local:
        for i, pi in zip(m, p):
            yield i, i, pi, pi
    elif anchor:
        anchor_bin = int((anchor[1]+anchor[2])/2//res)
        anchor_pad = int(round((anchor[2] - anchor[1])/2))
        for i, pi in zip(m, p):
            yield anchor_bin, i, anchor_pad, pi
    else:
        for i, j in zip(itertools.combinations(m, 2),
                        itertools.combinations(p, 2)):
            yield list(i)+list(j)

def get_positions_pairs(mids, res):
    m1 = (mids['Mids1']//res).astype(int).values
    m2 = (mids['Mids2']//res).astype(int).values
    p1 = (mids['Pad1']//res).astype(int).values
    p2 = (mids['Pad2']//res).astype(int).values
    for posdata in zip(m1, m2, p1, p2):
        yield posdata

def controlRegions(midcombs, res, minshift=10**5, maxshift=10**6, nshifts=1):
    minbin = minshift//res
    maxbin = maxshift//res
    for start, end, p1, p2 in midcombs:
        for i in range(nshifts):
            shift = np.random.randint(minbin, maxbin)
            sign = np.sign(np.random.random() - 0.5).astype(int)
            shift *= sign
            yield start+shift, end+shift, p1, p2

def get_expected_matrix(left_interval, right_interval, expected, local):
    lo_left, hi_left = left_interval
    lo_right, hi_right = right_interval
    exp_lo = lo_right - hi_left + 1
    exp_hi = hi_right - lo_left
    if exp_lo < 0:
        exp_subset = expected[0:exp_hi]
        if local:
            exp_subset = np.pad(exp_subset, (-exp_lo, 0), mode='reflect')
        else:
            exp_subset = np.pad(exp_subset, (-exp_lo, 0), mode='constant')
        i = len(exp_subset)//2
        exp_matrix = toeplitz(exp_subset[i::-1], exp_subset[i:])
    else:
        exp_subset = expected[exp_lo:exp_hi]
        i = len(exp_subset)//2
        exp_matrix = toeplitz(exp_subset[i::-1], exp_subset[i:])
    return exp_matrix

def make_outmap(pad, rescale=False, rescale_size=41):
    if rescale:
        return np.zeros((rescale_size, rescale_size))
    else:
        return np.zeros((2*pad + 1, 2*pad + 1))

def get_data(chrom, c, unbalanced, local):
    logging.debug('Loading data')
    data = c.matrix(sparse=True, balance=bool(1-unbalanced)).fetch(chrom)
    if local:
        data = data.tocsr()
    else:
        data = sparse.triu(data, 2).tocsr()
    return data

def _do_pileups(mids, data, binsize, pad, expected, mindist, maxdist, local,
                unbalanced, cov_norm, rescale, rescale_pad, rescale_size,
                coverage, anchor):
    mymap = make_outmap(pad, rescale, rescale_size)
    cov_start = np.zeros(mymap.shape[0])
    cov_end = np.zeros(mymap.shape[1])
    n = 0
    for stBin, endBin, stPad, endPad in mids:
        rot_flip = False
        flip_rot = False
        if stBin > endBin:
            stBin, stPad, endBin, endPad = endBin, endPad, stBin, stPad
            if anchor is None:
                rot_flip = True
            else:
                flip_rot = True
        if rescale:
            stPad = stPad + int(round(rescale_pad*2*stPad))
            endPad = endPad + int(round(rescale_pad*2*endPad))
        else:
            stPad = pad
            endPad = pad
        lo_left = stBin - stPad
        hi_left = stBin + stPad + 1
        lo_right = endBin - endPad
        hi_right = endBin + endPad + 1
        if mindist <= abs(endBin - stBin)*binsize < maxdist or local:
            if expected is False:
                try:
                    newmap = np.nan_to_num(data[lo_left:hi_left,
                                                lo_right:hi_right].toarray())
                except (IndexError, ValueError) as e:
                    continue
            else:
                newmap = get_expected_matrix((lo_left, hi_left),
                                             (lo_right, hi_right),
                                              expected, local)
            if newmap.shape != mymap.shape and not rescale: #AFAIK only happens at ends of chroms
                height, width = newmap.shape
                h, w = mymap.shape
                x = w - width
                y = h - height
                newmap = np.pad(newmap, [(y, 0), (0, x)], 'constant') #Padding to adjust to the right shape
            if rescale:
                if newmap.size==0:
                    newmap = np.zeros((rescale_size, rescale_size))
                else:
                    newmap = numutils.zoom_array(newmap, (rescale_size,
                                                         rescale_size))
            if rot_flip:
                newmap = np.rot90(np.flipud(newmap), -1)
            elif flip_rot:
                newmap = np.flipud(np.rot90(newmap))
            mymap += np.nan_to_num(newmap)
            if unbalanced and cov_norm and expected is False:
                new_cov_start = coverage[lo_left:hi_left]
                new_cov_end = coverage[lo_right:hi_right]
                if rescale:
                    if len(new_cov_start)==0:
                        new_cov_start = np.zeros(rescale_size)
                    if len(new_cov_end)==0:
                        new_cov_end = np.zeros(rescale_size)
                    new_cov_start = numutils.zoom_array(new_cov_start,
                                                       (rescale_size,))
                    new_cov_end = numutils.zoom_array(new_cov_end,
                                                     (rescale_size,))
                else:
                    l = len(new_cov_start)
                    r = len(new_cov_end)
                    new_cov_start = np.pad(new_cov_start, (mymap.shape[0]-l, 0),
                                                       'constant')
                    new_cov_end = np.pad(new_cov_end,
                                     (0, mymap.shape[1]-r), 'constant')
                cov_start += np.nan_to_num(new_cov_start)
                cov_end += +np.nan_to_num(new_cov_end)
            n += 1
    if local:
        mymap = np.triu(mymap, 0)
        mymap += np.rot90(np.fliplr(np.triu(mymap, 1)))
    return mymap, n, cov_start, cov_end

def pileups(chrom_mids, c, pad=7, ctrl=False, local=False,
            minshift=10**5, maxshift=10**6, nshifts=1, expected=False,
            mindist=0, maxdist=10**9, combinations=True, anchor=None,
            unbalanced=False, cov_norm=False,
            rescale=False, rescale_pad=50, rescale_size=41):
    chrom, mids = chrom_mids

    mymap = make_outmap(pad, rescale, rescale_size)
    cov_start = np.zeros(mymap.shape[0])
    cov_end = np.zeros(mymap.shape[1])

    if not len(mids) > 1:
        logging.info('Nothing to sum up in chromosome %s' % chrom)
        return make_outmap(pad, rescale, rescale_size), 0, cov_start, cov_end

    if expected is not False:
        data = False
        expected = expected[expected['chrom']==chrom]['balanced.avg'].values
        logging.info('Doing expected')
    else:
        data = get_data(chrom, c, unbalanced, local)

    if unbalanced and cov_norm and expected is False:
        coverage = np.nan_to_num(np.ravel(np.sum(data, axis=0))) + \
                   np.nan_to_num(np.ravel(np.sum(data, axis=1)))
    else:
        coverage=False

    if anchor:
        assert chrom==anchor[0]
#        anchor_bin = (anchor[1]+anchor[2])/2//c.binsize
        logging.info('Anchor: %s:%s-%s' % anchor)
    else:
        anchor = None

    if combinations:
        assert np.all(mids['chr']==chrom)
    else:
        assert np.all(mids['chr1']==chrom) & np.all(mids['chr1']==chrom)

    if ctrl:
        if combinations:
            mids = controlRegions(get_combinations(mids, c.binsize, local,
                                                    anchor),
                                   c.binsize, minshift, maxshift, nshifts)
        else:
            mids = controlRegions(get_positions_pairs(mids, c.binsize),
                                   c.binsize, minshift, maxshift, nshifts)
    else:
        if combinations:
            mids = get_combinations(mids, c.binsize, local, anchor)
        else:
            mids = get_positions_pairs(mids, c.binsize)
    mymap, n, cov_start, cov_end = _do_pileups(mids=mids, data=data, pad=pad,
                                               binsize=c.binsize,
                                               expected=expected,
                                               mindist=mindist,
                                               maxdist=maxdist,
                                               local=local,
                                               unbalanced=unbalanced,
                                               cov_norm=cov_norm,
                                               coverage=coverage,
                                               rescale=rescale,
                                               rescale_pad=rescale_pad,
                                               rescale_size=rescale_size,
                                               anchor=anchor)
    logging.info('%s: %s' % (chrom, n))
    return mymap, n, cov_start, cov_end

def chrom_mids(chroms, mids, combinations):
    for chrom in chroms:
        if combinations:
            yield chrom, mids[mids['chr']==chrom]
        else:
            yield chrom, mids[mids['chr1']==chrom]

def norm_coverage(loop, cov_start, cov_end):
    coverage = np.outer(cov_start, cov_end)
    coverage /= np.nanmean(coverage)
    loop /= coverage
    loop[np.isnan(loop)]=0
    return loop

def pileupsWithControl(mids, filename, pad=100, nproc=1, chroms=None,
                       local=False,
                       minshift=100000, maxshift=100000, nshifts=10,
                       expected=None,
                       mindist=0, maxdist=np.inf,
                       combinations=True, anchor=None, unbalanced=False,
                       cov_norm=False,
                       rescale=False, rescale_pad=1, rescale_size=99):
    c = cooler.Cooler(filename)
    if chroms is None:
        chroms = c.chromnames
    p = Pool(nproc)
    #Loops
    f = partial(pileups, c=c, pad=pad, ctrl=False, local=local,
                minshift=minshift, maxshift=maxshift, nshifts=nshifts,
                expected=False,
                mindist=mindist, maxdist=maxdist, combinations=combinations,
                anchor=anchor, unbalanced=unbalanced, cov_norm=cov_norm,
                rescale=rescale, rescale_pad=rescale_pad,
                rescale_size=rescale_size)
    chrommids = chrom_mids(chroms, mids, combinations)
    loops, ns, cov_starts, cov_ends = list(zip(*p.map(f, chrommids)))
    loop = np.sum(loops, axis=0)
    n = np.sum(ns)
    if cov_norm:
        cov_start = np.sum(cov_starts, axis=0)
        cov_end = np.sum(cov_starts, axis=0)
        loop = norm_coverage(loop, cov_start, cov_end)
    loop /= n
    logging.info('Total number of piled up windows: %s' % n)
    #Controls
    if nshifts>0:
        chrommids = chrom_mids(chroms, mids, combinations)
        f = partial(pileups, c=c, pad=pad, ctrl=True, local=local,
                    expected=False,
                    minshift=minshift, maxshift=maxshift, nshifts=nshifts,
                    mindist=mindist, maxdist=maxdist, combinations=combinations,
                    anchor=anchor, unbalanced=unbalanced, cov_norm=cov_norm,
                    rescale=rescale, rescale_pad=rescale_pad,
                    rescale_size=rescale_size)
        ctrls, ns, cov_starts, cov_ends = list(zip(*p.map(f, chrommids)))
        ctrl = np.sum(ctrls, axis=0)
        n = np.sum(ns)
        if cov_norm:
            cov_start = np.sum(cov_starts, axis=0)
            cov_end = np.sum(cov_starts, axis=0)
            ctrl = norm_coverage(ctrl, cov_start, cov_end)
        ctrl /= n
        loop /= ctrl
    elif expected is not False:
        chrommids = chrom_mids(chroms, mids, combinations)
        f = partial(pileups, c=c, pad=pad, ctrl=False, local=local,
            expected=expected,
            minshift=minshift, maxshift=maxshift, nshifts=nshifts,
            mindist=mindist, maxdist=maxdist, combinations=combinations,
            anchor=anchor, unbalanced=unbalanced, cov_norm=cov_norm,
            rescale=rescale, rescale_pad=rescale_pad,
            rescale_size=rescale_size)
        exps, ns, cov_starts, cov_ends = list(zip(*p.map(f, chrommids)))
        exp = np.sum(exps, axis=0)
        n = np.sum(ns)
        exp /= n
        loop /= exp
    p.close()
    return loop

def pileupsByWindow(chrom_mids, c, pad=7, ctrl=False,
                    minshift=10**5, maxshift=10**6, nshifts=1,
                    expected=False,
                    mindist=0, maxdist=10**9,
                    unbalanced=False, cov_norm=False,
                    rescale=False, rescale_pad=50, rescale_size=41):
    chrom, mids = chrom_mids

    if expected is not False:
        data = False
        expected = np.nan_to_num(expected[expected['chrom']==chrom]['balanced.avg'].values)
        logging.info('Doing expected')
    else:
        data = get_data(chrom, c, unbalanced, local=False)

#    if unbalanced and cov_norm and expected is False:
#        coverage = np.nan_to_num(np.ravel(np.sum(data, axis=0))) + \
#                   np.nan_to_num(np.ravel(np.sum(data, axis=1)))
#    else:
    coverage=False

    curmids = mids[mids["chr"] == chrom]
    mymaps = {}
    if not len(curmids) > 1:
#        mymap.fill(np.nan)
        return mymaps
    for i, (m, p) in curmids[['Mids', 'Pad']].astype(int).iterrows():
        if ctrl:
            current = controlRegions(get_combinations(curmids, c.binsize,
                                                    anchor=(chrom, m, m)),
                                       c.binsize, minshift, maxshift, nshifts)
        else:
             current = get_combinations(curmids, c.binsize, anchor=(chrom, m, m))
        mymap, n, cov_starts, cov_ends = _do_pileups(mids=current, data=data,
                                                     binsize=c.binsize,
                                                     pad=pad,
                                                     expected=expected,
                                                     mindist=mindist,
                                                     maxdist=maxdist,
                                                     local=False,
                                                     unbalanced=unbalanced,
                                                     cov_norm=cov_norm,
                                                     rescale=rescale,
                                                     rescale_pad=rescale_pad,
                                                     rescale_size=rescale_size,
                                                     coverage=coverage,
                                                     anchor=None)
        if n > 0:
            mymap = mymap/n
        else:
            mymap = make_outmap(pad, rescale, rescale_pad)
        mymaps[(m-p, m+p)] = mymap
    return mymaps

def pileupsByWindowWithControl(mids, filename, pad=100, nproc=1, chroms=None,
                               minshift=100000, maxshift=100000, nshifts=10,
                               expected=None,
                               mindist=0, maxdist=np.inf,
                               unbalanced=False,
                               cov_norm=False,
                               rescale=False, rescale_pad=1, rescale_size=99):
    p = Pool(nproc)
    c = cooler.Cooler(filename)
    if chroms is None:
        chroms = c.chromnames
    #Loops
    f = partial(pileupsByWindow, c=c, pad=pad, ctrl=False,
                minshift=minshift, maxshift=maxshift, nshifts=nshifts,
                expected=False,
                mindist=mindist, maxdist=maxdist, unbalanced=unbalanced,
                cov_norm=False,
                rescale=rescale, rescale_pad=rescale_pad,
                rescale_size=rescale_size)
    chrommids = chrom_mids(chroms, mids, True)
    loops = {chrom:lps for chrom, lps in zip(chroms,
                                             p.map(f, chrommids))}
    #Controls
    if nshifts>0:
        f = partial(pileupsByWindow, c=c, pad=pad, ctrl=True,
                    minshift=minshift, maxshift=maxshift, nshifts=nshifts,
                    expected=expected,
                    mindist=mindist, maxdist=maxdist, unbalanced=unbalanced,
                    cov_norm=cov_norm,
                    rescale=rescale, rescale_pad=rescale_pad,
                    rescale_size=rescale_size)
        chrommids = chrom_mids(chroms, mids, True)
        ctrls = {chrom:lps for chrom, lps in zip(chroms,
                                             p.map(f, chrommids))}
    elif expected is not False:
        f = partial(pileupsByWindow, c=c, pad=pad, ctrl=False,
            expected=expected,
            minshift=minshift, maxshift=maxshift, nshifts=nshifts,
            mindist=mindist, maxdist=maxdist,
            unbalanced=unbalanced, cov_norm=False,
            rescale=rescale, rescale_pad=rescale_pad,
            rescale_size=rescale_size)
        chrommids = chrom_mids(chroms, mids, True)
        ctrls = {chrom:lps for chrom, lps in zip(chroms,
                                             p.map(f, chrommids))}
    p.close()

    finloops = {}
    for chrom in loops.keys():
        for pos, lp in loops[chrom].items():
            finloops[(chrom, pos[0], pos[1])] = lp/ctrls[chrom][pos]
    return finloops

if __name__ == "__main__":
    import argparse
    import logging
    parser = argparse.ArgumentParser()
    parser.add_argument("coolfile", type=str,
                        help="Cooler file with your Hi-C data")
    parser.add_argument("baselist", type=str,
                        help="""A 3-column tab-delimited bed file with
                        coordinates which intersections to pile-up.
                        Alternatively, a 6-column double-bed file (i.e.
                        chr1,start1,end1,chr2,start2,end2) with coordinates of
                        centers of windows that will be piled-up. Can be piped
                        in via stdin, then use "-".""")
##### Extra arguments
    parser.add_argument("--pad", default=100, type=int, required=False,
                        help="""Padding of the windows (i.e. final size of the
                        matrix is 2×pad+res), in kb""")
### Control of controls
    parser.add_argument("--minshift", default=10**5, type=int, required=False,
                        help="""Shortest distance for randomly shifting
                        coordinates when creating controls""")
    parser.add_argument("--maxshift", default=10**6, type=int, required=False,
                        help="""Longest distance for randomly shifting
                        coordinates when creating controls""")
    parser.add_argument("--nshifts", default=10, type=int, required=False,
                        help="""Number of control regions per averaged
                        window""")
    parser.add_argument("--expected", default=None, type=str, required=False,
                        help="""File with expected (output of
                        cooltools compute-expected). If None, don't use expected
                        and use randomly shifted controls""")
### Filtering
    parser.add_argument("--mindist", type=int, required=False,
                        help="""Minimal distance of intersections to use. If
                        not specified, uses --pad as mindist""")
    parser.add_argument("--maxdist", type=int, required=False,
                        help="""Maximal distance of intersections to use""")
    parser.add_argument("--minsize", type=int, required=False,
                        help="""Minimal length of features to use for local
                        analysis""")
    parser.add_argument("--maxsize", type=int, required=False,
                        help="""Maximal length of features to use for local
                        analysis""")
    parser.add_argument("--excl_chrs", default='chrY,chrM', type=str,
                        required=False,
                        help="""Exclude these chromosomes from analysis""")
    parser.add_argument("--incl_chrs", default='all', type=str, required=False,
                        help="""Include these chromosomes; default is all.
                        excl_chrs overrides this.""")
    parser.add_argument("--subset", default=0, type=int, required=False,
                        help="""Take a random sample of the bed file - useful
                        for files with too many featuers to run as is, i.e.
                        some repetitive elements. Set to 0 or lower to keep all
                        data.""")
### Modes of action
    parser.add_argument("--anchor", default=None, type=str, required=False,
                        help="""A UCSC-style coordinate to use as an anchor to
                        create intersections with coordinates in the baselist
                        """)
    parser.add_argument("--by_window", action='store_true', default=False,
                        required=False,
                        help="""Create a pile-up for each coordinate in the
                        baselist. Will save a master-table with coordinates,
                        their enrichments and cornerCV, which is reflective of
                        noisiness""")
    parser.add_argument("--save_all", action='store_true', default=False,
                        required=False,
                        help="""If by-window, save all individual pile-ups in a
                        separate json file""")
    parser.add_argument("--local", action='store_true', default=False,
                        required=False,
                        help="""Create local pileups, i.e. along the
                        diagonal""")
    parser.add_argument("--unbalanced", action='store_true',
                        required=False,
                        help="""Do not use balanced data.
                        Useful for single-cell Hi-C data together with
                        --coverage_norm, not recommended otherwise.""")
    parser.add_argument("--coverage_norm", action='store_true',
                        required=False,
                        help="""If --unbalanced, also add coverage
                        normalization based on chromosome marginals""")
### Rescaling
    parser.add_argument("--rescale", action='store_true', default=False,
                        required=False,
                        help="""Do not use centres of features and pad, and
                        rather use the actual feature sizes and rescale
                        pileups to the same shape and size""")
    parser.add_argument("--rescale_pad", default=1.0, required=False, type=float,
                        help="""If --rescale, padding in fraction of feature
                        length""")
    parser.add_argument("--rescale_size", type=int,
                        default=99, required=False,
                        help="""If --rescale, this is used to determine the
                        final size of the pileup, i.e. it ill be size×size. Due
                        to technical limitation in the current implementation,
                        has to be an odd number""")


    parser.add_argument("--n_proc", default=1, type=int, required=False,
                        help="""Number of processes to use. Each process works
                        on a separate chromosome, so might require quite a bit
                        more memory, although the data are always stored as
                        sparse matrices""")
### Output
    parser.add_argument("--outdir", default='.', type=str, required=False,
                        help="""Directory to save the data in""")
    parser.add_argument("--outname", default='auto', type=str, required=False,
                        help="""Name of the output file. If not set, is
                        generated automatically to include important
                        information.""")
    parser.add_argument("-l", "--log", dest="logLevel",
                        choices=['DEBUG', 'INFO', 'WARNING',
                                 'ERROR', 'CRITICAL'],
                        default='INFO',
                        help="Set the logging level.")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.logLevel))

    logging.info(args)
    if args.n_proc==0:
        nproc=-1
    else:
        nproc=args.n_proc

    c = cooler.Cooler(args.coolfile)

    if not os.path.isfile(args.baselist) and args.baselist != '-':
        raise FileExistsError("Loop(base) coordinate file doesn't exist")

    coolname = args.coolfile.split('::')[0].split('/')[-1].split('.')[0]
    if args.baselist != '-':
        bedname = args.baselist.split('/')[-1].split('.bed')[0].split('_mm9')[0].split('_mm10')[0]
    else:
        bedname = 'stdin'
        import sys
        args.baselist = sys.stdin
    if args.expected is not None:
        if args.nshifts > 0:
            logging.warning('With specified expected will not use controls')
            args.nshifts = 0
        if not os.path.isfile(args.expected):
            raise FileExistsError("Expected file doesn't exist")
        expected = pd.read_csv(args.expected, sep='\t', header=0)
    else:
        expected = False

    pad = args.pad*1000//c.binsize

    if args.mindist is None:
        mindist=pad*c.binsize
    else:
        mindist=args.mindist

    if args.maxdist is None:
        maxdist=np.inf
    else:
        maxdist=args.maxdist

    if args.incl_chrs=='all':
        incl_chrs = c.chromnames
    else:
        incl_chrs = args.incl_chrs.split(',')

    if args.by_window and args.rescale:
        raise NotImplementedError("""Rescaling with by-window pileups is not
                                  supported""")

    if args.rescale and args.rescale_size%2==0:
        raise ValueError("Please provide an odd rescale_size")

    if args.anchor is not None:
        if '_' in args.anchor:
            anchor, anchor_name = args.anchor.split('_')
            anchor = cooler.util.parse_region_string(anchor)
        else:
            anchor = cooler.util.parse_region_string(args.anchor)
            anchor_name = args.anchor
    else:
        anchor = None

    if anchor:
        fchroms = [anchor[0]]
    else:
        chroms = c.chromnames
        fchroms = []
        for chrom in chroms:
            if chrom not in args.excl_chrs.split(',') and chrom in incl_chrs:
                fchroms.append(chrom)


    bases = pd.read_csv(args.baselist, sep='\t',
                            names=['chr1', 'start1', 'end1',
                                   'chr2', 'start2', 'end2'],
                        index_col=False)
    if np.all(pd.isnull(bases[['chr2', 'start2', 'end2']].values)):
        bases = bases[['chr1', 'start1', 'end1']]
        bases.columns = ['chr', 'start', 'end']
        if not np.all(bases['end']>=bases['start']):
            raise ValueError('Some ends in the file are smaller than starts')
        if args.local:
            if args.minsize is None:
                args.minsize = 0
            if args.maxsize is None:
                args.maxsize = np.inf
            length = bases['end']-bases['start']
            bases = bases[(length >= args.minsize) & (length <= args.maxsize)]
        combinations = True
    else:
        if not np.all(bases['chr1']==bases['chr2']):
            logging.warning("Found inter-chromosomal loci pairs, discarding them")
            bases = bases[bases['chr1']==bases['chr2']]
        if anchor:
            raise ValueError("Can't use anchor with both sides of loops defined")
        elif args.local:
            raise ValueError("Can't make local with both sides of loops defined")
#        if not np.all(bases['end1']>=bases['start1']) or\
#           not np.all(bases['end2']>=bases['start2']):
#            raise ValueError('Some interval ends in the file are smaller than starts')
#        if not np.all(bases[['start2', 'end2']].mean(axis=1)>=bases[['start1', 'end1']].mean(axis=1)):
#            raise ValueError('Some centres of right ends in the file are\
#                             smaller than centres in the left ends')
        combinations = False
    mids = get_mids(bases, resolution=c.binsize, combinations=combinations)
    if args.subset > 0 and args.subset < len(mids):
        mids = mids.sample(args.subset)

    if args.outname=='auto':
        outname = '%s-%sK_over_%s' % (coolname, c.binsize/1000, bedname)
        if args.nshifts>0:
            outname += '_%s-shifts' % args.nshifts
        if args.expected is not None:
            outname += '_expected'
        if args.nshifts <= 0 and args.expected is None:
            outname += '_noNorm'
        if anchor:
            outname += '_from_%s' % anchor_name
        if args.local:
            outname += '_local'
            if args.minsize > 0 or args.maxsize < np.inf:
                outname += '_len_%s-%s' % (args.minsize, args.maxsize)
        elif args.mindist is not None or args.maxdist is not None:
            outname += '_dist_%s-%s' % (mindist, maxdist)
        if args.rescale:
            outname += '_rescaled'
        if args.unbalanced:
            outname += '_unbalanced'
        if args.coverage_norm:
            outname += '_covnorm'
        if args.subset > 0:
            outname += '_subset-%s' % args.subset
        if args.by_window:
            outname = 'Enrichment_%s.txt' % outname
        else:
            outname += '.np.txt'
    else:
        outname = args.outname

    if args.by_window:
        if not combinations:
            raise ValueError("Can't make by-window pileups without making combinations")
        if args.local:
            raise ValueError("Can't make local by-window pileups")
        if anchor:
            raise ValueError("Can't make by-window combinations with an anchor")
        if args.coverage_norm:
            raise NotImplementedError("""Can't make by-window combinations with
                                      coverage normalization - please use
                                      balanced data instead""")
        if args.outname!='auto':
            logging.warning("Always using autonaming for by-window pileups")

        finloops = pileupsByWindowWithControl(mids=mids,
                                              filename=args.coolfile,
                                              pad=pad,
                                              nproc=nproc,
                                              chroms=fchroms,
                                              minshift=args.minshift,
                                              maxshift=args.maxshift,
                                              nshifts=args.nshifts,
                                              expected=expected,
                                              mindist=mindist,
                                              maxdist=maxdist,
                                              unbalanced=args.unbalanced,
                                              cov_norm=args.coverage_norm,
                                              rescale=args.rescale,
                                              rescale_pad=args.rescale_pad,
                                              rescale_size=args.rescale_size)
        if args.save_all:
            outdict = {'%s:%s-%s' % key : val.tolist() for key,val in finloops.items()}
            import json
            with open(os.path.join(args.outdir, outname)[:-4] + '.json', 'w') as fp:
                json.dump(outdict, fp)#, sort_keys=True, indent=4)

        def prepare_single(item, outname=outname):
            key, amap = item
#            coords = (key[0], int(key[1]), int(key[2]))
            enr1 = get_enrichment(amap, 1)
            enr3 = get_enrichment(amap, 3)
            cv3 = cornerCV(amap, 3)
            cv5 = cornerCV(amap, 5)
#            if args.save_all:
#                outname = outname + '_%s:%s-%s.np.txt' % coords
#                try:
#                    np.savetxt(os.path.join(args.outdir, 'individual', outname),
#                               amap)
#                except FileNotFoundError:
#                    os.mkdir(os.path.join(args.outdir, 'individual'))
#                    np.savetxt(os.path.join(args.outdir, 'individual', outname),
#                               amap)
            return list(key)+[enr1, enr3, cv3, cv5]

        p = Pool(nproc)
        data = p.map(prepare_single, finloops.items())
        p.close()
        data = pd.DataFrame(data, columns=['chr', 'start', 'end',
                                           'Enrichment1', 'Enrichment3', 'CV3', 'CV5'])
        data = data.reindex(index=order_by_index(data.index,
                                        index_natsorted(zip(data['chr'],
                                                              data['start']))))
        try:
            data.to_csv(os.path.join(args.outdir, outname),
                        sep='\t', index=False)
        except FileNotFoundError:
            os.mkdir(args.outdir)
            data.to_csv(os.path.join(args.outdir, outname),
                        sep='\t', index=False)
    else:
        loop = pileupsWithControl(mids=mids, filename=args.coolfile,
                                       pad=pad, nproc=nproc,
                                       chroms=fchroms, local=args.local,
                                       minshift=args.minshift,
                                       maxshift=args.maxshift,
                                       nshifts=args.nshifts,
                                       expected=expected,
                                       mindist=mindist,
                                       maxdist=maxdist,
                                       combinations=combinations,
                                       anchor=anchor,
                                       unbalanced=args.unbalanced,
                                       cov_norm=args.coverage_norm,
                                       rescale=args.rescale,
                                       rescale_pad=args.rescale_pad,
                                       rescale_size=args.rescale_size)
        try:
            np.savetxt(os.path.join(args.outdir, outname), loop)
        except FileNotFoundError:
            try:
                os.mkdir(args.outdir)
            except FileExistsError:
                pass
            np.savetxt(os.path.join(args.outdir, outname), loop)
