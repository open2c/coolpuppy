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

def cornerCV(amap, i=4):
    corners = np.concatenate((amap[0:i, 0:i], amap[-i:, -i:]))
    corners[~np.isfinite(corners)]=0
    return np.std(corners)/np.mean(corners)

def normCis(amap, i=3):
    return amap/np.nanmean((amap[0:i, 0:i]+amap[-i:, -i:]))*2

def get_enrichment_3(amap):
    c = int(np.floor(amap.shape[0]/2))
    return np.nanmean(amap[c-1:c+2, c-1:c+2])

def get_mids(intervals, combinations=True):
    if combinations:
        intervals = intervals.sort_values(['Chromosome', 'Start'])
        mids = np.round((intervals['End']+intervals['Start'])/2).astype(int)
        mids = pd.DataFrame({'Chromosome':intervals['Chromosome'],
                             'Mids':mids}).drop_duplicates()
    else:
        intervals = intervals.sort_values(['Chromosome1', 'Chromosome2',
                                           'Start1', 'Start2'])
        mids1 = np.round((intervals['End1']+intervals['Start1'])/2).astype(int)
        mids2 = np.round((intervals['End2']+intervals['Start2'])/2).astype(int)
        mids = pd.DataFrame({'Chromosome1':intervals['Chromosome1'],
                             'Mids1':mids1,
                             'Chromosome2':intervals['Chromosome2'],
                             'Mids2':mids2}).drop_duplicates()
    return mids

def get_combinations(mids, res, local=False, anchor=None):
    if local and anchor:
        raise ValueError("Can't have a local pileup with an anchor")
    m = (mids['Mids']//res).values.astype(int)
    if local:
        for i in m:
            yield i, i
    elif anchor:
        anchor_bin = int((anchor[1]+anchor[2])/2//res)
        for i in m:
            yield anchor_bin, i
    else:
        for i in itertools.combinations(m, 2):
                yield i

def get_positions_pairs(mids, res):
    m1 = (mids['Mids1']//res).astype(int).values
    m2 = (mids['Mids2']//res).astype(int).values
    for a, b in zip(m1, m2):
        yield a, b

def controlLoops(midcombs, res, minshift=10**5, maxshift=10**6, nshifts=1):
    minbin = minshift//res
    maxbin = maxshift//res
    for start, end in midcombs:
        for i in range(nshifts):
            shift = np.random.randint(minbin, maxbin)
            sign = np.sign(np.random.random() - 0.5).astype(int)
            shift *= sign
            yield start+shift, end+shift

def averageLoops(chrom, c, mids, pad=7, ctrl=False, local=False,
                 minshift=10**5, maxshift=10**6, nshifts=1,
                 mindist=0, maxdist=10**9, combinations=True, anchor=None,
                 individual=False):
    print(chrom)

    data = sparse.triu(c.matrix(sparse=True, balance=True).fetch(chrom)).tocsr()

    if anchor:
        assert chrom==anchor[0]
#        anchor_bin = (anchor[1]+anchor[2])/2//c.binsize
        print(anchor)
    else:
        anchor = None
    mymap = np.zeros((2*pad + 1, 2*pad + 1), np.float64)

    if combinations:
        current = mids[mids["Chromosome"] == chrom]
    else:
        current = mids[(mids["Chromosome1"] == chrom) &
                       (mids["Chromosome2"] == chrom)]
    if not len(current) > 1:
#        mymap.fill(np.nan)
        return mymap, 0

    if ctrl:
        if combinations:
            current = controlLoops(get_combinations(current, c.binsize, local,
                                                    anchor),
                                   c.binsize, minshift, maxshift, nshifts)
        else:
            current = controlLoops(get_positions_pairs(current, c.binsize),
                                   c.binsize, minshift, maxshift, nshifts)
    else:
        if combinations:
            current = get_combinations(current, c.binsize, local, anchor)
        else:
            current = get_positions_pairs(current, c.binsize)
    n = 0
    for stBin, endBin in current:
#        if not local and abs(endBin - stBin) < pad*2:
#            continue
        if mindist <= abs(endBin - stBin)*c.binsize < maxdist or local:
            try:
                mymap += np.nan_to_num(data[stBin - pad:stBin + pad + 1,
                                           endBin - pad:endBin + pad + 1].toarray())
                n += 1
            except (IndexError, ValueError) as e:
                continue
    print(n)
    if n > 0:
        return mymap/n, n
    else: #Don't want to get infs and nans
        return mymap, n

def averageLoopsByWindow(chrom, mids, c, pad=7, ctrl=False,
                         minshift=10**5, maxshift=10**6, nshifts=1,
                         mindist=0, maxdist=10**9):
    print(chrom)
    if c is None:
        assert isinstance(chrom, np.ndarray)
        data = chrom
    else:
        data = sparse.triu(c.matrix(sparse=True, balance=True).fetch(chrom)).tocsr()

    curmids = mids[mids["Chromosome"] == chrom]
    mymaps = {}
    if not len(curmids) > 1:
#        mymap.fill(np.nan)
        return mymaps
    for m in curmids['Mids'].values:
        print(m)
        if ctrl:
            current = controlLoops(get_combinations(curmids, c.binsize,
                                                    anchor=(chrom, m, m)),
                                       c.binsize, minshift, maxshift, nshifts)
        else:
             current = get_combinations(curmids, c.binsize, anchor=(chrom, m, m))
        mymap = np.zeros((2*pad + 1, 2*pad + 1), np.float64)
        n = 0
        for stBin, endBin in current:
#            if abs(endBin - stBin) < pad*2:
#                continue
            if mindist <= abs(endBin - stBin)*c.binsize < maxdist:
                try:
                    mymap += np.nan_to_num(data[stBin - pad:stBin + pad + 1,
                                                endBin - pad:endBin + pad + 1].toarray())
                    n += 1
                except (IndexError, ValueError) as e:
                    continue
            else:
                continue
        print('n=%s' % n)
        if n > 0:
            mymap = mymap/n
        mymaps[m] = mymap
    return mymaps

def averageLoopsWithControl(mids, filename, pad, nproc, chroms, local,
                            minshift, maxshift, nshifts, mindist, maxdist,
                            combinations, anchor):
    p = Pool(nproc)
    c = cooler.Cooler(filename)
    #Loops
    f = partial(averageLoops, mids=mids, c=c, pad=pad, ctrl=False, local=local,
                minshift=minshift, maxshift=maxshift, nshifts=nshifts,
                mindist=mindist, maxdist=maxdist, combinations=combinations,
                anchor=anchor)
    loops, ns = list(zip(*p.map(f, chroms)))
    loop = np.average(loops, axis=0, weights=ns) #Weights from how many windows we actually used
    #Controls
    f = partial(averageLoops, mids=mids, c=c, pad=pad, ctrl=True, local=local,
                minshift=minshift, maxshift=maxshift, nshifts=nshifts,
                mindist=mindist, maxdist=maxdist, combinations=combinations,
                anchor=anchor)
    ctrls, ns = list(zip(*p.map(f, chroms)))
    ctrl = np.average(ctrls, axis=0, weights=ns)
    p.close()
    return loop/ctrl

def averageLoopsByWindowWithControl(mids, filename, pad, nproc, chroms,
                            minshift, maxshift, nshifts, mindist, maxdist):
    p = Pool(nproc)
    c = cooler.Cooler(filename)
    #Loops
    f = partial(averageLoopsByWindow, c=c, mids=mids, pad=pad, ctrl=False,
                minshift=minshift, maxshift=maxshift, nshifts=nshifts,
                mindist=mindist, maxdist=maxdist)
    loops = {chrom:lps for chrom, lps in zip(fchroms, p.map(f, chroms))}
    #Controls
    f = partial(averageLoopsByWindow, c=c, mids=mids, pad=pad, ctrl=True,
                minshift=minshift, maxshift=maxshift, nshifts=nshifts,
                mindist=mindist, maxdist=maxdist)
    ctrls = {chrom:lps for chrom, lps in zip(fchroms, p.map(f, chroms))}
    p.close()

    finloops = {}
    for chrom in loops.keys():
        for pos, lp in loops[chrom].items():
            finloops[(chrom, pos)] = lp/ctrls[chrom][pos]
    return finloops

def prepare_single(item):
    key, amap = item
    if np.any(amap<0):
        print(amap)
        amap = np.zeros_like(amap)
    coords = (key[0], int(key[1]//c.binsize*c.binsize),
                      int(key[1]//c.binsize*c.binsize + c.binsize))
    enr = get_enrichment_3(normCis(amap))
    cv = cornerCV(amap)
    if args.save_all:
        outname = baseoutname + '_%s:%s-%s.np.txt' % coords
        try:
            np.savetxt(os.path.join(args.outdir, 'individual', outname),
                       amap)
        except FileNotFoundError:
            os.mkdir(os.path.join(args.outdir, 'individual'))
            np.savetxt(os.path.join(args.outdir, 'individual', outname),
                       amap)
    return list(coords)+[enr, cv]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("coolfile", type=str)
    parser.add_argument("baselist", type=str)
    parser.add_argument("--pad", default=7, type=int, required=False)
    parser.add_argument("--minshift", default=10**5, type=int, required=False)
    parser.add_argument("--maxshift", default=10**6, type=int, required=False)
    parser.add_argument("--nshifts", default=10, type=int, required=False)
    parser.add_argument("--mindist", type=int, required=False)
    parser.add_argument("--maxdist", type=int, required=False)
    parser.add_argument("--excl_chrs", default='chrY,chrM', type=str,
                        required=False)
    parser.add_argument("--incl_chrs", default='all', type=str, required=False)
    parser.add_argument("--anchor", default=None, type=str, required=False) #UCSC-style
# Use each bed entry as an anchor, save all pileups as separate files
    parser.add_argument("--by_window", action='store_true', default=False,
                        required=False)
# If by_window and save_all, save all individual pileups; if not save_all,
# only save a master-table with window coordinates and enrichments.
# Can save a very large number of files, so use cautiosly!
    parser.add_argument("--save_all", action='store_true', default=False,
                        required=False)
    parser.add_argument("--local", action='store_true', default=False,
                        required=False)
    parser.add_argument("--n_proc", default=1, type=int, required=False)
    parser.add_argument("--outdir", default='.', type=str, required=False)
    parser.add_argument("--outname", default='auto', type=str, required=False)
    args = parser.parse_args()
    print(args)
    if args.n_proc==0:
        nproc=-1
    else:
        nproc=args.n_proc

    c = cooler.Cooler(args.coolfile)

    coolname = args.coolfile.split('/')[-1].split('.')[0]
    bedname = args.baselist.split('/')[-1].split('.')[0].split('_mm9')[0]

    if args.mindist is None:
        mindist=0
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
                            names=['Chromosome1', 'Start1', 'End1',
                                   'Chromosome2', 'Start2', 'End2'],
                        index_col=False)
    if np.all(pd.isnull(bases[['Chromosome2', 'Start2', 'End2']])):
        bases = bases[['Chromosome1', 'Start1', 'End1']]
        bases.columns = ['Chromosome', 'Start', 'End']
        mids = get_mids(bases, combinations=True)
        combinations = True
    else:
        assert np.all(bases['Chromosome1']==bases['Chromosome2'])
        if anchor:
            raise ValueError("Can't use anchor with both sides of loops defined")
        elif args.local:
            raise ValueError("Can't make local with both sides of loops defined")
        mids = get_mids(bases, combinations=False)
        combinations = False

    if args.by_window:
        if not combinations:
            raise ValueError("Can't make by-window pileups without making combinations")
        if args.local:
            raise ValueError("Can't make local by-window pileups")
        if anchor:
            raise ValueError("Can't make by-window combinations with an anchor")
        if args.outname!='auto':
            import warnings
            warnings.warn("Always using autonaming for by-window pileups")

        finloops = averageLoopsByWindowWithControl(mids=mids,
                                                   filename=args.coolfile,
                                                   pad=args.pad,
                                                   nproc=nproc,
                                                   chroms=fchroms,
                                                   minshift=args.minshift,
                                                   maxshift=args.maxshift,
                                                   nshifts=args.nshifts,
                                                   mindist=mindist,
                                                   maxdist=maxdist)
        data = []
        baseoutname = '%s-%sK_over_%s' % (coolname, c.binsize/1000, bedname)
        if args.mindist is not None or args.maxdist is not None:
            baseoutname = baseoutname + '_dist_%s-%s' % (mindist, maxdist)

        p = Pool(nproc)
        data = p.map(prepare_single, finloops.items())
        p.close()
        data = pd.DataFrame(data, columns=['Chromosome', 'Start', 'End',
                                           'Enrichment', 'CV'])
        data = data.reindex(index=order_by_index(data.index,
                                        index_natsorted(zip(data['Chromosome'],
                                                              data['Start']))))
        try:
            data.to_csv(os.path.join(args.outdir,
                                     'Enrichments_%s.tsv' % baseoutname),
                        sep='\t', index=False)
        except FileNotFoundError:
            os.mkdir(args.outdir)
            data.to_csv(os.path.join(args.outdir,
                                     'Enrichments_%s.tsv' % baseoutname),
                        sep='\t', index=False)
    else:
        loop = averageLoopsWithControl(mids=mids, filename=args.coolfile,
                                       pad=args.pad, nproc=nproc,
                                       chroms=fchroms, local=args.local,
                                       minshift=args.minshift,
                                       maxshift=args.maxshift,
                                       nshifts=args.nshifts,
                                       mindist=mindist,
                                       maxdist=maxdist,
                                       combinations=combinations,
                                       anchor=anchor)
        if args.outname=='auto':
            outname = '%s-%sK_over_%s' % (coolname, c.binsize/1000, bedname)
            if anchor:
                outname += '_from_%s' % anchor_name
            if args.mindist is not None or args.maxdist is not None:
                outname += '_dist_%s-%s' % (mindist, maxdist)
            if args.local:
                outname += '_local'
            outname += '.np.txt'

        else:
            outname = args.outname
        try:
            np.savetxt(os.path.join(args.outdir, outname), loop)
        except FileNotFoundError:
            try:
                os.mkdir(args.outdir)
            except FileExistsError:
                pass
            np.savetxt(os.path.join(args.outdir, outname), loop)
