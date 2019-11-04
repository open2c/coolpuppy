# -*- coding: utf-8 -*-
import numpy as np
import cooler
import pandas as pd
import itertools
from multiprocessing import Pool
from functools import partial
import logging
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

def filter_bed(df, minsize, maxsize, chroms):
    length = df['end'] - df['start']
    df = df[(length>=minsize) & (length<=maxsize)]
    if chroms != 'all':
        df = df[df['chr'].isin(chroms)]
    if not np.all(df['end']>=df['start']):
        raise ValueError('Some ends in the file are smaller than starts')
    return df

def filter_bedpe(df, mindist, maxdist, chroms):
    mid1 = np.mean(df[['start1', 'end1']], axis=1)
    mid2 = np.mean(df[['start2', 'end2']], axis=1)
    length = mid2 - mid1
    df = df[(length>=mindist) & (length<=maxdist)]
    if chroms != 'all':
        df = df[(df['chr1'].isin(chroms)) & (df['chr2'].isin(chroms))]
    return df

#def check_bed_bedpe(df):
#    if np.any(df[['chr2', 'start2', 'end2']].isnull().all()):
#        # If the file has <6 columns, assume bed
#        return 'bed'
#    else:
#        # Else assume bedpe
#        return 'bedpe'

def auto_read_bed(f,  kind='auto', chroms='all', minsize=0, maxsize=np.inf,
                                  mindist=0, maxdist=np.inf, stdin=False):
    if kind == 'auto': #Guessing whether it's bed or bedpe style file
        if stdin:
            row1 = f.__next__().split('\t')
        else:
            with open(f, 'r') as fobject:
                row1 = fobject.readline().split('\t')
        if len(row1) ==6:
            filetype = 'bedpe'
            row1 = [row1[0], int(row1[1]), int(row1[2]),
                    row1[3], int(row1[4]), int(row1[5])]
        elif len(row1) == 3:
            filetype = 'bed'
            row1 = [row1[0], int(row1[1]), int(row1[2])]
        else:
            raise ValueError("""Input bed(pe) file has unexpected number of
                             columns: got {}, expect 3 (bed) or 6 (bedpe)
                             """.format(len(row1)))
#        testdf = pd.read_csv(f, sep='\t',
#                                names=['chr1', 'start1', 'end1',
#                                       'chr2', 'start2', 'end2'],
#                            index_col=False, chunksize = 10).__next__()
#        kind = check_bed_bedpe(testdf)

    if filetype=='bed' or kind == 'bed':
        filter_func = partial(filter_bed, minsize=minsize, maxsize=maxsize,
                              chroms=chroms)
        names = ['chr', 'start', 'end']
        row1 = filter_func(pd.DataFrame([row1], columns=names))
    elif filetype=='bedpe' or kind =='bedpe': #bedpe
        filter_func = partial(filter_bedpe, mindist=mindist, maxdist=maxdist,
                              chroms=chroms)
        names = ['chr1', 'start1', 'end1', 'chr2', 'start2', 'end2']
        row1 = filter_func(pd.DataFrame([row1], columns=names))
    else:
        raise ValueError("""Unsupported input kind: {}.
                         Expect {} or {}""".format(kind, 'bed', 'bedpe'))
    bases = []

    appended = False
    if kind == 'auto':

        if row1.shape[0]==1:
            bases.append(row1)
            appended = True

    for chunk in pd.read_csv(f, sep='\t',
                            names=names, index_col=False,
                            chunksize = 10**4):
        bases.append(filter_func(chunk))
    bases = pd.concat(bases)
    if appended: # Would mean we read it twice when checking and in the first chunk
        bases = bases.iloc[1:]
    if filetype=='bed' or kind == 'bed':
        bases['chr'] = bases['chr'].astype(str)
        bases[['start', 'end']] = bases[['start', 'end']].astype(np.uint64)
    if filetype=='bedpe' or kind == 'bedpe':
        bases[['chr1', 'chr2']] = bases[['chr1', 'chr2']].astype(str)
        bases[['start1', 'end1', 'start2', 'end2']] = bases[['start1', 'end1', 'start2', 'end2']].astype(np.uint64)
    return bases

def bedpe2bed(df, ends=True, how='center'):
    # If ends==True, will combine all coordinates from both loop ends into one bed-style df
    # Else, will convert to long intervals covering the whole loop, for e.g. rescaled averaging
    # how: center - take center of loop bases; inner or outer
    if ends:
        df1 = df[['chr1', 'start1', 'end1']]
        df1.columns = ['chr', 'start', 'end']
        df2 = df[['chr2', 'start2', 'end2']]
        df2.columns = ['chr', 'start', 'end']
        return pd.concat([df1, df2]).sort_values(['chr', 'start', 'end']).reset_index(drop=True)

    if how=='center':
        df['start'] = np.mean(df['start1'], df['end1'], axis=0)
        df['end'] = np.mean(df['start2'], df['end2'], axis=0)
    elif how == 'outer':
        df = df[['chr1', 'start1', 'end2']]
        df.columns = ['chr', 'start', 'end']
    elif how == 'inner':
        df = df[['chr1', 'end1', 'start2']]
        df.columns = ['chr', 'start', 'end']
    return df


def get_mids(intervals, resolution, kind='bed'):
    if kind=='bed':
        intervals = intervals.sort_values(['chr', 'start'])
        mids = np.round((intervals['end']+intervals['start'])/2).astype(int)
        widths = np.round((intervals['end']-intervals['start'])).astype(int)
        mids = pd.DataFrame({'chr':intervals['chr'],
                             'Mids':mids,
                             'Bin':mids//resolution,
                             'Pad':widths/2}).drop_duplicates(['chr', 'Bin'])#.drop('Bin', axis=1)
    elif kind=='bedpe':
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
                            ).drop_duplicates(['chr1', 'chr2',
                                'Bin1', 'Bin2'])#.drop(['Bin1', 'Bin2'], axis=1)
    else:
        raise ValueError("""
                         kind can only be "bed" or "bedpe"
                         """)
    return mids

def get_combinations(mids, res, mids2=None, ordered_mids=True,
                     local=False, anchor=None):
    if (local and anchor) or (local and (mids2 is not None)):
        raise ValueError("""Can't have a local pileup with an anchor or with
                            two bed files""")
    m =  mids['Bin'].values.astype(int)
    p = (mids['Pad']//res).values.astype(int)

    if mids2 is not None:
        m2 =  mids2['Bin'].values.astype(int)
        p2 = (mids2['Pad']//res).values.astype(int)

    if local:
        for i, pi in zip(m, p):
            yield i, i, pi, pi
    elif anchor:
        anchor_bin = int((anchor[1]+anchor[2])/2//res)
        anchor_pad = int(round((anchor[2] - anchor[1])/2))
        for i, pi in zip(m, p):
            yield anchor_bin, i, anchor_pad, pi
    elif mids2 is None:
        for i, j in zip(itertools.combinations(m, 2),
                        itertools.combinations(p, 2)):
            yield list(i)+list(j)
    elif (mids2 is not None) and ordered_mids:
        for i, j in zip(itertools.product(m, m2),
                        itertools.product(p, p2)):
            if i[1] > i[0]:
                yield list(i)+list(j)
    elif (mids2 is not None) and (not ordered_mids):
        for i, j in itertools.chain(zip(itertools.product(m, m2),
                                        itertools.product(p, p2)),
                                    zip(itertools.product(m2, m),
                                        itertools.product(p2, p))):
            yield list(i)+list(j)

def get_positions_pairs(mids, res):
    m1 = mids['Bin1'].astype(int).values
    m2 = mids['Bin2'].astype(int).values
    p1 = (mids['Pad1']//res).astype(int).values
    p2 = (mids['Pad2']//res).astype(int).values
    for posdata in zip(m1, m2, p1, p2):
        yield posdata

def prepare_single(item):
    key, (n, amap) = item
    enr1 = get_enrichment(amap, 1)
    enr3 = get_enrichment(amap, 3)
    cv3 = cornerCV(amap, 3)
    cv5 = cornerCV(amap, 5)
    return list(key)+[n, enr1, enr3, cv3, cv5]

def controlRegions(midcombs, res, minshift=10**5, maxshift=10**6, nshifts=1,
                   seed=None):
    if seed is not None:
        np.random.seed(seed)
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
#        if local:
        exp_subset = np.pad(exp_subset, (-exp_lo, 0), mode='reflect')
#        else:
#            exp_subset = np.pad(exp_subset, (-exp_lo, 0), mode='constant')
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

def get_data(chrom, c, balance, local):
    logging.debug('Loading data')
    data = c.matrix(sparse=True, balance=balance).fetch(chrom)
#    if local:
    data = data.tocsr()
#    else:
#        data = sparse.triu(data, 2).tocsr()
    return data

def _do_pileups(mids, data, binsize, pad, expected, mindist, maxdist, local,
                balance, cov_norm, rescale, rescale_pad, rescale_size,
                coverage, anchor):
    mymap = make_outmap(pad, rescale, rescale_size)
    cov_start = np.zeros(mymap.shape[0])
    cov_end = np.zeros(mymap.shape[1])
    n = 0
    for stBin, endBin, stPad, endPad in mids:
        rot_flip = False
        rot = False
        if stBin > endBin:
            stBin, stPad, endBin, endPad = endBin, endPad, stBin, stPad
            if anchor is None:
                rot_flip = True
            else:
                rot = True
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
                    newmap = data[lo_left:hi_left,
                                  lo_right:hi_right].toarray()
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
                newmap = np.rot90(np.flipud(newmap), 1)
            elif rot:
                newmap = np.rot90(newmap, -1)
            mymap = np.nansum([mymap, newmap], axis=0)
            if cov_norm and (expected is False) and (balance is False):
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

def chrom_mids(chroms, mids, kind):
    for chrom in chroms:
        if kind=='bed':
            yield chrom, mids[mids['chr']==chrom]
        else:
            yield chrom, mids[mids['chr1']==chrom]

def pileups(chrommids, c, pad=7, ctrl=False, local=False,
            two_beds=False, ordered_mids=True,
            minshift=10**5, maxshift=10**6, nshifts=1, expected=False,
            mindist=0, maxdist=10**9, kind='bed', anchor=None,
            balance=True, cov_norm=False,
            rescale=False, rescale_pad=50, rescale_size=41,
            seed=None):
    if two_beds:
        (chrom1, mids), (chrom2, mids2) = chrommids
        assert chrom1==chrom2
        chrom = chrom1
    else:
        chrom, mids = chrommids
        mids2 = None
    mymap = make_outmap(pad, rescale, rescale_size)
    cov_start = np.zeros(mymap.shape[0])
    cov_end = np.zeros(mymap.shape[1])

    if not len(mids) > 1:
        logging.info('Nothing to sum up in chromosome %s' % chrom)
        return make_outmap(pad, rescale, rescale_size), 0, cov_start, cov_end

    if expected is not False:
        data = False
        expected = np.nan_to_num(expected[expected['chrom']==chrom]['balanced.avg'].values) #Always named like this by cooltools, irrespective of --weight-name
        logging.info('Doing expected')
    else:
        data = get_data(chrom, c, balance, local)

    if cov_norm and (expected is False) and (balance is False):
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

    if kind=='bed' and not two_beds:
        assert np.all(mids['chr']==chrom)
    elif kind=='bed' and two_beds:
        assert np.all(mids['chr']==chrom)
        assert np.all(mids2['chr']==chrom)
    else:
        assert kind=='bedpe'
        assert np.all(mids['chr1']==chrom) & np.all(mids['chr1']==chrom)

    if ctrl:
        if kind == 'bed':
            mids = controlRegions(get_combinations(mids=mids, res=c.binsize,
                                                   mids2=mids2,
                                                   ordered_mids=ordered_mids,
                                                   local=local,
                                                   anchor=anchor),
                                   c.binsize, minshift, maxshift, nshifts,
                                   seed)
        else:
            mids = controlRegions(get_positions_pairs(mids, c.binsize),
                                   c.binsize, minshift, maxshift, nshifts,
                                   seed)
    else:
        if kind == 'bed':
            mids = get_combinations(mids=mids, res=c.binsize,
                                    mids2=mids2,
                                    ordered_mids=ordered_mids,
                                    local=local,
                                    anchor=anchor)
        else:
            mids = get_positions_pairs(mids, c.binsize)
    mymap, n, cov_start, cov_end = _do_pileups(mids=mids, data=data, pad=pad,
                                               binsize=c.binsize,
                                               expected=expected,
                                               mindist=mindist,
                                               maxdist=maxdist,
                                               local=local,
                                               balance=balance,
                                               cov_norm=cov_norm,
                                               coverage=coverage,
                                               rescale=rescale,
                                               rescale_pad=rescale_pad,
                                               rescale_size=rescale_size,
                                               anchor=anchor)
    logging.info('%s: %s' % (chrom, n))
    return mymap, n, cov_start, cov_end

def norm_coverage(loop, cov_start, cov_end):
    coverage = np.outer(cov_start, cov_end)
    coverage /= np.nanmean(coverage)
    loop /= coverage
    loop[np.isnan(loop)]=0
    return loop

def pileupsWithControl(mids, filename, mids2=None, pad=100, nproc=1,
                       ordered_mids=True,
                       chroms=None, local=False,
                       minshift=100000, maxshift=1000000, nshifts=10,
                       expected=None,
                       mindist=0, maxdist=np.inf,
                       kind='bed', anchor=None, balance=True,
                       cov_norm=False,
                       rescale=False, rescale_pad=1, rescale_size=99,
                       seed=None):

    if mids2 is not None:
        two_beds = True
    else:
        two_beds = False
    c = cooler.Cooler(filename)
    if chroms is None:
        chroms = c.chromnames
    p = Pool(nproc)
    #Loops
    f = partial(pileups, c=c, pad=pad, ctrl=False, local=local,
                two_beds=two_beds, ordered_mids=ordered_mids,
                minshift=minshift, maxshift=maxshift, nshifts=nshifts,
                expected=False,
                mindist=mindist, maxdist=maxdist, kind=kind,
                anchor=anchor, balance=balance, cov_norm=cov_norm,
                rescale=rescale, rescale_pad=rescale_pad,
                rescale_size=rescale_size, seed=seed)
    chrommids = chrom_mids(chroms, mids, kind)
    if mids2 is not None:
        chrommids2 = chrom_mids(chroms, mids2, 'bed')
        chrommids = zip(chrommids, chrommids2)
        two_beds = True
    else:
        two_beds = False
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
        chrommids = chrom_mids(chroms, mids, kind)
        if mids2 is not None:
            chrommids2 = chrom_mids(chroms, mids2, 'bed')
            chrommids = zip(chrommids, chrommids2)
        f = partial(pileups, c=c, pad=pad, ctrl=True, local=local,
                    two_beds=two_beds, ordered_mids=ordered_mids,
                    expected=False,
                    minshift=minshift, maxshift=maxshift, nshifts=nshifts,
                    mindist=mindist, maxdist=maxdist, kind=kind,
                    anchor=anchor, balance=balance, cov_norm=cov_norm,
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
        chrommids = chrom_mids(chroms, mids, kind)
        if mids2 is not None:
            chrommids2 = chrom_mids(chroms, mids2, 'bed')
            chrommids = zip(chrommids, chrommids2)
        f = partial(pileups, c=c, pad=pad, ctrl=False, local=local,
                    two_beds=two_beds, ordered_mids=ordered_mids,
                    expected=expected,
                    minshift=minshift, maxshift=maxshift, nshifts=nshifts,
                    mindist=mindist, maxdist=maxdist, kind=kind,
                    anchor=anchor, balance=balance, cov_norm=cov_norm,
                    rescale=rescale, rescale_pad=rescale_pad,
                    rescale_size=rescale_size)
        exps, ns, cov_starts, cov_ends = list(zip(*p.map(f, chrommids)))
        exp = np.sum(exps, axis=0)
        n = np.sum(ns)
        exp /= n
        loop /= exp
    p.close()
    loop[~np.isfinite(loop)] = 0
    return loop

def pileupsByWindow(chrom_mids, c, pad=7, ctrl=False,
                    minshift=10**5, maxshift=10**6, nshifts=1,
                    expected=False,
                    mindist=0, maxdist=10**9,
                    balance=True, cov_norm=False,
                    rescale=False, rescale_pad=50, rescale_size=41,
                    seed=None):
    chrom, mids = chrom_mids

    if expected is not False:
        data = False
        expected = np.nan_to_num(expected[expected['chrom']==chrom]['balanced.avg'].values)
        logging.info('Doing expected')
    else:
        data = get_data(chrom, c, balance, local=False)

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
    for i, (b, m, p) in curmids[['Bin', 'Mids', 'Pad']].astype(int).iterrows():
        if ctrl:
            current = controlRegions(get_combinations(curmids, c.binsize,
                                                    anchor=(chrom, m, m)),
                                       c.binsize, minshift, maxshift, nshifts,
                                       seed)
        else:
             current = get_combinations(curmids, c.binsize, anchor=(chrom, m, m))
        mymap, n, cov_starts, cov_ends = _do_pileups(mids=current, data=data,
                                                     binsize=c.binsize,
                                                     pad=pad,
                                                     expected=expected,
                                                     mindist=mindist,
                                                     maxdist=maxdist,
                                                     local=False,
                                                     balance=balance,
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
        mymaps[(m-p, m+p)] = n, mymap
    return mymaps

def pileupsByWindowWithControl(mids, filename, pad=100, nproc=1, chroms=None,
                               minshift=100000, maxshift=100000, nshifts=10,
                               expected=None,
                               mindist=0, maxdist=np.inf,
                               balance=True,
                               cov_norm=False,
                               rescale=False, rescale_pad=1, rescale_size=99,
                               seed=None):
    p = Pool(nproc)
    c = cooler.Cooler(filename)
    if chroms is None:
        chroms = c.chromnames
    #Loops
    f = partial(pileupsByWindow, c=c, pad=pad, ctrl=False,
                minshift=minshift, maxshift=maxshift, nshifts=nshifts,
                expected=False,
                mindist=mindist, maxdist=maxdist, balance=balance,
                cov_norm=False,
                rescale=rescale, rescale_pad=rescale_pad,
                rescale_size=rescale_size,
                seed=seed)
    chrommids = chrom_mids(chroms, mids, 'bed')
    loops = {chrom:lps for chrom, lps in zip(chroms,
                                             p.map(f, chrommids))}
    #Controls
    if nshifts>0:
        f = partial(pileupsByWindow, c=c, pad=pad, ctrl=True,
                    minshift=minshift, maxshift=maxshift, nshifts=nshifts,
                    expected=expected,
                    mindist=mindist, maxdist=maxdist, balance=balance,
                    cov_norm=cov_norm,
                    rescale=rescale, rescale_pad=rescale_pad,
                    rescale_size=rescale_size,
                    seed=seed)
        chrommids = chrom_mids(chroms, mids, 'bed')
        ctrls = {chrom:lps for chrom, lps in zip(chroms,
                                             p.map(f, chrommids))}
    elif expected is not False:
        f = partial(pileupsByWindow, c=c, pad=pad, ctrl=False,
            expected=expected,
            minshift=minshift, maxshift=maxshift, nshifts=nshifts,
            mindist=mindist, maxdist=maxdist,
            balance=balance, cov_norm=False,
            rescale=rescale, rescale_pad=rescale_pad,
            rescale_size=rescale_size,
            seed=seed)
        chrommids = chrom_mids(chroms, mids, 'bed')
        ctrls = {chrom:lps for chrom, lps in zip(chroms,
                                             p.map(f, chrommids))}
    p.close()

    finloops = {}
    for chrom in loops.keys():
        for pos, lp in loops[chrom].items():
            loop = lp[1]/ctrls[chrom][pos][1]
            loop[~np.isfinite(loop)] = 0
            finloops[(chrom, pos[0], pos[1])] = lp[0], loop
    return finloops
