from .coolpup import *
import cooler
import pandas as pd
import os
from natsort import index_natsorted, order_by_index
import argparse
import logging
import numpy as np
from multiprocessing import Pool

def main():
    parser = argparse.ArgumentParser(
                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("coolfile", type=str,
                        help="Cooler file with your Hi-C data")
    parser.add_argument("baselist", type=str,
                        help="""A 3-column bed file or a 6-column double-bed
                        file (i.e. chr1,start1,end1,chr2,start2,end2).
                        Should be tab-delimited.
                        With a bed file, will consider all cis combinations
                        of intervals. To pileup features along the diagonal
                        instead, use the --local argument.
                        Can be piped in via stdin, then use "-".""")
##### Extra arguments
    parser.add_argument("--pad", default=100, type=int, required=False,
                        help="""Padding of the windows around the centres of
                        specified features (i.e. final size of the matrix is
                        2×pad+res), in kb.
                        Ignored with --rescale, use --rescale_pad instead.""")
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
                        final size of the pileup, i.e. it will be size×size. Due
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

    if args.outdir=='.':
        args.outdir = os.getcwd()

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

if __name__ == "__main__":
    main()
