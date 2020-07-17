# -*- coding: utf-8 -*-
from coolpuppy import *
from coolpuppy import __version__
import cooler
import pandas as pd
import os
from natsort import index_natsorted, order_by_index, natsorted
import argparse
import logging
import numpy as np
from multiprocessing import Pool
import sys
import pdb, traceback

# from ._version.py import __version__


def parse_args_coolpuppy():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("coolfile", type=str, help="Cooler file with your Hi-C data")
    parser.add_argument(
        "baselist",
        type=str,
        help="""A 3-column bed file or a 6-column double-bed file
                i.e. chr1,start1,end1,chr2,start2,end2.
                Should be tab-delimited.

                With a bed file, will consider all cis combinations
                of intervals. To pileup features along the diagonal
                instead, use the ``--local`` argument.

                Can be piped in via stdin, then use "-"
                """,
    )
    ##### Extra arguments
    parser.add_argument(
        "--bed2",
        type=str,
        help="""A 3-column bed file.
                Should be tab-delimited.
                Will consider all cis combinations of intervals
                between baselist and bed2""",
        required=False,
    )
    parser.add_argument(
        "--bed2_unordered",
        action="store_false",
        dest="bed2_ordered",
        help="""Whether to use baselist as left ends, and bed2 as right ends of regions
             """,
        required=False,
    )
    # parser.set_defaults(bed2_ordered=True)
    parser.add_argument(
        "--pad",
        default=100,
        type=int,
        required=False,
        help="""Padding of the windows around the centres of specified features
                i.e. final size of the matrix is 2×pad+res, in kb.
                Ignored with ``--rescale``, use ``--rescale_pad`` instead""",
    )
    ### Control of controls
    parser.add_argument(
        "--minshift",
        default=10 ** 5,
        type=int,
        required=False,
        help="""Shortest shift for random controls, bp
             """,
    )
    parser.add_argument(
        "--maxshift",
        default=10 ** 6,
        type=int,
        required=False,
        help="""Longest shift for random controls, bp
             """,
    )
    parser.add_argument(
        "--nshifts",
        default=10,
        type=int,
        required=False,
        help="""Number of control regions per averaged window
             """,
    )
    parser.add_argument(
        "--expected",
        default=None,
        type=str,
        required=False,
        help="""File with expected (output of ``cooltools compute-expected``).
                If None, don't use expected and use randomly shifted controls""",
    )
    # Filtering
    parser.add_argument(
        "--mindist",
        type=int,
        required=False,
        help="""Minimal distance of interactions to use, bp.
                If not specified, uses 2*pad+2 (in bins) as mindist""",
    )
    parser.add_argument(
        "--maxdist",
        type=int,
        required=False,
        help="""Maximal distance of interactions to use""",
    )
    parser.add_argument(
        "--ignore_diags",
        type=int,
        default=2,
        required=False,
        help="""How many diagonals to ignore""",
    )
    parser.add_argument(
        "--minsize",
        type=int,
        required=False,
        help="""Minimal length of features to use for local analysis""",
    )
    parser.add_argument(
        "--maxsize",
        type=int,
        required=False,
        help="""Maximal length of features to use for local analysis""",
    )
    parser.add_argument(
        "--excl_chrs",
        default="chrY,chrM",
        type=str,
        required=False,
        help="""Exclude these chromosomes from analysis""",
    )
    parser.add_argument(
        "--incl_chrs",
        default="all",
        type=str,
        required=False,
        help="""Include these chromosomes; default is all.
                ``--excl_chrs`` overrides this""",
    )
    parser.add_argument(
        "--subset",
        default=0,
        type=int,
        required=False,
        help="""Take a random sample of the bed file.
                Useful for files with too many featuers to run as is, i.e.
                some repetitive elements. Set to 0 or lower to keep all data""",
    )
    # Modes of action
    parser.add_argument(
        "--anchor",
        default=None,
        type=str,
        required=False,
        help="""A UCSC-style coordinate.
                Use as an anchor to create intersections with coordinates in the
                baselist""",
    )
    parser.add_argument(
        "--by_window",
        action="store_true",
        default=False,
        required=False,
        help="""Perform by-window pile-ups.
                Create a pile-up for each coordinate in the baselist.
                Will save a master-table with coordinates, their enrichments and
                corner coefficient of variation, which is reflective of noisiness""",
    )
    parser.add_argument(
        "--save_all",
        action="store_true",
        default=False,
        required=False,
        help="""If ``--by-window``, save all individual pile-ups in a separate json file
             """,
    )
    parser.add_argument(
        "--local",
        action="store_true",
        default=False,
        required=False,
        help="""Create local pileups, i.e. along the diagonal""",
    )
    parser.add_argument(
        "--unbalanced",
        action="store_true",
        required=False,
        help="""Do not use balanced data.
                Useful for single-cell Hi-C data together with ``--coverage_norm``,
                not recommended otherwise""",
    )
    parser.add_argument(
        "--coverage_norm",
        action="store_true",
        required=False,
        help="""
        If ``--unbalanced``, add coverage normalization using chromosome marginals""",
    )
    # Rescaling
    parser.add_argument(
        "--rescale",
        action="store_true",
        default=False,
        required=False,
        help="""Rescale all features to the same size.
                Do not use centres of features and pad, and rather use the actual
                feature sizes and rescale pileups to the same shape and size""",
    )
    parser.add_argument(
        "--rescale_pad",
        default=1.0,
        required=False,
        type=float,
        help="""If --rescale, padding in fraction of feature length""",
    )
    parser.add_argument(
        "--rescale_size",
        type=int,
        default=99,
        required=False,
        help="""Size to rescale to.
                If ``--rescale``, used to determine the final size of the pileup,
                i.e. it will be size×size. Due to technical limitation in the current
                implementation, has to be an odd number""",
    )
    parser.add_argument(
        "--weight_name",
        default="weight",
        type=str,
        required=False,
        help="""Name of the norm to use for getting balanced data""",
    )
    parser.add_argument(
        "--n_proc",
        default=1,
        type=int,
        required=False,
        help="""Number of processes to use.
                Each process works on a separate chromosome, so might require quite a
                bit more memory, although the data are always stored as sparse matrices
                """,
    )
    # Output
    parser.add_argument(
        "--outdir",
        default=".",
        type=str,
        required=False,
        help="""Directory to save the data in""",
    )
    parser.add_argument(
        "--outname",
        default="auto",
        type=str,
        required=False,
        help="""Name of the output file.
                If not set, it is generated automatically to include important
                information""",
    )
    # Technicalities
    parser.add_argument(
        "--seed",
        default=None,
        type=int,
        required=False,
        help="""Set specific seed value to ensure reproducibility""",
    )
    parser.add_argument(
        "-l",
        "--log",
        dest="logLevel",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level",
    )
    parser.add_argument(
        "--post_mortem",
        action="store_true",
        default=False,
        required=False,
        help="""Enter debugger if there is an error""",
    )
    parser.add_argument("-v", "--version", action="version", version=__version__)
    return parser


def main():
    parser = parse_args_coolpuppy()
    args = parser.parse_args()

    if args.post_mortem:

        def _excepthook(exc_type, value, tb):
            traceback.print_exception(exc_type, value, tb)
            print()
            pdb.pm()

        sys.excepthook = _excepthook

    logging.basicConfig(format="%(message)s", level=getattr(logging, args.logLevel))

    logging.info(args)

    if args.seed is not None:
        np.random.seed(args.seed)

    if args.n_proc == 0:
        nproc = -1
    else:
        nproc = args.n_proc

    c = cooler.Cooler(args.coolfile)

    if not os.path.isfile(args.baselist) and args.baselist != "-":
        raise FileExistsError("Loop(base) coordinate file doesn't exist")

    if args.unbalanced:
        balance = False
    else:
        balance = args.weight_name

    coolname = os.path.splitext(os.path.basename(c.filename))[0]
    if args.baselist != "-":
        bedname = os.path.splitext(os.path.basename(args.baselist))[0]
        baselist = args.baselist
    else:
        bedname = "stdin"
        baselist = sys.stdin
        args.baselist = 'stdin'
    if args.bed2 is not None:
        bedname += "_vs_" + os.path.splitext(os.path.basename(args.bed2))[0]

    if args.nshifts > 0:
        control = True
    else:
        control = False

    if args.expected is not None:
        if args.nshifts > 0:
            logging.warning("With specified expected will not use controls")
            control = False
        if not os.path.isfile(args.expected):
            raise FileExistsError("Expected file doesn't exist")
        expected = pd.read_csv(args.expected, sep="\t", header=0, dtype={'region':str,
                                                                         'chrom':str})
    else:
        expected = False
    if args.mindist is None:
        mindist = "auto"
    else:
        mindist = args.mindist

    if args.maxdist is None:
        maxdist = np.inf
    else:
        maxdist = args.maxdist

    if args.minsize is None:
        minsize = 0
    else:
        minsize = args.minsize

    if args.maxsize is None:
        maxsize = np.inf
    else:
        maxsize = args.maxsize

    if args.incl_chrs == "all":
        incl_chrs = np.array(c.chromnames).astype(str)
    else:
        incl_chrs = args.incl_chrs.split(",")

    if args.by_window and args.rescale:
        raise NotImplementedError(
            """Rescaling with by-window pileups is not
                                  supported"""
        )

    if args.rescale and args.rescale_size % 2 == 0:
        raise ValueError("Please provide an odd rescale_size")

    if args.anchor is not None:
        if "_" in args.anchor:
            anchor, anchor_name = args.anchor.split("_")
            anchor = cooler.util.parse_region_string(anchor)
        else:
            anchor = cooler.util.parse_region_string(args.anchor)
            anchor_name = args.anchor
    else:
        anchor = None

    if anchor:
        fchroms = [anchor[0]]
    else:
        chroms = np.array(c.chromnames).astype(str)
        fchroms = []
        for chrom in chroms:
            if chrom not in args.excl_chrs.split(",") and chrom in incl_chrs:
                fchroms.append(chrom)
    if args.anchor is not None:
        anchor = cooler.util.parse_region_string(args.anchor)

    CC = CoordCreator(
        baselist=baselist,
        resolution=c.binsize,
        bed2=args.bed2,
        bed2_ordered=args.bed2_ordered,
        anchor=anchor,
        pad=args.pad * 1000,
        chroms=fchroms,
        minshift=args.minshift,
        maxshift=args.maxshift,
        nshifts=args.nshifts,
        minsize=minsize,
        maxsize=maxsize,
        mindist=mindist,
        maxdist=maxdist,
        local=args.local,
        subset=args.subset,
        seed=args.seed,
    )
    CC.process()

    PU = PileUpper(
        clr=c,
        CC=CC,
        balance=balance,
        expected=expected,
        control=control,
        coverage_norm=args.coverage_norm,
        rescale=args.rescale,
        rescale_pad=args.rescale_pad,
        rescale_size=args.rescale_size,
        ignore_diags=args.ignore_diags,
    )

    if args.outdir == ".":
        args.outdir = os.getcwd()

    if args.outname == "auto":
        outname = f"{coolname}-{c.binsize / 1000}K_over_{bedname}"
        if args.nshifts > 0 and args.expected is None:
            outname += f"_{args.nshifts}-shifts"
        if args.expected is not None:
            outname += "_expected"
        if args.nshifts <= 0 and args.expected is None:
            outname += "_noNorm"
        if anchor:
            outname += f"_from_{anchor_name}"
        if args.local:
            outname += "_local"
            if minsize > 0 or maxsize < np.inf:
                outname += f"_len_{minsize}-{maxsize}"
        elif args.mindist is not None or args.maxdist is not None:
            outname += f"_dist_{mindist}-{maxdist}"
        if args.rescale:
            outname += "_rescaled"
        if args.unbalanced:
            outname += "_unbalanced"
        if args.coverage_norm:
            outname += "_covnorm"
        if args.subset > 0:
            outname += f"_subset-{args.subset}"
        if args.by_window:
            outname = f"Enrichment_{outname}.txt"
        else:
            outname += ".np.txt"
    else:
        outname = args.outname

    if args.by_window:
        if CC.kind != "bed":
            raise ValueError("Can't make by-window pileups without making combinations")
        if args.local:
            raise ValueError("Can't make local by-window pileups")
        if anchor:
            raise ValueError("Can't make by-window combinations with an anchor")
        #        if args.coverage_norm:
        #            raise NotImplementedError("""Can't make by-window combinations with
        #                                      coverage normalization - please use
        #                                      balanced data instead""")
        finloops = PU.pileupsByWindowWithControl(nproc=nproc)

        p = Pool(nproc)
        data = p.map(prepare_single, finloops.items())
        p.close()
        data = pd.DataFrame(
            data,
            columns=[
                "chr",
                "start",
                "end",
                "N",
                "Enrichment1",
                "Enrichment3",
                "CV3",
                "CV5",
            ],
        )
        data = data.reindex(
            index=order_by_index(
                data.index, index_natsorted(zip(data["chr"], data["start"]))
            )
        )
        try:
            data.to_csv(os.path.join(args.outdir, outname), sep="\t", index=False)
        except FileNotFoundError:
            os.mkdir(args.outdir)
            data.to_csv(os.path.join(args.outdir, outname), sep="\t", index=False)
        finally:
            logging.info(
                f"Saved enrichment table to {os.path.join(args.outdir, outname)}"
            )

        if args.save_all:
            outdict = {
                "%s:%s-%s" % key: (val[0], val[1].tolist())
                for key, val in finloops.items()
            }
            import json

            json_path = (
                os.path.join(args.outdir, os.path.splitext(outname)[0]) + ".json"
            )
            with open(json_path, "w") as fp:
                json.dump(outdict, fp)  # , sort_keys=True, indent=4)
                logging.info(f"Saved individual pileups to {json_path}")
    else:
        pup, n = PU.pileupsWithControl(nproc)
        headerdict = vars(args)
        headerdict['resolution'] = int(c.binsize)
        headerdict['n'] = int(n)
        try:
            save_array_with_header(pup, headerdict, os.path.join(args.outdir, outname))
        except FileNotFoundError:
            try:
                os.mkdir(args.outdir)
            except FileExistsError:
                pass
            save_array_with_header(pup, headerdict, os.path.join(args.outdir, outname))
        finally:
            logging.info(f"Saved output to {os.path.join(args.outdir, outname)}")
