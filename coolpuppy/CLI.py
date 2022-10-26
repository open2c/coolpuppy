# -*- coding: utf-8 -*-
from .coolpup import pileup
from .lib.io import save_pileup_df, sniff_for_header
from .lib.util import validate_csv

# from coolpuppy import *
from coolpuppy._version import __version__
from cooltools.lib import common, io
import cooler
import numpy as np
import pandas as pd
import bioframe
import os
import argparse
import logging

import sys
import pdb, traceback


def parse_args_coolpuppy():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("cool_path", type=str, help="Cooler file with your Hi-C data")
    parser.add_argument(
        "features",
        type=str,
        help="""A 3-column bed file or a 6-column double-bed file
                i.e. chr1,start1,end1,chr2,start2,end2.
                Should be tab-delimited.

                With a bed file, will consider all combinations
                of intervals. To pileup features along the diagonal
                instead, use the ``--local`` argument.

                Can be piped in via stdin, then use "-"
                """,
    )
    ##### Extra arguments
    parser.add_argument(
        "--features_format",
        "--features-format",
        "--format",
        "--basetype",
        type=str,
        choices=["bed", "bedpe", "auto"],
        help="""Format of the features.
                Options:
                bed: chrom, start, end
                bedpe: chrom1, start1, end1, chrom2, start2, end2
                auto (default): determined from the file name extension
                Has to be explicitly provided is features is piped through stdin""",
        default="auto",
        required=False,
    )
    parser.add_argument(
        "--view",
        type=str,
        help="""Path to a file which defines which regions of the chromosomes to use""",
        default=None,
        required=False,
    )
    parser.add_argument(
        "--flank",
        "--pad",
        default=100_000,
        type=int,
        required=False,
        help="""Flanking of the windows around the centres of specified features
                i.e. final size of the matrix is 2 × flank+res, in bp.
                Ignored with ``--rescale``, use ``--rescale_flank`` instead""",
    )
    ### Control of controls
    parser.add_argument(
        "--minshift",
        default=10**5,
        type=int,
        required=False,
        help="""Shortest shift for random controls, bp
             """,
    )
    parser.add_argument(
        "--maxshift",
        default=10**6,
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
        type=validate_csv,
        required=False,
        help="""File with expected (output of ``cooltools compute-expected``).
                If None, don't use expected and use randomly shifted controls""",
    )
    parser.add_argument(
        "--not_ooe",
        "--not-ooe",
        dest="ooe",
        default=True,
        action="store_false",
        help="""If expected is provided, will accumulate all expected snippets just like for randomly shifted controls, instead of normalizing each snippet individually""",
    )
    # Filtering
    parser.add_argument(
        "--mindist",
        type=int,
        required=False,
        help="""Minimal distance of interactions to use, bp.
                If not provided, uses 2*flank+2 (in bins) as mindist to avoid first two
                diagonals""",
    )
    parser.add_argument(
        "--maxdist",
        type=int,
        required=False,
        help="""Maximal distance of interactions to use""",
    )
    parser.add_argument(
        "--ignore_diags",
        "--ignore-diags",
        type=int,
        default=2,
        required=False,
        help="""How many diagonals to ignore""",
    )
    parser.add_argument(
        "--subset",
        default=0,
        type=int,
        required=False,
        help="""Take a random sample of the bed file.
                Useful for files with too many featuers to run as is, i.e. some repetitive elements. Set to 0 or lower to keep all data""",
    )
    # Modes of action
    parser.add_argument(
        "--by_window",
        "--by-window",
        action="store_true",
        default=False,
        required=False,
        help="""Perform by-window pile-ups.
                Create a pile-up for each coordinate in the features.
                Not compatible with --by_strand and --by_distance.

                Only works with bed format features, and generates pairwise
                combinations of each feature against the rest.""",
    )
    parser.add_argument(
        "--by_strand",
        "--by-strand",
        action="store_true",
        default=False,
        required=False,
        help="""Perform by-strand pile-ups.
                Create a separate pile-up for each strand combination in the features.""",
    )
    parser.add_argument(
        "--by_distance",
        "--by-distance",
        nargs="*",
        required=False,
        help="""Perform by-distance pile-ups.
                Create a separate pile-up for each distance band. If empty, will use default 
                (0,50000,100000,200000,...) edges. Specify edges using multiple argument
                values, e.g. `--by_distance 1000000 2000000` """,
    )
    parser.add_argument(
        "--groupby",
        nargs="*",
        required=False,
        help="""Additional columns of features to use for groupby, space separated. 
                If feature_format=='bed', each columns should be specified twice with suffixes 
                '1' and '2', i.e. if features have a column 'group', specify 'group1 group2'.,
                e.g. --groupby chrom1 chrom2""",
    )
    parser.add_argument(
        "--flip_negative_strand",
        "--flip-negative-strand",
        action="store_true",
        default=False,
        required=False,
        help="""Flip snippets so the positive strand always points to bottom-right.
                Requires strands to be annotated for each feature (or two strands for
                bedpe format features)""",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        default=False,
        required=False,
        help="""Create local pileups, i.e. along the diagonal""",
    )
    parser.add_argument(
        "--coverage_norm",
        "--coverage-norm",
        default="",
        type=str,
        required=False,
        nargs="?",
        const="total",
        help="""
            Normalize the final pileup by accumulated coverage as an alternative to balancing.
            Useful for single-cell Hi-C data. Can be a string: "cis" or "total" to use 
            "cov_cis_raw" or "cov_tot_raw" columns in the cooler bin table, respectively.
            If they are not present, will calculate coverage with same ignore_diags as
            used in coolpup.py and store result in the cooler.
            Alternatively, if a different string is provided, will attempt to use a
            column with the that name in the cooler bin table, and will raise a
            ValueError if it does not exist.
            If no argument is given following the option string, will use "total".
            Only allowed when using empty --clr_weight_name""",
    )
    parser.add_argument(
        "--trans",
        action="store_true",
        default=False,
        required=False,
        help="""Perform inter-chromosomal (trans) pileups. 
                This ignores all contacts in cis.""",
    )
    parser.add_argument(
        "--store_stripes",
        action="store_true",
        default=False,
        required=False,
        help="""Store horizontal and vertical stripes in pileup output""",
    )

    # Rescaling
    parser.add_argument(
        "--rescale",
        action="store_true",
        default=False,
        required=False,
        help="""Rescale all features to the same size.
                Do not use centres of features and flank, and rather use the actual
                feature sizes and rescale pileups to the same shape and size""",
    )
    parser.add_argument(
        "--rescale_flank",
        "--rescale_pad",
        "--rescale-flank",
        "--rescale-pad",
        default=1.0,
        required=False,
        type=float,
        help="""If --rescale, flanking in fraction of feature length""",
    )
    parser.add_argument(
        "--rescale_size",
        "--rescale-size",
        type=int,
        default=99,
        required=False,
        help="""Size to rescale to.
                If ``--rescale``, used to determine the final size of the pileup,
                i.e. it will be size×size. Due to technical limitation in the current
                implementation, has to be an odd number""",
    )
    # Balancing
    parser.add_argument(
        "--clr_weight_name",
        "--weight_name",
        "--clr-weight-name",
        "--weight-name",
        default="weight",
        type=str,
        required=False,
        nargs="?",
        const=None,
        help="""Name of the norm to use for getting balanced data.
                Provide empty argument to calculate pileups on raw data
                (no masking bad pixels).""",
    )
    # Output
    parser.add_argument(
        "-o",
        "--outname",
        "--output",
        default="auto",
        type=str,
        required=False,
        help="""Name of the output file.
                If not set, file is saved in the current directory and the name is
                generated automatically to include important information and avoid
                overwriting files generated with different settings.""",
    )
    # Technicalities
    parser.add_argument(
        "-p",
        "--nproc",
        "--n_proc",
        "--n-proc",
        default=1,
        type=int,
        required=False,
        dest="n_proc",
        help="""Number of processes to use.
                Each process works on a separate chromosome, so might require quite a
                bit more memory, although the data are always stored as sparse matrices.
                Set to 0 to use all available cores.
                """,
    )
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
        "--post-mortem",
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
    if args.by_distance is not None:
        if len(args.by_distance) > 0:
            try:
                distance_edges = [int(item) for item in args.by_distance]
            except:
                raise ValueError(
                    "Distance edges must be integers. Separate edges with spaces."
                )
        else:
            distance_edges = "default"
            args.by_distance = True
    else:
        args.by_distance = False
        distance_edges = False

    logger = logging.getLogger("coolpuppy")
    logger.setLevel(getattr(logging, args.logLevel))

    logger.debug(args)

    if args.seed is not None:
        np.random.seed(args.seed)

    if args.n_proc == 0:
        nproc = -1
    else:
        nproc = args.n_proc

    clr = cooler.Cooler(args.cool_path)

    coolname = os.path.basename(clr.filename)
    if args.features != "-":
        bedname, ext = os.path.splitext(os.path.basename(args.features))
        features = args.features
        buf, names, ncols = sniff_for_header(features)
        if args.features_format == "auto":
            schema = ext[1:]
        else:
            schema = args.features_format
        if schema == "bed":
            schema = "bed12"
            features_format = "bed"
            dtypes = {"chrom": str,
                      "start": np.int64,
                      "end": np.int64,}
        else:
            features_format = "bedpe"
            dtypes = {
                "chrom1": str,
                "start1": np.int64,
                "end1": np.int64,
                "chrom2": str,
                "start2": np.int64,
                "end2": np.int64,
            }
        if (features_format == "bedpe") & (ncols < 6):
            raise ValueError("Too few columns")
        elif ncols < 3:
            raise ValueError("Too few columns")
        if names is not None:
            features = pd.read_table(buf, dtype=dtypes)
        else:
            features = bioframe.read_table(
                features, schema=schema, index_col=False, dtype=dtypes
            )
    else:
        if args.features_format == "auto":
            raise ValueError(
                "Can't determine format when features is piped in, please specify"
            )
        schema = args.features_format
        if schema == "bed":
            schema = "bed12"
            features_format = "bed"
            dtypes = {"chrom": str,
                      "start": np.int64,
                      "end": np.int64,}
        else:
            features_format = "bedpe"
            dtypes = {
                "chrom1": str,
                "start1": np.int64,
                "end1": np.int64,
                "chrom2": str,
                "start2": np.int64,
                "end2": np.int64,
            }
        bedname = "stdin"
        buf, names, ncols = sniff_for_header(sys.stdin)
        if (features_format == "bedpe") & (ncols < 6):
            raise ValueError("Too few columns")
        elif ncols < 3:
            raise ValueError("Too few columns")
        if names is not None:
            features = pd.read_table(buf, dtype=dtypes)
        else:
            features = bioframe.read_table(buf, schema=schema, index_col=False, dtype=dtypes)
        

    if args.view is None:
        # full chromosome case
        view_df = common.make_cooler_view(clr)
    else:
        # Read view_df dataframe, and verify against cooler
        view_df = io.read_viewframe_from_file(args.view, clr, check_sorting=True)

    if args.expected is None:
        expected = None
        expected_value_col = None
    else:
        expected_path, expected_value_col = args.expected
        expected_value_cols = [
            expected_value_col,
        ]
        if args.trans:
            expected = io.read_expected_from_file(
                expected_path,
                contact_type="trans",
                expected_value_cols=expected_value_cols,
                verify_view=view_df,
                verify_cooler=clr,
            )
        else:
            expected = io.read_expected_from_file(
                expected_path,
                contact_type="cis",
                expected_value_cols=expected_value_cols,
                verify_view=view_df,
                verify_cooler=clr,
            )
        args.nshifts = 0

    if args.mindist is None:
        mindist = "auto"
    else:
        mindist = args.mindist

    if args.maxdist is None:
        maxdist = np.inf
    else:
        maxdist = args.maxdist

    if args.rescale and args.rescale_size % 2 == 0:
        raise ValueError("Please provide an odd rescale_size")
    if not args.rescale:
        rescale_flank = None
    else:
        rescale_flank = args.rescale_flank

    if args.by_window:
        if schema != "bed12":
            raise ValueError("Can't make by-window pileups without making combinations")
        if args.local:
            raise ValueError("Can't make local by-window pileups")

    pups = pileup(
        clr=clr,
        features=features,
        features_format=features_format,
        view_df=view_df,
        expected_df=expected,
        expected_value_col=expected_value_col,
        clr_weight_name=args.clr_weight_name,
        flank=args.flank,
        minshift=args.minshift,
        maxshift=args.maxshift,
        nshifts=args.nshifts,
        ooe=args.ooe,
        mindist=mindist,
        maxdist=maxdist,
        min_diag=args.ignore_diags,
        subset=args.subset,
        by_window=args.by_window,
        by_strand=args.by_strand,
        by_distance=distance_edges,
        groupby=[] if args.groupby is None else args.groupby,
        flip_negative_strand=args.flip_negative_strand,
        local=args.local,
        coverage_norm=args.coverage_norm,
        trans=args.trans,
        rescale=args.rescale,
        rescale_flank=rescale_flank,
        rescale_size=args.rescale_size,
        store_stripes=args.store_stripes,
        nproc=nproc,
        seed=args.seed,
    )

    if args.outname == "auto":
        outname = f"{coolname}-{clr.binsize / 1000}K_over_{bedname}"
        if args.nshifts > 0 and args.expected is None:
            outname += f"_{args.nshifts}-shifts"
        if args.expected is not None:
            outname += "_expected"
        if args.nshifts <= 0 and args.expected is None:
            outname += "_noNorm"
        if args.local:
            outname += "_local"
        elif args.mindist is not None or args.maxdist is not None:
            outname += f"_dist_{mindist}-{maxdist}"
        if args.rescale:
            outname += "_rescaled"
        if args.coverage_norm:
            outname += "_covnorm"
        if args.subset > 0:
            outname += f"_subset-{args.subset}"
        if args.by_window:
            outname += "_by-window"
        if args.by_strand:
            outname += "_by-strand"
        if args.trans:
            outname += "_trans"
        if args.groupby:
            outname += f"_by-{'_'.join(args.groupby)}"
        outname += ".clpy"
    else:
        outname = args.outname

    if args.expected:
        pups["expected_file"] = expected_path
    if args.view:
        pups["view_file"] = args.view
    pups["features"] = args.features
    save_pileup_df(outname, pups)
    logger.info(f"Saved output to {outname}")
