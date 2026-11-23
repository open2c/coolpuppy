# -*- coding: utf-8 -*-
from .lib.io import load_pileup_df, save_pileup_df
from .lib.puputils import divide_pups

from ._version import __version__
import argparse
import logging
import os

logger = logging.getLogger("coolpuppy")


def parse_args_divide_pups():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "input_pups", type=str, nargs="+", help="""Two pileups to divide"""
    )
    parser.add_argument("-v", "--version", action="version", version=__version__)

    # Output
    parser.add_argument(
        "-o",
        "--outname",
        default="auto",
        type=str,
        required=False,
        help="""Name of the output file.
                If not set, file is saved in the current directory and the name is
                generated automatically.""",
    )
    return parser


def main():
    parser = parse_args_divide_pups()
    args = parser.parse_args()

    logging.basicConfig(format="%(levelname)s:%(name)s: %(message)s")
    logger.setLevel(logging.INFO)
    logger.debug(args)

    if len(args.input_pups) != 2:
        raise ValueError("Need exactly two input pups")
    else:
        pup1 = load_pileup_df(args.input_pups[0])
        pup2 = load_pileup_df(args.input_pups[1])
        pups = divide_pups(pup1, pup2)

    if args.outname == "auto":
        # Saved in the current directory, named after the two inputs
        name1, name2 = (
            os.path.splitext(os.path.basename(p))[0] for p in args.input_pups
        )
        outname = f"{name1}_over_{name2}.clpy"
    else:
        outname = args.outname

    save_pileup_df(outname, pups)
    logger.info(f"Saved output to {outname}")
