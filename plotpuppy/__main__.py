#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 14:05:06 2020

@author: Ilya Flyamer
"""
from coolpuppy.coolpup import (
    norm_cis,
    load_pileup_df,
    load_pileup_df_list,
    get_score,
)
from plotpuppy.plotpup import make_heatmap_grid
from coolpuppy._version import __version__

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib as mpl
import re
import argparse

from natsort import natsorted

import sys
import pdb, traceback


def sort_separation(sep_string_series, sep="Mb"):
    return sorted(set(sep_string_series.dropna()), key=lambda x: float(x.split(sep)[0]))


def parse_args_plotpuppy():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--cmap",
        type=str,
        required=False,
        default="coolwarm",
        help="""Colourmap to use
                (see https://matplotlib.org/users/colormaps.html)""",
    )
    parser.add_argument(
        "--symmetric",
        type=bool,
        required=False,
        default=True,
        help="""Whether to make colormap symmetric around 1, if log scale""",
    )
    parser.add_argument(
        "--vmin", type=float, required=False, help="""Value for the lowest colour"""
    )
    parser.add_argument(
        "--vmax", type=float, required=False, help="""Value for the highest colour"""
    )
    parser.add_argument(
        "--scale",
        type=str,
        default="log",
        required=False,
        choices={"linear", "log"},
        help="""Whether to use linear or log scaling for mapping colours""",
    )
    # parser.add_argument(
    #     "--cbar_mode",
    #     type=str,
    #     default="single",
    #     required=False,
    #     choices={"single", "edge", "each"},
    #     help="""Whether to show a single colorbar, one per row or one for each subplot
    #          """,
    # )
    # parser.add_argument(
    #     "--n_cols",
    #     type=int,
    #     default=0,
    #     required=False,
    #     help="""How many columns to use for plotting the data.
    #             If 0, automatically make the figure as square as possible""",
    # )
    parser.add_argument(
        "--cols",
        type=str,
        required=False,
        help="""Which value to map as columns""",
    )
    parser.add_argument(
        "--rows",
        type=str,
        required=False,
        help="""Which value to map as rows""",
    )
    parser.add_argument(
        "--col_order",
        type=lambda s: re.split(" |, ", s),
        required=False,
        help="""Order of columns to use, comma separated""",
    )
    parser.add_argument(
        "--row_order",
        type=lambda s: re.split(" |, ", s),
        required=False,
        help="""Order of rows to use, comma separated""",
    )
    parser.add_argument(
        "--query",
        type=str,
        required=False,
        action="append",
        help=""""Pandas query top select pups to plot from concatenated input files""",
    )
    parser.add_argument(
        "--norm_corners",
        type=int,
        required=False,
        default=0,
        help="""Whether to normalize pileups by their top left and bottom right corners.
                0 for no normalization, positive number to define the size of the corner
                squares whose values are averaged""",
    )
    parser.add_argument(
        "--score",
        type=bool,
        required=False,
        default=True,
        help="""Whether to calculate score and add it to the top right corner of each
                pileup. Will use the 'coolpup.get_score' function with 'center' and
                'ignore_central' arguments.""",
    )
    parser.add_argument(
        "--center",
        type=int,
        required=False,
        default=3,
        help="""How many central pixels to consider when calculating enrichment for
                off-diagonal pileups.""",
    )
    parser.add_argument(
        "--ignore_central",
        type=int,
        required=False,
        default=3,
        help="""ow many central bins to ignore when calculating insulation for
                local (on-diagonal) non-rescaled pileups.""",
    )
    parser.add_argument(
        "--quaich",
        required=False,
        default=False,
        action="store_true",
        help="""Activate if pileups are named accodring to Quaich naming convention
                to get information from the file name""",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        required=False,
        default=300,
        help="""DPI of the output plot. Try increasing if heatmaps look blurry""",
    )
    #    parser.add_argument("--n_rows", type=int, default=0,
    #                    required=False,
    #                    help="""How many rows to use for plotting the data""")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=False,
        default="pup.pdf",
        help="""Where to save the plot""",
    )
    parser.add_argument(
        "--post_mortem",
        action="store_true",
        default=False,
        required=False,
        help="""Enter debugger if there is an error""",
    )
    parser.add_argument(
        "--input_pups", type=str, nargs="+", help="""All files to plot"""
    )
    parser.add_argument("-v", "--version", action="version", version=__version__)
    return parser


def main():
    parser = parse_args_plotpuppy()
    args = parser.parse_args()
    if args.post_mortem:

        def _excepthook(exc_type, value, tb):
            traceback.print_exception(exc_type, value, tb)
            print()
            pdb.pm()

        sys.excepthook = _excepthook
    mpl.rcParams["svg.fonttype"] = "none"
    mpl.rcParams["pdf.fonttype"] = 42

    pups = load_pileup_df_list(args.input_pups, quaich=args.quaich, nice_metadata=True)
    if args.query is not None:
        for q in args.query:
            pups = pups.query(q)

    if args.norm_corners > 0:
        pups["data"] = pups["data"].apply(norm_cis, i=int(args.norm_corners))

    if args.score:
        pups["score"] = pups.apply(
            get_score, center=args.center, ignore_central=args.ignore_central, axis=1
        )
        score = "score"
    else:
        score = False

    if args.cols:
        if args.col_order:
            col_order = args.col_order
        elif args.cols == "separation":
            col_order = sort_separation(pups["separation"])
        else:
            col_order = natsorted(pups[args.cols].unique())
    else:
        col_order = None

    if args.rows:
        if args.row_order:
            row_order = args.row_order
        elif args.rows == "separation":
            row_order = sort_separation(pups["separation"])
        else:
            row_order = natsorted(pups[args.rows].unique())
    else:
        row_order = None

    fg = make_heatmap_grid(
        pups,
        cols=args.cols,
        rows=args.rows,
        score=score,
        col_order=col_order,
        row_order=row_order,
        vmin=args.vmin,
        vmax=args.vmax,
        sym=args.symmetric,
        cmap=args.cmap,
        scale=args.scale,
    )

    plt.savefig(args.output, bbox_inches="tight", dpi=args.dpi)
