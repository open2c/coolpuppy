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
    divide_pups,
)
from plotpuppy.plotpup import make_heatmap_grid, make_heatmap_stripes
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
        "--not_symmetric",
        "--not-symmetric",
        "--not_symmetrical",
        "--not-symmetrical",
        default=False,
        action="store_true",
        help="""Whether to **not** make colormap symmetric around 1, if log scale""",
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
    parser.add_argument(
        "--stripe",
        type=str,
        default=None,
        required=False,
        help="""For plotting stripe stackups""",
    )
    parser.add_argument(
        "--stripe_sort",
        type=str,
        default="sum",
        required=False,
        help="""Whether to sort stripe stackups by total signal (sum), central pixel signal (center_pixel), or not at all (None)""",
    )
    parser.add_argument(
        "--out_sorted_bedpe",
        type=str,
        default=None,
        required=False,
        help="""Output bedpe of sorted stripe regions""",
    )
    parser.add_argument(
        "--divide_pups",
        default=False,
        action="store_true",
        help="""Whether to divide two pileups and plot the result""",
    )
    parser.add_argument(
        "--font",
        type=str,
        default=False,
        required=False,
        help="""Font to use for plotting""",
    )

    parser.add_argument(
        "--font_scale",
        type=float,
        default=False,
        required=False,
        help="""Font scale to use for plotting. Defaults to 1""",
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
        help="""Order of columns to use, space or comma separated""",
    )
    parser.add_argument(
        "--row_order",
        type=lambda s: re.split(" |, ", s),
        required=False,
        help="""Order of rows to use, space or comma separated""",
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
        "--no_score",
        action="store_true",
        required=False,
        default=False,
        help="""If central pixel score should not be shown in top left corner""",
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
        help="""How many central bins to ignore when calculating insulation for
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
        "--height",
        type=float,
        required=False,
        help="""Height of the plot""",
    )
    parser.add_argument(
        "--plot_ticks",
        action="store_true",
        default=False,
        required=False,
        help="""Whether to plot ticks demarkating the center and flanking regions, only applicable for non-stripes""",
    )
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
    if args.divide_pups:
        if len(args.input_pups) != 2:
            raise ValueError("Need exactly two input pups when using --divide_pups")
        else:
            pup1 = load_pileup_df(args.input_pups[0])
            pup2 = load_pileup_df(args.input_pups[1])
            pups = divide_pups(pup1, pup2)
    else:
        pups = load_pileup_df_list(
            args.input_pups, quaich=args.quaich, nice_metadata=True
        )

    if args.query is not None:
        for q in args.query:
            pups = pups.query(q)

    if args.norm_corners > 0:
        pups["data"] = pups["data"].apply(norm_cis, i=int(args.norm_corners))

    if not args.no_score:
        pups["score"] = pups.apply(
            get_score, center=args.center, ignore_central=args.ignore_central, axis=1
        )
        score = "score"
    else:
        score = False

    if args.cols:
        if args.col_order:
            col_order = args.col_order
            pups = pups[pups[args.cols].isin(args.col_order)]
        elif args.cols == "separation":
            col_order = sort_separation(pups["separation"])
        else:
            col_order = pups[args.cols].unique()
    else:
        col_order = None

    if args.rows:
        if args.row_order:
            row_order = args.row_order
            pups = pups[pups[args.rows].isin(args.row_order)]
        elif args.rows == "separation":
            row_order = sort_separation(pups["separation"])
        else:
            row_order = pups[args.rows].unique()
    else:
        row_order = None

    if args.stripe_sort == "None":
        args.stripe_sort = None

    if args.not_symmetric:
        symmetric = False
    else:
        symmetric = True

    if args.height is None:
        if args.stripe:
            height = 2
        else:
            height = 1
    else:
        height = args.height

    if args.stripe:
        fg = make_heatmap_stripes(
            pups,
            cols=args.cols,
            rows=args.rows,
            col_order=col_order,
            row_order=row_order,
            vmin=args.vmin,
            vmax=args.vmax,
            sym=symmetric,
            cmap=args.cmap,
            scale=args.scale,
            height=height,
            stripe=args.stripe,
            stripe_sort=args.stripe_sort,
            out_sorted_bedpe=args.out_sorted_bedpe,
            font=args.font,
            font_scale=args.font_scale,
            plot_ticks=args.plot_ticks,
        )
    else:
        fg = make_heatmap_grid(
            pups,
            cols=args.cols,
            rows=args.rows,
            score=score,
            col_order=col_order,
            row_order=row_order,
            vmin=args.vmin,
            vmax=args.vmax,
            sym=symmetric,
            cmap=args.cmap,
            scale=args.scale,
            height=height,
            font=args.font,
            font_scale=args.font_scale,
            plot_ticks=args.plot_ticks,
        )

    plt.savefig(args.output, bbox_inches="tight", dpi=args.dpi)
