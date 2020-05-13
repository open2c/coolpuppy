#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 14:05:06 2020

@author: Ilya Flyamer
"""
# from coolpuppy import *
from coolpuppy import __version__
from coolpuppy import (
    norm_cis,
    load_array_with_header,
    get_enrichment,
    get_min_max,
    auto_rows_cols,
)

import numpy as np

import matplotlib

matplotlib.use("Agg")

from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as font_manager

from itertools import product
import argparse


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
    parser.add_argument(
        "--cbar_mode",
        type=str,
        default="single",
        required=False,
        choices={"single", "edge", "each"},
        help="""Whether to show a single colorbar, one per row or one for each subplot
             """,
    )
    parser.add_argument(
        "--n_cols",
        type=int,
        default=0,
        required=False,
        help="""How many columns to use for plotting the data.
                If 0, automatically make the figure as square as possible""",
    )
    parser.add_argument(
        "--col_names",
        type=str,
        required=False,
        help="""A comma separated list of column names""",
    )
    parser.add_argument(
        "--row_names",
        type=str,
        required=False,
        help="""A comma separated list of row names""",
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
        "--enrichment",
        type=int,
        required=False,
        default=1,
        help="""Whether to show the level of enrichment in the central pixels.
                0 to not show, odd positive number to define the size of the central
                square whose values are averaged""",
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
        "pileup_files", type=str, nargs="*", help="""All files to plot"""
    )
    parser.add_argument("-v", "--version", action="version", version=__version__)
    return parser


def main():
    mpl.rcParams["svg.fonttype"] = "none"
    mpl.rcParams["pdf.fonttype"] = 42

    parser = parse_args_plotpuppy()
    args = parser.parse_args()

    pups = [load_array_with_header(f)["data"] for f in args.pileup_files]

    if args.norm_corners > 0:
        pups = [norm_cis(pup) for pup in pups]

    n = len(pups)
    if args.n_cols == 0:
        n_rows, n_cols = auto_rows_cols(n)

    elif args.n_cols < n:
        n_rows = int(round(n / args.n_cols))
        n_cols = args.n_cols

    else:
        n_cols = args.n_cols
        n_rows = 1

    if args.col_names is not None:
        args.col_names = args.col_names.strip(", ").split(",")

    if args.row_names is not None:
        args.row_names = args.row_names.strip(", ").split(",")

    if args.col_names != None and n_cols != len(args.col_names):
        raise ValueError(
            f"""Number of column names is not equal to number of
                columns! You specified {n_cols} columns and {len(args.col_names)}
                column names"""
            % (n_cols, len(args.col_names))
        )
    if args.row_names is not None and n_rows != len(args.row_names):
        raise ValueError(
            """Number of row names is not equal to number of
                         rows!"""
        )

    if args.enrichment % 2 == 0 and args.enrichment > 0:
        raise ValueError(
            """Side of the square to calculate enrichment has
                         to be an odd number"""
        )

    f = plt.figure(dpi=args.dpi, figsize=(max(3.5, n_cols + 0.5), max(3, n_rows)))
    grid = ImageGrid(
        f,
        111,
        share_all=True,  # similar to subplot(111)
        nrows_ncols=(n_rows, n_cols),
        #                     direction='column',
        axes_pad=0.05,
        add_all=True,
        label_mode="L",
        cbar_location="right",
        cbar_mode=args.cbar_mode,
        cbar_size="5%",
        cbar_pad="3%",
    )
    axarr = np.array(grid).reshape((n_rows, n_cols))

    #    f, axarr = plt.subplots(n_rows, n_cols, sharex=True, sharey=True,# similar to subplot(111)
    #                            figsize=(max(3.5, n_cols+0.5), max(3, n_rows)),
    #                            dpi=300, squeeze=False,
    #                            constrained_layout=True
    #                            )
    sym = False
    if args.scale == "log":
        norm = LogNorm
        if args.symmetric:
            sym = True
    else:
        norm = Normalize

    n_grid = n_rows * n_cols
    extra = [None for i in range(n_grid - len(pups))]
    pupsarray = np.empty(n_rows * n_cols, dtype=object)
    for i, pup in enumerate(pups + extra):
        pupsarray[i] = pup
    pups = pupsarray.reshape((n_rows, n_cols))

    if args.cbar_mode == "single":
        vmin, vmax = get_min_max(pups, args.vmin, args.vmax, sym=sym)
    elif args.cbar_mode == "edge":
        colorscales = [get_min_max(row, args.vmin, args.vmax, sym=sym) for row in pups]
    elif args.cbar_mode == "each":
        grid.cbar_axes = np.asarray(grid.cbar_axes).reshape((n_rows, n_cols))

    cbs = []

    for i in range(n_rows):
        if args.cbar_mode == "edge":
            vmin, vmax = colorscales[i]
        for j in range(n_cols):
            #        n = i*n_cols+(j%n_cols)
            if pups[i, j] is not None:
                if args.cbar_mode == "each":
                    vmin = np.nanmin(pups[i, j])
                    vmax = np.nanmax(pups[i, j])
                ax = axarr[i, j]
                m = ax.imshow(
                    pups[i, j],
                    interpolation="none",
                    norm=norm(vmax=vmax, vmin=vmin),
                    cmap=args.cmap,
                    extent=(0, 1, 0, 1),
                )
                ax.set_xticks([])
                ax.set_yticks([])
                if args.enrichment > 0:
                    enr = round(get_enrichment(pups[i, j], args.enrichment), 2)
                    ax.text(
                        s=enr,
                        y=0.95,
                        x=0.05,
                        ha="left",
                        va="top",
                        size="x-small",
                        transform=ax.transAxes,
                    )
                if args.cbar_mode == "each":
                    cbs.append(plt.colorbar(m, cax=grid.cbar_axes[i, j]))
            else:
                axarr[i, j].axis("off")
                grid.cbar_axes[i, j].axis("off")
            if args.cbar_mode == "edge":
                cbs.append(plt.colorbar(m, cax=grid.cbar_axes[i]))

    if args.col_names is not None:
        for i, name in enumerate(args.col_names):
            axarr[-1, i].set_xlabel(name)
    if args.row_names is not None:
        for i, name in enumerate(args.row_names):
            axarr[i, 0].set_ylabel(name)
    if args.cbar_mode == "single":
        cbs.append(
            plt.colorbar(m, cax=grid.cbar_axes[0])
        )  # , format=FormatStrFormatter('%.2f'))
    #    plt.setp(cbs, ticks=mpl.ticker.LogLocator())
    #    if sym:
    #        cb.ax.yaxis.set_ticks([vmin, 1, vmax])
    plt.savefig(args.output, bbox_inches="tight")
