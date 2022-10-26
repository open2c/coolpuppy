# -*- coding: utf-8 -*-
from coolpuppy.lib import numutils, puputils

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.gridspec as grd
from matplotlib.colors import LogNorm, Normalize
from matplotlib import ticker
from matplotlib import cm
import seaborn as sns

from cooltools.lib import plotting

import logging
import warnings

logger = logging.getLogger("coolpuppy")

warnings.filterwarnings(action="ignore", message=".*tight_layout.*")
warnings.filterwarnings(action="ignore", message=".*Tight layout.*")
pd.options.mode.chained_assignment = None
import natsort
import copy


def auto_rows_cols(n):
    """Automatically determines number of rows and cols for n pileups

    Parameters
    ----------
    n : int
        Number of pileups.

    Returns
    -------
    rows : int
        How many rows to use.
    cols : int
        How many columsn to use.

    """
    rows = int(np.ceil(np.sqrt(n)))
    cols = int(np.ceil(n / rows))
    return rows, cols


def get_min_max(pups, vmin=None, vmax=None, sym=True, scale="log"):
    """Automatically determine minimal and maximal colour intensity for pileups

    Parameters
    ----------
    pups : np.array
        Numpy array of numpy arrays conaining pileups.
    vmin : float, optional
        Force certain minimal colour. The default is None.
    vmax : float, optional
        Force certain maximal colour. The default is None.
    sym : bool, optional
        Whether the output should be cymmetrical around 0. The default is True.

    Returns
    -------
    vmin : float
        Selected minimal colour.
    vmax : float
        Selected maximal colour.

    """
    if vmin is not None and vmax is not None:
        if sym:
            logger.info(
                "Can't set both vmin and vmax and get symmetrical scale. Plotting non-symmetrical"
            )
        return vmin, vmax
    else:
        comb = np.concatenate([pup.ravel() for pup in pups.ravel()])
        comb = comb[comb != -np.inf]
        comb = comb[comb != np.inf]
        comb = comb[comb != 0]
        if np.isnan(comb).all():
            raise ValueError("Data only contains NaNs or zeros")
    if vmin is None and vmax is None:
        vmax = np.nanmax(comb)
        vmin = np.nanmin(comb)
    elif vmin is not None:
        vmax = np.nanmax(comb)
    elif vmax is not None:
        vmin = np.nanmin(comb)
    if sym:

        if scale == "linear":
            logger.info(
                "Can't use symmetrical scale with linear. Plotting non-symmetrical"
            )
            pass
        else:
            vmax = np.max(np.abs([vmin, vmax]))
            if vmax >= 1:
                vmin = 2 ** -np.log2(vmax)
            else:
                raise ValueError(
                    "Maximum value is less than 1.0, can't plot using symmetrical scale"
                )
    return vmin, vmax


def add_heatmap(
    data,
    flank,
    rescale,
    rescale_flank,
    n,
    max_coordinates,
    height=1,
    aspect="auto",
    color=None,
    cmap="coolwarm",
    norm=LogNorm(0.5, 2),
    plot_ticks=False,
    stripe=False,
    font_scale=1,
):
    """
    Adds the array contained in data.values[0] to the current axes as a heatmap of stripes
    """
    if len(data) > 1:
        raise ValueError(
            "Multiple pileups for one of the conditions, ensure unique correspondence for each col/row combination"
        )
    elif len(data) == 0:
        return
    flank = int(flank)
    ax = plt.gca()
    if stripe:
        extents = [-flank / max_coordinates[1], flank / max_coordinates[1], -int(n), 0]
        ax.set_ylim(-max_coordinates[2], 0)
    else:
        extents = [
            -flank / max_coordinates[1],
            flank / max_coordinates[1],
            -flank / max_coordinates[1],
            flank / max_coordinates[1],
        ]
        ax.set_xlim(-(flank / max_coordinates[1]), (flank / max_coordinates[1]))
        ax.set_ylim(-(flank / max_coordinates[1]), (flank / max_coordinates[1]))
    ax.imshow(
        data.values[0],
        cmap=cmap,
        norm=norm,
        aspect=aspect,
        interpolation="none",
        extent=extents,
    )
    if plot_ticks:
        ax.tick_params(
            axis="both",
            which="major",
            labelsize=font_scale * (4.94 + height),
            width=1 + (height / 2),
            length=1 + height,
        )

        if not rescale.any():
            ticks_pixels = np.linspace(
                -max_coordinates[0] / max_coordinates[1],
                max_coordinates[0] / max_coordinates[1],
                5,
            )
            ticks_kbp = ((ticks_pixels * max_coordinates[1]) // 1000).astype(int)
            plt.xticks(ticks_pixels.tolist(), ticks_kbp.tolist())
            if not stripe:
                plt.yticks(ticks_pixels.tolist(), [])
            else:
                ticks_n = np.floor(np.linspace(0, -max_coordinates[2], 5)).astype(int)
                plt.yticks(ticks_n, np.append("", np.negative(ticks_n)[1:]))
    else:
        plt.yticks([], [])
        plt.xticks([], [])


def add_stripe_lineplot(
    data,
    resolution,
    flank,
    rescale,
    rescale_flank,
    height=1,
    aspect="auto",
    color=None,
    cmap="coolwarm",
    scale="log",
    norm=LogNorm(0.5, 2),
    plot_ticks=False,
    stripe=False,
    font_scale=1,
    colnames=None,
):
    """
    Adds the array contained in data.values[0] to the current axes as a heatmap of stripes and an average lineplot on top.
    Only works with one condition at a time.
    """
    gs = grd.GridSpec(
        2, 2, height_ratios=[1, 5], width_ratios=[8.25, 1], wspace=0.05, hspace=0.05
    )
    flank = int(flank)
    resolution = int(resolution)
    ticks_pixels = np.linspace(0, flank * 2 // resolution, 5)
    ticks_kbp = ((ticks_pixels - ticks_pixels[-1] / 2) * resolution // 1000).astype(int)
    mean = np.nanmean(data.values[0], axis=0)
    if scale == "log":
        mean = np.log(mean)
        mean = np.where(mean == -np.inf, 0, mean)
    kbp = np.linspace(-flank, flank, int(flank / resolution * 2 + 1)) / 1000
    ax2 = plt.subplot(gs[0])
    ax2.plot(kbp, mean)
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    plt.xlim(-(flank + (resolution / 2)) / 1000, (flank + (resolution / 2)) / 1000)
    if plot_ticks:
        ax2.tick_params(
            axis="both",
            which="major",
            labelsize=font_scale * (4.94 + height),
            width=1 + (height / 2),
            length=1 + height,
        )
        plt.yticks([np.round(min(mean), 4), np.round(max(mean), 4)])
    else:
        plt.yticks([], [])
    plt.xticks([], [])
    if colnames is not None:
        plt.title(colnames[0])
    ax = plt.subplot(gs[2])
    p = ax.imshow(
        data.values[0], cmap=cmap, norm=norm, interpolation="none", aspect=aspect
    )
    if plot_ticks:
        ax.tick_params(
            axis="both",
            which="major",
            labelsize=font_scale * (4.94 + height),
            width=1 + (height / 2),
            length=1 + height,
        )
        if not rescale.any():
            plt.xticks(ticks_pixels.tolist(), ticks_kbp.tolist())
            ticks_n = np.floor(np.linspace(0, data.values[0].shape[0], 5)).astype(int)
            plt.yticks(ticks_n, np.append("", -np.negative(ticks_n)[1:]))
    else:
        plt.xticks([], [])
        plt.yticks([], [])
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)


def add_score(score, height=1, color=None, font_scale=1):
    """
    Adds the value contained in score.values[0] to the current axes as a label in top left corner
    """
    if score is not None:
        ax = plt.gca()
        ax.text(
            s=f"{score.values[0]:.3g}",
            y=0.95,
            x=0.05,
            ha="left",
            va="top",
            size=font_scale * (4.94 + height),
            transform=ax.transAxes,
        )


def sort_separation(sep_string_series, sep="Mb"):
    s = set(sep_string_series.dropna())
    s.discard("all")
    return sorted(
        s,
        key=lambda x: float(x.split(sep)[0]),
    )


def plot_stripes(
    pupsdf,
    cols=None,
    rows=None,
    col_order=None,
    row_order=None,
    vmin=None,
    vmax=None,
    sym=True,
    cmap="coolwarm",
    cmap_emptypixel=(0.98, 0.98, 0.98),
    scale="log",
    height=1,
    aspect="auto",
    stripe="corner_stripe",
    stripe_sort="sum",
    out_sorted_bedpe=None,
    font="DejaVu Sans",
    font_scale=1,
    plot_ticks=False,
    colnames=None,
    rownames=None,
    lineplot=False,
    **kwargs,
):
    pupsdf = pupsdf.copy()
    if np.any(pupsdf.index.duplicated()):
        pupsdf = pupsdf.reset_index(drop=True)
    if not set(["vertical_stripe", "horizontal_stripe"]).issubset(pupsdf.columns):
        raise ValueError("No stripes stored in pup")
    if cols == "separation" and col_order is None:
        col_order = sort_separation(pupsdf["separation"])
        ncols = len(col_order)
    elif cols is not None and col_order is None:
        col_order = list(set(pupsdf[cols].dropna()))
        ncols = len(col_order)
    elif col_order is not None:
        if isinstance(col_order, str):
            col_order = [col_order]
        ncols = len(col_order)
    else:
        ncols = 1

    if rows == "separation" and row_order is None:
        row_order = sort_separation(pupsdf["separation"])
        nrows = len(row_order)
    elif rows is not None and row_order is None:
        row_order = list(set(pupsdf[rows].dropna()))
        nrows = len(row_order)
    elif row_order is not None:
        if isinstance(row_order, str):
            row_order = [row_order]
        nrows = len(row_order)
    else:
        nrows = 1

    if cols is None and rows is None:
        if pupsdf.shape[0] > 1:
            if "orientation" in pupsdf.columns:
                rows = "orientation"
                row_order = list(set(pupsdf[rows].dropna()))
                nrows = len(row_order)
            if "separation" in pupsdf.columns:
                cols = "separation"
                col_order = sort_separation(pupsdf["separation"])
                ncols = len(col_order)

    logger.debug(f"Plotting stripe stackups with {ncols} columns and {nrows} rows")

    vmin, vmax = get_min_max(pupsdf["data"].values, vmin, vmax, sym=sym, scale=scale)

    if scale == "log":
        norm = LogNorm
    elif scale == "linear":
        norm = Normalize
    else:
        raise ValueError(
            f"Unknown scale value, only log or linear implemented, but got {scale}"
        )

    right = ncols / (ncols + 0.25)

    pupsdf = pupsdf.reset_index()

    # Generate corner stripes
    cntr = int(np.floor(pupsdf["data"][0].shape[0] / 2))
    pupsdf["corner_stripe"] = pupsdf["horizontal_stripe"]
    for i in range(len(pupsdf)):
        pupsdf["corner_stripe"][i] = np.concatenate(
            (
                pupsdf["horizontal_stripe"][i][:, :(cntr)],
                pupsdf["vertical_stripe"][i][:, cntr:],
            ),
            axis=1,
        )

    # Sorting stripes
    if not stripe_sort == None:
        different = False
        for i in range(len(pupsdf)):
            pupsdf["coordinates"][i] = np.array(pupsdf["coordinates"][i], dtype=object)
            pupsdf["corner_stripe"][i] = np.array(pupsdf["corner_stripe"][i])
            pupsdf["vertical_stripe"][i] = np.array(pupsdf["vertical_stripe"][i])
            pupsdf["horizontal_stripe"][i] = np.array(pupsdf["horizontal_stripe"][i])
            ind_regions = natsort.index_natsorted(pupsdf["coordinates"][i])
            pupsdf.loc[
                i,
                [
                    "coordinates",
                    "corner_stripe",
                    "vertical_stripe",
                    "horizontal_stripe",
                ],
            ] = pupsdf.loc[
                i,
                [
                    "coordinates",
                    "corner_stripe",
                    "vertical_stripe",
                    "horizontal_stripe",
                ],
            ].apply(
                lambda x: x[ind_regions]
            )
        for i in range(len(pupsdf)):
            if not np.array_equal(pupsdf["coordinates"][0], pupsdf["coordinates"][i]):
                different = True
                warnings.warn(
                    "Cannot sort stripes, rows or columns contain different regions. Plot one by one if you want to sort",
                    stacklevel=2,
                )
        if not different:
            if stripe_sort == "sum":
                ind_sort = np.argsort(-np.nansum(pupsdf[stripe][0], axis=1))
            elif stripe_sort == "center_pixel":
                cntr = int(np.floor(pupsdf[stripe][0].shape[1] / 2))
                ind_sort = np.argsort(-pupsdf[stripe][0][:, cntr])
            else:
                raise ValueError("stripe_sort can only be None, sum, or center_pixel")
            for i in range(len(pupsdf)):
                pupsdf.loc[
                    i,
                    [
                        "coordinates",
                        "corner_stripe",
                        "vertical_stripe",
                        "horizontal_stripe",
                    ],
                ] = pupsdf.loc[
                    i,
                    [
                        "coordinates",
                        "corner_stripe",
                        "vertical_stripe",
                        "horizontal_stripe",
                    ],
                ].apply(
                    lambda x: x[ind_sort]
                )
            if isinstance(out_sorted_bedpe, str):
                pd.DataFrame(pupsdf.loc[0, "coordinates"]).to_csv(
                    out_sorted_bedpe, sep="\t", header=None, index=False
                )

    sns.set(font=font, font_scale=font_scale, style="ticks")

    fg = sns.FacetGrid(
        pupsdf,
        col=cols,
        row=rows,
        row_order=row_order,
        col_order=col_order,
        margin_titles=True,
        height=height,
        gridspec_kws={
            "right": right,
        },
        **kwargs,
    )

    norm = norm(vmin, vmax)

    cmap = copy.copy(cm.get_cmap(cmap))
    cmap.set_bad(cmap_emptypixel)

    max_coordinates = [
        pupsdf.loc[pd.to_numeric(pupsdf["flank"]).idxmax(), "flank"],
        pupsdf.loc[pd.to_numeric(pupsdf["flank"]).idxmax(), "resolution"],
        max(pupsdf.loc[pupsdf.drop(columns="index").iloc[:, 1] != "all", "n"]),
    ]

    if stripe in ["horizontal_stripe", "vertical_stripe", "corner_stripe"]:
        if lineplot and nrows == 1 and ncols == 1:
            fg.map(
                add_stripe_lineplot,
                stripe,
                "resolution",
                "flank",
                "rescale",
                "rescale_flank",
                scale=scale,
                norm=norm,
                cmap=cmap,
                aspect=aspect,
                height=height,
                plot_ticks=plot_ticks,
                stripe=stripe,
                font_scale=font_scale,
                colnames=colnames,
            )
            logger.debug(f"Plotting lineplot on top of stripes")
        else:
            fg.map(
                add_heatmap,
                stripe,
                "flank",
                "rescale",
                "rescale_flank",
                "n",
                norm=norm,
                cmap=cmap,
                aspect=aspect,
                height=height,
                plot_ticks=plot_ticks,
                stripe=stripe,
                font_scale=font_scale,
                max_coordinates=max_coordinates,
            )
            if lineplot:
                logger.info(
                    "Can only do lineplot for single conditions (no rows/columns). Doing normal stripe plot instead."
                )
    else:
        raise ValueError(
            "stripe can only be 'vertical_stripe', 'horizontal_stripe' or 'corner_stripe'"
        )
    if plot_ticks:
        fg.fig.subplots_adjust(wspace=0.2, hspace=0.05)
        fg.set_titles(col_template="", row_template="")
        if nrows > 1 and ncols > 1:
            for (row_val, col_val), ax in fg.axes_dict.items():
                if row_val == row_order[0]:
                    ax.set_title(col_val)
                if row_val == row_order[-1]:
                    ax.set_xlabel("")
                if col_val == col_order[0]:
                    ax.set_ylabel(row_val, rotation=0, ha="right", va="center")
        else:
            if nrows == 1 and ncols > 1:
                for col_val, ax in fg.axes_dict.items():
                    ax.set_xlabel("")
                    ax.set_ylabel("")
                    ax.set_title(col_val)
            elif nrows > 1 and ncols == 1:
                for row_val, ax in fg.axes_dict.items():
                    ax.set_ylabel(row_val, rotation=0, ha="right", va="center")
                    ax.set_xlabel("")
            else:
                plt.title("")
                plt.ylabel("")
                plt.xlabel("")
    else:
        fg.fig.subplots_adjust(wspace=0.05, hspace=0.05)
        if not lineplot:
            fg.map(lambda color: plt.gca().set_xticks([]))
            fg.map(lambda color: plt.gca().set_yticks([]))
        fg.set_titles(col_template="", row_template="")
        if nrows > 1 and ncols > 1:
            for (row_val, col_val), ax in fg.axes_dict.items():
                if row_val == row_order[-1]:
                    ax.set_xlabel(col_val)
                if col_val == col_order[0]:
                    ax.set_ylabel(row_val, rotation=0, ha="right")
        else:
            if nrows == 1 and ncols > 1:
                for col_val, ax in fg.axes_dict.items():
                    ax.set_xlabel(col_val)
                    ax.set_ylabel("")
            elif nrows > 1 and ncols == 1:
                for row_val, ax in fg.axes_dict.items():
                    ax.set_xlabel("")
                    ax.set_ylabel(row_val, rotation=0, ha="right")
            else:
                plt.title("")
                plt.xlabel("")
                plt.ylabel("")

    if colnames is not None:
        if len(colnames) != ncols:
            logger.info(f"{len(colnames)} colnames but {ncols} columns, ignoring")
        else:
            i = 0
            if nrows > 1 and ncols > 1:
                for (row_val, col_val), ax in fg.axes_dict.items():
                    if not plot_ticks and row_val == row_order[-1]:
                        ax.set_xlabel(colnames[i])
                        i += 1
                    elif plot_ticks and row_val == row_order[0]:
                        ax.set_title(colnames[i])
                        i += 1
            else:
                if nrows == 1 and ncols > 1:
                    for col_val, ax in fg.axes_dict.items():
                        if plot_ticks:
                            ax.set_title(colnames[i])
                        else:
                            ax.set_xlabel(colnames[i])
                        i += 1
                elif nrows > 1 and ncols == 1:
                    for row_val, ax in fg.axes_dict.items():
                        if not plot_ticks and row_val == row_order[-1]:
                            ax.set_xlabel(colnames[i])
                            i += 1
                        elif plot_ticks and row_val == row_order[0]:
                            ax.set_title(colnames[i])
                            i += 1
                else:
                    if not lineplot:
                        plt.title(colnames[i])

    if rownames is not None:
        if len(rownames) != nrows:
            logger.info(f"{len(rownames)} rownames but {nrows} columns, ignoring")
        else:
            i = 0
            if nrows > 1 and ncols > 1:
                for (row_val, col_val), ax in fg.axes_dict.items():
                    if col_val == col_order[0]:
                        ax.set_ylabel(rownames[i], rotation=0, ha="right", va="center")
                        i += 1
            else:
                if nrows == 1 and ncols > 1:
                    for col_val, ax in fg.axes_dict.items():
                        ax.set_ylabel(rownames[i], rotation=0, ha="right", va="center")
                elif nrows > 1 and ncols == 1:
                    for row_val, ax in fg.axes_dict.items():
                        ax.set_ylabel(rownames[i], rotation=0, ha="right", va="center")
                        i += 1
                else:
                    plt.ylabel(rownames[i], rotation=0, ha="right", va="center")

    plt.draw()
    ax_bottom = fg.axes[-1, -1]
    bottom = ax_bottom.get_position().y0
    ax_top = fg.axes[0, -1]
    top = ax_top.get_position().y1
    right = ax_top.get_position().x1
    ax_left = fg.axes[-1, 0]
    left = ax_left.get_position().x0
    cax = fg.fig.add_axes([right + 0.01, bottom, (1 - right - 0.01) / 5, top - bottom])
    if plot_ticks:
        if pupsdf["rescale"].any():
            string = "rescaled"
        else:
            string = "pos. [kb]"
        fg.fig.text(
            (right + left) / 2,
            (0.05 + (0.01 * nrows) - (0.25 / height / nrows)),
            s=string,
            ha="center",
            fontsize=font_scale * 2 * (4.94 + height),
        )
    if sym and scale == "log":
        ticks = [vmin, 1, vmax]
    else:
        ticks = [vmin, vmax]
    cb = plt.colorbar(
        cm.ScalarMappable(norm, cmap),
        ticks=ticks,
        cax=cax,
        format=ticker.FuncFormatter(lambda x, pos: f"{x:.2g}"),
    )
    cax.minorticks_off()
    return fg


def plot(
    pupsdf,
    cols=None,
    rows=None,
    score="score",
    center=3,
    ignore_central=3,
    col_order=None,
    row_order=None,
    vmin=None,
    vmax=None,
    sym=True,
    norm_corners=0,
    cmap="coolwarm",
    cmap_emptypixel=(0.98, 0.98, 0.98),
    scale="log",
    height=1,
    aspect=1,
    font="DejaVu Sans",
    font_scale=1,
    plot_ticks=False,
    colnames=None,
    rownames=None,
    **kwargs,
):
    pupsdf = pupsdf.copy()
    if np.any(pupsdf.index.duplicated()):
        pupsdf = pupsdf.reset_index(drop=True)
    cmap = copy.copy(cm.get_cmap(cmap))
    cmap.set_bad(cmap_emptypixel)

    sns.set(font=font, font_scale=font_scale, style="ticks")
    if norm_corners:
        pupsdf["data"] = pupsdf.apply(
            lambda x: numutils.norm_cis(x["data"], norm_corners), axis=1
        )
    if cols == "separation" and col_order is None:
        col_order = sort_separation(pupsdf["separation"])
        ncols = len(col_order)
    elif cols is not None and col_order is None:
        col_order = list(set(pupsdf[cols].dropna()))
        ncols = len(col_order)
    elif col_order is not None:
        if isinstance(col_order, str):
            col_order = [col_order]
        ncols = len(col_order)
    else:
        ncols = 1

    if rows == "separation" and row_order is None:
        row_order = sort_separation(pupsdf["separation"])
        nrows = len(row_order)
    elif rows is not None and row_order is None:
        row_order = list(set(pupsdf[rows].dropna()))
        nrows = len(row_order)
    elif row_order is not None:
        if isinstance(row_order, str):
            row_order = [row_order]
        nrows = len(row_order)
    else:
        nrows = 1
    if cols is None and rows is None:
        if pupsdf.shape[0] > 1:
            if "orientation" in pupsdf.columns:
                rows = "orientation"
                row_order = list(set(pupsdf[rows].dropna()))
                nrows = len(row_order)
            if "separation" in pupsdf.columns:
                cols = "separation"
                col_order = sort_separation(pupsdf["separation"])
                ncols = len(col_order)

    logger.debug(f"Plotting pileup with {ncols} columns and {nrows} rows")

    vmin, vmax = get_min_max(pupsdf["data"].values, vmin, vmax, sym=sym, scale=scale)

    if scale == "log":
        norm = LogNorm
    elif scale == "linear":
        norm = Normalize
    else:
        raise ValueError(
            f"Unknown scale value, only log or linear implemented, but got {scale}"
        )
    right = ncols / (ncols + 0.25)

    fg = sns.FacetGrid(
        pupsdf,
        col=cols,
        row=rows,
        row_order=row_order,
        col_order=col_order,
        margin_titles=True,
        aspect=1,
        height=height,
        gridspec_kws={
            "right": right,
        },
        **kwargs,
    )
    norm = norm(vmin, vmax)
    max_coordinates = [
        pupsdf.loc[pd.to_numeric(pupsdf["flank"]).idxmax(), "flank"],
        pupsdf.loc[pd.to_numeric(pupsdf["flank"]).idxmax(), "resolution"],
    ]
    fg.map(
        add_heatmap,
        "data",
        "flank",
        "rescale",
        "rescale_flank",
        "n",
        max_coordinates=max_coordinates,
        norm=norm,
        cmap=cmap,
        aspect=aspect,
        height=height,
        plot_ticks=plot_ticks,
        font_scale=font_scale,
    )

    if score:
        pupsdf["score"] = pupsdf.apply(
            puputils.get_score, center=center, ignore_central=ignore_central, axis=1
        )
        fg.map(add_score, "score", height=height, font_scale=font_scale)

    if plot_ticks:
        fg.fig.subplots_adjust(wspace=0.2, hspace=0.05)
        fg.set_titles(col_template="", row_template="")
        if nrows > 1 and ncols > 1:
            for (row_val, col_val), ax in fg.axes_dict.items():
                if row_val == row_order[0]:
                    ax.set_title(col_val)
                    saveax = ax
                if row_val == row_order[-1]:
                    ax.set_xlabel("")
                if col_val == col_order[0]:
                    ax.set_ylabel(row_val, rotation=0, ha="right", va="center")
        else:
            if nrows == 1 and ncols > 1:
                for col_val, ax in fg.axes_dict.items():
                    ax.set_xlabel("")
                    ax.set_ylabel("")
                    ax.set_title(col_val)
            elif nrows > 1 and ncols == 1:
                for row_val, ax in fg.axes_dict.items():
                    ax.set_ylabel(row_val, rotation=0, ha="right", va="center")
                    ax.set_xlabel("")
            else:
                plt.title("")
                plt.ylabel("")
                plt.xlabel("")

    else:
        fg.fig.subplots_adjust(wspace=0.05, hspace=0.05)
        fg.map(lambda color: plt.gca().set_xticks([]))
        fg.map(lambda color: plt.gca().set_yticks([]))
        fg.set_titles(col_template="", row_template="")
        if nrows > 1 and ncols > 1:
            for (row_val, col_val), ax in fg.axes_dict.items():
                if row_val == row_order[-1]:
                    ax.set_xlabel(col_val)
                if col_val == col_order[0]:
                    ax.set_ylabel(row_val, rotation=0, ha="right", va="center")
        else:
            if nrows == 1 and ncols > 1:
                for col_val, ax in fg.axes_dict.items():
                    ax.set_xlabel(col_val)
                    ax.set_ylabel("")
            elif nrows > 1 and ncols == 1:
                for row_val, ax in fg.axes_dict.items():
                    ax.set_xlabel("")
                    ax.set_ylabel(row_val, rotation=0, ha="right", va="center")
            else:
                plt.xlabel("")
                plt.ylabel("")
                plt.title("")

    if colnames is not None:
        if len(colnames) != ncols:
            logger.info(f"{len(colnames)} colnames but {ncols} columns, ignoring")
        else:
            i = 0
            if nrows > 1 and ncols > 1:
                for (row_val, col_val), ax in fg.axes_dict.items():
                    if not plot_ticks and row_val == row_order[-1]:
                        ax.set_xlabel(colnames[i])
                        i += 1
                    elif plot_ticks and row_val == row_order[0]:
                        ax.set_title(colnames[i])
                        i += 1
            else:
                if nrows == 1 and ncols > 1:
                    for col_val, ax in fg.axes_dict.items():
                        if plot_ticks:
                            ax.set_title(colnames[i])
                        else:
                            ax.set_xlabel(colnames[i])
                        i += 1
                elif nrows > 1 and ncols == 1:
                    for row_val, ax in fg.axes_dict.items():
                        if not plot_ticks and row_val == row_order[-1]:
                            ax.set_xlabel(colnames[i])
                            i += 1
                        elif plot_ticks and row_val == row_order[0]:
                            ax.set_title(colnames[i])
                            i += 1
                else:
                    plt.title(colnames[i])

    if rownames is not None:
        if len(rownames) != nrows:
            logger.info(f"{len(rownames)} rownames but {nrows} columns, ignoring")
        else:
            i = 0
            if nrows > 1 and ncols > 1:
                for (row_val, col_val), ax in fg.axes_dict.items():
                    if col_val == col_order[0]:
                        ax.set_ylabel(rownames[i], rotation=0, ha="right", va="center")
                        i += 1
            else:
                if nrows == 1 and ncols > 1:
                    for col_val, ax in fg.axes_dict.items():
                        ax.set_ylabel(rownames[i], rotation=0, ha="right", va="center")
                elif nrows > 1 and ncols == 1:
                    for row_val, ax in fg.axes_dict.items():
                        ax.set_ylabel(rownames[i], rotation=0, ha="right", va="center")
                        i += 1
                else:
                    plt.ylabel(rownames[i], rotation=0, ha="right", va="center")

    plt.draw()
    ax_bottom = fg.axes[-1, -1]
    bottom = ax_bottom.get_position().y0
    ax_top = fg.axes[0, -1]
    top = ax_top.get_position().y1
    right = ax_top.get_position().x1
    ax_left = fg.axes[-1, 0]
    left = ax_left.get_position().x0
    cax = fg.fig.add_axes(
        [right + 0.005, bottom, (1 - right - 0.005) / 5, top - bottom]
    )
    if plot_ticks:
        if pupsdf["rescale"].any():
            string = "rescaled"
        else:
            string = "pos. [kb]"
        fg.fig.text(
            (right + left) / 2,
            (0.1 - (0.25 / height / nrows)),
            s=string,
            ha="center",
            fontsize=font_scale * 2 * (4.94 + height),
        )
    if sym and scale == "log":
        ticks = [vmin, 1, vmax]
    else:
        ticks = [vmin, vmax]

    cb = plt.colorbar(
        cm.ScalarMappable(norm, cmap),
        ticks=ticks,
        cax=cax,
        format=ticker.FuncFormatter(lambda x, pos: f"{x:.2g}"),
    )
    cax.minorticks_off()
    return fg
