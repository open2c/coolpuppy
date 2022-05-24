# -*- coding: utf-8 -*-

import numpy as np
from coolpuppy import coolpup
import matplotlib.pyplot as plt
import pandas as pd

# from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.colors import LogNorm, Normalize
from matplotlib import ticker
from matplotlib import cm
import seaborn as sns
from cooltools.lib import plotting
import random
import logging
import warnings

warnings.filterwarnings(action="ignore", message=".*tight_layout.*")
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


def get_min_max(pups, vmin=None, vmax=None, sym=True):
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
        vmax = np.max(np.abs([vmin, vmax]))
        vmin = 2 ** -np.log2(vmax)
    return vmin, vmax


def add_heatmap(data, color=None, cmap="coolwarm", norm=LogNorm(0.5, 2)):
    """
    Adds the array contained in data.values[0] to the current axes as a heatmap
    """
    if len(data) > 1:
        raise ValueError(
            "Multiple pileups for one of the conditions, ensure unique correspondence for each col/row combination"
        )
    elif len(data) == 0:
        return
    ax = plt.gca()
    ax.imshow(data.values[0], cmap=cmap, norm=norm, interpolation="none")


def add_stripe_heatmap(
    data,
    resolution,
    flank,
    rescale,
    rescale_flank,
    color=None,
    cmap="coolwarm",
    norm=LogNorm(0.5, 2),
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
    ax = plt.gca()
    ax.imshow(data.values[0], cmap=cmap, norm=norm, aspect="auto", interpolation="none")

    resolution = int(resolution)
    flank = int(flank)

    if not rescale.any():
        ticks_pixels = np.linspace(0, flank * 2 // resolution, 5)
        ticks_kbp = ((ticks_pixels - ticks_pixels[-1] / 2) * resolution // 1000).astype(
            int
        )
        plt.xticks(ticks_pixels.tolist(), ticks_kbp.tolist())
    else:
        if not rescale_flank.any():
            plt.xticks([], [])


def add_score(score, color=None):
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
            size="x-small",
            transform=ax.transAxes,
        )


def sort_separation(sep_string_series, sep="Mb"):
    return sorted(set(sep_string_series.dropna()), key=lambda x: float(x.split(sep)[0]))


def make_heatmap_stripes(
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
    height=2,
    stripe="corner_stripe",
    stripe_sort="center_pixel",
    out_sorted_bedpe=None,
    font=False,
    **kwargs,
):
    pupsdf = pupsdf.copy()

    if cols == "separation":
        col_order = sort_separation(pupsdf["separation"])
        ncols = len(col_order)
    elif cols is not None and col_order is None:
        col_order = list(set(pupsdf[cols].dropna()))
        ncols = len(col_order)
    elif col_order is not None:
        ncols = len(col_order)
    else:
        ncols = 1

    if rows == "separation":
        row_order = sort_separation(pupsdf["separation"])
        nrows = len(row_order)
    elif rows is not None and row_order is None:
        row_order = list(set(pupsdf[rows].dropna()))
        nrows = len(row_order)
    elif row_order is not None:
        nrows = len(row_order)
    else:
        nrows = 1

    if cols is None and rows is None:
        nrows, ncols = auto_rows_cols(pupsdf.shape[0])

    vmin, vmax = get_min_max(pupsdf["data"].values, vmin, vmax, sym=sym)

    if scale == "log":
        norm = LogNorm
        if vmin == 0:
            vmin = 1e-6
    elif scale == "linear":
        norm = Normalize
    else:
        raise ValueError(
            f"Unknown scale value, only log or linear implemented, but got {scale}"
        )

    right = ncols / (ncols + 0.25)

    # Sorting stripes
    if not stripe_sort == None:
        pupsdf = pupsdf.reset_index()
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
            # if not np.all(pupsdf["coordinates"][0] == pupsdf["coordinates"][i]):
            if not np.array_equal(pupsdf["coordinates"][0], pupsdf["coordinates"][i]):
                different = True
                warnings.warn(
                    "Cannot sort, samples have different regions. Plot one by one if you want to sort",
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
    if font:
        sns.set(font=font, style="white")

    fg = sns.FacetGrid(
        pupsdf,
        col=cols,
        row=rows,
        row_order=row_order,
        col_order=col_order,
        margin_titles=True,
        height=height,
        **kwargs,
    )

    norm = norm(vmin, vmax)

    cmap = copy.copy(cm.get_cmap(cmap))
    cmap.set_bad(cmap_emptypixel)

    if stripe in ["horizontal_stripe", "vertical_stripe", "corner_stripe"]:
        fg.map(
            add_stripe_heatmap,
            stripe,
            "resolution",
            "flank",
            "rescale",
            "rescale_flank",
            norm=norm,
            cmap=cmap,
        )
    else:
        raise ValueError(
            "stripe can only be 'vertical_stripe', 'horizontal_stripe' or 'corner_stripe'"
        )

    fg.fig.subplots_adjust(wspace=0.2, right=right)

    fg.set_titles(row_template="", col_template="")
    if nrows > 1 and ncols > 1:
        for (row_val, col_val), ax in fg.axes_dict.items():
            if row_val == row_order[0]:
                ax.set_title(col_val)
            if row_val == row_order[-1]:
                if pupsdf["rescale"].any():
                    ax.set_xlabel("rescaled\n" + stripe)
                else:
                    ax.set_xlabel("relative position, kbp\n" + stripe)
            if col_val == col_order[0]:
                ax.set_ylabel(row_val)
    else:
        if nrows == 1 and ncols > 1:
            for col_val, ax in fg.axes_dict.items():
                if pupsdf["rescale"].any():
                    ax.set_xlabel("rescaled\n" + stripe)
                else:
                    ax.set_xlabel("relative position, kbp\n" + stripe)
                ax.set_ylabel("")
                ax.set_title(col_val)
        elif nrows > 1 and ncols == 1:
            for row_val, ax in fg.axes_dict.items():
                ax.set_ylabel(row_val, rotation=0, ha="right")
                if pupsdf["rescale"].any():
                    ax.set_xlabel("rescaled\n" + stripe)
                else:
                    ax.set_xlabel("relative position, kbp\n" + stripe)
        else:
            plt.title("")
            plt.ylabel("")
            if pupsdf["rescale"].any():
                plt.xlabel("rescaled\n" + stripe)
            else:
                plt.xlabel("relative position, kbp\n" + stripe)

    plt.draw()
    ax_bottom = fg.axes[-1, -1]
    bottom = ax_bottom.get_position().y0
    ax_top = fg.axes[0, -1]
    top = ax_top.get_position().y1
    height = top - bottom
    right = ax_top.get_position().x1
    cax = fg.fig.add_axes([right + 0.01, bottom, (1 - right - 0.01) / 5, height])
    if sym:
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


def make_heatmap_grid(
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
    font=False,
    **kwargs,
):
    pupsdf = pupsdf.copy()

    cmap = copy.copy(cm.get_cmap(cmap))
    cmap.set_bad(cmap_emptypixel)

    if font:
        sns.set(font=font, style="white")

    if norm_corners:
        pupsdf["data"] = pupsdf.apply(
            lambda x: coolpup.norm_cis(x["data"], norm_corners), axis=1
        )
    if cols == "separation":
        col_order = sort_separation(pupsdf["separation"])
        ncols = len(col_order)
    elif cols is not None and col_order is None:
        col_order = list(set(pupsdf[cols].dropna()))
        #     pupsdf = pupsdf[pupsdf[cols].isin(col_order + ["data"])]
        ncols = len(col_order)
    elif col_order is not None:
        ncols = len(col_order)
    else:
        ncols = 1
        # colvals = ['']

    if rows == "separation":
        row_order = sort_separation(pupsdf["separation"])
        nrows = len(row_order)
    elif rows is not None and row_order is None:
        row_order = list(set(pupsdf[rows].dropna()))
        # else:
        #     pupsdf = pupsdf[pupsdf[rows].isin(row_order)]
        nrows = len(row_order)
    elif row_order is not None:
        nrows = len(row_order)
    else:
        nrows = 1
        # rowvals = ['']
    if cols is None and rows is None:
        nrows, ncols = auto_rows_cols(pupsdf.shape[0])

    vmin, vmax = get_min_max(pupsdf["data"].values, vmin, vmax, sym=sym)

    if scale == "log":
        norm = LogNorm
        if vmin == 0:
            vmin = 1e-6
    elif scale == "linear":
        norm = Normalize
    else:
        raise ValueError(
            f"Unknown scale value, only log or linear implemented, but got {scale}"
        )

    right = ncols / (ncols + 0.25)

    # sns.set(font_scale=5)
    fg = sns.FacetGrid(
        pupsdf,
        col=cols,
        row=rows,
        row_order=row_order,
        col_order=col_order,
        aspect=1,
        margin_titles=True,
        gridspec_kws={
            "right": right,
            "hspace": 0.05,
            "wspace": 0.05,
            #'top':0.95,
            #'bottom':0.05
        },
        height=height,
        **kwargs,
    )
    norm = norm(vmin, vmax)

    fg.map(add_heatmap, "data", norm=norm, cmap=cmap)

    if score:
        pupsdf["score"] = pupsdf.apply(
            coolpup.get_score, center=center, ignore_central=ignore_central, axis=1
        )
        fg.map(add_score, "score")

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
        elif nrows > 1 and ncols == 1:
            for row_val, ax in fg.axes_dict.items():
                ax.set_ylabel(row_val, rotation=0, ha="right")
        else:
            plt.xlabel("")

    plt.draw()
    ax_bottom = fg.axes[-1, -1]
    bottom = ax_bottom.get_position().y0
    ax_top = fg.axes[0, -1]
    top = ax_top.get_position().y1
    height = top - bottom
    right = ax_top.get_position().x1
    cax = fg.fig.add_axes([right + 0.005, bottom, (1 - right - 0.005) / 5, height])
    if sym:
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
