# -*- coding: utf-8 -*-

import numpy as np
from coolpuppy import coolpup
import matplotlib.pyplot as plt

# from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.colors import LogNorm, Normalize
from matplotlib import ticker
from matplotlib import cm
import seaborn as sns


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
    ax.imshow(data.values[0], cmap=cmap, norm=norm)  #


#     sns.heatmap(data.values[0], cmap=cmap, norm=norm, ax=ax, square=True, cbar=False,
#                xticklabels=False, yticklabels=False)


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


def make_heatmap_grid(
    pupsdf,
    cols=None,
    rows=None,
    score="score",
    col_order=None,
    row_order=None,
    vmin=None,
    vmax=None,
    sym=True,
    norm_corners=0,
    cmap="coolwarm",
    scale="log",
    height=1,
    **kwargs,
):
    pupsdf = pupsdf.copy()

    if norm_corners:
        pupsdf["data"] = pupsdf.apply(
            lambda x: coolpup.norm_cis(x["data"], norm_corners), axis=1
        )

    if cols is not None and col_order is None:
        col_order = list(set(pupsdf[cols].dropna()))
        #     pupsdf = pupsdf[pupsdf[cols].isin(col_order + ["data"])]
        ncols = len(col_order)
    elif col_order is not None:
        ncols = len(col_order)
    else:
        ncols = 1
        # colvals = ['']
    if rows is not None and row_order is None:
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

    if scale == "log":
        norm = LogNorm
    elif scale == "linear":
        norm = Normalize
    else:
        raise ValueError(
            f"Unknown scale value, only log or linear implemented, but got {scale}"
        )

    vmin, vmax = get_min_max(pupsdf["data"].values, vmin, vmax, sym=sym)

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
            pass
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