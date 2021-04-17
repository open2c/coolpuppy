# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import coolpuppy
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.colors import LogNorm, Normalize


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


def make_heatmap_grid(pupsdf, cols=None, rows=None, score='score',
                      col_order=None, row_order=None,
                      vmin=None, vmax=None, sym=True, cbar_mode='single',
                      norm_corners=0,
                      cmap='coolwarm', scale='log', dpi=150):
    pupsdf = pupsdf.copy()
    
    if norm_corners:
        pupsdf['data'] = pupsdf.apply(lambda x: coolpuppy.norm_cis(x['data'], norm_corners), axis=1)
    
    if cols is not None:
        if col_order is None:
            colvals = list(set(pupsdf[cols].dropna()))
        else:
            colvals = col_order
            pupsdf = pupsdf[pupsdf[cols].isin(colvals+['data'])]
        ncols = len(colvals)
    else:
        ncols = 1
        # colvals = ['']
    if rows is not None:
        if row_order is None:
            rowvals = list(set(pupsdf[rows].dropna()))
        else:
            rowvals = row_order
            pupsdf = pupsdf[pupsdf[rows].isin(rowvals+['data'])]
        nrows = len(rowvals)
    else:
        nrows = 1
        # rowvals = ['']
    if cols is None and rows is None:
        nrows, ncols = auto_rows_cols(pupsdf.shape[0])    
    elif cols is not None and rows is not None:
        pupsdf = pd.pivot(pupsdf, columns=cols, index=rows, values='data')
    elif rows is None:
        pupsdf['index'] = '0'
        rowvals = ['0']
        pupsdf = pd.pivot(pupsdf, columns=cols, index='index', values='data')
    elif cols is None:
        pupsdf['cols'] = '0'
        colvals = ['0']
        pupsdf = pd.pivot(pupsdf, columns='cols', index=rows, values='data')
#         pupsdf = pupsdf[['data', score]]
    
    if scale == "log":
        norm = LogNorm
    elif scale == 'linear':
        norm = Normalize
    else:
        raise ValueError(f'Unknown scale value, only log or linear implemented, but got {scale}')
    
    f = plt.figure(dpi=dpi, figsize=(max(3.5, ncols + 0.5), max(3, nrows)))
    grid = ImageGrid(
        f,
        111,
        share_all=True,
        nrows_ncols=(nrows, ncols),
        axes_pad=0.05,
        label_mode="L",
        cbar_location="right",
        cbar_mode=cbar_mode,
        cbar_size="5%",
        cbar_pad="3%",
    )
    axarr = np.array(grid).reshape((nrows, ncols))
    
    if cbar_mode == "single":
        vmin, vmax = coolpuppy.get_min_max(pupsdf.values, vmin, vmax, sym=sym)
    elif cbar_mode == "edge":
        colorscales = {rowname:coolpuppy.get_min_max(row.values, vmin, vmax, sym=sym) for rowname, row in pupsdf.groupby(rows)}
    elif cbar_mode == "each":
        grid.cbar_axes = np.asarray(grid.cbar_axes).reshape((nrows, ncols))
    
    cbs = []
    
    for rowid, rowname in enumerate(rowvals):
        # axarr[rowid, 0].set_ylabel(rowname)
        if cbar_mode == "edge":
            vmin, vmax = colorscales[rowname]
        for colid, colname in enumerate(colvals):
            ax = axarr[rowid, colid]
            try:
                mat = pupsdf.loc[rowname, colname]
                if cbar_mode == 'each':
                    vmin, vmax = coolpuppy.get_min_max([mat], vmin, vmax, sym=sym)
                im = ax.imshow(mat,
                          interpolation="none",
                          norm=norm(vmax=vmax, vmin=vmin),
                          cmap=cmap,
                          extent=(0, 1, 0, 1)
                          )
                if score:
                    enr = pupsdf.loc[rowname, colname][score]
                    ax.text(
                        s=f"{enr:.3g}",
                        y=0.95,
                        x=0.05,
                        ha="left",
                        va="top",
                        size="x-small",
                        transform=ax.transAxes,
                    )
                if cbar_mode == 'edge':
                    cbs.append(plt.colorbar(im, cax=grid.cbar_axes[rowid]))
                elif cbar_mode == 'each':
                    cbs.append(plt.colorbar(im, cax=grid.cbar_axes[rowid, colid]))
                ax.set_xticks([])
                ax.set_yticks([])
            except KeyError:
                ax.axis("off")
                if cbar_mode == 'each':
                    grid.cbar_axes[rowid, colid].axis("off")
    if cbar_mode == "single":
        cbs.append(
            plt.colorbar(im, cax=grid.cbar_axes[0])
        )
    if rows:
        for rowid, rowname in enumerate(rowvals):
            axarr[rowid, 0].set_ylabel(rowname.replace('_', '\n'), rotation=0, ha='right', va='center')
    if cols:
        for colid, colname in enumerate(colvals):
            axarr[-1, colid].set_xlabel(colname.replace('_', '\n'))
    return f, axarr, cbs