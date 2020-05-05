# -*- coding: utf-8 -*-

import numpy as np


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
