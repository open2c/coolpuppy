# -*- coding: utf-8 -*-
import numpy as np
from cooltools import numutils as ctutils


def _copy_array_halves(x):
    cntr = int(np.floor(x.shape[1] / 2))
    x[:, : (cntr + 1)] = np.fliplr(x[:, cntr:])
    return x


def corner_cv(amap, i=4):
    """Get coefficient of variation for upper left and lower right corners of a pileup
    to estimate how noisy it is

    Parameters
    ----------
    amap : 2D array
        Pileup.
    i : int, optional
        How many bins to use from each upper left and lower right corner: final corner
        shape is i^2.
        The default is 4.

    Returns
    -------
    CV : float
        Coefficient of variation for the corner pixels.

    """
    corners = np.concatenate((amap[0:i, 0:i], amap[-i:, -i:]))
    corners = corners[np.isfinite(corners)]
    return np.std(corners) / np.mean(corners)


def norm_cis(amap, i=3):
    """Normalize the pileup by mean of pixels from upper left and lower right corners

    Parameters
    ----------
    amap : 2D array
        Pileup.
    i : int, optional
        How many bins to use from each upper left and lower right corner: final corner
        shape is i^2. 0 will not normalize.
        The default is 3.

    Returns
    -------
    amap : 2D array
        Normalized pileup.

    """
    if i > 0:
        return amap / np.nanmean((amap[0:i, 0:i] + amap[-i:, -i:])) * 2
    else:
        return amap


def get_enrichment(amap, n):
    """Get values from the center of a pileup for a square with side *n*

    Parameters
    ----------
    amap : 2D array
        Pileup.
    n : int
        Side of the central square to use.

    Returns
    -------
    enrichment : float
        Mean of the pixels in the central square.

    """
    c = amap.shape[0] // 2
    return np.nanmean(amap[c - n // 2 : c + n // 2 + 1, c - n // 2 : c + n // 2 + 1])


def get_local_enrichment(amap, flank=1):
    """Get values for a square from the central part of a pileup, ignoring padding

    Parameters
    ----------
    amap : 2D array
        Pileup.
    flank : int
        Relative padding used, i.e. if 1 the central third is used, if 2 the central
        fifth is used.
        The default is 1.

    Returns
    -------
    enrichment : float
        Mean of the pixels in the central square.

    """
    c = amap.shape[0] / (flank * 2 + 1)
    assert int(c) == c
    c = int(c)
    return np.nanmean(amap[c:-c, c:-c])


def get_domain_score(amap, flank=1):
    """Divide sum of values in a square from the central part of a matrix by the upper
    and right rectangles corresponding to interactions of the central region with
    its surroundings.

    Parameters
    ----------
    amap : 2D array
        Pileup.
    flank : int
        Relative padding used, i.e. if 1 the central third is used, if 2 the central
        fifth is used.
        The default is 1.

    Returns
    -------
    score : float
        Domain score.

    """
    c = amap.shape[0] / (flank * 2 + 1)
    assert int(c) == c
    c = int(c)
    central = np.nansum(amap[c:-c, c:-c])
    top = np.nansum(amap[:c, c:-c])
    right = np.nansum(amap[c:-c, -c:])
    return central / (top + right) * 2


def get_insulation_strength(amap, ignore_central=0, ignore_diags=2):
    """Divide values in upper left and lower right corners over upper right and lower
    left, ignoring the central bins.

    Parameters
    ----------
    amap : 2D array
        Pileup.
    ignore_central : int, optional
        How many central bins to ignore. Has to be odd or 0. The default is 0.

    Returns
    -------
    float
        Insulation strength.

    """
    for d in range(ignore_diags):
        amap = ctutils.fill_diag(amap, np.nan, d)
        if d != 0:
            amap = ctutils.fill_diag(amap, np.nan, -d)
    if ignore_central != 0 and ignore_central % 2 != 1:
        raise ValueError(f"ignore_central has to be odd (or 0), got {ignore_central}")
    i = (amap.shape[0] - ignore_central) // 2
    intra = np.nanmean(np.concatenate([amap[:i, :i].ravel(), amap[-i:, -i:].ravel()]))
    inter = np.nanmean(np.concatenate([amap[:i, -i:].ravel(), amap[-i:, :i].ravel()]))
    return intra / inter


def _prepare_single(item):
    """(Deprecated) Generate enrichment and corner CV, reformat into a list

    Parameters
    ----------
    item : tuple
        Key, (n, pileup).

    Returns
    -------
    list
        Concatenated list of key, n, enrichment1, enrichment3, cv3, cv5.

    """
    key, (n, amap) = item
    enr1 = get_enrichment(amap, 1)
    enr3 = get_enrichment(amap, 3)
    cv3 = corner_cv(amap, 3)
    cv5 = corner_cv(amap, 5)
    return list(key) + [n, enr1, enr3, cv3, cv5]
