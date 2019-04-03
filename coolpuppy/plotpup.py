#!/usr/bin/env python

# -*- coding: utf-8 -*-

import numpy as np

def normCis(amap, i=3):
    return amap/np.nanmean((amap[0:i, 0:i]+amap[-i:, -i:]))*2

def get_enrichment(amap, n):
    c = int(np.floor(amap.shape[0]/2))
    return np.nanmean(amap[c-n//2:c+n//2+1, c-n//2:c+n//2+1])

def auto_rows_cols(n):
    rows = int(np.ceil(np.sqrt(n)))
    cols = int(np.ceil(n/rows))
    return rows, cols

def get_min_max(pups, vmin=None, vmax=None, sym=True):
    if vmin is not None and vmax is not None:
        return vmin, vmax
    else:
        comb = []
        for pup in pups:
            comb = np.append(comb, pup.flatten())
        comb = np.asarray(comb)
    if vmin is None and vmax is None:
        vmax = np.nanmax(comb)
        vmin = np.nanmin(comb)
    elif vmin is not None:
        vmax = np.nanmax(comb)
    elif vmax is not None:
        vmin = np.nanmin(comb)
    if sym:
        vmax = np.max(np.abs([vmin, vmax]))
        vmin = 2**-np.log2(vmax)
    return vmin, vmax