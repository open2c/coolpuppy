from .numutils import get_enrichment, get_domain_score, get_insulation_strength

import numpy as np
import pandas as pd
from more_itertools import collapse
import logging
import warnings

logger = logging.getLogger("coolpuppy")


def _add_snip(outdict, key, snip, extra_funcs=None):
    data = snip["data"]
    isnan = np.isnan(data)
    if key not in outdict:
        outdict[key] = {k: snip[k] for k in ["cov_start", "cov_end"]}
        # Own copy with NaNs as 0, so it can be accumulated in place
        outdict[key]["data"] = np.where(isnan, 0.0, data)
        outdict[key]["coordinates"] = [snip["coordinates"]]
        outdict[key]["horizontal_stripe"] = [snip["horizontal_stripe"]]
        outdict[key]["vertical_stripe"] = [snip["vertical_stripe"]]
        outdict[key]["num"] = np.isfinite(data).astype(int)
        outdict[key]["n"] = 1
    else:
        acc = outdict[key]
        # Same result as np.nansum([acc, data], axis=0), without stacking arrays
        acc["data"] += np.where(isnan, 0.0, data)
        acc["num"] += np.isfinite(data)
        acc["cov_start"] = np.nansum([acc["cov_start"], snip["cov_start"]], axis=0)
        acc["cov_end"] = np.nansum([acc["cov_end"], snip["cov_end"]], axis=0)
        acc["n"] += 1
        acc["horizontal_stripe"].append(snip["horizontal_stripe"])
        acc["vertical_stripe"].append(snip["vertical_stripe"])
        acc["coordinates"].append(snip["coordinates"])
    if extra_funcs is not None:
        for key2, func in extra_funcs.items():
            outdict[key] = func(outdict[key], snip)


def get_score(pup, center=3, ignore_central=3):
    """Calculate a reasonable score for any kind of pileup
    For non-local (off-diagonal) pileups, calculates average signal in the central
    pixels (based on 'center').
    For local non-rescaled pileups calculates insulation strength, and ignores the
    central bins (based on 'ignore_central')
    For local rescaled pileups calculates enrichment in the central rescaled area
    relative to the two neighouring areas on the sides.

    Parameters
    ----------
    pup : pd.Series or dict
        Series or dict with pileup in 'data' and annotations in other keys.
        Will correctly calculate enrichment score with annotations in 'local' (book),
        'rescale' (bool) and 'rescale_flank' (float)
    enrichment : int, optional
        Passed to 'get_enrichment' to calculate the average strength of central pixels.
        The default is 3.
    ignore_central : int, optional
        How many central bins to ignore for calculation of insulation in local pileups.
        The default is 3.

    Returns
    -------
    float
        Score.

    """
    if not pup["local"]:
        logger.debug(f"Calculating enrichment for the central {center} pixels")
        return get_enrichment(pup["data"], center)
    else:
        if pup["rescale"]:
            logger.debug(
                "Calculating domain enrichment for the central rescaled domain versus surrounding"
            )
            return get_domain_score(pup["data"], pup["rescale_flank"])
        else:
            logger.debug(
                "Calculating insulation score, i.e., upper left and lower right corners over upper right and lower left corners"
            )
            return get_insulation_strength(pup["data"], ignore_central)


def sum_pups(pup1, pup2, extra_funcs={}):
    """
    Preserves data, stripes, cov_start, cov_end, n, num and coordinates
    Assumes n=1 if not present, and calculates num if not present
    If store_stripes is set to False, stripes and coordinates will be empty

    extra_funcs allows to give arbitrary functions to accumulate extra information
    from the two pups. Each function is called as ``func(pup1_copy, pup2)`` and
    should return the updated first argument (see `accumulate_values`); only the
    function's own key is taken from its result. The inputs are not modified.
    """
    data1 = np.nan_to_num(pup1["data"])
    data2 = np.nan_to_num(pup2["data"])
    pup = {
        "data": data1 + data2,
        "cov_start": pup1["cov_start"] + pup2["cov_start"],
        "cov_end": pup1["cov_end"] + pup2["cov_end"],
        "n": pup1.get("n", 1) + pup2.get("n", 1),
        "num": pup1.get("num", np.isfinite(pup1["data"]).astype(int))
        + pup2.get("num", np.isfinite(pup2["data"]).astype(int)),
        "horizontal_stripe": list(pup1["horizontal_stripe"])
        + list(pup2["horizontal_stripe"]),
        "vertical_stripe": list(pup1["vertical_stripe"]) + list(pup2["vertical_stripe"]),
        "coordinates": list(pup1["coordinates"]) + list(pup2["coordinates"]),
    }
    if extra_funcs:
        for key, func in extra_funcs.items():
            merged = func(dict(pup1), pup2)
            if key in merged:
                pup[key] = merged[key]
    return pd.Series(pup)


_DIVIDE_DROP_COLUMNS = [
    "control_n",
    "control_num",
    "n",
    "num",
    "clr",
    "chroms",
    "minshift",
    "expected_file",
    "group",
    "maxshift",
    "mindist",
    "maxdist",
    "subset",
    "seed",
    "data",
    "horizontal_stripe",
    "vertical_stripe",
    "cooler",
    "features",
    "outname",
    "coordinates",
]

# Columns that identify a row of a pileup dataframe, in order of preference
_DIVIDE_KEY_COLUMNS = [["group"], ["chrom", "start", "end"], ["separation"], ["orientation"]]


def _values_equal(a, b):
    try:
        if np.all(pd.isna(a)) and np.all(pd.isna(b)):
            return True
    except (TypeError, ValueError):
        pass
    try:
        return bool(np.all(np.asarray(a == b)))
    except Exception:
        return False


def _divide_pup_rows(row1, row2):
    """Divide one pileup (a row of a pileup dataframe, as a Series) by another."""
    drop = [c for c in _DIVIDE_DROP_COLUMNS if c in row1.index]
    out = row1.drop(labels=drop)
    for col in out.index:
        if col in row2.index and not _values_equal(row1[col], row2[col]):
            warnings.warn(f"Note that {col} is different between the two pileups")
    with np.errstate(divide="ignore", invalid="ignore"):
        out["data"] = row1["data"] / row2["data"]
    out["clrs"] = f"{row1.get('clr', '')}/{row2.get('clr', '')}"
    out["n"] = row1["n"] + row2["n"]
    if "group" in row1.index:
        out["group"] = row1["group"]
    stripes = ["vertical_stripe", "horizontal_stripe"]
    if all(s in row1.index for s in stripes) and all(s in row2.index for s in stripes):
        coords1 = np.asarray(row1["coordinates"])
        coords2 = np.asarray(row2["coordinates"])
        if coords1.shape == coords2.shape and np.array_equal(coords1, coords2):
            out["coordinates"] = row1["coordinates"]
            for stripe in stripes:
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = np.asarray(row1[stripe], dtype=float) / np.asarray(
                        row2[stripe], dtype=float
                    )
                out[stripe] = np.where(np.isfinite(ratio), ratio, 0)
        else:
            logger.info("Stripes cannot be divided, coordinates differ between pups")
            if "store_stripes" in out.index:
                out["store_stripes"] = False
    elif "store_stripes" in out.index:
        out["store_stripes"] = False
    return out


def _row_keys(pup, cols):
    return [
        tuple(str(v) for v in row) for row in pup[cols].itertuples(index=False, name=None)
    ]


def divide_pups(pup1, pup2):
    """
    Divide two pups and get the resulting pup. Requires that the pups have identical
    shapes, resolutions, flanks, etc. If pups contain stripes, these will only be
    divided if stripes have identical coordinates.

    Pileup dataframes with multiple rows (e.g. from `groupby`, `by_distance`,
    `by_strand` or `by_window`) are divided row by row, matching rows by their
    "group", "chrom"/"start"/"end" (by-window pileups), "separation" or
    "orientation" - whichever matches the most rows between the two pileups.
    """
    pup1 = pup1.reset_index(drop=True)
    pup2 = pup2.reset_index(drop=True)
    if pup1.shape[0] == 1 and pup2.shape[0] == 1:
        return pd.DataFrame([_divide_pup_rows(pup1.iloc[0], pup2.iloc[0])]).reset_index(
            drop=True
        )

    # Use the key columns that match the most rows between the two pileups (the
    # first candidate wins ties). E.g. ++ and -- pileups split by distance have
    # different "group" values but matching "separation" values.
    key_cols, best = None, 0
    for cols in _DIVIDE_KEY_COLUMNS:
        if not all(c in pup1.columns and c in pup2.columns for c in cols):
            continue
        overlap = len(set(_row_keys(pup1, cols)) & set(_row_keys(pup2, cols)))
        if key_cols is None or overlap > best:
            key_cols, best = cols, overlap
    if key_cols is None:
        raise ValueError(
            "Pileups with multiple rows can only be divided if they have matching "
            "'group' columns"
        )
    keys1 = _row_keys(pup1, key_cols)
    keys2 = _row_keys(pup2, key_cols)
    if len(set(keys1)) != len(keys1) or len(set(keys2)) != len(keys2):
        raise ValueError(
            f"Cannot match rows between pileups: {key_cols} values are not unique"
        )
    index2 = {key: i for i, key in enumerate(keys2)}
    unmatched = set(keys1) ^ set(keys2)
    if unmatched:
        warnings.warn(
            f"{len(unmatched)} groups are only present in one of the pileups and "
            "will be skipped"
        )
    rows = [
        _divide_pup_rows(pup1.iloc[i], pup2.iloc[index2[key]])
        for i, key in enumerate(keys1)
        if key in index2
    ]
    if not rows:
        raise ValueError("No matching groups between the two pileups")
    return pd.DataFrame(rows).reset_index(drop=True)


def norm_coverage(snip):
    """Normalize a pileup by coverage arrays

    Parameters
    ----------
    loop : 2D array
        Pileup.
    cov_start : 1D array
        Accumulated coverage of the left side of the pileup.
    cov_end : 1D array
        Accumulated coverage of the bottom side of the pileup.

    Returns
    -------
    loop : 2D array
        Normalized pileup.

    """
    coverage = np.outer(snip["cov_start"], snip["cov_end"])
    coverage = coverage / np.nanmean(coverage)
    snip["data"] /= coverage
    snip["data"][np.isnan(snip["data"])] = 0
    return snip


def bin_distance(snip, band_edges="default"):
    """


    Parameters
    ----------
    snip : pd.Series
        Series containing any annotations. Has to have ['distance']
    band_edges : list or array-like, or "default", optional
        Edges of distance bands used to assign the distance band.
        Default is np.append([0], 50000 * 2 ** np.arange(30))

    Returns
    -------
    snip : pd.Series
        The same snip with added ['distance_band'] annotation.

    """
    if isinstance(band_edges, str) and band_edges == "default":
        band_edges = np.append([0], 50000 * 2 ** np.arange(30))
    band_edges = np.asarray(band_edges).tolist()
    i = np.searchsorted(band_edges, snip["distance"])
    snip["distance_band"] = tuple(band_edges[i - 1 : i + 1])
    return snip


def group_by_region(snip):
    snip1 = snip.copy()
    snip1["group"] = tuple([snip1["chrom1"], snip1["start1"], snip1["end1"]])
    snip2 = snip.copy()
    snip2["group"] = tuple([snip2["chrom2"], snip2["start2"], snip2["end2"]])
    yield from (snip1, snip2)


def _combine_rows(row1, row2, normalize_order=True):
    """Deprecated, unused"""
    d = row2.center - row1.center
    if d < 0 and normalize_order:
        row1, row2 = row2, row1
        d = -d
    # row1.index = [i+'1' for i in row1.index]
    # row2.index = [i+'2' for i in row2.index]
    double_row = pd.Series(
        index=[i + "1" for i in row1.index]
        + [i + "2" for i in row2.index]
        + ["distance"],
        data=np.concatenate([row1.values, row2.values, [d]]),
    )
    # double_row['distance'] = d
    return double_row


def accumulate_values(dict1, dict2, key):
    """
    Useful as an extra_sum_func
    """
    if key not in dict2:
        # e.g. an empty pileup from a region without any valid snippets
        return dict1
    if key in dict1:
        dict1[key] = list(collapse([dict1[key], dict2[key]]))
    else:
        dict1[key] = list(collapse([dict2[key]]))
    return dict1
