from .numutils import get_enrichment, get_domain_score, get_insulation_strength

import numpy as np
import pandas as pd
from more_itertools import collapse
import logging
import warnings

logger = logging.getLogger("coolpuppy")


def _add_snip(outdict, key, snip, extra_funcs=None):
    if key not in outdict:
        outdict[key] = {key: snip[key] for key in ["data", "cov_start", "cov_end"]}
        outdict[key]["coordinates"] = [snip["coordinates"]]
        outdict[key]["horizontal_stripe"] = [snip["horizontal_stripe"]]
        outdict[key]["vertical_stripe"] = [snip["vertical_stripe"]]
        outdict[key]["num"] = np.isfinite(snip["data"]).astype(int)
        outdict[key]["n"] = 1
    else:
        outdict[key]["data"] = np.nansum([outdict[key]["data"], snip["data"]], axis=0)
        outdict[key]["num"] += np.isfinite(snip["data"]).astype(int)
        outdict[key]["cov_start"] = np.nansum(
            [outdict[key]["cov_start"], snip["cov_start"]], axis=0
        )
        outdict[key]["cov_end"] = np.nansum(
            [outdict[key]["cov_end"], snip["cov_end"]], axis=0
        )
        outdict[key]["n"] += 1
        outdict[key]["horizontal_stripe"] = outdict[key]["horizontal_stripe"] + [
            snip["horizontal_stripe"]
        ]
        outdict[key]["vertical_stripe"] = outdict[key]["vertical_stripe"] + [
            snip["vertical_stripe"]
        ]
        outdict[key]["coordinates"] = outdict[key]["coordinates"] + [
            snip["coordinates"]
        ]
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
    from the two pups.
    """
    pup1["data"] = np.nan_to_num(pup1["data"])
    pup2["data"] = np.nan_to_num(pup2["data"])
    pup = {
        "data": pup1["data"] + pup2["data"],
        "cov_start": pup1["cov_start"] + pup2["cov_start"],
        "cov_end": pup1["cov_end"] + pup2["cov_end"],
        "n": pup1.get("n", 1) + pup2.get("n", 1),
        "num": pup1.get("num", np.isfinite(pup1["data"]).astype(int))
        + pup2.get("num", np.isfinite(pup2["data"]).astype(int)),
        "horizontal_stripe": pup1["horizontal_stripe"] + pup2["horizontal_stripe"],
        "vertical_stripe": pup1["vertical_stripe"] + pup2["vertical_stripe"],
        "coordinates": pup1["coordinates"] + pup2["coordinates"],
    }
    if extra_funcs:
        for key, func in extra_funcs.items():
            pup = func(pup1, pup2)
    return pd.Series(pup)


def divide_pups(pup1, pup2):
    """
    Divide two pups and get the resulting pup. Requires that the pups have identical shapes, resolutions, flanks, etc. 
    If pups contain stripes, these will only be divided if stripes have identical coordinates.
    Supports multiple conditions - will match conditions between pups and divide them pairwise.
    """
    drop_columns = [
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
    
    # Handle multiple conditions
    if pup1.shape[0] > 1 or pup2.shape[0] > 1:
        # Find columns that could be used to match conditions
        # Exclude data columns and columns that might differ between pups
        match_columns = []
        for col in pup1.columns:
            if col not in drop_columns and col in pup2.columns:
                try:
                    # Check if column values are hashable and can be used for matching
                    if hasattr(pup1[col].iloc[0], '__hash__'):
                        pup1[col].unique()
                        pup2[col].unique()
                        match_columns.append(col)
                except (TypeError, AttributeError):
                    continue
        
        # Try to find the best column(s) for matching conditions
        # Prefer columns like 'separation', 'distance_band', etc.
        preferred_match_cols = ['separation', 'distance_band', 'group']
        match_cols_to_use = []
        for col in preferred_match_cols:
            if col in match_columns:
                match_cols_to_use.append(col)
                break
        
        if not match_cols_to_use:
            # Use any available match columns
            match_cols_to_use = match_columns[:1] if match_columns else []
        
        if not match_cols_to_use:
            raise ValueError("Cannot find matching columns to align conditions between pileups")
        
        # Check if conditions match
        pup1_conditions = pup1[match_cols_to_use].drop_duplicates().sort_values(match_cols_to_use)
        pup2_conditions = pup2[match_cols_to_use].drop_duplicates().sort_values(match_cols_to_use)
        
        if not pup1_conditions.equals(pup2_conditions):
            raise ValueError(f"Conditions do not match between the two pileups. "
                           f"Pup1 has {len(pup1_conditions)} unique conditions, "
                           f"Pup2 has {len(pup2_conditions)} unique conditions based on {match_cols_to_use}")
        
        # Merge pups on matching columns to align conditions
        pup1_indexed = pup1.copy()
        pup2_indexed = pup2.copy()
        
        # Create temporary index for merging
        for col in match_cols_to_use:
            pup1_indexed[f'_match_{col}'] = pup1_indexed[col]
            pup2_indexed[f'_match_{col}'] = pup2_indexed[col]
        
        # Process each condition
        div_pup_list = []
        for _, condition in pup1_conditions.iterrows():
            # Get matching rows from both pups
            mask1 = True
            mask2 = True
            for col in match_cols_to_use:
                mask1 = mask1 & (pup1[col] == condition[col])
                mask2 = mask2 & (pup2[col] == condition[col])
            
            pup1_cond = pup1[mask1].reset_index(drop=True)
            pup2_cond = pup2[mask2].reset_index(drop=True)
            
            if len(pup1_cond) != 1 or len(pup2_cond) != 1:
                raise ValueError(f"Expected exactly one row per condition, but got {len(pup1_cond)} and {len(pup2_cond)}")
            
            # Divide this condition pair
            drop_columns_cond = list(set(drop_columns) & set(pup1_cond.columns))
            div_pup_cond = pup1_cond.drop(columns=drop_columns_cond)
            
            for col in div_pup_cond.columns:
                if col not in match_cols_to_use:
                    try:
                        if np.all(np.sort(pup1_cond[col]) != np.sort(pup2_cond[col])):
                            warnings.warn(f"Note that {col} is different between the two pileups for condition {condition}")
                    except:
                        pass
            
            div_pup_cond["data"] = pup1_cond["data"] / pup2_cond["data"]
            div_pup_cond["clrs"] = str(pup1_cond["clr"].iloc[0]) + "/" + str(pup2_cond["clr"].iloc[0])
            div_pup_cond["n"] = pup1_cond["n"] + pup2_cond["n"]
            
            if set(["vertical_stripe", "horizontal_stripe"]).issubset(pup1_cond.columns):
                if np.all(np.sort(pup1_cond["coordinates"].iloc[0]) == np.sort(pup2_cond["coordinates"].iloc[0])):
                    div_pup_cond["coordinates"] = pup1_cond["coordinates"]
                    for stripe in ["vertical_stripe", "horizontal_stripe"]:
                        div_pup_cond[stripe] = pup1_cond[stripe] / pup2_cond[stripe]
                        div_pup_cond[stripe] = div_pup_cond[stripe].apply(
                            lambda x: np.where(np.isin(x, [np.inf, np.nan]), 0, x)
                        )
                else:
                    logging.info(f"Stripes cannot be divided for condition {condition}, coordinates differ between pups")
            
            div_pup_list.append(div_pup_cond)
        
        # Combine all divided conditions
        div_pup = pd.concat(div_pup_list, ignore_index=True)
        return div_pup
    
    # Original single condition logic
    pup1 = pup1.reset_index(drop=True)
    pup2 = pup2.reset_index(drop=True)
    drop_columns = list(set(drop_columns) & set(pup1.columns))
    div_pup = pup1.drop(columns=drop_columns)
    for col in div_pup.columns:
        if np.all(np.sort(pup1[col]) != np.sort(pup2[col])):
            warnings.warn(f"Note that {col} is different between the two pileups")
    div_pup["data"] = pup1["data"] / pup2["data"]
    div_pup["clrs"] = str(pup1["clr"]) + "/" + str(pup2["clr"])
    div_pup["n"] = pup1["n"] + pup2["n"]
    if set(["vertical_stripe", "horizontal_stripe"]).issubset(pup1.columns):
        if np.all(np.sort(pup1["coordinates"]) == np.sort(pup2["coordinates"])):
            div_pup["coordinates"] = pup1["coordinates"]
            for stripe in ["vertical_stripe", "horizontal_stripe"]:
                div_pup[stripe] = pup1[stripe] / pup2[stripe]
                div_pup[stripe] = div_pup[stripe].apply(
                    lambda x: np.where(np.isin(x, [np.inf, np.nan]), 0, x)
                )
        else:
            logging.info("Stripes cannot be divided, coordinates differ between pups")
    return div_pup


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
    if band_edges == "default":
        band_edges = np.append([0], 50000 * 2 ** np.arange(30))
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
    assert key in dict2, f"{key} not in dict2"
    if key in dict1:
        dict1[key] = list(collapse([dict1[key], dict2[key]]))
    else:
        dict1[key] = [dict2[key]]
    return dict1
