# -*- coding: utf-8 -*-
import numpy as np
from scipy import sparse
import pandas as pd
import h5py
import re
import os
import yaml
import io
import gzip
import csv
import logging
from coolpuppy._version import __version__

logger = logging.getLogger("coolpuppy")


def save_pileup_df(filename, df, metadata=None, mode="w", compression="lzf"):
    """
    Saves a dataframe with metadata into a binary HDF5 file`

    Parameters
    ----------
    filename : str
        File to save to.
    df : pd.DataFrame
        DataFrame to save into binary hdf5 file.
    metadata : dict, optional
        Dictionary with meatadata.
    mode : str, optional
        Mode for the first time access to the output file: 'w' to overwrite if file
        exists, or 'a' to fail if output file already exists
    compression : str, optional
        Compression to use for saving, e.g. 'gzip'. Defaults to 'lzf'

    Returns
    -------
    None.

    Notes
    -----
    Replaces `None` in metadata values with `False`, since HDF5 doesn't support `None`

    """
    if metadata is None:
        metadata = {}
    if mode == "a" and os.path.exists(filename):
        # Appending to an existing file can't be made atomic
        _write_pileup_df(filename, df, metadata, mode, compression)
        return
    # Write to a temporary file next to the target and move it into place only once
    # it's complete, so an interrupted save never leaves a partial .clpy file
    # (which later fails to load with "object 'attrs' doesn't exist")
    tmp = f"{filename}.tmp{os.getpid()}"
    try:
        _write_pileup_df(tmp, df, metadata, "w", compression)
        os.replace(tmp, filename)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return


def _write_pileup_df(filename, df, metadata, mode, compression):
    df[
        df.columns[
            ~df.columns.isin(
                ["data", "vertical_stripe", "horizontal_stripe", "coordinates"]
            )
        ]
    ].to_hdf(filename, key="annotation", mode=mode)

    with h5py.File(filename, "a") as f:
        width = df["data"].iloc[0].shape[0]
        height = width * df["data"].shape[0]
        ds = f.create_dataset(
            "data",
            dtype="f4",  # h5py's former implicit default, kept for compatibility
            compression=compression,
            chunks=(width, width),
            shape=(height, width),
        )
        for i, arr in df["data"].reset_index(drop=True).items():
            ds[i * width : (i + 1) * width, :] = arr
        has_stripes = all(
            col in df.columns
            for col in ["vertical_stripe", "horizontal_stripe", "coordinates"]
        )
        if has_stripes and "store_stripes" in df.columns and df["store_stripes"].any():
            for i, arr in df["vertical_stripe"].reset_index(drop=True).items():
                _write_sparse(
                    f, "vertical_stripe_" + str(i), sparse.csr_matrix(arr), compression
                )
            for i, arr in df["horizontal_stripe"].reset_index(drop=True).items():
                _write_sparse(
                    f, "horizontal_stripe_" + str(i), sparse.csr_matrix(arr), compression
                )
            for i, arr in df["coordinates"].reset_index(drop=True).items():
                f.create_dataset(
                    "coordinates_" + str(i),
                    compression=compression,
                    shape=(len(arr), 6),
                    data=arr.astype(object),
                )
        group = f.create_group("attrs")
        if metadata is not None:
            for key, val in metadata.items():
                if val is None:
                    val = False
                group.attrs[key] = val
        group.attrs["version"] = __version__
    return


# Sparse matrices are stored in the layout used by the (unmaintained) h5sparse
# package, so files stay readable by older coolpuppy versions and vice versa:
# a group with attributes h5sparse_format/h5sparse_shape and datasets
# data/indices/indptr.
_SPARSE_FORMATS = {"csr": sparse.csr_matrix, "csc": sparse.csc_matrix}


def _write_sparse(parent, name, mat, compression=None):
    group = parent.create_group(name)
    group.attrs["h5sparse_format"] = "csr"
    group.attrs["h5sparse_shape"] = mat.shape
    group.create_dataset("data", data=mat.data, compression=compression)
    group.create_dataset(
        "indices", data=mat.indices.astype(np.int32), compression=compression
    )
    group.create_dataset(
        "indptr", data=mat.indptr.astype(np.int64), compression=compression
    )
    return group


def _read_sparse(group):
    fmt = group.attrs["h5sparse_format"]
    if isinstance(fmt, bytes):
        fmt = fmt.decode()
    return _SPARSE_FORMATS[fmt](
        (group["data"][()], group["indices"][()], group["indptr"][()]),
        shape=tuple(group.attrs["h5sparse_shape"]),
    )


def _decode_strings(arr):
    """Decode an array of bytes/str objects to unicode without truncation."""
    arr = np.asarray(arr)
    if arr.dtype.kind in ("S", "O"):
        return np.vectorize(
            lambda x: x.decode() if isinstance(x, bytes) else str(x), otypes=[str]
        )(arr) if arr.size else arr.astype(str)
    return arr.astype(str)


def load_pileup_df(filename, quaich=False, skipstripes=False):
    """
    Loads a dataframe saved using `save_pileup_df`

    Parameters
    ----------
    filename : str
        File to load from.
    quaich : bool, optional
        Whether to assume standard quaich file naming to extract sample name and bedname.
        The default is False.

    Returns
    -------
    annotation : pd.DataFrame
        Pileups are in the "data" column, all metadata in other columns

    """
    with h5py.File(filename, "r", libver="latest") as f:
        if "data" not in f or "annotation" not in f:
            raise ValueError(
                f"{filename} is incomplete (missing 'data' or 'annotation'), probably "
                "because writing it was interrupted. Please regenerate it."
            )
        if "attrs" in f:
            metadata = dict(zip(f["attrs"].attrs.keys(), f["attrs"].attrs.values()))
        else:
            logger.warning(
                f"{filename} has no metadata ('attrs'), probably because writing it was "
                "interrupted; consider regenerating it"
            )
            metadata = {}
        dstore = f["data"]
        data = []
        for chunk in dstore.iter_chunks():
            chunk = dstore[chunk]
            data.append(chunk)
        annotation = pd.read_hdf(filename, "annotation")
        annotation["data"] = data
        vertical_stripe = []
        horizontal_stripe = []
        coordinates = []
        if not skipstripes:
            try:
                for i in range(len(data)):
                    vstripe = "vertical_stripe_" + str(i)
                    hstripe = "horizontal_stripe_" + str(i)
                    coords = "coordinates_" + str(i)
                    vertical_stripe.append(_read_sparse(f[vstripe]).toarray())
                    horizontal_stripe.append(_read_sparse(f[hstripe]).toarray())
                    coordinates.append(_decode_strings(f[coords][:]))
                annotation["vertical_stripe"] = vertical_stripe
                annotation["horizontal_stripe"] = horizontal_stripe
                annotation["coordinates"] = coordinates
            except KeyError:
                pass
    for key, val in metadata.items():
        if key != "version":
            annotation[key] = val
        elif val != __version__:
            logger.debug(
                f"pileup generated with v{val}. Current version is v{__version__}"
            )
    if quaich:
        basename = os.path.basename(filename)
        sample, bedname = re.search(
            r"^(.*)-(?:[0-9]+)_over_(.*)_(?:[0-9]+-shifts|expected).*\.clpy", basename
        ).groups()
        annotation["sample"] = sample
        annotation["bedname"] = bedname
    return annotation


def load_pileup_df_list(files, quaich=False, nice_metadata=True, skipstripes=False):
    """

    Parameters
    ----------
    files : iterable
        Files to read pileups from.
    quaich : bool, optional
        Whether to assume standard quaich file naming to extract sample name and bedname.
        The default is False.
    nice_metadata : bool, optional
        Whether to add nicer metadata for direct plotting. The default is True.
        Adds a "norm" column ("expected", "shifts" or "none").


    Returns
    -------
    pups : pd.DataFrame
        Combined dataframe with all pileups and annotations from all files.

    """
    pups = pd.concat(
        [load_pileup_df(path, quaich=quaich, skipstripes=skipstripes) for path in files]
    ).reset_index(drop=True)
    if nice_metadata:
        pups["norm"] = np.where(
            pups["expected"], ["expected"] * pups.shape[0], ["shifts"] * pups.shape[0]
        ).astype(str)
        pups.loc[
            np.logical_not(np.logical_or(pups["nshifts"] > 0, pups["expected"])), "norm"
        ] = "none"
    return pups


def save_array_with_header(array, header, filename):
    """Save a numpy array with a YAML header generated from a dictionary

    Parameters
    ----------
    array : np.array
        Array to save.
    header : dict
        Dictionaty to save into the header.
    filename : string
        Name of file to save array and metadata into.

    """
    header = yaml.dump(header).strip()
    np.savetxt(filename, array, header=header)


def load_array_with_header(filename):
    """Load array from files generated using `save_array_with_header`.
    They are simple txt files with an optional header in the first lines, commented
    using "# ". If uncommented, the header is in YAML.

    Parameters
    ----------
    filename : string
        File to load from.

    Returns
    -------
    data : dict
        Dictionary with information from the header. Access the associated data in an
        array using data['data'].

    """
    with open(filename) as f:
        read_data = f.read()

    lines = read_data.split("\n")
    header = "\n".join([line[2:] for line in lines if line.startswith("# ")])
    if len(header) > 0:
        metadata = yaml.load(header, Loader=yaml.FullLoader)
    else:
        metadata = {}
    data = "\n".join([line for line in lines if not line.startswith("# ")])
    with io.StringIO(data) as f:
        metadata["data"] = np.loadtxt(f)
    return metadata


def is_gz_file(filepath):
    with open(filepath, "rb") as test_f:
        return test_f.read(2) == b"\x1f\x8b"


def sniff_for_header(file, sep="\t", comment="#"):
    """
    Warning: reads the entire file into a StringIO buffer!
    """
    if isinstance(file, str):
        if is_gz_file(file):
            with gzip.open(file, "rt") as f:
                buf = io.StringIO(f.read())
        else:
            with open(file, "r") as f:
                buf = io.StringIO(f.read())
    else:
        buf = io.StringIO(file.read())

    sample_lines = []
    for line in buf:
        if not line.startswith(comment):
            sample_lines.append(line)
            break
    for _ in range(10):
        sample_lines.append(buf.readline())
    buf.seek(0)

    has_header = csv.Sniffer().has_header("\n".join(sample_lines))
    if has_header:
        names = sample_lines[0].strip().split(sep)
    else:
        names = None
    
    ncols = len(sample_lines[0].strip().split(sep))

    return buf, names, ncols
