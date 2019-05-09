# coolpup.py
[![DOI](https://zenodo.org/badge/147190130.svg)](https://zenodo.org/badge/latestdoi/147190130)

.**cool** file **p**ile-**up**s with **py**thon.

# Introduction

A versatile tool to perform pile-up analysis on Hi-C data in .cool format (https://github.com/mirnylab/cooler). And who doesn't like cool pupppies?

.cool is a modern and flexible (and the best, in my opinion) format to store Hi-C data. 
It uses HDF5 to store sparse a representation of the Hi-C data, which allows low memory requirements when dealing with high resolution datasets. Another popular format to store Hi-C data, .hic, can be converted into .cool files using `hic2cool` (https://github.com/4dn-dcic/hic2cool).

See for details:

Abdennur, N., and Mirny, L. (2019). Cooler: scalable storage for Hi-C data and other genomically-labeled arrays. BioRxiv, 557660. doi: [10.1101/557660](https://doi.org/10.1101/557660)

## What are pileups?

This is the idea of how pileups work to check whether certain regions tend to interacts with each other:

<img src="https://github.com/Phlya/coolpuppy/blob/master/loop_quant.svg" width="800">

What's not shown here is normalization to the expected values. This can be done in two ways: either using a provided file with expected values of interactions at different distances (output of `cooltools compute-expected`), or directly from Hi-C data by dividing the pileups over randomly shifted control regions. If neither expected normalization approach is used (just set `--nshifts 0`), this becomes essentially identical to the APA approach (Rao et al., 2014), which can be used for averaging strongly interacting regions, e.g. annotated loops. For weaker interactors, decay of contact probability with distance will hide any focal enrichment that could be observed otherwise.

`coolpup.py` is particularly well suited performance-wise for analysing huge numbers of potential interactions, since it loads whole chromosomes into memory one by one (or in parallel to speed it up) to extract small submatrices quickly. Having to read everything into memory makes it relatively slow for small numbers of loops, but performance doesn't decrease until you reach a huge number of interactions.

# Getting started

## Installation
All requirements apart from `cooltools` are available from PyPI or conda. For `cooltools`, do

`
pip install https://github.com/mirnylab/cooltools/archive/master.zip
`

For coolpuppy (and other dependencies) simply do:

`pip install coolpuppy`

or

`pip install https://github.com/Phlya/coolpuppy/archive/master.zip`

to get the latest version from GitHub. This will make `coolpup.py` callable in your terminal, and importable in python as `coolpuppy`.

## Usage

Help message should help you get started to use the tool. It is a single command that has a lot of options and can do a lot of things!

```
Usage: coolpup.py [-h] [--pad PAD] [--minshift MINSHIFT] [--maxshift MAXSHIFT]
                  [--nshifts NSHIFTS] [--expected EXPECTED]
                  [--mindist MINDIST] [--maxdist MAXDIST] [--minsize MINSIZE]
                  [--maxsize MAXSIZE] [--excl_chrs EXCL_CHRS]
                  [--incl_chrs INCL_CHRS] [--subset SUBSET] [--anchor ANCHOR]
                  [--by_window] [--save_all] [--local] [--unbalanced]
                  [--coverage_norm] [--rescale] [--rescale_pad RESCALE_PAD]
                  [--rescale_size RESCALE_SIZE] [--weight_name WEIGHT_NAME]
                  [--n_proc N_PROC] [--outdir OUTDIR] [--outname OUTNAME]
                  [-l {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
                  coolfile baselist

positional arguments:
  coolfile              Cooler file with your Hi-C data
  baselist              A 3-column bed file or a 6-column double-bed file
                        (i.e. chr1,start1,end1,chr2,start2,end2). Should be
                        tab-delimited. With a bed file, will consider all cis
                        combinations of intervals. To pileup features along
                        the diagonal instead, use the --local argument. Can be
                        piped in via stdin, then use "-".

optional arguments:
  -h, --help            show this help message and exit
  --pad PAD             Padding of the windows around the centres of specified
                        features (i.e. final size of the matrix is 2×pad+res),
                        in kb. Ignored with --rescale, use --rescale_pad
                        instead. (default: 100)
  --minshift MINSHIFT   Shortest distance for randomly shifting coordinates
                        when creating controls (default: 100000)
  --maxshift MAXSHIFT   Longest distance for randomly shifting coordinates
                        when creating controls (default: 1000000)
  --nshifts NSHIFTS     Number of control regions per averaged window
                        (default: 10)
  --expected EXPECTED   File with expected (output of cooltools compute-
                        expected). If None, don't use expected and use
                        randomly shifted controls (default: None)
  --mindist MINDIST     Minimal distance of intersections to use. If not
                        specified, uses 2*pad+2 (in bins) as mindist (default:
                        None)
  --maxdist MAXDIST     Maximal distance of intersections to use (default:
                        None)
  --minsize MINSIZE     Minimal length of features to use for local analysis
                        (default: None)
  --maxsize MAXSIZE     Maximal length of features to use for local analysis
                        (default: None)
  --excl_chrs EXCL_CHRS
                        Exclude these chromosomes from analysis (default:
                        chrY,chrM)
  --incl_chrs INCL_CHRS
                        Include these chromosomes; default is all. excl_chrs
                        overrides this. (default: all)
  --subset SUBSET       Take a random sample of the bed file - useful for
                        files with too many featuers to run as is, i.e. some
                        repetitive elements. Set to 0 or lower to keep all
                        data. (default: 0)
  --anchor ANCHOR       A UCSC-style coordinate to use as an anchor to create
                        intersections with coordinates in the baselist
                        (default: None)
  --by_window           Create a pile-up for each coordinate in the baselist.
                        Will save a master-table with coordinates, their
                        enrichments and cornerCV, which is reflective of
                        noisiness (default: False)
  --save_all            If by-window, save all individual pile-ups in a
                        separate json file (default: False)
  --local               Create local pileups, i.e. along the diagonal
                        (default: False)
  --unbalanced          Do not use balanced data. Useful for single-cell Hi-C
                        data together with --coverage_norm, not recommended
                        otherwise. (default: False)
  --coverage_norm       If --unbalanced, also add coverage normalization based
                        on chromosome marginals (default: False)
  --rescale             Do not use centres of features and pad, and rather use
                        the actual feature sizes and rescale pileups to the
                        same shape and size (default: False)
  --rescale_pad RESCALE_PAD
                        If --rescale, padding in fraction of feature length
                        (default: 1.0)
  --rescale_size RESCALE_SIZE
                        If --rescale, this is used to determine the final size
                        of the pileup, i.e. it will be size×size. Due to
                        technical limitation in the current implementation,
                        has to be an odd number (default: 99)
  --weight_name WEIGHT_NAME
                        Name of the norm to use for getting balanced data
                        (default: weight)
  --n_proc N_PROC       Number of processes to use. Each process works on a
                        separate chromosome, so might require quite a bit more
                        memory, although the data are always stored as sparse
                        matrices (default: 1)
  --outdir OUTDIR       Directory to save the data in (default: .)
  --outname OUTNAME     Name of the output file. If not set, is generated
                        automatically to include important information.
                        (default: auto)
  -l {DEBUG,INFO,WARNING,ERROR,CRITICAL}, --log {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        Set the logging level. (default: INFO)

```

Currently, `coolpup.py` doesn't support inter-chromosomal pileups, but this is an addition that is planned for the future.

### Plotting results
For flexible plotting, I suggest to use `matplotlib`. However simple plotting capabilities are included in this package. Just run `plotpup.py` with desired options and list all the output files of `coolpup.py` you'd like to plot.
```
Usage: plotpup.py [-h] [--cmap CMAP] [--symmetric SYMMETRIC] [--vmin VMIN]
                  [--vmax VMAX] [--scale {log,linear}] [--n_cols N_COLS]
                  [--col_names COL_NAMES] [--row_names ROW_NAMES]
                  [--output OUTPUT]
                  [pileup_files [pileup_files ...]]

positional arguments:
  pileup_files          All files to plot (default: None)

optional arguments:
  -h, --help            show this help message and exit
  --cmap CMAP           Colourmap to use (see
                        https://matplotlib.org/users/colormaps.html) (default:
                        coolwarm)
  --symmetric SYMMETRIC
                        Whether to make colormap symmetric around 1, if log
                        scale (default: True)
  --vmin VMIN           Value for the lowest colour (default: None)
  --vmax VMAX           Value for the highest colour (default: None)
  --scale {log,linear}  Whether to use linear or log scaling for mapping
                        colours (default: log)
  --n_cols N_COLS       How many columns to use for plotting the data. If 0,
                        automatically make the figure as square as possible
                        (default: 0)
  --col_names COL_NAMES
                        A comma separated list of column names (default: None)
  --row_names ROW_NAMES
                        A comma separated list of row names (default: None)
  --output OUTPUT       Where to save the plot (default: pup.pdf)
  ```

## Citing coolpup.py

Until it has been published in a peer-reviewed journal, please cite our preprint

**Coolpup.py - a versatile tool to perform pile-up analysis of Hi-C data**

Ilya M. Flyamer, Robert S. Illingworth, Wendy A. Bickmore

https://www.biorxiv.org/content/10.1101/586537v1

## This tool has been used in the following publications

**DNA methylation directs polycomb-dependent 3D genome re- organisation in naive pluripotency**

Katy A McLaughlin, Ilya M Flyamer, John P Thomson, Heidi K Mjoseng, Ruchi Shukla, Iain Williamson, Graeme R Grimes, Robert S Illingworth, Ian R Adams, Sari Pennings, Richard R Meehan, Wendy A Bickmore

https://www.biorxiv.org/content/10.1101/527309v1
