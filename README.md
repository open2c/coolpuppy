# coolpup.py
[![DOI](https://zenodo.org/badge/147190130.svg)](https://zenodo.org/badge/latestdoi/147190130)

.**cool** file **p**ile-**up**s with **py**thon.

# Introduction

A versatile tool to perform pile-up analysis on Hi-C data in .cool format (https://github.com/mirnylab/cooler). And who doesn't like cool pupppies?

.cool is a modern and flexible (and the best, in my opinion) format to store Hi-C data. It uses HDF5 to store sparse a representation of the Hi-C data, which allows low memory requirements when dealing with high resolution datasets. Another popular format to store Hi-C data, .hic, can be converted into .cool files using `hic2cool` (https://github.com/4dn-dcic/hic2cool).

## What are pileups?

This is the idea of how pileups work to check whether certain regions tend to interacts with each other:

<img src="https://github.com/Phlya/coolpuppy/blob/master/loop_quant.svg" width="400">

What's not shown here is normalization to the expected values. This can be done in two ways: either using a provided file with expected values of interactions at different distances (output of `cooltools compute-expected`), or directly from Hi-C data by dividing the pileups over randomly shifted control regions. If neither expected normalization approach is used (just set `--nshifts 0`), this becomes essentially identical to the APA approach (Rao et al., 2014), which can be used for averaging strongly interacting regions, e.g. annotated loops. For weaker interactors, decay of contact probability with distance will hide any focal enrichment that could be observed otherwise.

`coolpup.py` is particularly well suited performance-wise for analysing huge numbers of potential interactions, since it loads whole chromosomes into memory one by one (or in parallel to speed it up) to extract small submatrices quickly. Having to read everything into memory makes it relatively slow for small numbers of loops, but performance doesn't decrease until you reach a huge number of interactions.

# Getting started

## Installation

`pip install coolpuppy` (or `pip install https://github.com/Phlya/coolpuppy/archive/master.zip` to get the latest version from GitHub) , will make `coolpup.py` callable in your terminal, and importable in python as `coolpup`.

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
                  [--rescale_size RESCALE_SIZE] [--n_proc N_PROC]
                  [--outdir OUTDIR] [--outname OUTNAME]
                  coolfile baselist

positional arguments:
  coolfile              Cooler file with your Hi-C data
  baselist              A 3-column tab-delimited bed file with coordinates
                        which intersections to pile-up. Alternatively, a
                        6-column double-bed file (i.e.
                        chr1,start1,end1,chr2,start2,end2) with coordinates of
                        centers of windows that will be piled-up

optional arguments:
  -h, --help            show this help message and exit
  --pad PAD             Padding of the windows (i.e. final size of the matrix
                        is 2×pad+res), in kb
  --minshift MINSHIFT   Shortest distance for randomly shifting coordinates
                        when creating controls
  --maxshift MAXSHIFT   Longest distance for randomly shifting coordinates
                        when creating controls
  --nshifts NSHIFTS     Number of control regions per averaged window
  --expected EXPECTED   File with expected (output of cooltools compute-
                        expected). If None, don't use expected and use
                        randomly shifted controls
  --mindist MINDIST     Minimal distance of intersections to use. If not
                        specified, uses --pad as mindist
  --maxdist MAXDIST     Maximal distance of intersections to use
  --minsize MINSIZE     Minimal length of features to use for local analysis
  --maxsize MAXSIZE     Maximal length of features to use for local analysis
  --excl_chrs EXCL_CHRS
                        Exclude these chromosomes form analysis
  --incl_chrs INCL_CHRS
                        Include these chromosomes; default is all. excl_chrs
                        overrides this.
  --subset SUBSET       Take a random sample of the bed file - useful for
                        files with too many featuers to run as is, i.e. some
                        repetitive elements. Set to 0 or lower to keep all
                        data.
  --anchor ANCHOR       A UCSC-style coordinate to use as an anchor to create
                        intersections with coordinates in the baselist
  --by_window           Create a pile-up for each coordinate in the baselist
  --save_all            If by-window, save all individual pile-ups as separate
                        text files. Can create a very large number of files,
                        so use cautiosly! If not used, will save a master-
                        table with coordinates, their enrichments and
                        cornerCV, which is reflective of noisiness
  --local               Create local pileups, i.e. along the diagonal
  --unbalanced          Do not use balanced data. Useful for single-cell Hi-C
                        data together with --coverage_norm, not recommended
                        otherwise.
  --coverage_norm       If --unbalanced, also add coverage normalization based
                        on chromosome marginals
  --rescale             Do not use centres of features and pad, and rather use
                        the actual feature sizes and rescale pileups to the
                        same shape and size
  --rescale_pad RESCALE_PAD
                        If --rescale, padding in fraction of feature length
  --rescale_size RESCALE_SIZE
                        If --rescale, this is used to determine the final size
                        of the pileup, i.e. it ill be size×size. Due to
                        technical limitation in the current implementation,
                        has to be an odd number
  --n_proc N_PROC       Number of processes to use. Each process works on a
                        separate chromosome, so might require quite a bit more
                        memory, although the data are always stored as sparse
                        matrices
  --outdir OUTDIR       Directory to save the data in
  --outname OUTNAME     Name of the output file. If not set, is generated
                        automatically to include important information.

```

Currently, `coolpup.py` doesn't support inter-chromosomal pileups, but this is an addition that is planned for the future.

## This tool has been used in the following publications

**DNA methylation directs polycomb-dependent 3D genome re- organisation in naive pluripotency**

Katy A McLaughlin, Ilya M Flyamer, John P Thomson, Heidi K Mjoseng, Ruchi Shukla, Iain Williamson, Graeme R Grimes, Robert S Illingworth, Ian R Adams, Sari Pennings, Richard R Meehan, Wendy A Bickmore

https://www.biorxiv.org/content/early/2019/01/22/527309
