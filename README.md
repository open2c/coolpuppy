# coolpup.py
[![DOI](https://zenodo.org/badge/147190130.svg)](https://zenodo.org/badge/latestdoi/147190130)

.**cool** file **p**ile-**up**s with **py**thon.

A versatile tool to perform pile-up analysis on Hi-C data in .cool format (https://github.com/mirnylab/cooler). And who doesn't like cool pupppies?

`pip install coolpuppy` will make coolpup.py callable in your terminal.

This is the idea of how pileups work, in case you are checking whether certain regions tend to interacts with each other:

<img src="https://github.com/Phlya/coolpuppy/blob/master/loop_quant.svg" width="400">

What's not shown here is normalization to the expected values, done here by dividing the pileups by randomly shifted control regions. In the future dividing by expected values from whole-chromosome by-diagonal average will be possible.

```
Usage: coolpup.py [-h] [--pad PAD] [--minshift MINSHIFT] [--maxshift MAXSHIFT]
                  [--nshifts NSHIFTS] [--mindist MINDIST] [--maxdist MAXDIST]
                  [--excl_chrs EXCL_CHRS] [--incl_chrs INCL_CHRS]
                  [--anchor ANCHOR] [--by_window] [--save_all] [--local]
                  [--subset SUBSET] [--unbalanced] [--coverage_norm]
                  [--n_proc N_PROC] [--outdir OUTDIR] [--outname OUTNAME]
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
  --mindist MINDIST     Minimal distance of intersections to use
  --maxdist MAXDIST     Maximal distance of intersections to use
  --excl_chrs EXCL_CHRS
                        Exclude these chromosomes form analysis
  --incl_chrs INCL_CHRS
                        Include these chromosomes; default is all. excl_chrs
                        overrides this.
  --anchor ANCHOR       A UCSC-style coordinate to use as an anchor to create
                        intersections with coordinates in the baselist
  --by_window           Create a pile-up for each coordinate in the baselist
  --save_all            If by-window, save all individual pile-ups as separate
                        text files. Can create a very large number of files,
                        so use cautiosly! If not used, will save a master-
                        table with coordinates, their enrichments and
                        cornerCV, which is reflective of noisiness
  --local               Create local pileups, i.e. along the diagonal
  --subset SUBSET       Take a random sample of the bed file - useful for
                        files with too many featuers to run as is, i..e some
                        repetitive elements. Set to 0 or lower to keep all
                        data.
  --unbalanced          Do not use balanced data - rather average cis coverage
                        of all regions, and use it to normalize the final
                        pileups. Useful for single-cell Hi-C data, not
                        recommended otherwise.
  --coverage_norm       If --unbalanced, also add coverage normalization based
                        on chromosome marginals
  --n_proc N_PROC       Number of processes to use. Each process works on a
                        separate chromosome, so might require quite a bit more
                        memory, although the data are always stored as sparse
                        matrices
  --outdir OUTDIR       Directory to save the data in
  --outname OUTNAME     Name of the output file. If not set, is generated
                        automatically to include important information.


```
