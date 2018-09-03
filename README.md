# CoolPUp
A versatile tool to perform pile-up analysis on Hi-C data in .cool format (https://github.com/mirnylab/cooler). And who doesn't like cool pupppies?

```
Usage: pileups.py [-h] [--pad PAD] [--minshift MINSHIFT]
                     [--maxshift MAXSHIFT] [--nshifts NSHIFTS]
                     [--mindist MINDIST] [--maxdist MAXDIST]
                     [--excl_chrs EXCL_CHRS] [--incl_chrs INCL_CHRS]
                     [--anchor ANCHOR] [--by_window] [--save_all] [--local]
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
                        is (2×pad+1, 2×pad+1))
  --minshift MINSHIFT   Shortest distance for randomly shifting coordinates
                        when creating controls
  --maxshift MAXSHIFT   Longest distance for randomly shifting coordinates
                        when creating controls
  --nshifts NSHIFTS     Number of control regions per averaged window
  --mindist MINDIST     Minimal distance of intersections to use
  --maxdist MAXDIST     Maximal distance of intersections to use
  --excl_chrs EXCL_CHRS
                        Exclude these chromosomes from the analysis
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
  --n_proc N_PROC       Number of processes to use. Each process works on a
                        separate chromosome, so might require quite a bit more
                        memory, although the data are always stored as sparse
                        matrices
  --outdir OUTDIR       Directory to save the data in
  --outname OUTNAME     Name of the output file. If not set, is generated
                        automatically to include important information.
```
