coolpup.py CLI
==============

Run ``coolpup.py`` to generate pileups and ``plotpup.py`` to generate a figure with one or multiple pileups.

coolpup.py command
------------------

Usage: coolpup.py
                  [-h] [--bed2 BED2] [--bed2_unordered] [--pad PAD]
                  [--minshift MINSHIFT] [--maxshift MAXSHIFT]
                  [--nshifts NSHIFTS] [--expected EXPECTED]
                  [--mindist MINDIST] [--maxdist MAXDIST] [--minsize MINSIZE]
                  [--maxsize MAXSIZE] [--excl_chrs EXCL_CHRS]
                  [--incl_chrs INCL_CHRS] [--subset SUBSET] [--anchor ANCHOR]
                  [--by_window] [--save_all] [--local] [--unbalanced]
                  [--coverage_norm] [--rescale] [--rescale_pad RESCALE_PAD]
                  [--rescale_size RESCALE_SIZE] [--weight_name WEIGHT_NAME]
                  [--n_proc N_PROC] [--outdir OUTDIR] [--outname OUTNAME]
                  [--seed SEED] [-l {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
                  [--post_mortem] [-v]
                  coolfile baselist

positional arguments:
  --coolfile              Cooler file with your Hi-C data
  --baselist              A 3-column bed file or a 6-column double-bed file
                        (i.e. chr1,start1,end1,chr2,start2,end2). Should be
                        tab-delimited. With a bed file, will consider all cis
                        combinations of intervals. To pileup features along
                        the diagonal instead, use the --local argument. Can be
                        piped in via stdin, then use "-".

optional arguments:
  -h, --help            show this help message and exit
  --bed2 BED2           A 3-column bed file. Should be tab-delimited. Will
                        consider all cis combinations of intervals between
                        baselist and bed2. (default: None)
  --bed2_unordered      Whether to only use baselist as left ends, and bed2 as
                        right ends of regions. (default: True)
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
  --seed SEED           Set specific seed value to ensure reproducibility
                        (default: None)
  -l {DEBUG,INFO,WARNING,ERROR,CRITICAL}, --log {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        Set the logging level. (default: INFO)
  --post_mortem         Enter debugger if there is an error (default: False)
  -v, --version         show program's version number and exit


plotpup.py command
------------------
Usage: plotpup.py
                  [-h] [--cmap CMAP] [--symmetric SYMMETRIC] [--vmin VMIN]
                  [--vmax VMAX] [--scale {linear,log}]
                  [--cbar_mode {edge,single,each}] [--n_cols N_COLS]
                  [--col_names COL_NAMES] [--row_names ROW_NAMES]
                  [--norm_corners NORM_CORNERS] [--enrichment ENRICHMENT]
                  [--output OUTPUT] [-v]
                  [pileup_files [pileup_files ...]]

positional arguments:
  --pileup_files          All files to plot (default: None)

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
  --scale {linear,log}  Whether to use linear or log scaling for mapping
                        colours (default: log)
  --cbar_mode {edge,single,each}
                        Whether to show a single colorbar, one per row or one
                        for each subplot (default: single)
  --n_cols N_COLS       How many columns to use for plotting the data. If 0,
                        automatically make the figure as square as possible
                        (default: 0)
  --col_names COL_NAMES
                        A comma separated list of column names (default: None)
  --row_names ROW_NAMES
                        A comma separated list of row names (default: None)
  --norm_corners NORM_CORNERS
                        Whether to normalize pileups by their top left and
                        bottom right corners. 0 for no normalization, positive
                        number to define the size of the corner squares whose
                        values are averaged (default: 0)
  --enrichment ENRICHMENT
                        Whether to show the level of enrichment in the central
                        pixels. 0 to not show, odd positive number to define
                        the size of the central square whose values are
                        averaged (default: 1)
  --output OUTPUT, -o OUTPUT
                        Where to save the plot (default: pup.pdf)
  -v, --version         show program's version number and exit
