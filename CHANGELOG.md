# Release notes

## [Upcoming release](https://github.com/open2c/coolpuppy/compare/v1.1.0...HEAD)

### Performance
* Pairwise combinations of bed features are now generated with a vectorised search
  instead of a Python loop over all offsets genome-wide. With `maxdist` this is
  10-100x faster (e.g. 182 s -> 2 s for loop anchors on the Scc1 test cooler); results
  are unchanged. Pairs are produced in bounded chunks, so memory stays bounded
  without `maxdist`. Trans combinations are also vectorised.
* Faster accumulation of snippets (in-place sums; stripes appended instead of copied)

### Bug fixes
* `extra_sum_funcs` (e.g. per-snippet TAD scores): summing pileups across regions
  discarded all but one region's data; regions without valid snippets raised
  `AssertionError: ... not in dict2` (#146)
* Domain score / local enrichment were wrong or raised `AssertionError` for
  `rescale_flank != 1` (#161). Results for `rescale_flank=1` are unchanged
* `divide_pups`/`dividepups.py`/`plotpup.py --divide_pups` can now divide pileups
  with multiple groups, matching rows by group, distance band or orientation,
  whichever matches (e.g. ++ over -- pileups split by distance) (#163); dividing
  pileups with stripes no longer crashes; fixed the `clrs` annotation and the
  default output name
* Shifted controls are now reproducible with `seed` regardless of `nproc`
  (each region gets its own random stream). Note: seeded controls differ from v1.1.0
* Trans shifted controls now fetch the snippet at the shifted coordinates of both anchors
* `nproc=0` / `-p 0` now uses all cores (it ran serially, or crashed with `coverage_norm`)
* `nproc > 1` no longer crashes on macOS/Windows with multiprocessing_logging >= 0.4
* `coverage_norm` works with cooltools >= 0.6 (coverage computation crashed)
* With expected and `ooe=False`, expected values are only accumulated over pixels
  where the observed data is valid (not bad bins or ignored diagonals)
* `by_distance` accepts a list of edges, as documented
* `CoordCreator.bedpe2bed(ends=False, how="center")` crashed
* Pileup files: saving is atomic, so an interrupted save no longer leaves a file that
  fails with `object 'attrs' doesn't exist` (#166); such files load with a warning.
  Chromosome names in stored coordinates are no longer truncated to 13 characters
* `plotpup.py --query` always crashed; `--col_order`/`--row_order` accept commas;
  `--input_pups` is required
* `plot()` with default arguments raised `KeyError: 'score'`; a custom score column
  name is now used
* `coolpup.py` detects bed/bedpe correctly for `.gz` and other extensions; INFO-level
  logging (`-l INFO`) is now printed
* One warning per pileup reports how many regions of interest were skipped for
  extending beyond the chromosome/region boundaries (per-region counts at DEBUG level)
* `plot_stripes` no longer triggers pandas' "Calling int on a single element Series"
  FutureWarning
* Removed the unused top-level `__init__.py`: with pytest it made the repository root
  importable as `coolpuppy`, so tests with `nproc > 1` hung on macOS/Windows (spawn)

### Documentation
* The API and CLI walkthroughs show how to divide pileups with several groups
  (CTCF pileups by orientation, for sites 0.1-1 Mb apart over sites >1 Mb apart)
  with `divide_pups` / `dividepups.py`. All example notebooks were rerun;
  the TAD pileup in `TAD_score.ipynb` now averages all 938 TADs instead of only the
  238 in the first chromosome arm (#146)

### Compatibility
* Requires Python >= 3.10 (CI now tests 3.10-3.12) and numpy >= 1.17. Cython is no
  longer needed to install coolpuppy
* Removed the dependency on the unmaintained `h5sparse` (it imports `pkg_resources`);
  the file format is unchanged and files remain readable by older versions
* Compatible with pandas 3 and numpy 2, and with matplotlib >= 3.9 (`cm.get_cmap`
  removed; `plotpup` also works around cooltools <= 0.7.1 importing the removed
  `cm.register_cmap`) and pandas Copy-on-Write (stripe plots silently ignored
  assignments). Pileups with expected no longer use cooltools' `ExpectedSnipper`,
  which fails with pandas 3. Removed the pandas/matplotlib upper pins. Distance band
  labels are plain numbers with numpy 2 (not `np.int64(...)`)

## [v1.1.0] (https://github.com/open2c/coolpuppy/compare/v1.0.0...v1.1.0)

* Added `ignore_group_order` and `--ignore_group_order` argument to flip and combine groups when using groupby, i.e. combining e.g. group1-group2 and group2-group1

* Changed how flipping and group reassignment is implemented

* Fixed zooming near diagonal

* `divide_pups` now allows dividing even if columns are different, gives warning instead

* Added additional tests

* Bug fixes and logs/warnings added

* Changes to walkthroughs

## [v1.0.0](https://github.com/open2c/coolpuppy/compare/0.9.5...v1.0.0)

This is a major release with a lot of changes. The main ones are included below, but probably there are a lot of smaller ones too.

### API changes
* New HDF5-based storage format, with functions to write and read the files in coolpuppy.lib.io

* New “user-facing” `pileup` function in python API for convenient single-step pileups, and with interface similar to `cooltools.pileup`

* Pileups can be restricted using a genomic view (`--view` in CLI and `view_df` in API) in line with cooltools/bioframe (note that the expected has to be calculated for the same view, if used)

* If using expected, by default each snippet is now normalized to expected and only averaged afterwards; this is controlled by `ooe` argument in API and `--not-ooe` argument in CLI)

* Added options to split snippets based on strands, genomic distance, or both

* Added option `flip_negative_strand` (API) and `--flip-negative-strand` (CLI) to flip features located on the negative strand

* Added option to groupby snippets by their properties and generate multiple pileups in one run (`groupby` in API and `--groupby` in CLI)

* Added option `trans=True` (API) and `--trans` (CLI)  to generate inter-chromosomal (trans) pileups

* Added option `store_stripes=True` (API) and `--store_stripes` (CLI) to store individual vertical and horizontal stripe pairs

* Added advanced option (`modify_2Dintervals_func` of `PileUpper.pileupsWithControls`) to apply an arbitrary function to pairs of intervals before they are used for generating pileups (i.e. bedpe-style intervals generated internally from either bedpe- or bed-style input features)

* Added advanced option to apply an arbitrary function to each snippet before averaging in API (`postprocess_func` of `PileUpper.pileupsWithControls`)

* Added function `divide_pups()` (API) and `dividepups.py` (CLI) to divide two pileups 

* CLI detects headers of bed and bedpe files

* Overall API and CLI argument names are aligned with cooltools wherever possible

### Plotting changes
* Changed names of `plotpup.make_heatmap_grid()` to `plotpup.plot()`
* Added option `plotpup.plot_stripes()` (API) and `--stripe` (CLI) for plotting stripes
* Added option `--divide_pups` (CLI) to plot the ratio of two pileups

### Documentation
* Extensive tutorials for both python API and CLI

### Maintenance
* Code restructured, so all CLI tools are in one package, and created .lib with shared internal functions in lib/io.py, lib/numutils.py, lib/puputils.py and lib/util.py

* Tests migrated to github actions, and also running nbsmoke on the CLI notebook to ensure it runs without errors

* Only support python >=3.8, cooltools >=0.5.2

### Miscellaneous
* Logging has been fixed so the --logLevel properly works
* Removed the launch_pileups.sh script
