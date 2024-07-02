# Release notes

## [Upcoming release](https://github.com/open2c/coolpuppy/compare/v1.1.0...HEAD)

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
