# Guide to pileup analysis

Coolpup.py is a tool for pileup analysis. But what are pile-ups?

If you don't know, you might have seen average ChIP-seq or ATAC-seq profiles which look something like this:

<img src="https://raw.githubusercontent.com/Phlya/coolpuppy/master/docs/source/figs/chip_profile.png" alt="ChIP profile" width="600px"/>

Pile-ups in Hi-C are essentially the same as average profiles, but in 2 dimensions, since
Hi-C data is a a matrix, not a linear track!

Therefore instead of a linear plot, pileups are usually represented as heatmaps - by mapping
values of different pixels in the average matrix to specific colours.

## Interactions between a set of regions

For example, we can again start with ChIP-seq peaks, but instead of averaging ChIP-seq data around them,
combine them with Hi-C data and check whether these regions are often found in proximity to each other. The algorithm is simple:
we find intersections of all peaks in the Hi-C matrix (with some padding around the peak), and average them. If the peaks often interact,
we will detect an enrichment in the center of the average matrix:

<img src="https://raw.githubusercontent.com/Phlya/coolpuppy/master/docs/source/figs/new_grid_loop_quant.png" alt="Grid averaging" width="300px"/>


Here is a real example:

<img src="https://raw.githubusercontent.com/Phlya/coolpuppy/master/docs/source/figs/example_pileup.png" alt="Example pileup" width="400px"/>

Here I averaged all (intra-chromosomal) interactions between highly enriched ChIP-seq peaks of RING1B in mouse ES cells.
I added 100 kbp padding to each bin containing the peak, and since I used 5 kbp resolution Hi-C data,
the total length of each side of this heatmap is 205 kbp. I also normalizes the result by what we would expect to find by chance,
and therefore the values indicate observed/expected enrichment. Because of that, the colour is log-scaled, so that the neutral
grey colour corresponds to 1 - no enrichment or depletion, while red and blue correspond to value above and below 1, respectively.

What is important, is that in the center we see higher values than on the edges: this means that regions
bound by RING1B tend to stick together more, than expected! The actual value in the central pixel is displayed on top left for reference.