#!/bin/sh
# Grid Engine options (lines prefixed with #$)
#$ -cwd
#$ -N pileups
#$ -l h_rt=4:00:00
#$ -l h_vmem=5G
#$ -pe sharedmem 4
#$ -j yes
#$ -V

# args: coolfile baselist outdir
python3 pileups.py $1 $2 --n_proc 4 --outdir $3
