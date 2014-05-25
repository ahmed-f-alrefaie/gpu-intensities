#!/bin/csh

#$ -wd /home/ucapfal/Scratch/output/intensities
set pwd=$1
set name=$2
set n_gpus=$3
set exec=$4
#Move to the working directory
cd $TMPDIR

setenv NUM_GPUS $3

module load cuda/5.0


$pwd/$exec < $pwd/$name.inp > $pwd/$name.out
