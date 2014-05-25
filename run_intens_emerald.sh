#!/bin/csh

set wd=$1
set name=$2
set n_gpus=$3
set exec=$4
#Move to the working directory
cd $wd

setenv NUM_GPUS $3

module load cuda/6.0.37
module load intel/14.0.2


./$exec < $wd/$name.inp > $wd/$name.out
