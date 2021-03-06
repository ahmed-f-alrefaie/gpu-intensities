#!/bin/bash -l

#heck if our working directory is on the central file server
#
####export exec=j-trove2105_red.x

export exec=main.x
####export exec=j-trove0606_cosmos.x

export pwd=`pwd`
export projectid=exomolaar
echo $pwd

export name=`echo $1 | sed -e 's/\.inp//'`
echo $name

export JOB=$3
echo $JOB

if [ -e "$name.o" ]; then
   /bin/rm $name.o
fi

if [ -e "$name.e" ]; then
   /bin/rm $name.e
fi

if [ -e "$name.out" ]; then
  if [ -e "$name.tmp" ]; then
    /bin/rm $name.tmp
  fi
  /bin/mv $name.out $name.tmp
fi

export nproc=$2


export jobtype="emerald3g"
export wclim=$3

if [ "$nproc" -gt "2" ]; then
  export nproc=2
fi


echo "Nproc=" $nproc


#PBS -l nodes=8:ppn=16,mem=512000mb,walltime=02:00:00
#PBS -m ae


echo "Nnodes=" 1, "Nproc=" $nproc, " Memory = "$MEM, "jobtype = " $jobtype, "wclimit = " $wclim
echo "Working dir is " $pwd

#qsub -r n -N $name -j oe -e $name.e -q $jobtype -l "walltime=$wclim:00:00,mem=$MEM,nodes=1:ppn=$nproc" \
#     -v "name=$name,pwd=$pwd,nproc=$nproc,exec=$exec" \
#     $pwd/run_trove.csh

#sbatch -A DIRAC-dp020 --nodes=1  --ntasks=$nproc --time=$wclim:00:00  -J $name -o $name.o -e $name.e   \
#     --workdir=$pwd --hint=compute_bound --no-requeue --mem=$MEM -p sandybridge \
#     $pwd/run_trove.csh $nproc $name $exec $pwd

#bsub -n $nproc -J $name -x -o $name.o -e $name.e -W $wclim:00 -m $jobtype run_intens_emerald.sh $pwd $name $nproc $exec

#sub -N $name lj oe -e $name.e -p exomolaar -l "h_rt=$wclim:00:00,mem=36G,gpu=2" run_intens_legion.sh $pwd $name $nproc $execq
qsub -N $name -wd $pwd -P projectid -l "h_rt=$wclim:00:00,mem=10G,gpu=2" $pwd/run_intens_legion.sh $pwd $name $nproc $exec


