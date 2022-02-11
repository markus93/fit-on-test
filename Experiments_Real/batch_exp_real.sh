#!/usr/bin/bash

# The location of this file is in data/ folder where raw-all folder is.

#SBATCH -J Batch_job

#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#The maximum walltime of the job is a 30 minutes
#SBATCH -t 0:03:00

#SBATCH --mem=1G



for i in {0..54}
   do
   for d in {0..9}
      do
      sbatch exp_main_real.sh $i 1000 42 $d
      sbatch exp_main_real2.sh $i 1000 42 $d
      sbatch exp_main_real3.sh $i 1000 42 $d
      done
   for d in {0..2}
      do
      sbatch exp_main_real.sh $i 3000 42 $d
      sbatch exp_main_real2.sh $i 3000 42 $d
      sbatch exp_main_real3.sh $i 3000 42 $d
      done
   sbatch exp_main_real.sh $i 10000 42 0
   sbatch exp_main_real2.sh $i 10000 42 0
   sbatch exp_main_real3.sh $i 10000 42 0
   done


echo "Job finished"
