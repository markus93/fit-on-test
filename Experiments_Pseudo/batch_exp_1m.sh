#!/usr/bin/bash

# The location of this file is in data/ folder where raw-all folder is.

#SBATCH -J Batch_job

#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#The maximum walltime of the job is a 30 minutes
#SBATCH -t 0:03:00

#SBATCH --mem=1G


for i in {0..155}
   do
   for s in {0..4}
      do
      sbatch exp_main_1m.sh $i 1000 $s
      sbatch exp_main_1m.sh $i 3000 $s
      sbatch exp_main_1m.sh $i 10000 $s
      sbatch exp_main_1m2.sh $i 1000 $s
      sbatch exp_main_1m2.sh $i 3000 $s
      sbatch exp_main_1m2.sh $i 10000 $s
      sbatch exp_main_1m3.sh $i 1000 $s
      sbatch exp_main_1m3.sh $i 3000 $s
      sbatch exp_main_1m3.sh $i 10000 $s
      done
   done
echo "Job finished"
