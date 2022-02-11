#!/usr/bin/bash

#SBATCH -J Batch_job

#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#The maximum walltime of the job is a 5 minutes
#SBATCH -t 0:05:00

#SBATCH --mem=1G

#export PATH=/gpfs/space/home/user/miniconda3/bin:$PATH

for s in {0..0} # seed 0..4
  do
  echo $s
  for d in {0..2} # data size 0..2
     do
     echo $d
     for i in {0..4} # cal fn
        do
        echo $i
        for c in {0..20} # 0..20 derivates
          do
          echo $c
          sbatch syn_exp_script.sh $i $d 0 $c $s
          done

        for c in {0..6} # 0..6 derivates
          do
          echo $c
          sbatch syn_exp_script.sh $i $d 1 $c $s
          done
      done
    done
  done

echo "Job finished"