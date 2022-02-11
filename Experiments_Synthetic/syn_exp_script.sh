#!/usr/bin/bash

#SBATCH -J ECE_new_exp

#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#The maximum walltime of the job is 10h 0min
#SBATCH -t 10:00:00

#SBATCH --mem=4G

#Leave this here if you need a GPU for your job
# #SBATCH --partition=gpu
# #SBATCH --gres=gpu:tesla:1

# export PATH=/gpfs/space/home/user/miniconda3/bin:$PATH  # Add path if needed
source activate ECE_fit

echo "Start 'other' training"
python -u synthetic_experiment_runner.py -i $1 -d $2 -b $3 -c $4 -s $5 -m 0 # other
echo "Start 'pw_nn4_bs' training"
python -u synthetic_experiment_runner.py -i $1 -d $2 -b $3 -c $4 -s $5 -m 1 # pw_nn4_bs
echo "Start 'pw_nn4_ce' training"
python -u synthetic_experiment_runner.py -i $1 -d $2 -b $3 -c $4 -s $5 -m 2 # pw_nn4_ce
echo "Start 'pw_nn6_logit_bs' training"
python -u synthetic_experiment_runner.py -i $1 -d $2 -b $3 -c $4 -s $5 -m 3 # pw_nn6_logit_bs
echo "Start 'pw_nn6_logit_ce' training"
python -u synthetic_experiment_runner.py -i $1 -d $2 -b $3 -c $4 -s $5 -m 4 # pw_nn6_logit_ce
echo "Start 'pwlf' training"
python -u synthetic_experiment_runner.py -i $1 -d $2 -b $3 -c $4 -s $5 -m 5 # other
echo "Job finished"