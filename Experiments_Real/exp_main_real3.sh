#!/usr/bin/bash

# The location of this file is in data/ folder where raw-all folder is.

#SBATCH -J Exp_ECE

#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#The maximum walltime of the job is a 2 hours
#SBATCH -t 6:00:00

#SBATCH --mem=6G

#Leave this here if you need a GPU for your job
# #SBATCH --partition=gpu
# #SBATCH --gres=gpu:tesla:1

#The module with all the NMT / deep learning packages

#export PATH=/gpfs/space/home/user/miniconda3/bin:$PATH  # Add correct path if needed
source activate ECE_fit

echo "Start model training 5m - PW4 - MSE and CE"
python -u main_NN_real_final.py -i $1 -ss $2 -s $3 -dp $4 --fit_pw --use_nn4 --use_ce_loss --use_sweep
python -u main_NN_real_final.py -i $1 -ss $2 -s $3 -dp $4 --fit_pw --use_nn4 --use_sweep
python -u main_NN_real_final.py -i $1 -ss $2 -s $3 -dp $4 --fit_pw --use_nn4 --use_ce_loss
python -u main_NN_real_final.py -i $1 -ss $2 -s $3 -dp $4 --fit_pw --use_nn4
echo "Job finished"
