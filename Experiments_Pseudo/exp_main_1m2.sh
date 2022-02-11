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

#export PATH=/gpfs/space/home/user/miniconda3/bin:$PATH - Add path if needed
source activate ECE_fit

echo "Start model training 5m - PW6 CE MSE"
python -u main_NN_1m_final.py -i $1 -ss $2 -s $3 --fit_pw --use_nn6 --use_logit --logistic_out --use_sweep
python -u main_NN_1m_final.py -i $1 -ss $2 -s $3 --fit_pw --use_nn6 --use_logit --logistic_out --use_ce_loss --use_sweep
python -u main_NN_1m_final.py -i $1 -ss $2 -s $3 --fit_pw --use_nn6 --use_logit --logistic_out --use_ce_loss
python -u main_NN_1m_final.py -i $1 -ss $2 -s $3 --fit_pw --use_nn6 --use_logit --logistic_out
echo "Job finished"
