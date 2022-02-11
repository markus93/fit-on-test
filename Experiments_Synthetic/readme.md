# Running Syn Data Experiments

## 1. Run experiments

To run all experiments run the script "batch_syn_exp_script.sh".

Run a singular experiment with some specific parameters.
"python synthetic_experiment_runner.py -i $1 -d $2 -b $3 -c $4 -s $5 -m $6"
$1 - Calibration function number (base shapes "sqrt", "square", "beta1", "beta2", "stairs"): 0,1,2,3,4.
$2 - Data size number (1000, 3000, 10000): 0,1,2.
$3 - Distribution number (uniform): 0.
$4 - Derivate expected calibration error (0.0, 0.005, .., 0.10): 0,1,2,3,..,20.
$5 - Data seed: 0,1,2,3,4
$6 - Methods to run (everything else, our PL variations): 0,1,2,3,4,5


## 2. Combine data

Run "python data_combiner.py".
