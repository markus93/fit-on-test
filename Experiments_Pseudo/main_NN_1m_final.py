# Imports
import numpy as np

from piecewise_linear import Piecewise_NN2, Piecewise_NN2_val, Piecewise_NN3, Piecewise_NN4, Piecewise_NN5, Piecewise_NN6
import argparse

import os
import pickle
from os.path import join
from time import time

#from run_kde import run_kde
#from run_pwlf import run_pwlf_test
from run_pw import run_PW_ECE
from run_ece import run_ece_tests
#from run_extra import run_isotonic, run_platt, run_beta

import pandas as pd

def split_data(inputs, split_size = 1000, seed = 5):

    for i, inp in enumerate(inputs):
        inputs[i] = np.array(inp)
    
    n = len(inputs[-1])
    N = np.arange(0, n)

    np.random.seed(seed)
    np.random.shuffle(N)
    

    if split_size == 1000:
        split_start = 0
    elif split_size == 3000:
        split_start = 1000
    else:
        split_start = 4000
 
    
    for i, inp in enumerate(inputs):
        inputs[i] = inp[N[split_start:(split_size+split_start)]]
                
    return inputs

if __name__ == "__main__":

    # Add arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_nr', '-i', type=int, default=0, help="Calibration function number")
    parser.add_argument('--seed_nr', '-s', type=int, default=0, help="Number of seed")
    parser.add_argument('--cv_folds', '-cv', type=int, default=10, help="Number of cross_validation folds (default 10), if smaller than 2, then no CV is done.")
    parser.add_argument('--split_size', '-ss', type=int, default=-1, help="Size of the smaller data part used.")
    parser.add_argument('--use_sweep', help='Use sweep approach for finding right number of bins.', action='store_true')
    parser.add_argument('--monotonic', help='Train using monotonicity.', action='store_true')
    parser.add_argument('--fit_ece', help='Fit ECE methods.', action='store_true')
    parser.add_argument('--fit_pw', help='Fit PW method.', action='store_true')
    parser.add_argument('--fit_kde', help='Fit KDE method.', action='store_true')
    parser.add_argument('--fit_pwlf', help='Fit PWLF method.', action='store_true')
    parser.add_argument('--use_val', help='Fit PW method with validation set.', action='store_true')
    parser.add_argument('--use_nn3', help='Fit PW method with validation set.', action='store_true')
    parser.add_argument('--use_nn4', help='Fit PW method with NN4 variant.', action='store_true')
    parser.add_argument('--use_nn5', help='Fit PW method with NN5 variant.', action='store_true')
    parser.add_argument('--use_nn6', help='Fit PW method with NN6 variant.', action='store_true')
    parser.add_argument('--use_logit', help='Convert input to logit scale.', action='store_true')
    parser.add_argument('--logit_input', help='Convert input to logit input.', action='store_true')
    parser.add_argument('--logistic_out', help='Convert input to logit input.', action='store_true')
    parser.add_argument('--use_ce_loss', help='Use CE loss.', action='store_true')
    parser.add_argument('--fit_extra', help='Fit extra (iso, beta, platt) methods.', action='store_true')


    args = parser.parse_args()
    i = args.data_nr
    seed = args.seed_nr
    cv_folds = args.cv_folds
    split_size = args.split_size
    use_sweep = args.use_sweep
    monotonic = args.monotonic
    fit_ece = args.fit_ece
    fit_pw = args.fit_pw
    fit_kde = args.fit_kde
    fit_pwlf = args.fit_pwlf
    use_val = args.use_val
    use_nn3 = args.use_nn3
    use_nn4 = args.use_nn4
    use_nn5 = args.use_nn5
    use_nn6 = args.use_nn6
    fit_extra = args.fit_extra
    logit_scale = args.use_logit
    logit_input = args.logit_input
    logistic_out = args.logistic_out
    use_ce_loss = args.use_ce_loss
    
    print("1m final 21.10")
    
    print("Using validation set:", use_val)
    
    if use_val:
        fn_method = Piecewise_NN2_val
    elif use_nn3:
        print("Using PW_NN4 method!")
        fn_method = Piecewise_NN3   
    elif use_nn4:
        print("Using PW_NN4 method!")
        fn_method = Piecewise_NN4
    elif use_nn5:
        print("Using PW_NN5 method!")
        fn_method = Piecewise_NN5
    elif use_nn6:
        print("Using PW_NN6 method!")
        fn_method = Piecewise_NN6
    else:  
        fn_method = Piecewise_NN2

    t_total = time()


    PATH_DATA = "data_1m_final_21_10"        
    all_n_bins = [n_bins for n_bins in range(1, 41, 1)]

    if not os.path.exists(PATH_DATA):
        os.makedirs(PATH_DATA)
  
    PATH_PROBS = "probs_1m_cgt"

    files = []
    for file in os.listdir(PATH_PROBS):
        if file.endswith(".p"):
            files.append(file)

    files = sorted(files)

    with open(join(PATH_PROBS, files[i]), "rb") as f:
        p, y, c_dict = pickle.load(f)

    
    parts = files[i].split(".")[0].split("_")
    
    # probs_1m_beta_resnet110_s7_tag_1vsRest5_with_c.p

    cal_method = parts[2]
    data_name = "_".join(parts[3:-2])
    # seed = int(parts[1][1:])

    print(cal_method)
    print(data_name)

    print("Split size:", split_size)
       

    s = time()
    Ps, Ys, Cs_slope, Cs_flat, Cs_sweep, Cs_iso = split_data([p, y, c_dict["c_slope_1"], 
                                                              c_dict["c_flat"], c_dict["c_sweep"], c_dict["c_isotonic"]],
                                                              split_size = split_size, seed = seed)
    
    Ps = np.array(Ps, dtype=np.float64)

    Cs_list = [Cs_slope, Cs_flat, Cs_sweep, Cs_iso]

    print(Ps[:10])
    
    n_data = len(Ys)

    print("Running part %i with number of instances: %i" % (i, n_data))    

    
    if fit_pw:
    
        input = Ps
    
        print("Fitting PW")
        run_PW_ECE(fn_method, input, Ps, Ys, Cs_list, cv_folds, seed, cal_method, data_name, data_path = PATH_DATA, data_part = -1, max_nodes = 15, #15
                   equal_size = True, monotonic = monotonic, use_sweep = use_sweep, use_nn3 = use_nn3, 
                   use_nn4 = use_nn4, use_nn5 = use_nn5, use_nn6 = use_nn6, logit_scale=logit_scale, logit_input = logit_input,
                   logistic_out = logistic_out, use_ce_loss = use_ce_loss)
    
    if fit_ece:
        print("Fitting ECE")
        for c_nr, c in enumerate(Cs_list):
            df_all = run_ece_tests(Ps, Ys, c, cal_method, data_name + "gt%i" % c_nr, n_data, seed, all_n_bins, data_path = PATH_DATA, data_part = -1)
        
    if fit_kde:
        print("Fitting KDE")
        df = run_kde(Ps, Ys, Cs_list, cal_method, data_name, n_data, seed, data_path = PATH_DATA)
        
    if fit_pwlf:
        print("PWLF - degree 1")
        run_pwlf_test(Ps, Ys, Cs_list, cv_folds, seed, cal_method, data_name, data_part = -1, max_nodes = 7, degree = 1, data_path = PATH_DATA, use_sweep = use_sweep) #7
        print("PWLF - degree 2")
        run_pwlf_test(Ps, Ys, Cs_list, cv_folds, seed, cal_method, data_name, data_part = -1, max_nodes = 5, degree = 2, data_path = PATH_DATA, use_sweep = use_sweep) #5

    if fit_extra:
    
        print("Fitting extra (isotonic, beta, platt)")
    
        for c_nr, c in enumerate(Cs_list):
            row = run_isotonic(Ps, Ys, c)
            row2 = run_beta(Ps, Ys, c)
            row3 = run_platt(Ps, Ys, c)
            
            df = pd.DataFrame([row, row2, row3]) 
            
            data_name2 = data_name + "gt%i" % c_nr
            data_part = -1
            df.to_pickle(join(PATH_DATA, f"df_seed_{seed}_{cal_method}_{n_data}_cv_{0}_{data_name2}_dp_{data_part}_iso_beta_platt.pkl"), protocol = 4)
    


    print("Time taken:", time() - s)
    
    print("Total time taken:", time() - t_total)


