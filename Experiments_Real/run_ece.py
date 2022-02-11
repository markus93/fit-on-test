import numpy as np
import pandas as pd

from data_generation import *
from calibration_functions import *
from calibration_function_derivates import *
from dataframe_helpers import *
from binnings import *

from os.path import join


def create_CV_trick_rows(df):
    
    data_rows = []
    selection = df[(df["n_folds"] == 5) | (df["n_folds"] == 10)]
    
    for idx, row in selection.iterrows():
        
        optimal = row["n_bins"]
        pos_new = row["n_bins"]
        min_bin_score = np.min(row["bin_scores"])
        
        assert min_bin_score == row["bin_scores"][optimal]
        
        for pos in range(optimal-1, 0, -1):
            max_diff = min_bin_score*0.001
            
            new_min_cand = row["bin_scores"][pos]

            if new_min_cand <= min_bin_score + max_diff:
                pos_new = pos

        s1 = df[(df["seed"] == row["seed"])]
        s2 = s1[(s1["expected_calibration_error"] == row["expected_calibration_error"])]
        s3 = s2[(s2["binning"] == row["binning"])]
        s4 = s3[(s3["n_bins"] == pos_new)]
        found = s4[(s4["n_folds"] == 0)].copy()

        found["n_folds"] = str(row["n_folds"]) + "_trick"
        found["old_n_bins"] = optimal
        data_rows.append(found.iloc[0])

    return pd.DataFrame(data_rows) 
    
    
def construct_data_row(binning, seed, n_folds, calibration_fun_name, n_data, 
                       true_calibration_error_abs, true_calibration_error_square, expected_calibration_error,
                       bin_scores=None, all_cv_scores=None):
    
    return {
        "seed": seed,
        "calibration_function": calibration_fun_name,
        "n_data": n_data,
        "n_folds": n_folds,
        "binning": binning.binning_name,
        "n_bins": binning.n_bins,
        
        "ECE_abs": binning.ECE_abs, 
        "ECE_abs_debiased": binning.ECE_abs_debiased,
        "true_calibration_error_abs": true_calibration_error_abs,

        "ECE_square": binning.ECE_square, 
        "ECE_square_debiased": binning.ECE_square_debiased, 
        "true_calibration_error_square": true_calibration_error_square,
        
        "flat_abs_c_hat_dist_c":np.mean(np.abs(binning.eval_flat(binning.p) - binning.c)),
        "flat_square_c_hat_dist_c":np.mean(np.square(binning.eval_flat(binning.p) - binning.c)),
        "slope_abs_c_hat_dist_c":np.mean(np.abs(binning.eval_slope_1(binning.p) - binning.c)),
        "slope_square_c_hat_dist_c":np.mean(np.square(binning.eval_slope_1(binning.p) - binning.c)), 
        
        "expected_calibration_error": expected_calibration_error,
        
        "bin_scores": bin_scores,
        "all_cv_scores": all_cv_scores
    }

def run_ece_tests(p, y, c, cal_method, data_name, n_data, seed, all_n_bins, data_path, data_part = -1):
    """
    all_n_data = [100, 300, ..]
    all_derivate_functions = find_all_derivates_for_calibration_functions(..)
    seeds = [0,1,2,3,..]
    beta_distribution = [1,1] # uniform
                        [1.1, 0.1] # beta1
    all_n_bins = [1,2,3,4,5,..]
    file_identifier = "uniform" # v6i m6ni muu s6na t2histamaks salvestatavat jaotust
    """
    
    data_rows = []
    

    expected_calibration_error = -1
    error_fun = np.abs       
    

    true_calibration_error_abs = np.mean(np.abs(p - c))
    true_calibration_error_square = np.mean(np.square(p - c))

    cv_ECEs = {}

    # Cross-validation
    for use_eq_width in [True, False]:
        for n_splits in [5, 10]:

            bin_scores, all_cv_scores = binning_n_bins_with_crossvalidation(p, y, use_eq_width, n_splits)
            n_bins_cv = np.argmin(bin_scores)

            if use_eq_width:
                binning = EqualWidthBinning(p, y, c, n_bins_cv)
            else:
                binning = EqualSizeBinning(p, y, c, n_bins_cv)

            data_rows.append(construct_data_row(binning, seed, n_splits, cal_method, n_data, 
                                               true_calibration_error_abs, true_calibration_error_square, expected_calibration_error,
                                               bin_scores, all_cv_scores))  

    # No cross-validation
    for n_bins in all_n_bins:

        # Eq width      
        binning = EqualWidthBinning(p, y, c, n_bins)
        data_rows.append(construct_data_row(binning, seed, 0, cal_method, n_data, 
           true_calibration_error_abs, true_calibration_error_square, expected_calibration_error))                                       

        # Eq size
        binning = EqualSizeBinning(p, y, c, n_bins)
        data_rows.append(construct_data_row(binning, seed, 0, cal_method, n_data, 
           true_calibration_error_abs, true_calibration_error_square, expected_calibration_error))  


    # Monotonic eq size
    binning = MonotonicEqualSizeBinning(p, y, c)

    n_bins = binning.n_bins

    
    data_rows.append(construct_data_row(binning, seed, 0, cal_method, n_data, 
           true_calibration_error_abs, true_calibration_error_square, expected_calibration_error))     

    df = pd.DataFrame(data_rows)     
    df_CV_trick = create_CV_trick_rows(df)
    df_all = pd.concat([df,df_CV_trick])

    df_all.to_pickle(join(data_path, f"binning_CV_seed_{seed}_{n_data}_{cal_method}_{data_name}_dp_{data_part}.pkl"), protocol = 4)
    
    return df_all