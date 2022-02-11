import numpy as np
import pandas as pd
import os
from os.path import join
import pickle
from binnings import EqualWidthBinning, EqualSizeBinning, MonotonicEqualSizeBinning
from pycalib.models import IsotonicCalibration

def gen_cal_cgt_new(probs_2m, Y_2m, file_name, N=1000000):

    ## Fit ground-truths on 1m datapoints
    binning_es = EqualSizeBinning(probs_2m, Y_2m, np.zeros(N), 100)
    fn_slope_1 = binning_es.construct_eval_function_slope_1()
    fn_flat = binning_es.construct_eval_function_flat()

    binning_sweep = MonotonicEqualSizeBinning(probs_2m, Y_2m, np.zeros(N))
    fn_sweep = binning_sweep.construct_eval_function_slope_1()

    isotonic = IsotonicCalibration()
    isotonic.fit(probs_2m, Y_2m)

    c_slope_1 = fn_slope_1(probs_2m)
    c_flat = fn_flat(probs_2m)
    c_sweep = fn_sweep(probs_2m)
    c_isotonic = isotonic.predict(probs_2m)

    c_s = {"c_slope_1":c_slope_1, "c_flat": c_flat, "c_sweep": c_sweep, "c_isotonic": c_isotonic}

    with open("probs_1m_cgt/%s" % (file), "wb") as f:
        pickle.dump((probs_2m, Y_2m, c_s), f, protocol=4)

PATH_PROBS = "probs_1m_final"

files = []
for file in os.listdir(PATH_PROBS):
    if file.endswith(".p"):
        files.append(file)

files = sorted(files)

for file in files:
    print(file)

    with open(join(PATH_PROBS, file), "rb") as f:
        probs_2m, Y_2m = pickle.load(f)

    gen_cal_cgt_new(probs_2m, Y_2m, file)
    
    

        