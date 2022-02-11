#!/usr/bin/env python
# coding: utf-8

# # 5m - Df unification (10 calib. fn-s)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import os
from os.path import join
import pickle
from copy import copy


def get_data_name(file):
    
    if "resnet110" in file:
        return "resnet110"
    elif "densenet40" in file:
        return "densenet40"
    else:
        return "wide32"


def get_strs(file, is_ece = True, is_kde = False, is_bip = False):
    
    extra = 0
    
    pieces = file.split(".")[0].split("_tag_")

    parts1 = pieces[0].split("_")
    parts2 = pieces[1].split("_")
    n_data = -1
    seed = -1
    
# binning_CV_seed_0_10000_VecS_wide32_s7_tag_confidencegt1_dp_-1.pkl

    if is_ece:
        cal_method = "_".join(parts1[5:6])
        data_name = get_data_name("_".join(parts1[6:]))
        tag_name = parts2[0][:-3]
        cgt_nr = int(parts2[0][-1])

        # KDE_seed_9_10000_VecS_resnet_wide32_s7_tag_1vsRest5_with_c_dp_-1.pkl
    elif is_kde:
        cal_method = "_".join(parts1[4:5])
        data_name = get_data_name("_".join(parts1[5:]))
        tag_name = parts2[0]
        cgt_nr = -1
        
        # df_seed_1_platt_resnet_3000_cv_0_wide32_s7_tag_confidence_with_cgt3_dp_-1_iso_beta_platt.pkl
        # df_seed_6_TempS_3000_cv_0_resnet_wide32_s7_1vsRest5_m_3_921420382311321_with_cgt0_dp_-1_iso_beta_platt
    elif is_bip:
        cal_method = "_".join(parts1[3:4])
        data_name = get_data_name("_".join(parts1[4:]))
        
        n_data = int(parts1[4])
            
        tag_name = parts2[0][:-3]
        cgt_nr = int(parts2[0][-1])
        seed = int(parts1[2])
        
        # 'df_seed_0_beta_10000_cv_0_densenet40_s7_tag_1vsRest1gt0_dp_-1_iso_beta_platt.pkl'
        #df_seed_0_beta_10000_cv_0_resnet110_s7_tag_confidencegt3_dp_-1_iso_beta_platt.pkl

        # df_seed_2_Isotonic_resnet110_10000_cv_0_s7_tag_confidence_with_c_dp_-1_PW_NN4_sweep.pkl            
    else:
        cal_method = "_".join(parts1[3:4])
        data_name = get_data_name("_".join(parts1[4:]))
        tag_name = parts2[0]
        cgt_nr = -1
        
    return (cal_method, data_name, tag_name, cgt_nr, n_data, seed)

# In[7]:

def get_cgts(df):

    all_cdc = []
    all_cdcs = []
    all_pdc = []
    all_pdcs = []

    for cdc, cdcs, pdc, pdcs in zip(df.c_hat_distance_c, df.c_hat_distance_c_square, df.p_distance_c, df.p_distance_c_square):
    
        if len(np.array(cdc)) != 4:
            print(cdc)
    
        all_cdc.append(np.array(cdc))
        all_cdcs.append(np.array(cdcs))
        all_pdc.append(np.array(pdc))
        all_pdcs.append(np.array(pdcs))
    
    all_cdc = np.array(all_cdc)
    all_cdcs = np.array(all_cdcs)
    all_pdc = np.array(all_pdc)
    all_pdcs = np.array(all_pdcs)
    
    dfs = []

    for i in range(4):
    
        if len(all_cdc.shape) == 1:
            print()

        df_new = df.copy()
        df_new.c_hat_distance_c = all_cdc[:,i]
        df_new.c_hat_distance_c_square = all_cdcs[:,i]
        df_new.p_distance_c = all_pdc[:,i]
        df_new.p_distance_c_square = all_pdcs[:,i]
        df_new.cgt_nr = i
        
        dfs.append(df_new)
        
    return pd.concat(dfs)


def prep_ECE(files_ECE, columns, path, id_tag):

    dfs = []

    for file in files_ECE:
        #print(file)
        cal_fn, data_name, tag_name, cgt_nr, _, _ = get_strs(file)
        with open(join(path, file), "rb") as f:
            df = pickle.load(f)
            df["calibration_function"] = cal_fn
            df["model_name"] = data_name
            df["tag_name"] = tag_name
            df["cgt_nr"] = cgt_nr
            dfs.append(df)

    df_ECE = pd.concat(dfs)


    # Binning column = full method name
    df_ECE["binning"] = df_ECE["binning"] + "_" + df_ECE["n_bins"].map(str) + "_" + df_ECE["n_folds"].map(str)
    # Remove CV marker from no CV rows
    df_ECE["binning"] = df_ECE['binning'].str.replace('(_0$)', "")

    # ECE drop useless columns
    df_ECE = df_ECE.drop(labels=['n_folds'], axis=1)

    # ECE rename columns to match PW
    df_ECE = df_ECE.rename({"ECE_abs":"c_hat_distance_p", "ECE_abs_debiased": "c_hat_distance_p_debiased",
                            "ECE_square":"c_hat_distance_p_square", "ECE_square_debiased":"c_hat_distance_p_square_debiased",
                           "true_calibration_error_abs":"p_distance_c", "true_calibration_error_square":"p_distance_c_square",
                           "slope_abs_c_hat_dist_c": "c_hat_distance_c", "slope_square_c_hat_dist_c": "c_hat_distance_c_square"}, axis=1)


    df_ECE = df_ECE[columns]
    df_ECE.to_pickle("res_ECE_%s.pkl" % id_tag, protocol=4)

def prep_PW(files_PW, columns, path, id_tag):


    dfs = []

    for file in files_PW:
        #print(file)
        cal_fn, data_name, tag_name, cgt_nr, _, _ = get_strs(file, is_ece = False)
        with open(join(path, file), "rb") as f:
            df = pickle.load(f)
            df["calibration_function"] = cal_fn
            df["model_name"] = data_name
            df["tag_name"] = tag_name
            df["cgt_nr"] = cgt_nr
            dfs.append(df)



    df_PW = pd.concat(dfs)
    
    #df_PW.to_pickle("res_PW_%s_test.pkl" % id_tag, protocol=4)
    

#    binnings = df_PW.binning.unique()

#    binning_with_trick = []

#    for binning in binnings:
#        if "trick" in binning:
#            binning_with_trick.append(binning)
            
#    for bwt in binning_with_trick:
#        df_PW = df_PW.loc[df_PW.binning != bwt]  # Drop trick
        
    print(df_PW.binning.unique())


    # Create dummy columns for our method
    df_PW["c_hat_distance_p_debiased"] = df_PW["c_hat_distance_p"]
    df_PW["c_hat_distance_p_square_debiased"] = df_PW["c_hat_distance_p_square"]

    # Unify calibration_function name column to match ECE_df
    df_PW["calibration_function"] = df_PW['calibration_function'].str.replace('(_[0-9].[0-9]+$)', "")

    df_PW = get_cgts(df_PW)
    df_PW = df_PW[columns]
    df_PW.to_pickle("res_PW_%s.pkl" % id_tag, protocol=4)

def prep_BIP(files_BIP, columns, path, id_tag):

    dfs = []

    for file in files_BIP:
        #print(file)
        cal_fn, data_name, tag_name, cgt_nr, n_data, seed = get_strs(file, is_ece = False, is_bip = True)
        with open(join(path, file), "rb") as f:
            df = pickle.load(f)
            df["calibration_function"] = cal_fn
            df["model_name"] = data_name
            df["tag_name"] = tag_name
            df["cgt_nr"] = cgt_nr
            df["n_data"] = n_data
            df["seed"] = seed
            df["p_distance_c"] = -1
            df["p_distance_c_squared"] = -1
            dfs.append(df)

    df_BIP = pd.concat(dfs)
    df_BIP = df_BIP.sort_values(by=["binning", "n_data", "calibration_function", "model_name", "tag_name", "cgt_nr", "seed"])

    with open("res_PW_%s.pkl" % id_tag, "rb") as f:
        res_PW = pickle.load(f)

    bins_uniq = res_PW.binning.unique()
    print(bins_uniq)

    sel = res_PW.loc[res_PW.binning == bins_uniq[0]].sort_values(by=["binning", "n_data", "calibration_function", "model_name", "tag_name", "cgt_nr", "seed"])

    p_dists = sel.loc[:, ["p_distance_c", "p_distance_c_square"]].values
    p_dists_x3 = np.concatenate([p_dists, p_dists, p_dists])

    df_BIP["p_distance_c"] = p_dists_x3[:, 0]
    df_BIP["p_distance_c_square"] = p_dists_x3[:, 1]

    # df_BIP preprocessing

    # Create dummy columns for our method
    df_BIP["c_hat_distance_p_debiased"] = df_BIP["c_hat_distance_p"]
    df_BIP["c_hat_distance_p_square_debiased"] = df_BIP["c_hat_distance_p_square"]

    # Unify calibration_function name column to match ECE_df
    df_BIP["calibration_function"] = df_BIP['calibration_function'].str.replace('(_[0-9].[0-9]+$)', "")

    df_BIP = df_BIP[columns]
    df_BIP.to_pickle("res_BIP_%s.pkl" % id_tag, protocol=4)


def prep_KDE(files_KDE, columns, path, id_tag):


    dfs = []

    for file in files_KDE:
        #print(file)
        cal_fn, data_name, tag_name, cgt_nr, _, _ = get_strs(file, is_ece = False, is_kde = True) #cal_method, data_name, tag_name, cgt_nr, n_data, seed
        with open(join(path, file), "rb") as f:
            df = pickle.load(f)
            df["calibration_function"] = cal_fn
            df["model_name"] = data_name
            df["tag_name"] = tag_name
            df["cgt_nr"] = cgt_nr
            dfs.append(df)


    df_KDE = pd.concat(dfs)


    for i, row in df_KDE.iterrows():
        if isinstance(row.c_hat_distance_p, tuple):
            row.c_hat_distance_p = row.c_hat_distance_p[0]

    vals = np.array(df_KDE.loc[df_KDE.binning == "KDE_integral", "c_hat_distance_p"].values)
    vals = [i[0] for i in vals]


    df_KDE.loc[df_KDE.binning == "KDE_integral", "c_hat_distance_p"] = vals
    df_KDE = get_cgts(df_KDE)


    # Create dummy columns for our method
    df_KDE["c_hat_distance_p_debiased"] = df_KDE["c_hat_distance_p"]
    df_KDE["c_hat_distance_p_square_debiased"] = df_KDE["c_hat_distance_p_square"]

    # Unify calibration_function name column to match ECE_df
    df_KDE["calibration_function"] = df_KDE['calibration_function'].str.replace('(_[0-9].[0-9]+$)', "")
    
    df_KDE = df_KDE[columns]
    
    df_KDE.to_pickle("res_KDE_%s.pkl" % id_tag, protocol=4)


# MAIN


IDENT_TAG = "28_05_pre"
PATH_res = "data_1m_final_2805"
COLUMNS = ["model_name", "tag_name", "cgt_nr", "seed", "n_data", "binning", "n_bins", "c_hat_distance_p", "c_hat_distance_p_square", "c_hat_distance_p_debiased", 
          "c_hat_distance_p_square_debiased", "c_hat_distance_c", "c_hat_distance_c_square", "p_distance_c", "p_distance_c_square", "calibration_function"]



files_ECE = []
files_PW = []
files_KDE = []
files_BIP = []  # beta, iso, platt, 
for file in os.listdir(PATH_res):
    if file.endswith(".pkl") and not "_m_" in file:
        if file.startswith("binning"):
            files_ECE.append(file)
        elif file.startswith("df_seed"):
            if ("gt0_" in file) or ("gt1_" in file) or ("gt2_" in file) or ("gt3_" in file):
                files_BIP.append(file)
            else:
                files_PW.append(file)
        elif file.startswith("KDE"):
            files_KDE.append(file)



print("ECE files:", len(files_ECE)) # cgt - 612*4, 44 missing? # TODO why? - 44 puudu
print("KDE files:", len(files_KDE)) # Right amount
print("PW files:", len(files_PW)) # PW_NN_mono + PW_NN_SWEEP # Mis siin puudu? 612*10 = 6120` rIGHT AMount
print("BIP files:", len(files_BIP))  # Right amount

print("Start prepping")

#files_ECE = []


if len(files_ECE) != 0:
    prep_ECE(files_ECE, COLUMNS, PATH_res, IDENT_TAG)
    print("ECE prepped")
if len(files_PW) != 0:
    prep_PW(files_PW, COLUMNS, PATH_res, IDENT_TAG)
    print("PW prepped")
if len(files_BIP) != 0:
    prep_BIP(files_BIP, COLUMNS, PATH_res, IDENT_TAG)
    print("BIP prepped")
if len(files_KDE) != 0:
    prep_KDE(files_KDE, COLUMNS, PATH_res, IDENT_TAG)
    print("KDE prepped")


# ### Put all together

res_dfs = []


if len(files_KDE) != 0:
    with open("res_KDE_%s.pkl" % IDENT_TAG, "rb") as f:
        res_KDE = pd.read_pickle(f)
    res_dfs.append(res_KDE)
if len(files_PW) != 0:
    with open("res_PW_%s.pkl" % IDENT_TAG, "rb") as f:
        res_PW = pd.read_pickle(f)
    res_dfs.append(res_PW)

if len(files_ECE) != 0:
    with open("res_ECE_%s.pkl" % IDENT_TAG, "rb") as f:
        res_ECE = pd.read_pickle(f)
    res_dfs.append(res_ECE)
if len(files_BIP) != 0:
    with open("res_BIP_%s.pkl" % IDENT_TAG, "rb") as f:
        res_BIP = pd.read_pickle(f)
    res_dfs.append(res_BIP)



# In[94]:


all_df = pd.concat(res_dfs)
all_df.reset_index(inplace=True, drop = True)


# Filter BIN NR from method name for CV rows
all_df["binning"] = all_df['binning'].str.replace('monotonic_eq_size_.+', "monotonic_eq_size")
for bin_nr in [5,10]:
    for bin_type in ["width", "size"]:
        all_df["binning"] = all_df['binning'].str.replace(f'eq_{bin_type}_([0-9]+)_{bin_nr}_trick', f"eq_{bin_type}_CV{bin_nr}tr")
        all_df["binning"] = all_df['binning'].str.replace(f'eq_{bin_type}_([0-9]+)_{bin_nr}', f"eq_{bin_type}_CV{bin_nr}")


all_df["ECE_abs"] = np.abs(all_df["c_hat_distance_p_debiased"] - all_df["p_distance_c"])
all_df["ECE_square"] = np.abs(all_df["c_hat_distance_p_square_debiased"] - all_df["p_distance_c_square"])


# ## Save to pickle file


all_df.to_pickle("df_all_%s.pkl" % IDENT_TAG, protocol=4)

print("All data saved to %s" % ("df_all_%s.pkl" % IDENT_TAG))



