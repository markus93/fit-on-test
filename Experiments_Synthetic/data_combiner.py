import numpy as np
import pandas as pd

from calibration_functions import *

datas = []
metadatas = []

for folder in ["pw_nn4_bs", "pw_nn4_ce", "pw_nn6_logit_bs", "other", "pwlf"
               "pw_nn6_logit_ce"]:
    path = "results_" + folder
    print(folder)
    for cf in calibration_functions: #
        print(cf.__name__)
        for n_data in [1000, 3000, 10000]:

            for seed in [0,1,2,3,4]:

                dist = "uniform"
                for ce in np.arange(0.0, 0.1001, 0.005):
                    try:
                        data = pd.read_pickle(f"{path}/d{dist}_n{n_data}_s{seed}_{cf.__name__}_ce{ce}_{folder}.pkl")
                        metadata = pd.read_pickle(
                            f"{path}/META_d{dist}_n{n_data}_s{seed}_{cf.__name__}_ce{ce}_{folder}.pkl")

                        if folder == "pw_nn4_bs":
                            data.loc[data.binning.str[-2:] != "bs", "binning"] = data[data.binning.str[-2:] != "bs"].binning + "_bs"
                            metadata.loc[metadata.binning.str[-2:] != "bs", "binning"] = metadata[metadata.binning.str[-2:] != "bs"].binning + "_bs"
                        if folder == "pw_nn4_ce":
                            data.loc[data.binning.str[-2:] != "ce", "binning"] = data[data.binning.str[-2:] != "ce"].binning + "_ce"
                            metadata.loc[metadata.binning.str[-2:] != "ce", "binning"] = metadata[metadata.binning.str[-2:] != "ce"].binning + "_ce"
                        if folder == "pw_nn4_mono_bs":
                            data.loc[data.binning.str[-2:] != "bs", "binning"] = data[data.binning.str[-2:] != "bs"].binning + "_bs"
                            metadata.loc[metadata.binning.str[-2:] != "bs", "binning"] = metadata[metadata.binning.str[-2:] != "bs"].binning + "_bs"
                        if folder == "pw_nn4_mono_ce":
                            data.loc[data.binning.str[-2:] != "ce", "binning"] = data[data.binning.str[-2:] != "ce"].binning + "_ce"
                            metadata.loc[metadata.binning.str[-2:] != "ce", "binning"] = metadata[metadata.binning.str[-2:] != "ce"].binning + "_ce"
                        if folder == "pw_nn4_mono_logit_bs":
                            data.loc[data.binning.str[-2:] != "bs", "binning"] = data[data.binning.str[-2:] != "bs"].binning + "_bs"
                            metadata.loc[metadata.binning.str[-2:] != "bs", "binning"] = metadata[metadata.binning.str[-2:] != "bs"].binning + "_bs"
                        if folder == "pw_nn4_mono_logit_ce":
                            data.loc[data.binning.str[-2:] != "ce", "binning"] = data[data.binning.str[-2:] != "ce"].binning + "_ce"
                            metadata.loc[metadata.binning.str[-2:] != "ce", "binning"] = metadata[metadata.binning.str[-2:] != "ce"].binning + "_ce"

                        datas.append(data)
                        metadatas.append(metadata)
                    except:
                        print(f"{path}/d{dist}_n{n_data}_s{seed}_{cf.__name__}_ce{ce}_{folder}.pkl")

                dist = "beta"
                for ce in np.arange(0.0, 0.0301, 0.005):
                    try:
                        data = pd.read_pickle(f"{path}/d{dist}_n{n_data}_s{seed}_{cf.__name__}_ce{ce}_{folder}.pkl")
                        metadata = pd.read_pickle(
                            f"{path}/META_d{dist}_n{n_data}_s{seed}_{cf.__name__}_ce{ce}_{folder}.pkl")

                        if folder == "pw_nn4_bs":
                            data.loc[data.binning.str[-2:] != "bs", "binning"] = data[data.binning.str[-2:] != "bs"].binning + "_bs"
                            metadata.loc[metadata.binning.str[-2:] != "bs", "binning"] = metadata[metadata.binning.str[-2:] != "bs"].binning + "_bs"
                        if folder == "pw_nn4_ce":
                            data.loc[data.binning.str[-2:] != "ce", "binning"] = data[data.binning.str[-2:] != "ce"].binning + "_ce"
                            metadata.loc[metadata.binning.str[-2:] != "ce", "binning"] = metadata[metadata.binning.str[-2:] != "ce"].binning + "_ce"
                        if folder == "pw_nn4_mono_bs":
                            data.loc[data.binning.str[-2:] != "bs", "binning"] = data[data.binning.str[-2:] != "bs"].binning + "_bs"
                            metadata.loc[metadata.binning.str[-2:] != "bs", "binning"] = metadata[metadata.binning.str[-2:] != "bs"].binning + "_bs"
                        if folder == "pw_nn4_mono_ce":
                            data.loc[data.binning.str[-2:] != "ce", "binning"] = data[data.binning.str[-2:] != "ce"].binning + "_ce"
                            metadata.loc[metadata.binning.str[-2:] != "ce", "binning"] = metadata[metadata.binning.str[-2:] != "ce"].binning + "_ce"
                        if folder == "pw_nn4_mono_logit_bs":
                            data.loc[data.binning.str[-2:] != "bs", "binning"] = data[data.binning.str[-2:] != "bs"].binning + "_bs"
                            metadata.loc[metadata.binning.str[-2:] != "bs", "binning"] = metadata[metadata.binning.str[-2:] != "bs"].binning + "_bs"
                        if folder == "pw_nn4_mono_logit_ce":
                            data.loc[data.binning.str[-2:] != "ce", "binning"] = data[data.binning.str[-2:] != "ce"].binning + "_ce"
                            metadata.loc[metadata.binning.str[-2:] != "ce", "binning"] = metadata[metadata.binning.str[-2:] != "ce"].binning + "_ce"

                        datas.append(data)
                        metadatas.append(metadata)

                    except:
                        print(f"{path}/d{dist}_n{n_data}_s{seed}_{cf.__name__}_ce{ce}_{folder}.pkl")

df_all_metadata = pd.concat(metadatas)
df_all_data = pd.concat(datas)

df_all_data.to_pickle("DATANAME.pkl", protocol=4)
df_all_metadata.to_pickle("METADATANAME.pkl", protocol=4)

#####

from os import walk
import re

_, _, filenames = next(walk("./"))

times_taken = np.zeros(shape=(10, 10))

for f_name in filenames:

    if f_name[-3:] != "out":
        continue

    with open("./" + f_name) as f:
        data = f.read()
        der_done = [m.start() for m in re.finditer('Derivate done in', data)]

        if data.find("Traceback") != -1:
            print("Error in")
            print(f_name)

        for done_idx, done in enumerate(der_done):
            time_taken = float(data[done + 17:done + 25])
            times_taken[len(der_done)][done_idx] += time_taken

print(times_taken)
print(np.sum(times_taken) / 60 / 60)
print(np.sum(times_taken, axis=1) / 60 / 60)
print(np.sum(times_taken, axis=0) / 60 / 60)
