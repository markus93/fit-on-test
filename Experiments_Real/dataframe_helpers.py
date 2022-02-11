from scipy import stats
import pandas as pd

def construct_data_row_from_binning(binning, calibration_function_name,
                                    true_calibration_error,
                                    seed, n_data, error_type, expected_calibration_error):
    """ Constructs a data row (dictionary) for a binning scheme.
    """

    if error_type == "square":
        c_hat_distance_p = binning.c_hat_distance_p_square
    elif error_type == "absolute":
        c_hat_distance_p = binning.c_hat_distance_p_abs

    return {"seed": seed,
            "n_data": n_data,
            "calibration_function": calibration_function_name,
            "binning": binning.binning_name,
            "n_bins": binning.n_bins,
            "error_type": error_type,
            "c_hat_distance_p": c_hat_distance_p,
            "p_distance_c": true_calibration_error,
            "expected_calibration_error": expected_calibration_error
            }


def select(df, n_data, binning, error_type):
    return df.loc[(df["n_data"] == n_data) \
                  & (df["binning"] == binning) \
                  & (df["error_type"] == error_type)]


def construct_correlation_data_row(selection):
    spearman = stats.spearmanr(selection["c_hat_distance_p"], selection["p_distance_c"])[0]
    pearson = stats.pearsonr(selection["c_hat_distance_p"], selection["p_distance_c"])[0]

    return {"n_data": selection["n_data"].iloc[0],
            "binning": selection["binning"].iloc[0],
            "n_bins": selection["n_bins"].iloc[0],
            "error_type": selection["error_type"].iloc[0],
            "spearman": spearman,
            "pearson": pearson
            }


def load_data(seeds, calibration_functions):

    dataframes = [pd.read_pickle(f"seed_{seed}_{calibration_fun.__name__}.pkl")
                  for seed in seeds
                  for calibration_fun in calibration_functions]

    return pd.concat(dataframes, ignore_index=True)
