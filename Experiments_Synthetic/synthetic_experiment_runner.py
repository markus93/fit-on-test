import pandas as pd
import time
import argparse

from data_generation import *
from calibration_functions import *
from calibration_function_derivates import *
from binnings import *
from piecewise_linear import Piecewise_NN6, Piecewise_NN4

from run_method import run_method

import os.path


def run_tests(cf_name, derivate, beta_dist, dist_name, n_data, data_seed, methods, methods_name):

    dataframes = {"metadata": None, "data": None}
    rows = {"metadata": [], "data": []}

    s = time()

    derivate_fun_name, derivate_fun, expected_calibration_error, error_fun = derivate
    p, y, c = generate_data(beta_dist, n_data, derivate_fun, seed=data_seed)
    p_test, _, c_test = generate_data(beta_dist, 1_000_000, derivate_fun, seed=0)

    print(f"CE {expected_calibration_error}")
    true_calibration_error_abs = np.mean(np.abs(p - c))
    true_calibration_error_square = np.mean(np.square(p - c))

    f_name = f"results_{methods_name}/d{dist_name}_n{n_data}_s{data_seed}_{cf_name}_ce{expected_calibration_error}_{methods_name}.pkl"
    if os.path.exists(f_name):
        print("Already exists!")
        return

    for method in methods:
        print("Starting " + method["method_name"])
        data_rows_method, metadata_rows_method = run_method(method=method, p=p, y=y, c=c,
                                                            calibration_function=derivate_fun,
                                                            p_test=p_test, c_test=c_test)
        rows["data"].extend(data_rows_method)
        rows["metadata"].extend(metadata_rows_method)

    for data_type, rows in rows.items():
        df = pd.DataFrame(rows)
        df["expected_calibration_error"] = expected_calibration_error
        df["p_distance_c"] = true_calibration_error_abs
        df["p_distance_c_square"] = true_calibration_error_square
        df["seed"] = data_seed
        df["calibration_function"] = cf_name
        df["n_data"] = n_data
        df["distribution"] = dist_name
        dataframes[data_type] = df

    print(f"Derivate done in {time() - s}s")

    dataframes["data"].to_pickle(
        f_name,
        protocol=4)

    dataframes["metadata"].to_pickle(
        f"results_{methods_name}/META_d{dist_name}_n{n_data}_s{data_seed}_{cf_name}_ce{expected_calibration_error}_{methods_name}.pkl",
        protocol=4)


if __name__ == "__main__":
    # Methods
    max_nodes_pw_nn = 15
    model_seed = 0
    equal_size_start = True
    n_cv_folds = 10
    lr = 0.01
    patience = 20

    other_methods = [{"method_name": "eq_size", "n_bins": n_bins, "n_cv_folds": None} for n_bins in range(1, 41, 1)] \
                    + [{"method_name": "eq_width", "n_bins": n_bins, "n_cv_folds": None} for n_bins in range(1, 41, 1)] \
                    + [{"method_name": "eq_size", "n_cv_folds": n_cv_folds},
                       {"method_name": "eq_width", "n_cv_folds": n_cv_folds},
                       {"method_name": "monotonic_eq_size", "n_bins": None, "n_cv_folds": None},
                       {"method_name": "KDE"},
                       {"method_name": "cross_entropy_loss"},
                       {"method_name": "brier_score"},
                       {"method_name": "beta"},
                       {"method_name": "platt"},
                       {"method_name": "isotonic"},
                       {"method_name": "KCE"} # REMOVE THIS IF NO JULIA
                       ]

    pw_nn4_bs_methods = [{"method_name": "PW_NN", "n_cv_folds": None, "use_sweep": True, "monotonic": False,
                          "max_nodes": max_nodes_pw_nn, "fn_method": Piecewise_NN4, "seed": model_seed,
                          "equal_size": equal_size_start, "nn_number": 4, "use_ce_loss": False, "logit_scale": False,
                          "logistic_out": False, "lr": lr, "patience": patience},
                         {"method_name": "PW_NN", "n_cv_folds": n_cv_folds, "use_sweep": False, "monotonic": False,
                          "max_nodes": max_nodes_pw_nn, "fn_method": Piecewise_NN4, "seed": model_seed,
                          "equal_size": equal_size_start, "nn_number": 4, "use_ce_loss": False, "logit_scale": False,
                          "logistic_out": False, "lr": lr, "patience": patience}
                         ]

    pw_nn4_ce_methods = [{"method_name": "PW_NN", "n_cv_folds": None, "use_sweep": True, "monotonic": False,
                          "max_nodes": max_nodes_pw_nn, "fn_method": Piecewise_NN4, "seed": model_seed,
                          "equal_size": equal_size_start, "nn_number": 4, "use_ce_loss": True, "logit_scale": False,
                          "logistic_out": False, "lr": lr, "patience": patience},
                         {"method_name": "PW_NN", "n_cv_folds": n_cv_folds, "use_sweep": False, "monotonic": False,
                          "max_nodes": max_nodes_pw_nn, "fn_method": Piecewise_NN4, "seed": model_seed,
                          "equal_size": equal_size_start, "nn_number": 4, "use_ce_loss": True, "logit_scale": False,
                          "logistic_out": False, "lr": lr, "patience": patience}]

    pw_nn6_bs_methods = [{"method_name": "PW_NN", "n_cv_folds": None, "use_sweep": True, "monotonic": False,
                          "max_nodes": max_nodes_pw_nn, "fn_method": Piecewise_NN6, "seed": model_seed,
                          "equal_size": equal_size_start, "nn_number": 6, "use_ce_loss": False, "logit_scale": True,
                          "logistic_out": True, "lr": lr, "patience": patience},
                         {"method_name": "PW_NN", "n_cv_folds": n_cv_folds, "use_sweep": False, "monotonic": False,
                          "max_nodes": max_nodes_pw_nn, "fn_method": Piecewise_NN6, "seed": model_seed,
                          "equal_size": equal_size_start, "nn_number": 6, "use_ce_loss": False, "logit_scale": True,
                          "logistic_out": True, "lr": lr, "patience": patience}]

    pw_nn6_ce_methods = [{"method_name": "PW_NN", "n_cv_folds": None, "use_sweep": True, "monotonic": False,
                          "max_nodes": max_nodes_pw_nn, "fn_method": Piecewise_NN6, "seed": model_seed,
                          "equal_size": equal_size_start, "nn_number": 6, "use_ce_loss": True, "logit_scale": True,
                          "logistic_out": True, "lr": lr, "patience": patience},
                         {"method_name": "PW_NN", "n_cv_folds": n_cv_folds, "use_sweep": False, "monotonic": False,
                          "max_nodes": max_nodes_pw_nn, "fn_method": Piecewise_NN6, "seed": model_seed,
                          "equal_size": equal_size_start, "nn_number": 6, "use_ce_loss": True, "logit_scale": True,
                          "logistic_out": True, "lr": lr, "patience": patience}]

    pwlf_methods = [{"method_name": "pwlf", "n_cv_folds": None, "use_sweep": True, "max_nodes": 7,
                     "seed": model_seed, "degree": 1},
                    {"method_name": "pwlf", "n_cv_folds": n_cv_folds, "use_sweep": False, "max_nodes": 7,
                     "seed": model_seed, "degree": 1}
                    ] \
                   + [{"method_name": "pwlf", "n_cv_folds": None, "use_sweep": True, "max_nodes": 5,
                       "seed": model_seed, "degree": 2},
                      {"method_name": "pwlf", "n_cv_folds": n_cv_folds, "use_sweep": False, "max_nodes": 5,
                       "seed": model_seed, "degree": 2}
                      ]

    methods = [(other_methods, "other"),
               (pw_nn4_bs_methods, "pw_nn4_bs"),
               (pw_nn4_ce_methods, "pw_nn4_ce"),
               (pw_nn6_bs_methods, "pw_nn6_logit_bs"),
               (pw_nn6_ce_methods, "pw_nn6_logit_ce"),
               (pwlf_methods, "pwlf")]

    # Add arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--cal_fn_idx', '-i', type=int, default=0, help="Calibration function number (0,1,2,3,4)")
    parser.add_argument('--n_data_idx', '-d', type=int, default=0, help="Number of data points generated (0,1,2)")
    parser.add_argument('--methods_idx', '-m', type=int, default=0, help="Method group to run")
    parser.add_argument('--dist_nr', '-b', type=int, default=0, help="Use uniform (0) or beta(1.1,0.1) (1)")
    parser.add_argument('--ce_nr', '-c', type=int, default=0, help="Expected CE number (0,1,2,..,20)")
    parser.add_argument('--data_seed', '-s', type=int, default=0, help="Data seed to run")

    args = parser.parse_args()
    n_data = [1000, 3000, 10_000][args.n_data_idx]
    dist_nr = args.dist_nr
    ce_nr = args.ce_nr
    data_seed = args.data_seed

    cal_fn = calibration_functions[args.cal_fn_idx]
    cf_name = cal_fn.__name__
    calibration_error = [(error, np.abs) for error in np.arange(0.0, 0.10001, 0.005)][ce_nr]

    method_selection = methods[args.methods_idx][0]
    method_selection_name = methods[args.methods_idx][1]

    if args.dist_nr == 0:
        print("Beta uniform distribution used!")
        beta_dist = [1, 1]
        dist_name = "uniform"
    elif args.dist_nr == 1:
        print("Beta(1.1,0.1) distribution used")
        beta_dist = [1.1, 0.1]
        dist_name = "beta"

    print(f"cal fn: {cal_fn}")
    print(f"n_data: {n_data}")
    print(f"methods: {method_selection_name}")
    print(f"dist: {dist_name}")
    print(f"ce: {calibration_error[0]}")
    print(f"seed: {data_seed}")

    all_derivate_functions = find_all_derivates_for_calibration_functions([cal_fn], [calibration_error],
                                                                          beta_alpha=beta_dist[0],
                                                                          beta_beta=beta_dist[1])
    derivate = all_derivate_functions[cf_name][0]

    run_tests(cf_name=cf_name, derivate=derivate, beta_dist=beta_dist, dist_name=dist_name, n_data=n_data,
              data_seed=data_seed, methods=method_selection, methods_name=method_selection_name)
