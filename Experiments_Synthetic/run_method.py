import gc
from keras import backend as K
import tensorflow as tf
from sklearn.metrics import log_loss

from binnings import *
from kde import KDE_estimator
from data_generation import generate_data

from pycalib.models import IsotonicCalibration, SigmoidCalibration
from betacal import BetaCalibration
from piecewise_linear import node_scores_xy_with_crossvalidation

import pwlf
from pwlf_cv import node_scores_pwlf_with_crossvalidation
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression


def construct_data_row(n_folds=None, binning_name=None, n_bins=None,
                       c_hat_distance_p=None, c_hat_distance_p_debiased=None,
                       c_hat_distance_p_square=None, c_hat_distance_p_square_debiased=None,
                       c_hat_distance_c=None, c_hat_distance_c_square=None,

                       test_c_hat_distance_p=None, test_c_hat_distance_p_debiased=None,
                       test_c_hat_distance_p_square=None, test_c_hat_distance_p_square_debiased=None,
                       test_c_hat_distance_c=None, test_c_hat_distance_c_square=None):
    return {
        "n_folds": n_folds,
        "binning": binning_name,
        "n_bins": n_bins,

        "c_hat_distance_p": c_hat_distance_p,
        "c_hat_distance_p_debiased": c_hat_distance_p_debiased,

        "c_hat_distance_p_square": c_hat_distance_p_square,
        "c_hat_distance_p_square_debiased": c_hat_distance_p_square_debiased,

        "c_hat_distance_c": c_hat_distance_c,
        "c_hat_distance_c_square": c_hat_distance_c_square,

        "test_c_hat_distance_p": test_c_hat_distance_p,
        "test_c_hat_distance_p_debiased": test_c_hat_distance_p_debiased,

        "test_c_hat_distance_p_square": test_c_hat_distance_p_square,
        "test_c_hat_distance_p_square_debiased": test_c_hat_distance_p_square_debiased,

        "test_c_hat_distance_c": test_c_hat_distance_c,
        "test_c_hat_distance_c_square": test_c_hat_distance_c_square
    }


def construct_data_row_from_raw_data(n_folds=None, binning_name=None, n_bins=None,
                                     c_hat=None, p=None, c=None,
                                     c_hat_test=None, p_test=None, c_test=None):
    c_hat_diff_p = c_hat - p
    c_hat_diff_c = c_hat - c

    c_hat_diff_p_test = c_hat_test - p_test
    c_hat_diff_c_test = c_hat_test - c_test

    return construct_data_row(n_folds=n_folds, binning_name=binning_name, n_bins=n_bins,
                              c_hat_distance_p=np.mean(np.abs(c_hat_diff_p)),
                              c_hat_distance_p_debiased=np.mean(np.abs(c_hat_diff_p)),
                              c_hat_distance_p_square=np.mean(np.square(c_hat_diff_p)),
                              c_hat_distance_p_square_debiased=np.mean(np.square(c_hat_diff_p)),
                              c_hat_distance_c=np.mean(np.abs(c_hat_diff_c)),
                              c_hat_distance_c_square=np.mean(np.square(c_hat_diff_c)),

                              test_c_hat_distance_p=np.mean(np.abs(c_hat_diff_p_test)),
                              test_c_hat_distance_p_debiased=np.mean(np.abs(c_hat_diff_p_test)),
                              test_c_hat_distance_p_square=np.mean(np.square(c_hat_diff_p_test)),
                              test_c_hat_distance_p_square_debiased=np.mean(np.square(c_hat_diff_p_test)),
                              test_c_hat_distance_c=np.mean(np.abs(c_hat_diff_c_test)),
                              test_c_hat_distance_c_square=np.mean(np.square(c_hat_diff_c_test))
                              )


def construct_data_row_from_binning(binning, n_folds, p_test, c_test, name_addition=""):
    c_hat = binning.eval_slope_1(binning.p)

    c_hat_test = binning.eval_slope_1(p_test)
    c_hat_diff_p_test = c_hat_test - p_test
    c_hat_diff_c_test = c_hat_test - c_test

    return construct_data_row(n_folds=n_folds, binning_name=binning.binning_name + name_addition, n_bins=binning.n_bins,
                              c_hat_distance_p=binning.ECE_abs,
                              c_hat_distance_p_debiased=binning.ECE_abs_debiased,
                              c_hat_distance_p_square=binning.ECE_square,
                              c_hat_distance_p_square_debiased=binning.ECE_square_debiased,
                              c_hat_distance_c=np.mean(np.abs(c_hat - binning.c)),
                              c_hat_distance_c_square=np.mean(np.square(c_hat - binning.c)),

                              test_c_hat_distance_p=np.mean(np.abs(c_hat_diff_p_test)),
                              test_c_hat_distance_p_debiased=np.mean(np.abs(c_hat_diff_p_test)),
                              test_c_hat_distance_p_square=np.mean(np.square(c_hat_diff_p_test)),
                              test_c_hat_distance_p_square_debiased=np.mean(np.square(c_hat_diff_p_test)),
                              test_c_hat_distance_c=np.mean(np.abs(c_hat_diff_c_test)),
                              test_c_hat_distance_c_square=np.mean(np.square(c_hat_diff_c_test))
                              )


def construct_metadata_row_from_fit_model_data(model_name, n_nodes, n_nodes_trick, cv_folds, node_scores,
                                               node_loss, node_ECEs_abs, node_ECEs_square, all_weights,
                                               weights_final, weights_trick, all_cv_scores, last_epoch):
    return {"binning": model_name,
            "n_bins": n_nodes,
            "n_bins_trick": n_nodes_trick,
            "cv_folds": cv_folds,
            "node_scores": node_scores,
            "node_train_loss": node_loss,
            "cv_score": np.min(node_scores),
            "node_ECEs_abs": node_ECEs_abs,
            "node_ECEs_square": node_ECEs_square,
            "model_weights": all_weights,
            "model_weights_final": weights_final,
            "model_weights_trick": weights_trick,
            "all_cv_scores": all_cv_scores,
            "last_epoch": last_epoch
            }


def cv_trick(bin_scores):
    optimal = np.argmin(bin_scores)
    pos_new = np.argmin(bin_scores)
    min_bin_score = np.min(bin_scores)
    max_diff = min_bin_score * 0.001

    for pos in range(optimal - 1, -1, -1):
        new_min_cand = bin_scores[pos]
        if new_min_cand <= min_bin_score + max_diff:
            pos_new = pos

    return pos_new


def gc_model(model):
    # Garbage collect

    del model.model
    del model

    gc.collect()
    K.clear_session()
    tf.compat.v1.reset_default_graph()


def run_PW_NN_nonCV_method(method, p, y, c, p_test, c_test):
    n_cv_folds = method["n_cv_folds"]
    monotonic = method["monotonic"]
    fn_method = method["fn_method"]
    seed = method["seed"]
    equal_size = method["equal_size"]
    use_ce_loss = method["use_ce_loss"]
    logit_scale = method["logit_scale"]
    logistic_out = method["logistic_out"]
    nn_number = method["nn_number"]
    n_bins = method["n_bins"]
    lr = method["lr"]
    patience = method["patience"]

    n_nodes = n_bins - 1
    n_data = len(p)

    name_addition = method["name_addition"]
    add_to_name_mono = "_monotonic" if monotonic else ""
    add_to_name_logit = "_logit" if logit_scale else ""
    add_to_name_loss = "_ce" if use_ce_loss else "_bs"
    add_to_name_lr = f"_lr{lr}"
    add_to_name_patience = f"_p{patience}"
    model_name = "PW_NN" + str(nn_number) + add_to_name_mono + name_addition + add_to_name_logit + add_to_name_loss + add_to_name_lr + add_to_name_patience

    model = fn_method(k=n_nodes, max_epochs=1500, random_state=seed, equal_size=equal_size, monotonic=monotonic,
                      use_ce_loss=use_ce_loss, logit_scale=logit_scale, logistic_out=logistic_out, lr=lr, patience=patience)
    h = model.fit(p, y, verbose=False, batch_size=min(n_data // 4, 512))

    last_epoch = len(h.history['loss'])
    weights_final = model.model.get_weights()

    c_hat = model.predict(p)
    c_hat_test = model.predict(p_test)
    data_row = construct_data_row_from_raw_data(n_folds=n_cv_folds, binning_name=model_name,
                                                n_bins=n_bins, c_hat=c_hat, p=p, c=c,
                                                c_hat_test=c_hat_test, p_test=p_test, c_test=c_test)

    gc_model(model)  # Garbage collection

    return data_row, last_epoch, weights_final


def run_PW_NN_CV_method(method, p, y, c, p_test, c_test):
    n_cv_folds = method["n_cv_folds"]
    monotonic = method["monotonic"]
    max_nodes = method["max_nodes"]
    fn_method = method["fn_method"]
    seed = method["seed"]
    equal_size = method["equal_size"]
    use_ce_loss = method["use_ce_loss"]
    logit_scale = method["logit_scale"]
    logistic_out = method["logistic_out"]
    nn_number = method["nn_number"]
    lr = method["lr"]
    patience = method["patience"]

    n_data = len(p)

    add_to_name_mono = "_monotonic" if monotonic else ""
    add_to_name_logit = "_logit" if logit_scale else ""
    add_to_name_loss = "_ce" if use_ce_loss else "_bs"
    add_to_name_lr = f"_lr{lr}"
    add_to_name_patience = f"_p{patience}"
    model_name = "PW_NN" + str(nn_number) + add_to_name_mono + add_to_name_logit + add_to_name_loss + add_to_name_lr + add_to_name_patience

    start_cv = time()
    node_scores, all_weights, all_cv_scores, node_ECEs_square, node_ECEs_abs, node_loss = node_scores_xy_with_crossvalidation(
        method=fn_method, p_hat=p, y=y,
        n_splits=n_cv_folds, seed=seed,
        max_nodes=min(n_data // 200, max_nodes), equal_size=equal_size,
        monotonic=monotonic,
        use_ce_loss=use_ce_loss,
        logit_scale=logit_scale,
        logistic_out=logistic_out,
        lr=lr,
        patience=patience
    )
    print("Cross-validation took %f seconds" % (time() - start_cv))

    n_nodes = np.argmin(node_scores)
    n_bins = n_nodes + 1
    method["n_bins"] = n_bins
    method["name_addition"] = ""
    data_row, last_epoch, weights_final = run_PW_NN_nonCV_method(method=method, p=p, y=y, c=c, p_test=p_test,
                                                                 c_test=c_test)

    n_nodes_tr = cv_trick(node_scores)
    n_bins_tr = n_nodes_tr + 1
    method["n_bins"] = n_bins_tr
    method["name_addition"] = "tr"
    data_row_tr, last_epoch_tr, weights_final_tr = run_PW_NN_nonCV_method(method=method, p=p, y=y, c=c, p_test=p_test,
                                                                          c_test=c_test)

    metadata_row = construct_metadata_row_from_fit_model_data(model_name=model_name, n_nodes=n_nodes,
                                                              n_nodes_trick=n_nodes_tr, cv_folds=n_cv_folds,
                                                              node_scores=node_scores,
                                                              node_loss=node_loss, node_ECEs_abs=node_ECEs_abs,
                                                              node_ECEs_square=node_ECEs_square,
                                                              all_weights=all_weights,
                                                              weights_final=weights_final,
                                                              weights_trick=weights_final_tr,
                                                              all_cv_scores=all_cv_scores, last_epoch=last_epoch)

    return data_row, data_row_tr, metadata_row


def run_PW_NN_sweep_method(method, p, y, c, p_test, c_test):
    max_nodes = method["max_nodes"]
    fn_method = method["fn_method"]
    seed = method["seed"]
    equal_size = method["equal_size"]
    nn_number = method["nn_number"]
    use_ce_loss = method["use_ce_loss"]
    logit_scale = method["logit_scale"]
    logistic_out = method["logistic_out"]
    lr = method["lr"]
    patience = method["patience"]

    n_data = len(p)

    add_to_name_logit = "_logit" if logit_scale else ""
    add_to_name_loss = "_ce" if use_ce_loss else "_bs"
    add_to_name_lr = f"_lr{lr}"
    add_to_name_patience = f"_p{patience}"
    model_name = "PW_NN" + str(nn_number) + "_sweep" + add_to_name_logit + add_to_name_loss + add_to_name_lr + add_to_name_patience

    assert (not method["monotonic"]), "Trying to sweep monotonic PW_NN method"

    all_weights = []

    for n_nodes in range(max_nodes + 1):
        model = fn_method(k=n_nodes, max_epochs=1500, random_state=seed, equal_size=equal_size, monotonic=False,
                          use_ce_loss=use_ce_loss, logit_scale=logit_scale, logistic_out=logistic_out, lr=lr, patience=patience)
        h = model.fit(p, y, verbose=False, batch_size=min(n_data // 4, 512))
        last_epoch = len(h.history['loss'])
        print("Last epoch", last_epoch)

        weights = model.model.get_weights()
        all_weights.append(weights)

        if nn_number == 5:
            y_w = np.array(weights[-3])  # weights for y  # Weights for first and last breakpoint is not used.
        else:
            y_w = np.array(weights[-1])  # weights for y

        if not np.all((y_w[1:] - y_w[:-1]) > 0):
            print("N_nodes %i is not monotonic" % n_nodes)
            break
        else:
            model_last = model
            last_last_epoch = last_epoch

    print("Get predictions for n_nodes %i!" % (n_nodes - 1))

    c_hat = model_last.predict(p)
    c_hat_test = model_last.predict(p_test)
    data_row = construct_data_row_from_raw_data(n_folds=None, binning_name=model_name,
                                                n_bins=model_last.k + 1, c_hat=c_hat, p=p, c=c,
                                                c_hat_test=c_hat_test, p_test=p_test, c_test=c_test)
    node_loss = np.mean(np.square(c_hat - y))
    weights_final = model_last.model.get_weights()

    metadata_row = construct_metadata_row_from_fit_model_data(model_name=model_name, n_nodes=model_last.k,
                                                              n_nodes_trick=None, cv_folds=None, node_scores=None,
                                                              node_loss=node_loss, node_ECEs_abs=None,
                                                              node_ECEs_square=None, all_weights=all_weights,
                                                              weights_final=weights_final, weights_trick=None,
                                                              all_cv_scores=None, last_epoch=last_last_epoch)
    gc_model(model)

    return data_row, metadata_row


def run_PW_NN_method(method, p, y, c, p_test, c_test):
    n_cv_folds = method["n_cv_folds"]
    use_sweep = method["use_sweep"]

    if use_sweep:
        data_row, metadata_row = run_PW_NN_sweep_method(method=method, p=p, y=y, c=c, p_test=p_test, c_test=c_test)
        return [data_row], [metadata_row]
    elif n_cv_folds is None:
        data_row, _, _ = run_PW_NN_nonCV_method(method=method, p=p, y=y, c=c, p_test=p_test, c_test=c_test)
        return [data_row], []
    else:
        data_row, data_row_tr, metadata_row = run_PW_NN_CV_method(method=method, p=p, y=y, c=c, p_test=p_test,
                                                                  c_test=c_test)
        return [data_row, data_row_tr], [metadata_row]


def run_binning_CV_method(method, p, y, c, p_test, c_test):
    method_name = method["method_name"]
    n_cv_folds = method["n_cv_folds"]
    name_addition = f"_CV{n_cv_folds}"

    assert (method_name == "eq_width" or method_name == "eq_size"), "Method name incorrect in CV binning!"

    use_eq_width = method_name == "eq_width"
    if use_eq_width:
        binning_method = EqualWidthBinning
    else:
        binning_method = EqualSizeBinning

    bin_scores, all_cv_scores = binning_n_bins_with_crossvalidation(p=p, y=y, use_eq_width=use_eq_width,
                                                                    n_splits=n_cv_folds)
    n_bins = np.argmin(bin_scores)
    n_bins_tr = cv_trick(bin_scores)

    binning_cv = binning_method(p, y, c, n_bins)
    binning_cv_tr = binning_method(p, y, c, n_bins_tr)

    data_row_cv = construct_data_row_from_binning(binning=binning_cv, n_folds=n_cv_folds, p_test=p_test, c_test=c_test,
                                                  name_addition=name_addition)
    data_row_cv_tr = construct_data_row_from_binning(binning=binning_cv_tr, n_folds=n_cv_folds, p_test=p_test,
                                                     c_test=c_test,
                                                     name_addition=name_addition + "tr")

    return data_row_cv, data_row_cv_tr


def run_binning_nonCV_method(method, p, y, c, p_test, c_test):
    method_name = method["method_name"]
    n_bins = method["n_bins"]

    if method_name == "eq_size":
        binning = EqualSizeBinning(p, y, c, n_bins)
        name_addition = f"_{n_bins}"
    elif method_name == "eq_width":
        binning = EqualWidthBinning(p, y, c, n_bins)
        name_addition = f"_{n_bins}"
    elif method_name == "monotonic_eq_size":
        binning = MonotonicEqualSizeBinning(p, y, c)
        name_addition = ""

    return construct_data_row_from_binning(binning, n_folds=None, p_test=p_test, c_test=c_test,
                                           name_addition=name_addition)


def run_binning_method(method, p, y, c, p_test, c_test):
    n_cv_folds = method["n_cv_folds"]

    if n_cv_folds is None:
        data_row = run_binning_nonCV_method(method=method, p=p, y=y, c=c, p_test=p_test, c_test=c_test)
        return [data_row], []
    else:
        data_row_cv, data_row_cv_tr = run_binning_CV_method(method=method, p=p, y=y, c=c, p_test=p_test, c_test=c_test)
        return [data_row_cv, data_row_cv_tr], []


def run_kde_method(p, y, c, calibration_function, p_test, c_test):
    kde_estimator = KDE_estimator(p=p, y=y, c=c, calibration_function=calibration_function, p_test=p_test,
                                  c_test=c_test)

    data_row_pointwise = construct_data_row(n_folds=None, binning_name="kde_pointwise", n_bins=None,
                                            c_hat_distance_p=kde_estimator.pointwise_ece_abs,
                                            c_hat_distance_p_debiased=kde_estimator.pointwise_ece_abs,
                                            c_hat_distance_p_square=kde_estimator.pointwise_ece_sq,
                                            c_hat_distance_p_square_debiased=kde_estimator.pointwise_ece_sq,
                                            c_hat_distance_c=kde_estimator.pointwise_c_hat_dist_c_abs,
                                            c_hat_distance_c_square=kde_estimator.pointwise_c_hat_dist_c_sq,

                                            test_c_hat_distance_p=kde_estimator.pointwise_ece_abs_test,
                                            test_c_hat_distance_p_debiased=kde_estimator.pointwise_ece_abs_test,
                                            test_c_hat_distance_p_square=kde_estimator.pointwise_ece_sq_test,
                                            test_c_hat_distance_p_square_debiased=kde_estimator.pointwise_ece_sq_test,
                                            test_c_hat_distance_c=kde_estimator.pointwise_c_hat_dist_c_abs_test,
                                            test_c_hat_distance_c_square=kde_estimator.pointwise_c_hat_dist_c_sq_test
                                            )

    data_row_integral = construct_data_row(n_folds=None, binning_name="kde_integral", n_bins=None,
                                           c_hat_distance_p=kde_estimator.integral_ece_abs,
                                           c_hat_distance_p_debiased=kde_estimator.integral_ece_abs,
                                           c_hat_distance_p_square=kde_estimator.integral_ece_sq,
                                           c_hat_distance_p_square_debiased=kde_estimator.integral_ece_sq,
                                           c_hat_distance_c=kde_estimator.integral_c_hat_dist_c_abs,
                                           c_hat_distance_c_square=kde_estimator.integral_c_hat_dist_c_sq,

                                           test_c_hat_distance_p=None,
                                           test_c_hat_distance_p_debiased=None,
                                           test_c_hat_distance_p_square=None,
                                           test_c_hat_distance_p_square_debiased=None,
                                           test_c_hat_distance_c=None,
                                           test_c_hat_distance_c_square=None
                                           )

    return [data_row_pointwise, data_row_integral], []


def run_isotonic(p, y, c, p_test, c_test):
    isotonic = IsotonicCalibration()
    isotonic.fit(p, y)
    c_hat = isotonic.predict(p)
    c_hat_test = isotonic.predict(p_test)

    data_row = construct_data_row_from_raw_data(n_folds=None, binning_name="isotonic",
                                                n_bins=len(isotonic.X_thresholds_) - 1,
                                                c_hat=c_hat, p=p, c=c,
                                                c_hat_test=c_hat_test, p_test=p_test, c_test=c_test)
    return [data_row], []


def run_platt(p, y, c, p_test, c_test):
    p_clipped = np.clip(p, 1e-8, 1 - 1e-8)
    p_logit = np.log(p_clipped / (1 - p_clipped))

    p_clipped_test = np.clip(p_test, 1e-8, 1 - 1e-8)
    p_logit_test = np.log(p_clipped_test / (1 - p_clipped_test))

    platt = SigmoidCalibration()
    platt.fit(p_logit.reshape(-1, 1), y)

    c_hat = platt.predict_proba(p_logit.reshape(-1, 1))
    c_hat_test = platt.predict_proba(p_logit_test.reshape(-1, 1))

    data_row = construct_data_row_from_raw_data(n_folds=None, binning_name="platt", n_bins=None,
                                                c_hat=c_hat, p=p, c=c,
                                                c_hat_test=c_hat_test, p_test=p_test, c_test=c_test)
    return [data_row], []


def run_beta(p, y, c, p_test, c_test):
    bc = BetaCalibration(parameters="abm")
    bc.fit(p.reshape(-1, 1), y)
    c_hat = bc.predict(p.reshape(-1, 1))
    c_hat_test = bc.predict(p_test.reshape(-1, 1))

    data_row = construct_data_row_from_raw_data(n_folds=None, binning_name="beta", n_bins=None,
                                                c_hat=c_hat, p=p, c=c,
                                                c_hat_test=c_hat_test, p_test=p_test, c_test=c_test)
    return [data_row], []


def run_brier_score(p, y):
    brier_score = np.mean(np.square(p - y))

    data_row = construct_data_row(n_folds=None, binning_name="brier_score", n_bins=None,
                                  c_hat_distance_p=brier_score,
                                  c_hat_distance_p_debiased=brier_score,
                                  c_hat_distance_p_square=brier_score,
                                  c_hat_distance_p_square_debiased=brier_score,
                                  c_hat_distance_c=brier_score,
                                  c_hat_distance_c_square=brier_score)
    return [data_row], []


def run_cross_entropy_loss(p, y):
    cross_entropy_loss = log_loss(y, p)

    data_row = construct_data_row(n_folds=None, binning_name="cross_entropy_loss", n_bins=None,
                                  c_hat_distance_p=cross_entropy_loss,
                                  c_hat_distance_p_debiased=cross_entropy_loss,
                                  c_hat_distance_p_square=cross_entropy_loss,
                                  c_hat_distance_p_square_debiased=cross_entropy_loss,
                                  c_hat_distance_c=cross_entropy_loss,
                                  c_hat_distance_c_square=cross_entropy_loss)

    return [data_row], []


def run_pwlf_nonCV_method(method, p, y, c, p_test, c_test):
    n_cv_folds = method["n_cv_folds"]
    n_bins = method["n_bins"]
    degree = method["degree"]
    name_addition = method["name_addition"]

    model_name = "pwlf_d" + str(degree) + name_addition

    if n_bins == 1:
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        model.fit(p.reshape(-1, 1), y)
        c_hat = model.predict(p.reshape(-1, 1))
        c_hat_test = model.predict(p_test.reshape(-1, 1))
    else:
        model = pwlf.PiecewiseLinFit(p, y, degree=degree)
        model.fit(n_bins)
        c_hat = model.predict(p)
        c_hat_test = model.predict(p_test)

    data_row = construct_data_row_from_raw_data(n_folds=n_cv_folds, binning_name=model_name,
                                                n_bins=n_bins, c_hat=c_hat, p=p, c=c,
                                                c_hat_test=c_hat_test, p_test=p_test, c_test=c_test)

    return data_row


def run_pwlf_CV_method(method, p, y, c, p_test, c_test):
    n_cv_folds = method["n_cv_folds"]
    max_nodes = method["max_nodes"]
    seed = method["seed"]
    n_data = len(p)
    degree = method["degree"]

    model_name = "pwlf_d" + str(degree)

    start_cv = time()
    node_scores, all_weights, all_cv_scores, node_ECEs_square, node_ECEs_abs, node_loss = node_scores_pwlf_with_crossvalidation(
        p_hat=p, y=y,
        n_splits=n_cv_folds, seed=seed,
        max_nodes=min(n_data // 20, max_nodes), degree=degree)

    print("Cross-validation took %f seconds" % (time() - start_cv))
    n_bins = np.argmin(node_scores)

    method["n_bins"] = n_bins
    method["name_addition"] = ""
    data_row = run_pwlf_nonCV_method(method=method, p=p, y=y, c=c, p_test=p_test, c_test=c_test)

    n_bins_tr = cv_trick(node_scores)
    method["n_bins"] = n_bins_tr
    method["name_addition"] = "tr"
    data_row_tr = run_pwlf_nonCV_method(method=method, p=p, y=y, c=c, p_test=p_test, c_test=c_test)

    metadata_row = construct_metadata_row_from_fit_model_data(model_name=model_name, n_nodes=n_bins,
                                                              n_nodes_trick=n_bins_tr, cv_folds=n_cv_folds,
                                                              node_scores=node_scores,
                                                              node_loss=node_loss, node_ECEs_abs=node_ECEs_abs,
                                                              node_ECEs_square=node_ECEs_square,
                                                              all_weights=all_weights,
                                                              weights_final=None,
                                                              weights_trick=None,
                                                              all_cv_scores=all_cv_scores, last_epoch=None)

    return data_row, data_row_tr, metadata_row


def run_pwlf_sweep_method(method, p, y, c, p_test, c_test):
    max_nodes = method["max_nodes"]
    degree = method["degree"]
    model_name = "pwlf_d" + str(degree) + "_sweep"

    all_weights = []

    for n_nodes in range(1, max_nodes + 1):
        if n_nodes == 1:
            model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
            model.fit(p.reshape(-1, 1), y)
            y_w = np.array([0, 1])  # weights for y
        else:
            model = pwlf.PiecewiseLinFit(p, y, degree=degree)
            breaks = model.fit(n_nodes)
            y_w = np.array(model.predict(breaks))  # weights for y

        print(y_w)
        if not np.all((y_w[1:] - y_w[:-1]) > 0):
            print("N_nodes %i is not monotonic" % n_nodes)
            break
        else:
            model_last = model

    if n_nodes <= 2:
        c_hat = model_last.predict(p.reshape(-1, 1))
        c_hat_test = model_last.predict(p_test.reshape(-1, 1))
    else:
        c_hat = model_last.predict(p)
        c_hat_test = model_last.predict(p_test)

    data_row = construct_data_row_from_raw_data(n_folds=None, binning_name=model_name,
                                                n_bins=n_nodes - 1, c_hat=c_hat, p=p, c=c,
                                                c_hat_test=c_hat_test, p_test=p_test, c_test=c_test)
    node_loss = np.mean(np.square(c_hat - y))

    metadata_row = construct_metadata_row_from_fit_model_data(model_name=model_name, n_nodes=n_nodes - 1,
                                                              n_nodes_trick=None, cv_folds=None, node_scores=None,
                                                              node_loss=node_loss, node_ECEs_abs=None,
                                                              node_ECEs_square=None, all_weights=all_weights,
                                                              weights_final=None, weights_trick=None,
                                                              all_cv_scores=None, last_epoch=None)

    return data_row, metadata_row


def run_pwlf_method(method, p, y, c, p_test, c_test):
    n_cv_folds = method["n_cv_folds"]
    use_sweep = method["use_sweep"]

    if use_sweep:
        data_row, metadata_row = run_pwlf_sweep_method(method=method, p=p, y=y, c=c, p_test=p_test, c_test=c_test)
        return [data_row], [metadata_row]
    elif n_cv_folds is None:
        data_row, _, _ = run_pwlf_nonCV_method(method=method, p=p, y=y, c=c, p_test=p_test, c_test=c_test)
        return [data_row], []
    else:
        data_row, data_row_tr, metadata_row = run_pwlf_CV_method(method=method, p=p, y=y, c=c, p_test=p_test,
                                                                 c_test=c_test)
        return [data_row, data_row_tr], [metadata_row]


def run_kce_method(p, y):
    from pycalibration import calerrors as ce
    skce_unbiased = ce.UnbiasedSKCE(ce.tensor(ce.RBFKernel(), ce.RBFKernel()))
    skce_blockunbiased = ce.BlockUnbiasedSKCE(ce.tensor(ce.RBFKernel(), ce.RBFKernel()))
    skce = ce.BiasedSKCE(ce.tensor(ce.RBFKernel(), ce.RBFKernel()))

    skce_p = p
    skce_y = np.array(y, dtype=bool)

    skce_estimate = skce(skce_p, skce_y)
    skce_unbiased_estimate = skce_unbiased(skce_p, skce_y)
    skce_blockunbiased_estimate = skce_blockunbiased(skce_p, skce_y)

    data_row = construct_data_row(n_folds=None, binning_name="KCE", n_bins=None,
                                  c_hat_distance_p=None,
                                  c_hat_distance_p_debiased=None,
                                  c_hat_distance_p_square=skce_estimate,
                                  c_hat_distance_p_square_debiased=skce_unbiased_estimate,
                                  c_hat_distance_c=None,
                                  c_hat_distance_c_square=None,

                                  test_c_hat_distance_p=None,
                                  test_c_hat_distance_p_debiased=None,
                                  test_c_hat_distance_p_square=None,
                                  test_c_hat_distance_p_square_debiased=None,
                                  test_c_hat_distance_c=None,
                                  test_c_hat_distance_c_square=None)

    data_row_block = construct_data_row(n_folds=None, binning_name="KCE_block", n_bins=None,
                                        c_hat_distance_p=None,
                                        c_hat_distance_p_debiased=None,
                                        c_hat_distance_p_square=skce_estimate,
                                        c_hat_distance_p_square_debiased=skce_blockunbiased_estimate,
                                        c_hat_distance_c=None,
                                        c_hat_distance_c_square=None,

                                        test_c_hat_distance_p=None,
                                        test_c_hat_distance_p_debiased=None,
                                        test_c_hat_distance_p_square=None,
                                        test_c_hat_distance_p_square_debiased=None,
                                        test_c_hat_distance_c=None,
                                        test_c_hat_distance_c_square=None)

    return [data_row, data_row_block], []


def run_method(method, p, y, c, calibration_function, p_test, c_test):
    method_name = method["method_name"]

    if method_name == "PW_NN":
        return run_PW_NN_method(method=method, p=p, y=y, c=c, p_test=p_test, c_test=c_test)
    elif method_name == "pwlf":
        return run_pwlf_method(method=method, p=p, y=y, c=c, p_test=p_test, c_test=c_test)
    elif method_name == "KDE":
        return run_kde_method(p=p, y=y, c=c, calibration_function=calibration_function, p_test=p_test, c_test=c_test)
    elif method_name == "KCE":
        return run_kce_method(p=p, y=y)
    elif method_name == "isotonic":
        return run_isotonic(p=p, y=y, c=c, p_test=p_test, c_test=c_test)
    elif method_name == "platt":
        return run_platt(p=p, y=y, c=c, p_test=p_test, c_test=c_test)
    elif method_name == "beta":
        return run_beta(p=p, y=y, c=c, p_test=p_test, c_test=c_test)
    elif method_name == "brier_score":
        return run_brier_score(p=p, y=y)
    elif method_name == "cross_entropy_loss":
        return run_cross_entropy_loss(p=p, y=y)
    else:
        return run_binning_method(method=method, p=p, y=y, c=c, p_test=p_test, c_test=c_test)
