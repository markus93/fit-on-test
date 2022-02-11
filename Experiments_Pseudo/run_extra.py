import numpy as np
import pandas as pd

from pycalib.models import IsotonicCalibration
from betacal import BetaCalibration
from pycalib.models import SigmoidCalibration

def construct_data_row(n_folds=None, binning_name=None, n_bins=None,
                       c_hat_distance_p=None, c_hat_distance_p_debiased=None,
                       c_hat_distance_p_square=None, c_hat_distance_p_square_debiased=None,
                       c_hat_distance_c=None, c_hat_distance_c_square=None, p_distance_c = None, p_distance_c_square = None):
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
        "p_distance_c": p_distance_c,
        "p_distance_c_square": p_distance_c_square
    }

def construct_data_row_from_raw_data(n_folds=None, binning_name=None, n_bins=None, c_hat=None, p=None, c=None):
    return construct_data_row(n_folds=n_folds, binning_name=binning_name, n_bins=n_bins,
                              c_hat_distance_p=np.mean(np.abs(c_hat - p)),
                              c_hat_distance_p_debiased=np.mean(np.abs(c_hat - p)),
                              c_hat_distance_p_square=np.mean(np.square(c_hat - p)),
                              c_hat_distance_p_square_debiased=np.mean(np.square(c_hat - p)),
                              c_hat_distance_c=np.mean(np.abs(c_hat - c)),
                              c_hat_distance_c_square=np.mean(np.square(c_hat - c)),
                              p_distance_c=np.mean(np.abs(p - c)),
                              p_distance_c_square=np.mean(np.square(p - c)))


def run_isotonic(p, y, c):
    isotonic = IsotonicCalibration()
    isotonic.fit(p, y)
    c_hat = isotonic.predict(p)

    data_row = construct_data_row_from_raw_data(n_folds=None, binning_name="isotonic",
                                                n_bins=None,
                                                c_hat=c_hat, p=p, c=c)
    return data_row
    

def logit(x):
    return np.log(x/(1-x))
    
def logit_to_scale(x, eps = 1e-16):
    l_max = logit(1-eps)
    x = np.array(x, dtype = np.float64)
    x_clip = np.clip(x, eps, 1-eps)
    return (logit(x_clip) + l_max)/(2*l_max)

def run_platt(p, y, c):

    p_log = logit_to_scale(np.copy(p))

    platt = SigmoidCalibration()
    platt.fit(p_log.reshape(-1, 1), y)
    c_hat = platt.predict_proba(p_log.reshape(-1, 1))

    data_row = construct_data_row_from_raw_data(n_folds=None, binning_name="platt", n_bins=None,
                                                c_hat=c_hat, p=p, c=c)
    return data_row

def run_beta(p, y, c):
    bc = BetaCalibration(parameters="abm")
    bc.fit(p.reshape(-1, 1), y)
    c_hat = bc.predict(p.reshape(-1, 1))

    data_row = construct_data_row_from_raw_data(n_folds=None, binning_name="beta", n_bins=None,
                                                c_hat=c_hat, p=p, c=c)
    return data_row

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