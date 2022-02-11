# Helper functions to find calibration function derivates

import numpy as np
import scipy.integrate as integrate
from scipy.stats import beta

def find_expected_calibration_error(function, error_function, beta_alpha=1, beta_beta=1):
    """ Returns the expected calibration error for given function under the assumption that data distribution is uniform
    """

    distance_from_diagonal = lambda x: error_function(x - function(x)) * beta.pdf(x, beta_alpha, beta_beta)
    expected_calibration_error, error_upper_bound = integrate.quad(distance_from_diagonal, 0, 1)

    if error_upper_bound >= 1e-5:
        raise Exception("Too large uncertainty when estimating calibration error.")
    else:
        return np.round(expected_calibration_error, 6)


def construct_calibration_fun(calibration_fun, fun_amount):
    """ Returns the mix of given function and the diagonal
    """
    return lambda x: (1 - fun_amount) * x + fun_amount * calibration_fun(x)


def find_suitable_fun_amount(calibration_fun, true_calibration_error, error_fun, beta_alpha=1, beta_beta=1):
    """ Searches for and returns the fun_amount ratio to match the desired calibration error
    """

    if true_calibration_error == 0:
        return 0.0

    # Binary search to find the suitable fun_amount
    min_fun = 0
    max_fun = 1
    fun_amount = 0.5

    while True:
        temp_fun = construct_calibration_fun(calibration_fun, fun_amount)
        expected_calibration_error = find_expected_calibration_error(temp_fun, error_fun, beta_alpha, beta_beta)

        if expected_calibration_error == true_calibration_error:
            break
        elif expected_calibration_error > true_calibration_error:
            max_fun = fun_amount
            fun_amount = (fun_amount + min_fun) / 2
        elif expected_calibration_error < true_calibration_error:
            min_fun = fun_amount
            fun_amount = (fun_amount + max_fun) / 2

    return np.round(fun_amount, 6)


def find_all_derivates_for_calibration_functions(calibration_functions, all_calibration_errors, beta_alpha=1, beta_beta=1):
    """
    Constructs all possible derivates specified by all_calibration_errors for the calibration functions.

    Returns a dictionary, where the keys are calibration function names. Items are arrays of 4-tuples.
    Each tuple represents a derivate and contains (derivate_fun_name, derivate_fun, expected_calibration_error, error_fun).
    """

    all_derivate_functions = {}

    for calibration_fun in calibration_functions:

        all_calibration_fun_derivates = []

        for expected_calibration_error, error_fun in all_calibration_errors:
            suitable_fun_amount = find_suitable_fun_amount(calibration_fun,
                                                           expected_calibration_error,
                                                           error_fun, beta_alpha, beta_beta)

            derivate_fun = construct_calibration_fun(calibration_fun, suitable_fun_amount)
            derivate_fun_name = calibration_fun.__name__ + "_" + str(suitable_fun_amount)

            all_calibration_fun_derivates.append(
                (derivate_fun_name, derivate_fun, expected_calibration_error, error_fun))

        all_derivate_functions[calibration_fun.__name__] = all_calibration_fun_derivates

    return all_derivate_functions
