# Imports
import numpy as np
from time import time

import pwlf
from sklearn.model_selection import KFold

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression


def node_scores_pwlf_with_crossvalidation(p_hat, y, n_splits=5, add_error=False, seed=0,
                                          max_nodes=15, degree=1):

    node_scores = [0] * (max_nodes + 1)
    node_ECEs_abs = [0] * (max_nodes + 1)
    node_ECEs_square = [0] * (max_nodes + 1)
    node_loss = [0] * (max_nodes + 1)

    node_scores[0] = np.inf
    node_ECEs_abs[0] = np.inf
    node_ECEs_square[0] = np.inf
    node_loss[0] = np.inf

    all_weights = []
    all_cv_scores = []

    for n_nodes in range(1, max_nodes + 1, 1):

        weights = []
        cv_scores = []
        cv_ECE_square = []
        cv_ECE_abs = []
        cv_loss = []

        print("Nodes:", n_nodes)
        start_cv = time()

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

        ij = 0

        for train_index, test_index in kf.split(p_hat):
            try:
                p_hat_train, p_hat_test = p_hat[train_index], p_hat[test_index]
                y_train, y_test = y[train_index], y[test_index]

                if n_nodes == 1:
                    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
                    model.fit(p_hat_train.reshape(-1, 1), y_train)
                    c_hat_test = model.predict(p_hat_test.reshape(-1, 1))
                    c_hat_train = model.predict(p_hat_train.reshape(-1, 1))
                else:
                    model = pwlf.PiecewiseLinFit(p_hat_train, y_train, degree=degree)
                    h = model.fit(n_nodes)
                    c_hat_test = model.predict(p_hat_test)
                    c_hat_train = model.predict(p_hat_train)

                cv_scores.append(np.mean((c_hat_test - y_test) ** 2))  # Squared error
                cv_ECE_square.append(np.mean((c_hat_test - p_hat_test) ** 2))
                cv_ECE_abs.append(np.mean(np.abs(c_hat_test - p_hat_test)))
                cv_loss.append(np.mean(np.square(c_hat_train - y_train)))  # Train loss

                print("Split:", ij)

                ij += 1
            except:
                print("error for " + str(n_nodes) + " with method pwlf")
                node_scores[n_nodes] += 9999

        node_scores[n_nodes] += np.mean(cv_scores)
        node_ECEs_square[n_nodes] = np.mean(cv_ECE_square)
        node_ECEs_abs[n_nodes] = np.mean(cv_ECE_abs)
        node_loss[n_nodes] = cv_loss
        all_cv_scores.append(cv_scores)
        all_weights.append(weights)

        print("Time for %i-fold CV for %i nodes was %f" % (n_splits, n_nodes, (time() - start_cv)))

    if add_error:
        error = 1 / len(p_hat) ** 0.5
        errors = [0 for i in range(max_nodes + 1)]
        for i in range(max_nodes + 1):
            errors[i] = error * i ** 0.125 * node_scores[i]

        node_scores = node_scores + np.asarray(errors)

    return node_scores, all_weights, all_cv_scores, node_ECEs_square, node_ECEs_abs, node_loss
