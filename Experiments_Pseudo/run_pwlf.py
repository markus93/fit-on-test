# Imports
import numpy as np
import pandas as pd

import os
import pickle
from os.path import join
from time import time

import pwlf
from sklearn.model_selection import KFold

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import log_loss




def node_scores_pwlf_with_crossvalidation(p_hat, y, n_splits = 5, add_error = False, seed = 0, 
                                        max_nodes = 15, degree = 1):
                                        
                                       
    
    node_scores = [0]*(max_nodes+1)
    node_ECEs_abs = [0]*(max_nodes+1)
    node_ECEs_square = [0]*(max_nodes+1)
    node_loss = [0]*(max_nodes+1)
    
    node_scores[0] = np.inf
    node_ECEs_abs[0] = np.inf
    node_ECEs_square[0] = np.inf
    node_loss[0] = np.inf

    all_weights = []
    all_cv_scores = []
    
    for n_nodes in range(1, max_nodes+1, 1):
    
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
                    model= make_pipeline(PolynomialFeatures(degree),LinearRegression())
                    model.fit(p_hat_train.reshape(-1,1), y_train)
                    c_hat_test = model.predict(p_hat_test.reshape(-1,1))
                    c_hat_train = model.predict(p_hat_train.reshape(-1,1))
                else:
                    model = pwlf.PiecewiseLinFit(p_hat_train, y_train, degree = degree)
                    h = model.fit(n_nodes)
                    c_hat_test = model.predict(p_hat_test)
                    c_hat_train = model.predict(p_hat_train)

                cv_scores.append(np.mean((c_hat_test - y_test)**2)) # Squared error
                cv_ECE_square.append(np.mean((c_hat_test - p_hat_test)**2))
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

        print("Time for %i-fold CV for %i nodes was %f" % (n_splits, n_nodes, (time()-start_cv)))
     
    if add_error:
        error = 1 / len(p_hat)**0.5
        errors = [0 for i in range(max_nodes+1)]
        for i in range(max_nodes+1):
            errors[i] = error * i**0.125 * node_scores[i]

        node_scores = node_scores + np.asarray(errors)
    
    
    return node_scores, all_weights, all_cv_scores, node_ECEs_square, node_ECEs_abs, node_loss


def create_data_rows(seed, n_data, data_part, cal_method, data_name, binning_name, n_nodes, c_hat_distance_p, c_hat_distance_c,
                     n_nodes_trick, cv_folds, c_hat_distance_p_square, c_hat_distance_p_trick, c_hat_distance_p_square_trick, 
                     c_hat_distance_c_square, c_hat_distance_c_trick, c_hat_distance_c_square_trick,
                     p_distance_c, p_distance_c_square,
                     node_scores, node_ECEs_abs, node_ECEs_square, all_weights, weights_final, weights_trick, all_cv_scores,
                     last_epoch, node_loss, CE, CE_trick, brier, brier_trick):

    data_row = {"seed": seed,
                "n_data": n_data,
                "data_part": data_part,
                "calibration_function": cal_method,
                "data_name": data_name,
                "binning": binning_name,
                "n_bins": n_nodes,
                "c_hat_distance_p": c_hat_distance_p,
                "c_hat_distance_p_square": c_hat_distance_p_square,
                "c_hat_distance_c": c_hat_distance_c,
                "c_hat_distance_c_square": c_hat_distance_c_square,
                "p_distance_c": p_distance_c,
                "p_distance_c_square": p_distance_c_square,
                "brier": brier,
                "cross_entropy": CE,
                }

    data_row2 = {"seed": seed,
                "n_data": n_data,
                "data_part": data_part,
                "calibration_function": cal_method,
                "data_name": data_name,
                "binning": binning_name,
                "n_bins": n_nodes,
                "n_bins_trick": n_nodes_trick,
                "cv_folds": cv_folds,
                "c_hat_distance_p": c_hat_distance_p,
                "c_hat_distance_p_square": c_hat_distance_p_square,
                "c_hat_distance_c": c_hat_distance_c,
                "c_hat_distance_c_square": c_hat_distance_c_square,
                "c_hat_distance_p_trick": c_hat_distance_p_trick,
                "c_hat_distance_p_square_trick": c_hat_distance_p_square_trick,
                "c_hat_distance_c_trick": c_hat_distance_c_trick,
                "c_hat_distance_c_square_trick": c_hat_distance_c_square_trick,                 
                "p_distance_c": p_distance_c,
                "p_distance_c_square": p_distance_c_square,
                "expected_calibration_error": -1,
                "node_scores": node_scores,
                "cv_score": np.min(node_scores),
                "node_ECEs_abs": node_ECEs_abs,
                "node_ECEs_square": node_ECEs_square,
                "node_train_loss:": node_loss,
                "model_weights": all_weights,
                "model_weights_final": weights_final,
                "model_weights_trick": weights_trick,
                "all_cv_scores": all_cv_scores,
                "last_epoch": last_epoch,
                "brier": brier,
                "cross_entropy": CE,
                "brier_trick": brier_trick,
                "cross_entropy_trick": CE_trick
                }
                
                
                
    if c_hat_distance_p_trick == -1:  # No trick used
        c_hat_distance_p_trick = c_hat_distance_p
        c_hat_distance_p_square_trick = c_hat_distance_p_square
        c_hat_distance_c_trick = c_hat_distance_c
        c_hat_distance_c_square_trick = c_hat_distance_c_square
        n_nodes_trick = n_nodes
        brier_trick = brier
        CE_trick = CE

    # Add datarow also for trick
    data_row_trick = {"seed": seed,
                "n_data": n_data,
                "data_part": data_part,
                "calibration_function": cal_method,
                "data_name": data_name,
                "binning": binning_name + "_trick",
                "n_bins": n_nodes_trick,
                "c_hat_distance_p": c_hat_distance_p_trick,
                "c_hat_distance_p_square": c_hat_distance_p_square_trick,
                "c_hat_distance_c": c_hat_distance_c_trick,
                "c_hat_distance_c_square": c_hat_distance_c_square_trick,
                "p_distance_c": p_distance_c,
                "p_distance_c_square": p_distance_c_square,
                "brier": brier_trick,
                "cross_entropy": CE_trick
                }
                
                
                
    return data_row, data_row_trick, data_row2


def apply_trick(node_scores, p, y, c_list, seed, n_data, cv_folds, multiplier = 0.001, degree = 1):
        
    n_nodes_trick = -1
    c_hat_distance_p_trick = -1
    c_hat_distance_p_square_trick = -1
    weights_trick = []
    c_hat_distance_c_trick = []
    c_hat_distance_c_square_trick = []
    CE_trick = -1
    brier_trick = -1
    
    if cv_folds >= 2:
    
        node_score_min = min(node_scores)
        node_score_pos = np.argmin(node_scores)
        new_found = False
        
        for pos in range(node_score_pos-1, -1, -1):
            #print(pos)
            max_diff = node_score_min*multiplier
            new_min_cand = node_scores[pos]
            if new_min_cand <= node_score_min + max_diff:
                pos_new = pos
                new_found = True
                
        if new_found:
            print("We found better candidate!")
            print("Position:",  pos_new)
            print(node_scores[pos_new])
            print("Previous:", node_score_pos)
            print(node_score_min)
            
            n_nodes_trick = pos_new
            
            if n_nodes_trick == 1:
                model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
                model.fit(p.reshape(-1,1), y)
                c_hat = model.predict(p.reshape(-1,1))
            else:
                model = pwlf.PiecewiseLinFit(p, y, degree = degree)
                model.fit(n_nodes_trick)
                c_hat = model.predict(p)
            
            #c_hat = model.predict(p)
            c_hat_distance_p_trick = np.mean(np.abs(c_hat - p))
            c_hat_distance_p_square_trick = np.mean(np.square(c_hat - p))

            CE_trick = log_loss(y, c_hat)
            brier_trick = np.mean(np.square(c_hat - y))
            
            c_hat_distance_c_trick = []
            c_hat_distance_c_square_trick = []
            
            for c in c_list:            
                c_hat_distance_c_trick.append(np.mean(np.abs(c_hat - c)))
                c_hat_distance_c_square_trick.append(np.mean(np.square(c_hat - c)))
            
            
    return (n_nodes_trick, c_hat_distance_p_trick, c_hat_distance_p_square_trick, [], 
    c_hat_distance_c_trick, c_hat_distance_c_square_trick, CE_trick, brier_trick)


# In[6]:


def run_pwlf_test(p, y, c_list, cv_folds, seed, cal_method, data_name, data_part = -1, max_nodes = 15, 
               n_nodes = 4, degree = 1, data_path = "data", use_sweep = False):

    n_data = len(y)
    data_rows = []
    data_rows2 = []
    print("Number of data points: ", n_data)
    
    if degree == 1:
        binning_name = "pwlf_d1"
    else:
        binning_name = "pwlf_d2"    
        
    if use_sweep:
        binning_name = "%s_sweep" % (binning_name)

    all_weights = []        
    
    p_distance_c = []
    p_distance_c_square = []
    
    for c in c_list:
        p_distance_c.append(np.mean(np.abs(p - c)))
        p_distance_c_square.append(np.mean(np.square(p - c)))
        
    if use_sweep:
    
        print("Using sweep!")
    
        for n_nodes in range(1, max_nodes + 1):
            if n_nodes == 1:
                model = make_pipeline(PolynomialFeatures(degree),LinearRegression())
                model.fit(p.reshape(-1,1), y)
                c_hat = model.predict(p.reshape(-1,1))                
                y_w = np.array([0,1])  # weights for y
            else:
                model = pwlf.PiecewiseLinFit(p, y, degree = degree)
                breaks = model.fit(n_nodes)
                c_hat = model.predict(p)
                y_w = np.array(model.predict(breaks))  # weights for y

            print(y_w)
            
            if not np.all((y_w[1:] - y_w[:-1]) > 0):
                print("N_nodes %i is not monotonic" % n_nodes)
                break
            else:
                model_last = model
            
        
        print("Get predictions for n_nodes %i!" % (n_nodes - 1))

        if n_nodes == 2:
            c_hat = model_last.predict(p.reshape(-1,1))
        else:
            c_hat = model_last.predict(p)
            
        c_hat_distance_p = np.mean(np.abs(c_hat - p))
        c_hat_distance_p_square = np.mean(np.square(c_hat - p))
        

        CE = log_loss(y, c_hat)
        brier = np.mean(np.square(c_hat - y))

        node_loss = np.mean(np.square(c_hat - y))
        

        n_nodes_trick = -1
        cv_folds = 0
        c_hat_distance_p_trick = -1
        c_hat_distance_p_square_trick = []
        c_hat_distance_c_trick = []
        c_hat_distance_c_square_trick = []
        node_scores = [0]
        node_ECEs_abs = []
        node_ECEs_square = []
        weights_trick = []
        weights_final = []
        all_cv_scores = []
        CE_trick = -1
        brier_trick = -1
        
        c_hat_distance_c = []
        c_hat_distance_c_square = []

        
        for c in c_list:        
            c_hat_distance_c.append(np.mean(np.abs(c_hat - c)))
            c_hat_distance_c_square.append(np.mean(np.square(c_hat - c)))


        # Make rows for DataFrame
        data_row, data_row_trick, data_row2 = create_data_rows(seed, n_data, data_part, cal_method, data_name, binning_name, 
                     n_nodes, c_hat_distance_p, c_hat_distance_c,
                     n_nodes_trick, cv_folds, c_hat_distance_p_square, c_hat_distance_p_trick, c_hat_distance_p_square_trick, 
                     c_hat_distance_c_square, c_hat_distance_c_trick, c_hat_distance_c_square_trick,
                     p_distance_c, p_distance_c_square,
                     node_scores, node_ECEs_abs, node_ECEs_square, all_weights, weights_final, weights_trick, all_cv_scores,
                     -1, node_loss, CE, CE_trick, brier, brier_trick)  
                         
                          
        


    # Cross-validation
    elif cv_folds >= 2:

        start_cv = time()
        node_scores, all_weights, all_cv_scores, node_ECEs_square, node_ECEs_abs, node_loss = node_scores_pwlf_with_crossvalidation(p_hat = p, y = y, 
                                                                                      n_splits = cv_folds, seed = seed, 
                                                                                      max_nodes = min(n_data//20, max_nodes), degree = degree)

        print("Cross-validation took %f seconds" % (time() - start_cv))
        n_nodes = np.argmin(node_scores)
        print("Best number of nodes is %i" % (n_nodes))

        # Check if sweep would lower the number of bins?

    else:
        print("CV nor Sweep is not used!")
        print("Number of nodes selected by user: %i" % n_nodes)
        node_scores = [-1]
        all_weights = []
        all_cv_scores = []

    
    if n_nodes == 1:
        model = make_pipeline(PolynomialFeatures(degree),LinearRegression())
        model.fit(p.reshape(-1,1), y)
        c_hat = model.predict(p.reshape(-1,1))
    else:
        model = pwlf.PiecewiseLinFit(p, y, degree = degree)
        model.fit(n_nodes)
        c_hat = model.predict(p)
    
    c_hat_distance_p = np.mean(np.abs(c_hat - p))
    c_hat_distance_p_square = np.mean(np.square(c_hat - p))
    

    CE = log_loss(y, c_hat)
    brier = np.mean(np.square(c_hat - y))
    
    c_hat_distance_c = []
    c_hat_distance_c_square = []
    
    for c in c_list:    
        c_hat_distance_c.append(np.mean(np.abs(c_hat - c)))
        c_hat_distance_c_square.append(np.mean(np.square(c_hat - c)))

    n_nodes_trick, c_hat_distance_p_trick, c_hat_distance_p_square_trick, weights_trick, c_hat_distance_c_trick, c_hat_distance_c_square_trick, \
    CE_trick, brier_trick = apply_trick(node_scores, p, y, c_list, seed, n_data, cv_folds, multiplier = 0.001, degree = degree)
    
    # Make rows for DataFrame
    data_row, data_row_trick, data_row2 = create_data_rows(seed, n_data, data_part, cal_method, data_name, binning_name, 
                 n_nodes, c_hat_distance_p, c_hat_distance_c, n_nodes_trick, cv_folds, c_hat_distance_p_square, 
                 c_hat_distance_p_trick, c_hat_distance_p_square_trick, c_hat_distance_c_square, c_hat_distance_c_trick, 
                 c_hat_distance_c_square_trick, p_distance_c, p_distance_c_square, node_scores, node_ECEs_abs, node_ECEs_square, [], 
                 [], [], all_cv_scores, -1, node_loss, CE, CE_trick, brier, brier_trick)                                                                                                  


    data_rows.append(data_row)
    data_rows.append(data_row_trick)
    data_rows2.append(data_row2)
    
    print(data_row2)

    df = pd.DataFrame(data_rows) 
    df.to_pickle(os.path.join(data_path, f"df_seed_{seed}_{cal_method}_{n_data}_cv_{cv_folds}_{data_name}_dp_{data_part}_{binning_name}.pkl"), protocol = 4)
    pickle.dump(data_rows2, open(os.path.join(data_path, f"drow2_seed_{seed}_{cal_method}_{n_data}_cv_{cv_folds}_{data_name}_dp_{data_part}_{binning_name}.pkl"), "wb"))