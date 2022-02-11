import numpy as np
import pandas as pd
from piecewise_linear import node_scores_xy_with_crossvalidation
from keras import backend as K
import tensorflow as tf
import gc
from time import time
from os.path import join
import pickle
from sklearn.metrics import log_loss

def gc_model(model):
    # Garbage collect
   
    del model.model
    del model

    gc.collect()
    K.clear_session()
    tf.compat.v1.reset_default_graph()


# In[5]:

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
                "cross_entropy": CE
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
                "error_type": "abs",
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
                "node_train_loss": node_loss,
                "cv_score": np.min(node_scores),
                "node_ECEs_abs": node_ECEs_abs,
                "node_ECEs_square": node_ECEs_square,
                "model_weights": all_weights,
                "model_weights_final": weights_final,
                "model_weights_trick": weights_trick,
                "all_cv_scores": all_cv_scores,
                "last_epoch": last_epoch,
                "brier": brier,
                "cross_entropy": CE,
                "brier_trick": brier_trick,
                "cross_entropy_trick": CE_trick}
                
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


def create_ensemble_and_results(seed, n_data, data_part, cal_method, data_name, binning_name, n_nodes, 
                                weights_ens, model, input, p, y, c_list, p_distance_c, p_distance_c_square, binning_extra = "_ens"):
       
    c_hats = []
                                
    for i, w in enumerate(weights_ens):
            model.model.set_weights(w)
            c_hats.append(model.predict(input))
      
    c_hat = np.mean(c_hats, axis=0)      
            
    c_hat_distance_p = np.mean(np.abs(c_hat - p))
    c_hat_distance_p_square = np.mean((c_hat - p)**2)
    
    CE = log_loss(y, c_hat)
    brier = np.mean(np.square(c_hat - y))
        
    c_hat_distance_c = []
    c_hat_distance_c_square = []
    
    for c in c_list:        
    
        c_hat_distance_c.append(np.mean(np.abs(c_hat - c)))
        c_hat_distance_c_square.append(np.mean(np.square(c_hat - c)))

    data_row = {"seed": seed, #
                "n_data": n_data, #
                "data_part": data_part, #
                "calibration_function": cal_method, #
                "data_name": data_name, #
                "binning": binning_name + binning_extra, #
                "n_bins": n_nodes, #
                "c_hat_distance_p": c_hat_distance_p, #
                "c_hat_distance_p_square": c_hat_distance_p_square, #
                "c_hat_distance_c": c_hat_distance_c, #
                "c_hat_distance_c_square": c_hat_distance_c_square, #
                "p_distance_c": p_distance_c, #
                "p_distance_c_square": p_distance_c_square, #
                "brier": brier, #
                "cross_entropy": CE #
                }
                
                
    return data_row

def apply_trick(method, node_scores, input, p, y, c_list, seed, n_data, cv_folds, equal_size = False, multiplier = 0.001, monotonic = False,
                logit_scale=False, logit_input = False, logistic_out = False, use_ce_loss = False):
        
    n_nodes_trick = -1
    c_hat_distance_p_trick = -1
    c_hat_distance_p_square_trick = -1
    weights_trick = []
    c_hat_distance_c_trick = -1
    c_hat_distance_c_square_trick = -1
    CE_trick = -1
    brier_trick = -1
    data_row_ens_trick = {}
    
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
            
            model = method(k=n_nodes_trick, max_epochs=1500, random_state = seed, equal_size = equal_size, monotonic = monotonic,
                           logit_scale=logit_scale, logit_input = logit_input, logistic_out = logistic_out, use_ce_loss = use_ce_loss)
            h = model.fit(input, y, verbose=False, batch_size=min(n_data//4, 512))
            last_epoch = len(h.history['loss'])
            print("Last epoch", last_epoch)
            c_hat = model.predict(input)
            c_hat_distance_p_trick = np.mean(np.abs(c_hat - p))
            c_hat_distance_p_square_trick = np.mean(np.square(c_hat - p))

            weights_trick = model.model.get_weights()
            CE_trick = log_loss(y, c_hat)
            brier_trick = np.mean(np.square(c_hat - y))
            
            c_hat_distance_c_trick = []
            c_hat_distance_c_square_trick = []
            
            for c in c_list:
                c_hat_distance_c_trick.append(np.mean(np.abs(c_hat - c)))
                c_hat_distance_c_square_trick.append(np.mean(np.square(c_hat - c)))
            
            gc_model(model)
            
    return (n_nodes_trick, c_hat_distance_p_trick, c_hat_distance_p_square_trick, weights_trick, c_hat_distance_c_trick, c_hat_distance_c_square_trick, 
    CE_trick, brier_trick)


# In[6]:


def run_PW_ECE(fn_method, input, p, y, c_list, cv_folds, seed, cal_method, data_name, data_path = ".", data_part = -1, max_nodes = 15, equal_size = True, 
               n_nodes = 4, monotonic = False, use_sweep = False, use_nn3 = False, use_nn4 = False, use_nn5 = False, use_nn6 = False,
               logit_scale=False, logit_input = False, logistic_out = False, use_ce_loss = False):

    n_data = len(y)
    data_rows = []
    data_rows2 = []
    print("Number of data points: ", n_data)

    if use_nn3:
        binning_start = "PW_NN3"
    elif use_nn4:
        binning_start = "PW_NN4"
    elif use_nn5:
        binning_start = "PW_NN5"
    elif use_nn6:
        binning_start = "PW_NN6"
    else:
        binning_start = "PW_NN2"
        
    if logit_scale:
        tag_logI = "logscale"
    elif logit_input:
        tag_logI = "logI"
    else:
        tag_logI = ""
        
    if logistic_out:
        tag_logO = "O"
    else:
        tag_logO = ""
        
    if use_ce_loss:
        tag_loss = "CE"
    else:
        tag_loss = "MSE"
        
    binning_end = "%s%s_%s" % (tag_logI, tag_logO, tag_loss)
    
    if monotonic:
        binning_name = "%s%s_monotonic" % (binning_start, binning_end)
    elif use_sweep:
        binning_name = "%s%s_sweep" % (binning_start, binning_end)
    else:
        binning_name = "%s%s" % (binning_start, binning_end)
        
    
    all_weights = []
    
    p_distance_c = []
    p_distance_c_square = []
    
    for c in c_list:
        p_distance_c.append(np.mean(np.abs(p - c)))
        p_distance_c_square.append(np.mean(np.square(p - c)))

        
    if use_sweep:
    
        print("Using sweep!")
    
        for n_nodes in range(max_nodes + 1):
            model = fn_method(k=n_nodes, random_state = seed, equal_size = equal_size, monotonic = False, logit_scale=logit_scale, logit_input = logit_input,
                              logistic_out = logistic_out, use_ce_loss = use_ce_loss)
            h = model.fit(input, y, verbose=False, batch_size=min(n_data//4, 512))
            last_epoch = len(h.history['loss'])
            print("Last epoch", last_epoch)
            
            weights = model.model.get_weights()
            all_weights.append(weights)
            

            if use_nn5:
                y_w = np.array(weights[-3])  # weights for y  # Weights for first and last breakpoint is not used.
            else:            
                y_w = np.array(weights[-1])  # weights for y

            print(y_w)
            
            if not np.all((y_w[1:] - y_w[:-1]) > 0):
                print("N_nodes %i is not monotonic" % n_nodes)
                break
            else:
                model_last = model
                last_epoch_sweep = last_epoch
            
        
        print("Get predictions for n_nodes %i!" % (n_nodes - 1))

        c_hat = model_last.predict(input)
        c_hat_distance_p = np.mean(np.abs(c_hat - p))
        c_hat_distance_p_square = np.mean(np.square(c_hat - p))
        

        CE = log_loss(y, c_hat)
        brier = np.mean(np.square(c_hat - y))

        node_loss = np.mean(np.square(c_hat - y))
        
        weights_final = model_last.model.get_weights()

        n_nodes_trick = -1
        cv_folds = 0
        c_hat_distance_p_trick = -1
        c_hat_distance_p_square_trick = -1
        c_hat_distance_c_trick = []
        c_hat_distance_c_square_trick = []
        node_scores = [0]
        node_ECEs_abs = []
        node_ECEs_square = []
        weights_trick = []
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
                     last_epoch_sweep, node_loss, CE, CE_trick, brier, brier_trick)  
                         
                          
        
        gc_model(model)
        #gc_model(model_last)

    # Cross-validation
    else:
        if cv_folds >= 2:

            start_cv = time()
            node_scores, all_weights, all_cv_scores, node_ECEs_square, node_ECEs_abs, node_loss = node_scores_xy_with_crossvalidation(method = fn_method, input = input, p_hat = p, y = y, 
                                                                                          n_splits = cv_folds, seed = seed, 
                                                                                          max_nodes = min(n_data//200, max_nodes), equal_size = equal_size,
                                                                                          monotonic = monotonic, logit_scale=logit_scale, logit_input = logit_input,
                                                                                          logistic_out = logistic_out, use_ce_loss = use_ce_loss)

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

        model = fn_method(k=n_nodes, max_epochs=1500, random_state = seed, equal_size = equal_size, monotonic = monotonic,
                          logit_scale=logit_scale, logit_input = logit_input, logistic_out = logistic_out, use_ce_loss = use_ce_loss)
        h = model.fit(input, y, verbose=False, batch_size=min(n_data//4, 512))
        last_epoch = len(h.history['loss'])
        print("Last epoch", last_epoch)
        c_hat = model.predict(input)
        c_hat_distance_p = np.mean(np.abs(c_hat - p))
        c_hat_distance_p_square = np.mean((c_hat - p)**2)
        
        CE = log_loss(y, c_hat)
        brier = np.mean(np.square(c_hat - y))
        
        weights_final = model.model.get_weights()
        
        c_hat_distance_c = []
        c_hat_distance_c_square = []
        
        for c in c_list:        
        
            c_hat_distance_c.append(np.mean(np.abs(c_hat - c)))
            c_hat_distance_c_square.append(np.mean(np.square(c_hat - c)))
            
        


        gc_model(model) # Garbage collection
        
        

        n_nodes_trick, c_hat_distance_p_trick, c_hat_distance_p_square_trick, weights_trick, c_hat_distance_c_trick, c_hat_distance_c_square_trick, \
        CE_trick, brier_trick = apply_trick(fn_method, node_scores, input, p, y, c_list, seed, n_data, cv_folds, multiplier = 0.001, equal_size = equal_size, monotonic = monotonic,
                                            logit_scale=logit_scale, logit_input = logit_input, logistic_out = logistic_out, use_ce_loss = use_ce_loss)
        
        # Make rows for DataFrame
        data_row, data_row_trick, data_row2 = create_data_rows(seed, n_data, data_part, cal_method, data_name, binning_name, 
                     n_nodes, c_hat_distance_p, c_hat_distance_c, n_nodes_trick, cv_folds, c_hat_distance_p_square, 
                     c_hat_distance_p_trick, c_hat_distance_p_square_trick, c_hat_distance_c_square, c_hat_distance_c_trick, 
                     c_hat_distance_c_square_trick, p_distance_c, p_distance_c_square, node_scores, node_ECEs_abs, node_ECEs_square, all_weights, 
                     weights_final, weights_trick, all_cv_scores, last_epoch, node_loss, CE, CE_trick, brier, brier_trick)   

        if len(all_weights) != 0:
            model_ens = fn_method(k=n_nodes, max_epochs=1500, random_state = seed, equal_size = equal_size, monotonic = monotonic,
                          logit_scale=logit_scale, logit_input = logit_input, logistic_out = logistic_out, use_ce_loss = use_ce_loss)
                          
            weights_ens = all_weights[n_nodes]
                          
            data_row_ens = create_ensemble_and_results(seed, n_data, data_part, cal_method, data_name, binning_name, n_nodes, 
                                    weights_ens, model_ens, input, p, y, c_list, p_distance_c, p_distance_c_square, binning_extra = "_ens")
                                    
            data_rows.append(data_row_ens)            
            gc_model(model_ens) # Garbage collection    

            if n_nodes_trick != -1:
                model_ens = fn_method(k=n_nodes_trick, max_epochs=1500, random_state = seed, equal_size = equal_size, monotonic = monotonic,
                          logit_scale=logit_scale, logit_input = logit_input, logistic_out = logistic_out, use_ce_loss = use_ce_loss)
                          
                weights_ens = all_weights[n_nodes_trick]
                              
                data_row_ens = create_ensemble_and_results(seed, n_data, data_part, cal_method, data_name, binning_name, n_nodes_trick, 
                                        weights_ens, model_ens, input, p, y, c_list, p_distance_c, p_distance_c_square, binning_extra = "_ens_trick")
                                        
                data_rows.append(data_row_ens)            
                gc_model(model_ens) # Garbage collection  


    data_rows.append(data_row)
    data_rows.append(data_row_trick)
    data_rows2.append(data_row2)
    
    print(data_row2)


    #return (data_rows, data_rows2)

    df = pd.DataFrame(data_rows) 
    df.to_pickle(join(data_path, f"df_seed_{seed}_{cal_method}_{n_data}_cv_{cv_folds}_{data_name}_dp_{data_part}_{binning_name}.pkl"), protocol = 4)
    pickle.dump(data_rows2, open(join(data_path, f"drow2_seed_{seed}_{cal_method}_{n_data}_cv_{cv_folds}_{data_name}_dp_{data_part}_{binning_name}.pkl"), "wb"))