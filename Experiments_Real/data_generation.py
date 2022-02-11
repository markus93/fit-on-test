import numpy as np

def generate_data(dirichlet, n_data, calibration_function, seed):
    """ Returns the predictions, labels and real probabilities for the first class only.
    """
    
    np.random.seed(seed)
    
    p_hat = np.sort(np.random.dirichlet(dirichlet, n_data)[:,0])
    c = np.asarray([calibration_function(pred) for pred in p_hat])
    y = np.array([np.random.choice([1,0], p=[pred, 1-pred]) for pred in c])
        
    return p_hat, y, c