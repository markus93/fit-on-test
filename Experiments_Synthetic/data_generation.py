import numpy as np

def generate_data(dirichlet, n_data, calibration_function, seed=None):
    """ Returns the predictions, labels and real probabilities for the first class only.
    """
    
    np.random.seed(seed+n_data)
    
    c = np.sort(np.random.dirichlet(dirichlet, n_data)[:,0])
    y = np.random.binomial(n=1, p=c)
    p = calibration_function(c)
        
    return p, y, c