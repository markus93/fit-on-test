"""
Input: p, y, c (+ n_bins if needed)

Output as object fields:
    1) data p/y/c/c^ (sorted by p)
        .p, .y, .c, .c_hat
    2) binned data (p/y/c/c^): [[p1, p2, p3], [p4, p5], ..]
        .binned_p, .binned_y, .binned_c, .binned_c_hat
    3) bin borders: [0, x1, x2, x3, .., 1]
        .bin_borders
    4) some metrics: absolute-ECE, square-ECE, absolute and square calibration error estimates
        .ECE_abs, .ECE_square, .c_hat_distance_p_square, c_hat_distance_p_abs
    5) bins as polygons for plotting
        .bins_as_polygons_slope_1, bins_as_polygons_flat
    6) binning scheme name (eg. "eq_size")
        .binning_name
    7) number of bins
        .n_bins
    
    ...

"""

import numpy as np
from copy import deepcopy, copy
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from time import time   
from scipy.stats import norm
import scipy.integrate as integrate

class Binning(object):
    """ Abstract binning class for binning schemes to sub-class
    """

    def __init__(self, p, y, c):
        #x=time()

        sorted_by_p = np.argsort(p)
        
        self.p = copy(p)[sorted_by_p]
        self.y = copy(y)[sorted_by_p]
        self.c = copy(c)[sorted_by_p] if c is not None else np.ones(len(p)) * np.infty    
        
        self.binned_ids = self.split_indices_into_bins()
        
        self.is_some_bin_empty = self.__is_some_bin_empty()
        if self.is_some_bin_empty:
            print("NB! Too large binning - one of the bins is empty!")

        self.binned_p = self.__split_array_by_binned_indices(self.p)
        self.binned_y = self.__split_array_by_binned_indices(self.y)
        self.binned_c = self.__split_array_by_binned_indices(self.c)
        self.bin_borders = self.get_bin_borders()

        self.eval_flat = self.__construct_eval_function_flat()
        self.eval_slope_1 = self.__construct_eval_function_slope_1()
        
        self.c_hat = self.eval_slope_1(self.p)
        self.binned_c_hat = self.__split_array_by_binned_indices(self.c_hat)
        
        self.ECE_abs = self.__get_ECE_abs()

        self.ECE_square = self.__get_ECE_square()
        self.ECE_square_debiased = self.__get_ECE_square_debiased()
        
        self.ECE_abs_debiased = self.__get_ECE_abs_debiased_integrate()
        #self.ECE_abs_debiased_integrate_true = self.__get_ECE_abs_debiased_integrate_true()
        #self.ECE_abs_debiased_sampling = self.__get_ECE_abs_debiased_sampling()

        self.c_hat_distance_p_square = self.__get_c_hat_distance_p_square()
        self.c_hat_distance_c_square = self.__get_c_hat_distance_c_square()
        self.c_hat_distance_p_abs = self.__get_c_hat_distance_p_abs()
        self.c_hat_distance_c_abs = self.__get_c_hat_distance_c_abs()

        self.binning_name = self.get_binning_name()
        self.n_bins = self.__get_n_bins()
        #print(time()-x - end)

    def __is_some_bin_empty(self):
        return np.any([len(b) == 0 for b in self.binned_ids])
    
    def __split_array_by_binned_indices(self, arr):
        if self.is_some_bin_empty: # my slow implementation
            indexer = lambda idx: arr[idx]
            binner = lambda bin_of_ids: np.asarray(list(map(indexer, bin_of_ids)))
            return np.asarray(list(map(binner, deepcopy(self.binned_ids))), dtype=object)
        else: # fast np implementation. breaks if empty bins
            return np.split(arr, [b[0] for b in self.binned_ids[1:]], axis=0)
       
    
    def construct_plt_polygons_slope_1(self):
        polygons = []
        
        for i in range(len(self.binned_p)):
            
            bin_y_diff = self.bin_borders[i+1]-self.bin_borders[i] if len(self.binned_p[i]) != 0 else 0
            
            polygon = plt.Polygon([[self.bin_borders[i], 0],
                                  [self.bin_borders[i+1], 0],
                                  [self.bin_borders[i+1], self.eval_slope_1(self.bin_borders[i]) + bin_y_diff],
                                  [self.bin_borders[i], self.eval_slope_1(self.bin_borders[i])]],
                                 edgecolor="black", lw=0.4, alpha=0.4)
            polygons.append(polygon)
        
        return polygons
    
    def construct_plt_polygons_flat(self):
        polygons = []

        for i in range(len(self.binned_p)):
            polygon = plt.Polygon([[self.bin_borders[i],0],
                                   [self.bin_borders[i+1],0],
                                   [self.bin_borders[i+1], self.eval_flat(self.bin_borders[i])],
                                   [self.bin_borders[i], self.eval_flat(self.bin_borders[i])]],
                                   edgecolor="black", lw=0.4, alpha=0.4)
            polygons.append(polygon)
            
        return polygons
        


    def __construct_eval_function_slope_1(self):
        
        def eval_function(pred):
            conditions = [(pred >= self.bin_borders[i]) & (pred < self.bin_borders[i+1]) for i in range(len(self.bin_borders)-2)]
            conditions.append(((pred >= self.bin_borders[-2]) & (pred <= 1)))
            
            functions = []
            for i in range(len(self.bin_borders)-1):
               
                if len(self.binned_p[i]) == 0:
                    f = lambda pred: 0
                else:
                    x = np.mean(self.binned_p[i])
                    y = np.mean(self.binned_y[i])
                    b = y - x

                    f = lambda pred, b=b: 1 * pred + b
                functions.append(f)
                
            return np.piecewise(pred, conditions, functions)
        
        return eval_function
        
    def construct_eval_function_slope_1(self):
        
        def eval_function(pred):
            conditions = [(pred >= self.bin_borders[i]) & (pred < self.bin_borders[i+1]) for i in range(len(self.bin_borders)-2)]
            conditions.append(((pred >= self.bin_borders[-2]) & (pred <= 1)))
            
            functions = []
            for i in range(len(self.bin_borders)-1):
               
                if len(self.binned_p[i]) == 0:
                    f = lambda pred: 0
                else:
                    x = np.mean(self.binned_p[i])
                    y = np.mean(self.binned_y[i])
                    b = y - x

                    f = lambda pred, b=b: 1 * pred + b
                functions.append(f)
                
            return np.piecewise(pred, conditions, functions)
        
        return eval_function
    
    def __construct_eval_function_flat(self):
        
        def eval_function(pred):
            conditions = [(pred >= self.bin_borders[i]) & (pred < self.bin_borders[i+1]) for i in range(len(self.bin_borders)-2)]
            conditions.append(((pred >= self.bin_borders[-2]) & (pred <= 1)))
            
            functions = []
            for i in range(len(self.bin_borders)-1):
                if len(self.binned_p[i]) == 0:
                    f = lambda pred: 0
                else:
                    y = np.mean(self.binned_y[i])
                    f = lambda pred, y=y: y
                functions.append(f)
                
            return np.piecewise(pred, conditions, functions)
        
        return eval_function

    def construct_eval_function_flat(self):
        
        def eval_function(pred):
            conditions = [(pred >= self.bin_borders[i]) & (pred < self.bin_borders[i+1]) for i in range(len(self.bin_borders)-2)]
            conditions.append(((pred >= self.bin_borders[-2]) & (pred <= 1)))
            
            functions = []
            for i in range(len(self.bin_borders)-1):
                if len(self.binned_p[i]) == 0:
                    f = lambda pred: 0
                else:
                    y = np.mean(self.binned_y[i])
                    f = lambda pred, y=y: y
                functions.append(f)
                
            return np.piecewise(pred, conditions, functions)
        
        return eval_function         
    
    def __get_ECE_abs(self):
        ECE = 0
        for i in range(len(self.binned_p)):
            if len(self.binned_p[i]) == 0:
                continue
            
            mean_p = np.mean(self.binned_p[i])
            mean_y = np.mean(self.binned_y[i])
            
            ECE += np.abs(mean_p - mean_y) * len(self.binned_p[i])
        ECE = ECE / len(self.p)
        return ECE
    
    def __get_ECE_abs_debiased_sampling(self):
        ECE = 0
        bias = 0
        
        for i in range(len(self.binned_p)):
            if len(self.binned_p[i]) == 0:
                continue
            
            mean_p = np.mean(self.binned_p[i])
            mean_y = np.mean(self.binned_y[i])
            
            ECE += np.abs(mean_p - mean_y) * len(self.binned_p[i])
            
            np.random.seed(4)
            R_k_samples = np.random.normal( mean_y, np.sqrt(mean_y * (1 - mean_y) / len(self.binned_p[i])), size=10_000_000 )
            
            bias += np.mean( np.abs(mean_p - R_k_samples) * len(self.binned_p[i]) )
            
        ECE = 2 * ECE - bias 
        ECE = ECE / len(self.p)
        
        return ECE
    
    def __get_ECE_abs_debiased_integrate(self): 
        
        def normpdf(x, mean, std):
            var = std**2

            denom = (2*np.pi*var)**.5
            num = np.exp(-(x-mean)**2/(2*var))

            return num/denom
        
        ECE = 0
        bias = 0
        for i in range(len(self.binned_p)):
            if len(self.binned_p[i]) == 0:
                continue
            
            mean_p = np.mean(self.binned_p[i])
            mean_y = np.mean(self.binned_y[i])
            
            ECE += np.abs(mean_p - mean_y) * len(self.binned_p[i])
       
    
            std = np.sqrt(mean_y * (1 - mean_y) / len(self.binned_p[i]))

            if mean_y == 0 or std == 0:
                bias += np.abs(mean_p - mean_y) * len(self.binned_p[i]) 
            else:
                x_range = np.linspace(mean_y - 5*std, mean_y + 5*std, 10_000) # 10k gives result trustworthy to approx ~0.00002

                bias += np.sum(normpdf(x_range, mean_y, std) * np.abs(mean_p - x_range) * len(self.binned_p[i]) ) * (10*std / len(x_range))


        ECE = 2 * ECE - bias 
        ECE = ECE / len(self.p)
        
        return ECE
    
    def __get_ECE_abs_debiased_integrate_true(self):
        ECE = 0
        bias = 0
        for i in range(len(self.binned_p)):
            if len(self.binned_p[i]) == 0:
                continue
            
            mean_p = np.mean(self.binned_p[i])
            mean_y = np.mean(self.binned_y[i])
            
            ECE += np.abs(mean_p - mean_y) * len(self.binned_p[i])
       
    
            std = np.sqrt(mean_y * (1 - mean_y) / len(self.binned_p[i]))

            if mean_y == 0 or std == 0:
                bias += np.abs(mean_p - mean_y) * len(self.binned_p[i]) 
            else:
                rv = norm( mean_y, std )
                fun = lambda x: rv.pdf(x) * np.abs(mean_p - x) * len(self.binned_p[i]) 

                bias += integrate.quad(fun, mean_y - 5*std, mean_y + 5*std, epsabs=1e-8, epsrel=1e-8)[0]                      

        ECE = 2 * ECE - bias 
        ECE = ECE / len(self.p)
        
        return ECE
        
    def __get_ECE_square(self):
        ECE = 0
        for i in range(len(self.binned_p)):
            if len(self.binned_p[i]) == 0:
                continue
                
            mean_p = np.mean(self.binned_p[i])
            mean_y = np.mean(self.binned_y[i])
            
            ECE += (mean_p - mean_y)**2 * len(self.binned_p[i])
        ECE = ECE / len(self.p)
        return ECE
    
    def __get_ECE_square_debiased(self):
        ECE = 0
        for i in range(len(self.binned_p)):
            if len(self.binned_p[i]) == 0:
                continue
                
            mean_p = np.mean(self.binned_p[i])
            mean_y = np.mean(self.binned_y[i])
                        
            ECE += (mean_p - mean_y)**2 * len(self.binned_p[i])
            
            if len(self.binned_p[i]) == 1:
                bias = 0
            else:
                bias = (mean_y * (1 - mean_y)) / (len(self.binned_p[i]) - 1)
                                                  
            ECE -= bias * len(self.binned_p[i])
            
        ECE = ECE / len(self.p)
        return ECE
    
    def __get_c_hat_distance_p_square(self):
        return np.mean((self.p - self.c_hat)**2)
    
    def __get_c_hat_distance_c_square(self):
        return np.mean((self.c - self.c_hat)**2)
    
    def __get_c_hat_distance_p_abs(self):
        return np.mean(np.abs(self.p - self.c_hat))
    
    def __get_c_hat_distance_c_abs(self):
        return np.mean(np.abs(self.c - self.c_hat))
    
    def __get_n_bins(self):
        return len(self.binned_p)
    
    def split_indices_into_bins(self):
        """ Should be private, but couldn't set as private as subclasses can't see it then.
        Calling this returns the object variable binned_ids
        """
        raise NotImplementedError("subclass must override")
    
    def get_bin_borders(self):
        """ 
        Calling this returns the objects variable bin_borders
        """
        raise NotImplementedError("subclass must override")
        
    def get_binning_name(self):
        """ 
        Calling this returns the objects variable binning name
        """
        raise NotImplementedError("subclass must override")


class EqualWidthBinning(Binning):

    def __init__(self, p, y, c, n_bins):
        self.n_bins = n_bins
        super().__init__(p, y, c)

    def split_indices_into_bins(self):
        bins = [[] for _ in range(self.n_bins)]
        bin_width = 1.0 / self.n_bins

        for i, pred in enumerate(self.p):
            idx = int(pred // bin_width)
            if pred >= 1.0:
                idx = self.n_bins - 1

            bins[idx].append(i)

        for i in range(len(bins)):
            bins[i] = np.asarray(bins[i])

        return np.asarray(bins, dtype=object)       
        
    def get_bin_borders(self):
        return np.asarray([1 / self.n_bins * i for i in range(self.n_bins + 1)])
    
    def get_binning_name(self):
        return "eq_width"


class EqualSizeBinning(Binning):
    
    def __init__(self, p, y, c, n_bins):
        self.n_bins = n_bins
        super().__init__(p, y, c)

    def split_indices_into_bins(self):
        return np.array_split(range(len(self.p)), self.n_bins)
        
    def get_bin_borders(self):
        """ Bin borders are right exclusive [), [), .., [), []
        """
        bin_borders = [0]
        for b in self.binned_p:
            if len(b) == 0:
                continue
            else:
                bin_borders.append(b[0])
        bin_borders.pop(1)
        bin_borders.append(1)
        
        return np.asarray(bin_borders)
    
    def get_binning_name(self):
        return "eq_size"


def binning_n_bins_with_crossvalidation(p, y, use_eq_width, n_splits):
    
    MAX_N_BINS = 41 # exclusive  
    
    bin_scores = [0]*MAX_N_BINS
    bin_scores[0] = np.infty
    
    all_cv_scores = []
    all_cv_scores.append([np.infty]*n_splits)
    
    for n_bins in range(1, MAX_N_BINS, 1):
        
        cv_scores = [] 

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=1)
        
        for train_index, test_index in kf.split(p):
            
            p_train, p_test = p[train_index], p[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            if use_eq_width:
                binning = EqualWidthBinning(p_train, y_train, None, n_bins)
            else:
                binning = EqualSizeBinning(p_train, y_train, None, n_bins)
                
            c_hat_test = binning.eval_slope_1(p_test)
            
            cv_scores.append(np.mean((c_hat_test - y_test)**2)) 
    
        all_cv_scores.append(cv_scores)
        bin_scores[n_bins] = np.mean(cv_scores)    
       
    return bin_scores, all_cv_scores
        
    
class FixedBordersBinning(Binning):
    """ Bin borders are right exclusive [), [), .., [), []
    """

    def __init__(self, p, y, c, bin_borders):
        
        if bin_borders[0] != 0 or bin_borders[-1] != 1:
            raise ValueError("Tried to fix binning not between 0 and 1")
        
        self.bin_borders = bin_borders
        super().__init__(p, y, c)

    def split_indices_into_bins(self):
        
        bins = [[] for _ in range(len(self.bin_borders) - 1)]
        bin_id = 0
        bin_start = self.bin_borders[bin_id]
        bin_end = self.bin_borders[bin_id+1]
        for i, pred in enumerate(self.p):
            while True:
                if pred >= bin_start and pred < bin_end:
                    bins[bin_id].append(i)
                    break
                else:
                    bin_id += 1
                    bin_start = self.bin_borders[bin_id]
                    bin_end = self.bin_borders[bin_id+1]
                    
                    if bin_end == 1:
                        bin_end += 0.01 # At 1 the right side must be inclusive
        
        for i in range(len(bins)):
            bins[i] = np.asarray(bins[i])
        
        return np.asarray(bins, dtype=object)
        
    def get_bin_borders(self):
        return self.bin_borders
    
    def get_binning_name(self):
        return "fixed_borders"
    
    
class MonotonicEqualSizeBinning(Binning):

    def __init__(self, p, y, c):
        super().__init__(p, y, c)

    def split_indices_into_bins(self):
  
        n_bins = 1
        while True:
            binning_to_try = EqualSizeBinning(self.p, self.y, self.c, n_bins)
            bin_heights = [np.mean(y_bin) for y_bin in binning_to_try.binned_y]
            is_monotonic = (np.diff(bin_heights) >= 0).all()
        
            if is_monotonic:
                n_bins += 1
            else:
                binning = EqualSizeBinning(self.p, self.y, self.c, n_bins - 1).binned_ids
                self.n_bins = n_bins - 1
                break
                
        return binning
        
    def get_bin_borders(self):
        return EqualSizeBinning(self.p, self.y, self.c, self.n_bins).bin_borders
    
    def get_binning_name(self):
        return "monotonic_eq_size"