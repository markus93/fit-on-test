import torch
import numpy as np
import torch.nn.parallel
from KDEpy import FFTKDE
from time import time
from math import isclose

# Code scrapped from
# https://github.com/zhang64-llnl/Mix-n-Match-Calibration/blob/master/demo_calibration.py

class KDE_estimator():
    
    def __init__(self, p, y, c=None, calibration_function=None, p_test=None, c_test=None):
        
        all_classes_p = self.__reshape_p_to_binary(p)
        one_hot_encoded_y = self.__one_hot_encode_y(y)

        all_classes_p_test = self.__reshape_p_to_binary(p_test)

        x_int, pp1, pp2, perc = self.__kde_setup(p=all_classes_p, label=one_hot_encoded_y)
        
        self.x_int = x_int
        self.pp1 = pp1
        self.pp2 = pp2
        self.perc = perc

        c_hat = self.__c_hat_kde(all_classes_p, x_int, pp1, pp2, perc)
        self.c_hat = c_hat

        c_hat_test = self.__c_hat_kde(all_classes_p_test, x_int, pp1, pp2, perc)

        self.integral_ece_abs, self.integral_ece_sq = self.__integral_ece_kde(x_int, pp1, pp2, perc)
        self.pointwise_ece_abs = np.mean(np.abs(c_hat - p)**1)
        self.pointwise_ece_sq = np.mean(np.abs(c_hat - p)**2)

        self.pointwise_ece_abs_test = np.mean(np.abs(c_hat_test - p_test)**1)
        self.pointwise_ece_sq_test = np.mean(np.abs(c_hat_test - p_test)**2)

        if calibration_function is not None:
            self.integral_c_hat_dist_c_abs, self.integral_c_hat_dist_c_sq = self.__integral_c_hat_distance_c(x_int, pp1, pp2, perc, calibration_function)
        if c is not None:
            self.pointwise_c_hat_dist_c_abs = np.mean(np.abs(c_hat - c)**1)
            self.pointwise_c_hat_dist_c_sq = np.mean(np.abs(c_hat - c)**2)
        if c_test is not None:
            self.pointwise_c_hat_dist_c_abs_test = np.mean(np.abs(c_hat_test - c_test)**1)
            self.pointwise_c_hat_dist_c_sq_test = np.mean(np.abs(c_hat_test - c_test)**2)
        
    def __one_hot_encode_y(self, y):
        # reshape labels from y=[0, 0] to label=[[0,1], [0,1]]
        return np.eye(2)[(y+1)%2]
        
        
    def __reshape_p_to_binary(self, p):  
        # reshape predictions from p=[0.1, 0.3] to p=[[0.1, 0.9], [0.3, 0.7]]
        p1 = p.reshape((-1,1))
        p2 = (1-p).reshape((-1,1))
        return np.concatenate([p1, p2], axis=1)        
        
        
    def __c_hat_kde(self, p, x_int, pp1, pp2, perc):

        c_hat = np.zeros(p.shape[0])
        for i in range(c_hat.shape[0]):
            conf = p[i, 1]
            if np.max([pp1[np.abs(x_int-conf).argmin()],pp2[np.abs(x_int-conf).argmin()]])>1e-6:
                accu = np.min([perc*pp1[np.abs(x_int-conf).argmin()]/pp2[np.abs(x_int-conf).argmin()],1.0])
                if np.isnan(accu)==False:
                    c_hat[i] = accu
            else:
                if i>1:
                    c_hat[i] = c_hat[i-1]

        return 1-c_hat
    
    
    def __integral_c_hat_distance_c(self, x_int, pp1, pp2, perc, calibration_function):
        
        integral_c = np.zeros(x_int.shape)
        integral_c_hat = np.zeros(x_int.shape)
        
        for i in range(x_int.shape[0]):
            conf = x_int[i]
            if np.max([pp1[np.abs(x_int-conf).argmin()],pp2[np.abs(x_int-conf).argmin()]])>1e-6:
                accu = np.min([perc*pp1[np.abs(x_int-conf).argmin()]/pp2[np.abs(x_int-conf).argmin()],1.0])
                if np.isnan(accu)==False:
                    
                    c = inverse_func(f=calibration_function, y=conf)
                    
                    integral_c[i] = c
                    integral_c_hat[i] = accu
            else:
                if i>1:
                    integral_c[i] = integral_c[i-1]
                    integral_c_hat[i] = integral_c_hat[i-1]

        integral_c_hat = 1-integral_c_hat[::-1]
        
        integral_abs = np.abs(integral_c-integral_c_hat)**1*pp2
        integral_sq = np.abs(integral_c-integral_c_hat)**2*pp2
        
        ind = np.where((x_int >= 0.0) & (x_int <= 1.0))
        c_hat_dist_c_abs = np.trapz(integral_abs[ind],x_int[ind])/np.trapz(pp2[ind],x_int[ind])
        c_hat_dist_c_sq = np.trapz(integral_sq[ind],x_int[ind])/np.trapz(pp2[ind],x_int[ind])
        
        return c_hat_dist_c_abs, c_hat_dist_c_sq
    
        
    def __integral_ece_kde(self, x_int, pp1, pp2, perc):
                        
        integral_abs = np.zeros(x_int.shape)
        integral_sq = np.zeros(x_int.shape)
        
        for i in range(x_int.shape[0]):
            conf = x_int[i]
            if np.max([pp1[np.abs(x_int-conf).argmin()],pp2[np.abs(x_int-conf).argmin()]])>1e-6:
                accu = np.min([perc*pp1[np.abs(x_int-conf).argmin()]/pp2[np.abs(x_int-conf).argmin()],1.0])
                if np.isnan(accu)==False:
                    integral_abs[i] = np.abs(conf-accu)**1*pp2[i]
                    integral_sq[i] = np.abs(conf-accu)**2*pp2[i]
            else:
                if i>1:
                    integral_abs[i] = integral_abs[i-1]
                    integral_sq[i] = integral_sq[i-1]

        ind = np.where((x_int >= 0.0) & (x_int <= 1.0))
        ece_abs = np.trapz(integral_abs[ind],x_int[ind])/np.trapz(pp2[ind],x_int[ind])
        ece_sq = np.trapz(integral_sq[ind],x_int[ind])/np.trapz(pp2[ind],x_int[ind])
        
        return ece_abs, ece_sq
    
    
    # Code from
    # https://github.com/zhang64-llnl/Mix-n-Match-Calibration/blob/master/demo_calibration.py
    def __mirror_1d(self, d, xmin=None, xmax=None):
        """If necessary apply reflecting boundary conditions."""
        if xmin is not None and xmax is not None:
            xmed = (xmin+xmax)/2
            return np.concatenate(((2*xmin-d[d < xmed]).reshape(-1,1), d, (2*xmax-d[d >= xmed]).reshape(-1,1)))
        elif xmin is not None:
            return np.concatenate((2*xmin-d, d))
        elif xmax is not None:
            return np.concatenate((d, 2*xmax-d))
        else:
            return d

        
    # Code from
    # https://github.com/zhang64-llnl/Mix-n-Match-Calibration/blob/master/demo_calibration.py
    def __kde_setup(self, p, label):

        # points from numerical integration
        p_int = np.copy(p)

        p = np.clip(p,1e-256,1-1e-256)
        p_int = np.clip(p_int,1e-256,1-1e-256)

        x_int = np.linspace(-0.6, 1.6, num=2**14)

        N = p.shape[0]

        # this is needed to convert labels from one-hot to conventional form
        label_index = np.array([np.where(r==1)[0][0] for r in label])
        with torch.no_grad():
            if p.shape[1] !=2:
                p_new = torch.from_numpy(p)
                p_b = torch.zeros(N,1)
                label_binary = np.zeros((N,1))
                for i in range(N):
                    pred_label = int(torch.argmax(p_new[i]).numpy())
                    if pred_label == label_index[i]:
                        label_binary[i] = 1
                    p_b[i] = p_new[i,pred_label]/torch.sum(p_new[i,:])
            else:
                p_b = torch.from_numpy((p/np.sum(p,1)[:,None])[:,1])
                label_binary = label_index

        method = 'triweight'

        dconf_1 = (p_b[np.where(label_binary==1)].reshape(-1,1)).numpy()
        kbw = np.std(p_b.numpy())*(N*2)**-0.2
        kbw = np.std(dconf_1)*(N*2)**-0.2
        # Mirror the data about the domain boundary
        low_bound = 0.0
        up_bound = 1.0
        dconf_1m = self.__mirror_1d(dconf_1,low_bound,up_bound)
        # Compute KDE using the bandwidth found, and twice as many grid points
        pp1 = FFTKDE(bw=kbw, kernel=method).fit(dconf_1m).evaluate(x_int)
        pp1[x_int<=low_bound] = 0  # Set the KDE to zero outside of the domain
        pp1[x_int>=up_bound] = 0  # Set the KDE to zero outside of the domain
        pp1 = pp1 * 2  # Double the y-values to get integral of ~1


        p_int = p_int/np.sum(p_int,1)[:,None]
        N1 = p_int.shape[0]
        with torch.no_grad():
            p_new = torch.from_numpy(p_int)
            pred_b_int = np.zeros((N1,1))
            if p_int.shape[1]!=2:
                for i in range(N1):
                    pred_label = int(torch.argmax(p_new[i]).numpy())
                    pred_b_int[i] = p_int[i,pred_label]
            else:
                for i in range(N1):
                    pred_b_int[i] = p_int[i,1]

        low_bound = 0.0
        up_bound = 1.0
        pred_b_intm = self.__mirror_1d(pred_b_int,low_bound,up_bound)
        # Compute KDE using the bandwidth found, and twice as many grid points
        pp2 = FFTKDE(bw=kbw, kernel=method).fit(pred_b_intm).evaluate(x_int)
        pp2[x_int<=low_bound] = 0  # Set the KDE to zero outside of the domain
        pp2[x_int>=up_bound] = 0  # Set the KDE to zero outside of the domain
        pp2 = pp2 * 2  # Double the y-values to get integral of ~1


        if p.shape[1] !=2: # top label (confidence)
            perc = np.mean(label_binary)
        else: # or joint calibration for binary cases
            perc = np.mean(label_index)
        
        return x_int, pp1, pp2, perc
    
    
def inverse_func(f, y):
    """
    Returns the x for f(x)=y where f is a monotonic function with input space in range 0..1
    """
    if y > 1:
        return 1
    if y < 0:
        return 0
    
    min_x = 0
    max_x = 1
    x = 0.5

    while True:
        
        fx = f(x)
        if isclose(y, fx, abs_tol=1e-7):
            break
        elif fx > y:
            max_x = x
            x = (x + min_x) / 2
        elif fx < y:
            min_x = x
            x = (x + max_x) / 2

    return x