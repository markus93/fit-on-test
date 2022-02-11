import keras
from keras.layers import Add, Dense, Input, Activation
from keras.callbacks import EarlyStopping
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from sklearn.model_selection import KFold
import gc
from binnings import EqualSizeBinning
from tensorflow import cumsum
import time
from keras.layers import Layer

K.set_floatx('float64')


class Reg_for_piecewise(keras.regularizers.Regularizer):

    def __init__(self, l1, bounded = True):
    
        # l1 - regularization parameters
        # bounded - do not let the values go out of the range [0,1]
    
        self.l1 = l1
        self.reg = 0.0
        self.bounded = bounded
        #self.reg = K.variable(0., dtype=np.float64)
    def __call__(self, x):
        self.reg = 0.0
        #self.reg.assign(0)
        #tf.add(self.reg, -self.reg)
        
        diff = x[1:]-x[:-1]
        length = K.int_shape(diff)[0]
        
        if self.bounded:
            self.reg = tf.cond(x[0] < 0, lambda: tf.add(self.reg, self.l1*(-x[0])), lambda: tf.add(self.reg, 0.0)) # Condition so that left-most breakpoint wouldn't be smaller than 0
            self.reg = tf.cond((1.0 - x[-1]) < 0, lambda: tf.add(self.reg, self.l1*(x[-1]-1)), lambda: tf.add(self.reg, 0.0)) # Condition so that right-most breakpoint wouldn't be bigger than 0

        for i in range(length):
            self.reg = tf.cond(diff[i] < 0, lambda: tf.add(self.reg, self.l1*(-diff[i])), lambda: tf.add(self.reg, 0.0))  # Condition so all the breakpoints are in order
            
        return self.reg

def MSE_v2(y_true, y_pred):
    return K.mean(K.square(K.flatten(tf.cast(y_true, dtype=tf.float64)) - K.flatten(y_pred)))  # TODO: debug the code, why we need to flatten here

def MAE_v2(y_true, y_pred):
    return K.mean(K.abs(K.flatten(tf.cast(y_true, dtype=tf.float64)) - K.flatten(y_pred)))  # TODO: debug the code, why we need to flatten here

def CE_v2(y_true, y_pred):
    y = K.flatten(tf.cast(y_true, dtype=tf.float64))
    p = K.flatten(tf.cast(y_pred, dtype=tf.float64))
    p = K.clip(p, 1e-16, 1 - 1e-16)
    return K.mean(-(y*K.log(p) + (1-y)*K.log(1-p)))


def act_gate(x):
    return K.cast(x >= 0, np.float64)

get_custom_objects().update({'act_gate': Activation(act_gate)})

class Gate(keras.layers.Layer):
    def __init__(self, breakpoints=3, input_dim=1, l1 = 100000, fit_x = True):
        super(Gate, self).__init__()
        self.breakpoints = breakpoints-2        
        if self.breakpoints > 0:
            self.b = self.add_weight(shape=(self.breakpoints,), initializer="zeros", trainable=fit_x, name="b", regularizer=Reg_for_piecewise(l1))
        self.b0 = self.add_weight(shape=(1,), initializer="zeros", trainable=False, name="b0")
        self.b1 = self.add_weight(shape=(1,), initializer="ones", trainable=False, name="b1")
        
    def call(self, inputs):
        if self.breakpoints > 0:
            gates = K.concatenate([self.b0, self.b, self.b1])
        else:
            gates = K.concatenate([self.b0, self.b1])
        return K.concatenate([gates - inputs, inputs - gates], axis=1)

class Piecewise_layer(keras.layers.Layer):
    def __init__(self, breakpoints=3, input_dim=1, monotonic = False, l1 = 100000):
        super(Piecewise_layer, self).__init__()
        self.monotonic = monotonic
        
        if self.monotonic:
            self.y = self.add_weight(shape=(breakpoints,), initializer="ones", trainable=True, name="ys", regularizer=Reg_for_piecewise(l1, False))
        else:
            self.y = self.add_weight(shape=(breakpoints,), initializer="ones", trainable=True, name="ys")
        # +2 more than one breakpoint, if the edge points are excluded.
        self.breakp = breakpoints
        self.e = 1e-16

    def call(self, inputs):
        
        res = None
        dists = inputs[0]
        gates = inputs[1]
                
        for i in range(self.breakp-1):
            left_p = K.abs(dists[:, i+1])
            right_p = K.abs(dists[:, self.breakp+i])
            gate = gates[:,i+1]*gates[:,self.breakp+i]  # L_{i+1} >= 0 and R_i >=0
            if i != 0: # and check if the x is on the breakpoint. From the left side
                gate *= 1-(gates[:,i]*gates[:,self.breakp+i])
            
            res_sub = gate*(self.y[i+1]*((K.abs(right_p) + self.e)/(K.abs(left_p + right_p) + self.e)) + 
                            self.y[i]*((K.abs(left_p) + self.e)/(K.abs(left_p + right_p) + self.e)))
            
            if res is None:
                res = res_sub
            else:
                res += res_sub
        
        return res
    
class Piecewise_NN2():
    
    def __init__(self, k = 1, max_epochs = 1500, patience = 10, lr = 0.001, random_state = 15, 
                 loss = MSE_v2, verbose = False, opt = "Adam", l1 = 100000, fit_x = True, equal_size = False, monotonic = False):
        """
        Initialize class
        
        Params:
            k (int): how many breakpoints are there. I.e breakpoints for piecewise function.
            max_epochs (int): maximum iterations done by optimizer.
            patience (int): how many worse epochs before early stopping
            lr (float): learning rate of Adam optimizer
            random_state (int): random seed for numpy and tensorflow
            loss (string/class): loss function to optimize
            fit_x (bool): If the gates/breakpoints can be trained, default: True.
            equal_size (bool): If the weights are initialized based on the data with equal number of elements in each bin.

        """
        
        self.max_epochs = max_epochs
        self.patience = patience
        self.k = k
        self.lr = lr
        self.random_state = random_state
        self.loss = loss
        self.l1 = l1
        self.fit_x = fit_x
        self.equal_size = equal_size
        self.monotonic = monotonic
       
        if opt == "Adam":
            self.opt = keras.optimizers.Adam(lr = self.lr)
        elif opt == "SGD":
            self.opt = keras.optimizers.SGD(lr = self.lr)
        else:
            self.opt = keras.optimizers.RMSprop(lr = self.lr)
        
        if k >= 0:
            self.model = self.create_model(k, verbose)
        else:
            self.model = None        
        
        #tf.random.set_seed(random_state)
        tf.set_random_seed(random_state)
        np.random.seed(random_state)
    
    def create_model(self, k, verbose = False):
        breakpoints = k + 2 # break points
        x = Input(shape=(1,), name="input")
        y = Gate(breakpoints,1, self.l1, self.fit_x)(x)
        z = Activation("act_gate")(y)
        w = Piecewise_layer(breakpoints,1, self.monotonic, self.l1)([y, z])
        
        model = keras.models.Model(inputs=x, outputs=w) 
        if breakpoints > 2:
            model.set_weights([np.linspace(0,1,breakpoints)[1:-1], np.array([0]), np.array([1]), np.linspace(0,1,breakpoints)]) # Initialise weights
        else: # Only 2 breakpoints, on each edge (0,1)
            model.set_weights([np.array([0]), np.array([1]), np.linspace(0,1,breakpoints)]) # Initialise weights        model.compile(loss=self.loss, optimizer=self.opt)

        model.compile(loss=self.loss, optimizer=self.opt)

        if verbose:
            model.summary()

        return model

    def fit(self, probs, true, batch_size = 32, verbose = False):
        """
        Trains the model and finds optimal parameters

        Params:
            probs: the input for model
            true: one-hot-encoding of true labels.
            batch_size: Number of instances in one batch
            verbose (bool): whether to print out anything or not

        Returns:
            hist: Keras history of learning process
        """

        if self.model is None:
            print("Warning no model created: Creating model with 1 breakpoint.")
            self.model = self.create_model(1, verbose)
            
        if self.equal_size == True:
            breakpoints = self.k + 2
            n_bins = breakpoints - 1

            esb = EqualSizeBinning(probs, true, np.array([-1]*len(true)), n_bins)
            bin_borders = esb.get_bin_borders()
            
            print("Using equal size with Borders:", bin_borders)
            
            if breakpoints > 2:
                self.model.set_weights([bin_borders[1:-1], np.array([0]), np.array([1]), bin_borders]) # Initialise weights
            else: # Only 2 breakpoints, on each edge (0,1)
                self.model.set_weights([np.array([0]), np.array([1]), bin_borders]) # Initialise weights        model.compile(loss=self.loss, optimizer=self.opt)


        early_stop = EarlyStopping(monitor='loss', min_delta=0, patience=self.patience, verbose=verbose, mode='auto')
        cbs = [early_stop]

        hist = self.model.fit(probs, true, epochs=self.max_epochs, callbacks=cbs, batch_size=batch_size, verbose=verbose)

        return hist
    
    def predict(self, probs, clip_preds = True):
        """
        Scales probabilities based on the model and returns calibrated probabilities
        
        Params:
            probs: probability values of data for each class (shape [samples, classes])
            clip_preds (bool): If the predictions should be clipped to be in region between 0 to 1.
            
        Returns:
            calibrated probabilities (nd.array with shape [samples, classes])
        """
        
        pred = self.model.predict(probs)
        
        if clip_preds:
            pred = np.clip(pred, 0, 1)
        
        return pred


class Piecewise_NN2_val():
    
    def __init__(self, k = 1, max_epochs = 1500, patience = 10, lr = 0.001, random_state = 15, 
                 loss = MSE_v2, verbose = False, opt = "Adam", l1 = 100000, fit_x = True, equal_size = False, 
                 monotonic = False, val_split = 0.2):
        """
        Initialize class
        
        Params:
            k (int): how many breakpoints are there. I.e breakpoints for piecewise function.
            max_epochs (int): maximum iterations done by optimizer.
            patience (int): how many worse epochs before early stopping
            lr (float): learning rate of Adam optimizer
            random_state (int): random seed for numpy and tensorflow
            loss (string/class): loss function to optimize
            fit_x (bool): If the gates/breakpoints can be trained, default: True.
            equal_size (bool): If the weights are initialized based on the data with equal number of elements in each bin.

        """
        
        self.max_epochs = max_epochs
        self.patience = patience
        self.k = k
        self.lr = lr
        self.random_state = random_state
        self.loss = loss
        self.l1 = l1
        self.fit_x = fit_x
        self.equal_size = equal_size
        self.monotonic = monotonic
        self.val_split = val_split
       
        if opt == "Adam":
            self.opt = keras.optimizers.Adam(lr = self.lr)
        elif opt == "SGD":
            self.opt = keras.optimizers.SGD(lr = self.lr)
        else:
            self.opt = keras.optimizers.RMSprop(lr = self.lr)
        
        if k >= 0:
            self.model = self.create_model(k, verbose)
        else:
            self.model = None        
        
        #tf.random.set_seed(random_state)
        tf.set_random_seed(random_state)
        np.random.seed(random_state)
    
    def create_model(self, k, verbose = False):
        breakpoints = k + 2 # break points
        x = Input(shape=(1,), name="input")
        y = Gate(breakpoints,1, self.l1, self.fit_x)(x)
        z = Activation("act_gate")(y)
        w = Piecewise_layer(breakpoints,1, self.monotonic, self.l1)([y, z])
        
        model = keras.models.Model(inputs=x, outputs=w) 
        if breakpoints > 2:
            model.set_weights([np.linspace(0,1,breakpoints)[1:-1], np.array([0]), np.array([1]), np.linspace(0,1,breakpoints)]) # Initialise weights
        else: # Only 2 breakpoints, on each edge (0,1)
            model.set_weights([np.array([0]), np.array([1]), np.linspace(0,1,breakpoints)]) # Initialise weights        model.compile(loss=self.loss, optimizer=self.opt)

        model.compile(loss=self.loss, optimizer=self.opt)

        if verbose:
            model.summary()

        return model

    def fit(self, probs, true, batch_size = 32, verbose = False):
        """
        Trains the model and finds optimal parameters

        Params:
            probs: the input for model
            true: one-hot-encoding of true labels.
            batch_size: Number of instances in one batch
            verbose (bool): whether to print out anything or not

        Returns:
            hist: Keras history of learning process
        """

        if self.model is None:
            print("Warning no model created: Creating model with 1 breakpoint.")
            self.model = self.create_model(1, verbose)
            
        if self.equal_size == True:
            breakpoints = self.k + 2
            n_bins = breakpoints - 1

            esb = EqualSizeBinning(probs, true, np.array([-1]*len(true)), n_bins)
            bin_borders = esb.get_bin_borders()
            
            print("Using equal size with Borders:", bin_borders)
            
            if breakpoints > 2:
                self.model.set_weights([bin_borders[1:-1], np.array([0]), np.array([1]), bin_borders]) # Initialise weights
            else: # Only 2 breakpoints, on each edge (0,1)
                self.model.set_weights([np.array([0]), np.array([1]), bin_borders]) # Initialise weights        model.compile(loss=self.loss, optimizer=self.opt)


        early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=self.patience, verbose=verbose, mode='auto')
        cbs = [early_stop]

        hist = self.model.fit(probs, true, epochs=self.max_epochs, callbacks=cbs, batch_size=batch_size, verbose=verbose,
                              validation_split = self.val_split)

        return hist
    
    def predict(self, probs, clip_preds = True):
        """
        Scales probabilities based on the model and returns calibrated probabilities
        
        Params:
            probs: probability values of data for each class (shape [samples, classes])
            clip_preds (bool): If the predictions should be clipped to be in region between 0 to 1.
            
        Returns:
            calibrated probabilities (nd.array with shape [samples, classes])
        """
        
        pred = self.model.predict(probs)
        
        if clip_preds:
            pred = np.clip(pred, 0, 1)
        
        return pred
        


### =========== Piecewise_NN4 ===========

def K_softmax(x):
    e_x = K.exp(x - K.max(x))
    return e_x / K.sum(e_x, axis=0)

def K_sigmoid(x):
    return 1/(1+K.exp(-x))
    
def sigmoid(x):
    return 1/(1+np.exp(-x))

def rev_sigmoid(x, eps = 0.001):
    x = np.array(x)
    x = np.clip(x, eps, 1 - eps)
    return np.log(x/(1-x))

def rev_softmax(x, eps = 1e-16):
    x = np.array(x)
    x[1:] -= x[:-1]  # Make it reverse cumulative
    x = np.clip(x, eps, 1)
    return np.log(x)

get_custom_objects().update({'act_gate': Activation(act_gate)})


class Gate2(keras.layers.Layer):
    def __init__(self, breakpoints=3, input_dim=1, fit_x = True):
        super(Gate2, self).__init__()
        self.breakpoints = breakpoints-1        
        self.b = self.add_weight(shape=(self.breakpoints,), initializer="zeros", trainable=fit_x, name="b")
        self.b0 = self.add_weight(shape=(1,), initializer="zeros", trainable=False, name="b0")
        
    def call(self, inputs):
        
        b_cm = cumsum(K_softmax(self.b))  # Softmax and cumulative sum, TODO should use it as an layer.
        gates = K.concatenate([self.b0, b_cm[:-1], [1]])
        
        return K.concatenate([gates - inputs, inputs - gates], axis=1)
        
        


class Piecewise_layer2(keras.layers.Layer):
    def __init__(self, breakpoints=3, input_dim=1, monotonic = False):
        super(Piecewise_layer2, self).__init__()
        self.monotonic = monotonic

        if self.monotonic:
            self.y = self.add_weight(shape=(breakpoints-1,), initializer="ones", trainable=True, name="ys")
            self.y0 = self.add_weight(shape=(1,), initializer="zeros", trainable=True, name="y0")
            self.y1 = self.add_weight(shape=(1,), initializer="ones", trainable=True, name="y1")
        
        else: 
            self.y = self.add_weight(shape=(breakpoints,), initializer="ones", trainable=True, name="ys")
            
        # +2 more than one breakpoint, if the edge points are excluded.
        self.breakp = breakpoints
        self.e = 1e-16

    def call(self, inputs):
        
        res = None
        dists = inputs[0]
        gates = inputs[1]
        
        if self.monotonic:
            y_act = cumsum(K_softmax(self.y))
            y_act = K.concatenate([self.y0, y_act*(self.y1 - self.y0) + self.y0])
        else:
            y_act = K_sigmoid(self.y)  # Sigmoid activate: TODO, use it as a layer
                
        for i in range(self.breakp-1):
            left_p = K.abs(dists[:, i+1])
            right_p = K.abs(dists[:, self.breakp+i])
            gate = gates[:,i+1]*gates[:,self.breakp+i]  # L_{i+1} >= 0 and R_i >=0
            if i != 0: # and check if the x is on the breakpoint. From the left side
                gate *= 1-(gates[:,i]*gates[:,self.breakp+i])
            
            res_sub = gate*(y_act[i+1]*((K.abs(right_p) + self.e)/(K.abs(left_p + right_p) + self.e)) + 
                            y_act[i]*((K.abs(left_p) + self.e)/(K.abs(left_p + right_p) + self.e)))
            
            if res is None:
                res = res_sub
            else:
                res += res_sub
        
        return res
        

class Piecewise_NN3():
    
    def __init__(self, k = 1, max_epochs = 1500, patience = 10, lr = 0.01, random_state = 15, 
                 loss = MSE_v2, verbose = False, opt = "Adam", l1 = 100000, fit_x = True, equal_size = False, monotonic = False):
        """
        Initialize class
        
        Params:
            k (int): how many breakpoints are there. I.e breakpoints for piecewise function.
            max_epochs (int): maximum iterations done by optimizer.
            patience (int): how many worse epochs before early stopping
            lr (float): learning rate of Adam optimizer
            random_state (int): random seed for numpy and tensorflow
            loss (string/class): loss function to optimize
            fit_x (bool): If the gates/breakpoints can be trained, default: True.
            equal_size (bool): If the weights are initialized based on the data with equal number of elements in each bin.

        """
        
        self.max_epochs = max_epochs
        self.patience = patience
        self.k = k
        self.lr = lr
        self.random_state = random_state
        self.loss = loss
        self.l1 = l1
        self.fit_x = fit_x
        self.equal_size = equal_size
        self.monotonic = monotonic
       
        if opt == "Adam":
            self.opt = keras.optimizers.Adam(lr = self.lr)
        elif opt == "SGD":
            self.opt = keras.optimizers.SGD(lr = self.lr)
        else:
            self.opt = keras.optimizers.RMSprop(lr = self.lr)
        
        if k >= 0:
            self.model = self.create_model(k, verbose)
        else:
            self.model = None        
        
        #tf.random.set_seed(random_state)
        tf.compat.v1.set_random_seed(random_state)
        np.random.seed(random_state)
    
    def create_model(self, k, verbose = False):
        breakpoints = k + 2 # break points
        
        x = Input(shape=(1,), name="input")
        y = Gate2(breakpoints,1, self.fit_x)(x)
        z = Activation("act_gate")(y)
        w = Piecewise_layer(breakpoints,1, self.monotonic)([y, z])
        
        model = keras.models.Model(inputs=x, outputs=w) 
        ws_cm = rev_softmax(np.linspace(0,1,breakpoints)[1:])
        
        y_act = np.linspace(0,1,breakpoints)
            
        model.set_weights([ws_cm, np.array([0]), y_act]) # Initialise weights
            
        model.compile(loss=self.loss, optimizer=self.opt)

        if verbose:
            model.summary()

        return model

    def fit(self, probs, true, batch_size = 32, verbose = False):
        """
        Trains the model and finds optimal parameters

        Params:
            probs: the input for model
            true: one-hot-encoding of true labels.
            batch_size: Number of instances in one batch
            verbose (bool): whether to print out anything or not

        Returns:
            hist: Keras history of learning process
        """

        if self.model is None:
            print("Warning no model created: Creating model with 1 breakpoint.")
            self.model = self.create_model(1, verbose)
            
        if self.equal_size == True:
            breakpoints = self.k + 2
            n_bins = breakpoints - 1

            esb = EqualSizeBinning(probs, true, np.array([-1]*len(true)), n_bins)
            bin_borders = esb.get_bin_borders()
            
            print("Using equal size with Borders:", bin_borders)
            ws_cm = rev_softmax(np.copy(bin_borders[1:]))
            
            y_act = bin_borders
            self.model.set_weights([ws_cm, np.array([0]), y_act]) # Initialise weights

        early_stop = EarlyStopping(monitor='loss', min_delta=0, patience=self.patience, verbose=verbose, mode='auto')
        cbs = [early_stop]

        hist = self.model.fit(probs, true, epochs=self.max_epochs, callbacks=cbs, batch_size=batch_size, verbose=verbose)

        return hist
    
    def predict(self, probs, clip_preds = True):
        """
        Scales probabilities based on the model and returns calibrated probabilities
        
        Params:
            probs: probability values of data for each class (shape [samples, classes])
            clip_preds (bool): If the predictions should be clipped to be in region between 0 to 1.
            
        Returns:
            calibrated probabilities (nd.array with shape [samples, classes])
        """
        
        pred = self.model.predict(probs)
        
        if clip_preds:
            pred = np.clip(pred, 0, 1)
        
        return pred        

        
def K_logit(x):
    return K.log(x/(1-x))        
        
def act_logit(x, eps = 1e-16):
    l_max = K.cast(K_logit(1-eps), dtype = np.float64)  # L_max and min are same
    x_clip = K.clip(x, eps, 1-eps)
    return (K_logit(x_clip) + l_max)/(2*l_max)  # -20 20 -> 0 40 -> 0 1
  
def K_clip_logit(x, eps = 1e-16):
    x_clip = K.clip(x, eps, 1-eps)
    return K_logit(x_clip)  # -20 20 -> 0 40 -> 0 1  
    
def logit(x):
    return np.log(x/(1-x))
    
def clip_logit(x):
    x_clipped = np.clip(x, 1e-16, 1-1e-16)
    return logit(x_clipped)
    
def logit_to_scale(x, eps = 1e-16):
    l_max = logit(1-eps)
    x = np.array(x, dtype = np.float64)
    x_clip = np.clip(x, eps, 1-eps)
    return (logit(x_clip) + l_max)/(2*l_max)

class Act_min_max_scale(Layer):
    def __init__(self, x_min = 0, x_max = 1, **kwargs):
        super(Act_min_max_scale, self).__init__(**kwargs)
        self.x_min = K.cast_to_floatx(x_min)
        self.x_max = K.cast_to_floatx(x_max)

    def call(self, inputs):
        x_clip = K.clip(inputs, self.x_min, self.x_max)
        return (x_clip - self.x_min)/(self.x_max - self.x_min) 

    def get_config(self):
        config = {'x_min': float(self.x_min), "x_max": float(self.x_max)}
        base_config = super(Act_min_max_scale, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape
        
        
class Act_logistic_out(Layer):
    def __init__(self, x_min = -50, x_max = 50, **kwargs):
        super(Act_logistic_out, self).__init__(**kwargs)
        self.x_min = K.cast_to_floatx(x_min)
        self.x_max = K.cast_to_floatx(x_max)

    def call(self, inputs):
        return 1/(1+K.exp(-((self.x_max - self.x_min)*inputs + self.x_min)))

    def get_config(self):
        config = {'x_min': float(self.x_min), "x_max": float(self.x_max)}
        base_config = super(Act_logistic_out, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


def min_max_scale(x, x_min, x_max):
    x_clip = np.clip(x, x_min, x_max)
    return (x_clip - x_min)/(x_max - x_min) 

def act_gate(x):
    return K.cast(x >= 0, np.float64)

get_custom_objects().update({'act_gate': Activation(act_gate), 'act_logit':Activation(act_logit)})

    
class Piecewise_NN4():
    
    def __init__(self, k = 1, max_epochs = 1500, patience = 20, lr = 0.01, random_state = 15, 
                 loss = MSE_v2, verbose = False, opt = "Adam", l1 = 100000, fit_x = True, 
                 equal_size = False, monotonic = False, logit_scale = False, eps = 1e-5, logit_input = False, 
                 logistic_out = False, use_ce_loss = False):
        """
        Initialize class
        
        Params:
            k (int): how many breakpoints are there. I.e breakpoints for piecewise function.
            max_epochs (int): maximum iterations done by optimizer.
            patience (int): how many worse epochs before early stopping
            lr (float): learning rate of Adam optimizer
            random_state (int): random seed for numpy and tensorflow
            loss (string/class): loss function to optimize
            fit_x (bool): If the gates/breakpoints can be trained, default: True.
            equal_size (bool): If the weights are initialized based on the data with equal number of elements in each bin.

        """
        
        self.max_epochs = max_epochs
        self.patience = patience
        self.k = k
        self.lr = lr
        self.random_state = random_state
        self.loss = loss
        self.l1 = l1
        self.fit_x = fit_x
        self.equal_size = equal_size
        self.monotonic = monotonic
        self.logit_scale = logit_scale
        self.eps = eps
        self.x_min = 0
        self.x_max = 1
        self.logit_input = logit_input
        self.logistic_out = logistic_out
       
        if use_ce_loss:
            self.loss = CE_v2
       
        if opt == "Adam":
            self.opt = keras.optimizers.Adam(lr = self.lr)
        elif opt == "SGD":
            self.opt = keras.optimizers.SGD(lr = self.lr)
        else:
            self.opt = keras.optimizers.RMSprop(lr = self.lr)
        
        if k >= 0:
            self.model = self.create_model(k, verbose)
        else:
            self.model = None        
        
        #tf.random.set_seed(random_state)
        tf.compat.v1.set_random_seed(random_state)
        np.random.seed(random_state)
    
    def create_model(self, k, verbose = False):
        breakpoints = k + 2 # break points
        
        x = Input(shape=(1,), name="input")
        
        if self.logit_scale:
            x2 = Activation("act_logit")(x)
            y = Gate2(breakpoints,1, self.fit_x)(x2)
        elif self.logit_input:
            x2 = Act_min_max_scale(self.x_min, self.x_max)(x)
            y = Gate2(breakpoints,1, self.fit_x)(x2)
        else:
            y = Gate2(breakpoints,1, self.fit_x)(x)
            
        z = Activation("act_gate")(y)
        
        if self.logistic_out:
            w0 = Piecewise_layer2(breakpoints,1, self.monotonic)([y, z])
            w = Act_logistic_out()(w0)
        else:
            w = Piecewise_layer2(breakpoints,1, self.monotonic)([y, z])

        
        model = keras.models.Model(inputs=x, outputs=w) 
        ws_cm = rev_softmax(np.linspace(0,1,breakpoints)[1:])
        
        if self.monotonic:
            y_act = rev_softmax(np.linspace(0,1,breakpoints)[1:])
            model.set_weights([ws_cm, np.array([0]), y_act, np.array([0]), np.array([1])]) # Initialise weights
        else:
            y_act = rev_sigmoid(np.linspace(0,1,breakpoints))
            model.set_weights([ws_cm, np.array([0]), y_act]) # Initialise weights
            
        model.compile(loss=self.loss, optimizer=self.opt)

        if verbose:
            model.summary()

        return model

    def fit(self, probs, true, batch_size = 32, verbose = False):
        """
        Trains the model and finds optimal parameters

        Params:
            probs: the input for model
            true: one-hot-encoding of true labels.
            batch_size: Number of instances in one batch
            verbose (bool): whether to print out anything or not

        Returns:
            hist: Keras history of learning process
        """
        
        if self.logit_input:
            self.x_min = -50 #K.min(probs)
            self.x_max = 50 #K.max(probs)
            self.model = self.create_model(self.k, verbose)

        elif self.model is None:
            print("Warning no model created: Creating model with 1 breakpoint.")
            self.model = self.create_model(1, verbose)
        
        
        if self.equal_size == True:
            breakpoints = self.k + 2
            n_bins = breakpoints - 1

            bin_borders = []
            brkpts = np.linspace(0,1,breakpoints)
            
            if self.logit_input:
                probs_brk = min_max_scale(probs, -50, 50) #np.min(probs), np.max(probs))
            elif self.logit_scale:
                probs_brk = logit_to_scale(probs)
            else:
                probs_brk = probs
            
            for brk in brkpts:
                bin_borders.append(np.quantile(probs_brk, brk))
            
            
            print("Using equal size with Borders:", bin_borders)  # 0.4 0.5 1
            ws_cm = rev_softmax(np.copy(bin_borders[1:]))  # 0.5 1 -> 0.5 0.5 -> np.log(0.5); np.log(0.5)
            
            if self.monotonic:
                y_act = rev_softmax(np.copy(bin_borders[1:])) # 0.4 0.5 1 -> 0.4 0.1 0.5 -> np.log(0.4); np.log(0.1), np.log(0.5)
                self.model.set_weights([ws_cm, np.array([0]), y_act, np.array([0]), np.array([1])]) # Initialise weights
            else:
                y_act = rev_sigmoid(bin_borders)
                self.model.set_weights([ws_cm, np.array([0]), y_act]) # Initialise weights
                

        early_stop = EarlyStopping(monitor='loss', min_delta=0, patience=self.patience, verbose=verbose, mode='auto')
        cbs = [early_stop]

        hist = self.model.fit(probs, true, epochs=self.max_epochs, callbacks=cbs, batch_size=batch_size, verbose=verbose)

        return hist
    
    def predict(self, probs, clip_preds = True):
        """
        Scales probabilities based on the model and returns calibrated probabilities
        
        Params:
            probs: probability values of data for each class (shape [samples, classes])
            clip_preds (bool): If the predictions should be clipped to be in region between 0 to 1.
            
        Returns:
            calibrated probabilities (nd.array with shape [samples, classes])
        """
        
        pred = self.model.predict(probs)
        
        return pred



## ============ PW_NN5 ===================

class Piecewise_layer3(keras.layers.Layer):
    def __init__(self, breakpoints=3, input_dim=1, monotonic = False):
        super(Piecewise_layer3, self).__init__()
        self.monotonic = monotonic

        self.y = self.add_weight(shape=(breakpoints,), initializer="ones", trainable=True, name="ys")
        self.y0 = self.add_weight(shape=(1,), initializer="zeros", trainable=True, name="y0")
        self.y1 = self.add_weight(shape=(1,), initializer="ones", trainable=True, name="y1")

        # +2 more than one breakpoint, if the edge points are excluded.
        self.breakp = breakpoints
        self.e = 1e-16

    def call(self, inputs):
        
        res = None
        dists = inputs[0]
        gates = inputs[1]
        
        if self.monotonic:
            y_act = cumsum(K_softmax(self.y))
        else:
            y_act = K_sigmoid(self.y)  # Sigmoid activate: TODO, use it as a layer
        
        y_act = K.concatenate([self.y0, y_act[1:-1], self.y1])

                
        for i in range(self.breakp-1):
            left_p = K.abs(dists[:, i+1])
            right_p = K.abs(dists[:, self.breakp+i])
            gate = gates[:,i+1]*gates[:,self.breakp+i]  # L_{i+1} >= 0 and R_i >=0
            if i != 0: # and check if the x is on the breakpoint. From the left side
                gate *= 1-(gates[:,i]*gates[:,self.breakp+i])
            
            res_sub = gate*(y_act[i+1]*((K.abs(right_p) + self.e)/(K.abs(left_p + right_p) + self.e)) + 
                            y_act[i]*((K.abs(left_p) + self.e)/(K.abs(left_p + right_p) + self.e)))
            
            if res is None:
                res = res_sub
            else:
                res += res_sub
        
        return res

class Piecewise_NN5():
    
    def __init__(self, k = 1, max_epochs = 1500, patience = 10, lr = 0.01, random_state = 15, 
                 loss = MSE_v2, verbose = False, opt = "Adam", l1 = 100000, fit_x = True, equal_size = False, monotonic = False):
        """
        Initialize class
        
        Params:
            k (int): how many breakpoints are there. I.e breakpoints for piecewise function.
            max_epochs (int): maximum iterations done by optimizer.
            patience (int): how many worse epochs before early stopping
            lr (float): learning rate of Adam optimizer
            random_state (int): random seed for numpy and tensorflow
            loss (string/class): loss function to optimize
            fit_x (bool): If the gates/breakpoints can be trained, default: True.
            equal_size (bool): If the weights are initialized based on the data with equal number of elements in each bin.

        """
        
        self.max_epochs = max_epochs
        self.patience = patience
        self.k = k
        self.lr = lr
        self.random_state = random_state
        self.loss = loss
        self.l1 = l1
        self.fit_x = fit_x
        self.equal_size = equal_size
        self.monotonic = monotonic
       
        if opt == "Adam":
            self.opt = keras.optimizers.Adam(lr = self.lr)
        elif opt == "SGD":
            self.opt = keras.optimizers.SGD(lr = self.lr)
        else:
            self.opt = keras.optimizers.RMSprop(lr = self.lr)
        
        if k >= 0:
            self.model = self.create_model(k, verbose)
        else:
            self.model = None        
        
        #tf.random.set_seed(random_state)
        tf.compat.v1.set_random_seed(random_state)
        np.random.seed(random_state)
    
    def create_model(self, k, verbose = False):
        breakpoints = k + 2 # break points
        
        x = Input(shape=(1,), name="input")
        y = Gate2(breakpoints,1, self.fit_x)(x)
        z = Activation("act_gate")(y)
        w = Piecewise_layer3(breakpoints,1, self.monotonic)([y, z])
        
        model = keras.models.Model(inputs=x, outputs=w) 
        ws_cm = rev_softmax(np.linspace(0,1,breakpoints)[1:])
        
        if self.monotonic:
            y_act = rev_softmax(np.linspace(0,1,breakpoints))
        else:
            y_act = rev_sigmoid(np.linspace(0,1,breakpoints))
        
        model.set_weights([ws_cm, np.array([0]), y_act, np.array([0]), np.array([1])]) # Initialise weights
            
        model.compile(loss=self.loss, optimizer=self.opt)

        if verbose:
            model.summary()

        return model

    def fit(self, probs, true, batch_size = 32, verbose = False):
        """
        Trains the model and finds optimal parameters

        Params:
            probs: the input for model
            true: one-hot-encoding of true labels.
            batch_size: Number of instances in one batch
            verbose (bool): whether to print out anything or not

        Returns:
            hist: Keras history of learning process
        """

        if self.model is None:
            print("Warning no model created: Creating model with 1 breakpoint.")
            self.model = self.create_model(1, verbose)
            
        if self.equal_size == True:
            breakpoints = self.k + 2
            n_bins = breakpoints - 1

            esb = EqualSizeBinning(probs, true, np.array([-1]*len(true)), n_bins)
            bin_borders = esb.get_bin_borders()
            
            print("Using equal size with Borders:", bin_borders)
            ws_cm = rev_softmax(np.copy(bin_borders[1:]))
            
            if self.monotonic:
                y_act = rev_softmax(bin_borders)
            else:
                y_act = rev_sigmoid(bin_borders)
            self.model.set_weights([ws_cm, np.array([0]), y_act, np.array([0]), np.array([1])]) # Initialise weights

        early_stop = EarlyStopping(monitor='loss', min_delta=0, patience=self.patience, verbose=verbose, mode='auto')
        cbs = [early_stop]

        hist = self.model.fit(probs, true, epochs=self.max_epochs, callbacks=cbs, batch_size=batch_size, verbose=verbose)

        return hist
    
    def predict(self, probs, clip_preds = True):
        """
        Scales probabilities based on the model and returns calibrated probabilities
        
        Params:
            probs: probability values of data for each class (shape [samples, classes])
            clip_preds (bool): If the predictions should be clipped to be in region between 0 to 1.
            
        Returns:
            calibrated probabilities (nd.array with shape [samples, classes])
        """
        
        pred = self.model.predict(probs)
        
        if clip_preds:
            pred = np.clip(pred, 0, 1)
        
        return pred
        
        
class Gate6(keras.layers.Layer):
    def __init__(self, breakpoints=3, input_dim=1, fit_x = True):
        super(Gate6, self).__init__()
        self.breakpoints = breakpoints-1        
        self.b = self.add_weight(shape=(self.breakpoints,), initializer="zeros", trainable=fit_x, name="b")
        self.b0 = self.add_weight(shape=(1,), initializer="zeros", trainable=False, name="b0")
        
    def call(self, inputs):
        
        b_cm = cumsum(K_softmax(self.b))  # Softmax and cumulative sum, TODO should use it as an layer.
        gates = K.concatenate([self.b0, b_cm[:-1], [1]])
        
        gates = K_clip_logit(gates) # 0 1
        
        return K.concatenate([gates - inputs, inputs - gates], axis=1)
        
class Piecewise_layer6(keras.layers.Layer):
    def __init__(self, breakpoints=3, input_dim=1, monotonic = False):
        super(Piecewise_layer6, self).__init__()
        self.monotonic = monotonic

        if self.monotonic:
            self.y = self.add_weight(shape=(breakpoints-1,), initializer="ones", trainable=True, name="ys")
            self.y0 = self.add_weight(shape=(1,), initializer="zeros", trainable=True, name="y0")
            self.y1 = self.add_weight(shape=(1,), initializer="ones", trainable=True, name="y1")
        
        else: 
            self.y = self.add_weight(shape=(breakpoints,), initializer="ones", trainable=True, name="ys")
            
        # +2 more than one breakpoint, if the edge points are excluded.
        self.breakp = breakpoints
        self.e = 1e-16

    def call(self, inputs):
        
        res = None
        dists = inputs[0]
        gates = inputs[1]
        
        if self.monotonic:
            y_act = cumsum(K_softmax(self.y))
            y_act = K.concatenate([self.y0, y_act*(self.y1 - self.y0) + self.y0])
        else:
            y_act = self.y  # Sigmoid activate: TODO, use it as a layer
                
        for i in range(self.breakp-1):
            left_p = K.abs(dists[:, i+1])
            right_p = K.abs(dists[:, self.breakp+i])
            gate = gates[:,i+1]*gates[:,self.breakp+i]  # L_{i+1} >= 0 and R_i >=0
            if i != 0: # and check if the x is on the breakpoint. From the left side
                gate *= 1-(gates[:,i]*gates[:,self.breakp+i])
            
            res_sub = gate*(y_act[i+1]*((K.abs(right_p) + self.e)/(K.abs(left_p + right_p) + self.e)) + 
                            y_act[i]*((K.abs(left_p) + self.e)/(K.abs(left_p + right_p) + self.e)))
            
            if res is None:
                res = res_sub
            else:
                res += res_sub
        
        return res
        
class Act_logistic_out6(Layer):
    def __init__(self, **kwargs):
        super(Act_logistic_out6, self).__init__(**kwargs)

    def call(self, inputs):
        return 1/(1+K.exp(-inputs))

    def compute_output_shape(self, input_shape):
        return input_shape


class Act_clip_logit(Layer):
    def __init__(self, **kwargs):
        super(Act_clip_logit, self).__init__(**kwargs)

    def call(self, inputs):
        return K_clip_logit(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape


class Act_clip(Layer):
    def __init__(self, **kwargs):
        super(Act_clip, self).__init__(**kwargs)

    def call(self, inputs):
        eps = 1e-15  # TODO: Fix the stability issue - K_clip_logit gives 36.841
        x_min = np.log(eps/(1-eps))
        x_max = np.log((1-eps)/eps)
        x_clip = K.clip(inputs, x_min, x_max)  
        return x_clip

    def compute_output_shape(self, input_shape):
        return input_shape

class Piecewise_NN6():
    
    def __init__(self, k = 1, max_epochs = 1500, patience = 20, lr = 0.01, random_state = 15, 
                 loss = MSE_v2, verbose = False, opt = "Adam", l1 = 100000, fit_x = True, 
                 equal_size = False, monotonic = False, logit_scale = False, eps = 1e-5, logit_input = False, 
                 logistic_out = False, use_ce_loss = False):
        """
        Initialize class
        
        Params:
            k (int): how many breakpoints are there. I.e breakpoints for piecewise function.
            max_epochs (int): maximum iterations done by optimizer.
            patience (int): how many worse epochs before early stopping
            lr (float): learning rate of Adam optimizer
            random_state (int): random seed for numpy and tensorflow
            loss (string/class): loss function to optimize
            fit_x (bool): If the gates/breakpoints can be trained, default: True.
            equal_size (bool): If the weights are initialized based on the data with equal number of elements in each bin.

        """
        
        self.max_epochs = max_epochs
        self.patience = patience
        self.k = k
        self.lr = lr
        self.random_state = random_state
        self.loss = loss
        self.l1 = l1
        self.fit_x = fit_x
        self.equal_size = equal_size
        self.monotonic = monotonic
        self.logit_scale = logit_scale
        self.eps = eps
        self.x_min = 0
        self.x_max = 1
        self.logit_input = logit_input
        self.logistic_out = logistic_out
       
        if use_ce_loss:
            self.loss = CE_v2
       
        if opt == "Adam":
            self.opt = keras.optimizers.Adam(lr = self.lr)
        elif opt == "SGD":
            self.opt = keras.optimizers.SGD(lr = self.lr)
        else:
            self.opt = keras.optimizers.RMSprop(lr = self.lr)
        
        if k >= 0:
            self.model = self.create_model(k, verbose)
        else:
            self.model = None        
        
        #tf.random.set_seed(random_state)
        tf.compat.v1.set_random_seed(random_state)
        np.random.seed(random_state)
    
    def create_model(self, k, verbose = False):
        breakpoints = k + 2 # break points
        
        x = Input(shape=(1,), name="input")
        
        if self.logit_scale:
            x2 = Act_clip_logit()(x)
            y = Gate6(breakpoints,1, self.fit_x)(x2)
        elif self.logit_input:
            x2 = Act_clip()(x)
            y = Gate6(breakpoints,1, self.fit_x)(x2)
        else:
            y = Gate6(breakpoints,1, self.fit_x)(x)
            
        z = Activation("act_gate")(y)
        
        if self.logistic_out:
            w0 = Piecewise_layer6(breakpoints,1, self.monotonic)([y, z])
            w = Act_logistic_out6()(w0)
        else:
            w = Piecewise_layer6(breakpoints,1, self.monotonic)([y, z])

        
        model = keras.models.Model(inputs=x, outputs=w) 
        ws_cm = rev_softmax(np.linspace(0,1,breakpoints)[1:])
        
        if self.monotonic:
            y_act = rev_softmax(np.linspace(0,1,breakpoints)[1:])
            model.set_weights([ws_cm, np.array([0]), y_act, np.array([0]), np.array([1])]) # Initialise weights
        else:
            y_act = rev_sigmoid(np.linspace(0,1,breakpoints))
            model.set_weights([ws_cm, np.array([0]), y_act]) # Initialise weights
            
        model.compile(loss=self.loss, optimizer=self.opt)

        if verbose:
            model.summary()

        return model

    def fit(self, probs, true, batch_size = 32, verbose = False):
        """
        Trains the model and finds optimal parameters

        Params:
            probs: the input for model
            true: one-hot-encoding of true labels.
            batch_size: Number of instances in one batch
            verbose (bool): whether to print out anything or not

        Returns:
            hist: Keras history of learning process
        """
        
        if self.logit_input:
            self.x_min = -50 #K.min(probs)
            self.x_max = 50 #K.max(probs)
            self.model = self.create_model(self.k, verbose)

        elif self.model is None:
            print("Warning no model created: Creating model with 1 breakpoint.")
            self.model = self.create_model(1, verbose)
        
        
        if self.equal_size == True:
            breakpoints = self.k + 2
            n_bins = breakpoints - 1

            bin_borders = []
            brkpts = np.linspace(0,1,breakpoints)
            
            if self.logit_input:
                #probs_brk = min_max_scale(probs, -50, 50) #np.min(probs), np.max(probs))
                eps = 1e-16
                x_min = np.log(eps/(1-eps))
                x_max = np.log((1-eps)/eps)
                probs_brk = np.clip(probs, x_min, x_max)
            elif self.logit_scale:
                probs_brk = clip_logit(probs)
            else:
                probs_brk = probs
            
            for brk in brkpts:
                bin_borders.append(np.quantile(probs_brk, brk))  # logitid -> -23, 12, 23
            
            
            print("Using equal size with Borders:", bin_borders)  # 0.4 0.5 1
            if self.logit_input or self.logit_scale:
                ws_cm = rev_softmax(sigmoid(np.copy(bin_borders[1:])))  # 0 - 1
            else:
                raise "Not implemented"
            
            if self.monotonic:
                y_act = rev_softmax(np.copy(bin_borders[1:])) # 0.4 0.5 1 -> 0.4 0.1 0.5 -> np.log(0.4); np.log(0.1), np.log(0.5)
                self.model.set_weights([ws_cm, np.array([0]), y_act, np.array([0]), np.array([1])]) # Initialise weights
            else:
                y_act = np.array(bin_borders)
                self.model.set_weights([ws_cm, np.array([0]), y_act]) # Initialise weights
                

        early_stop = EarlyStopping(monitor='loss', min_delta=0, patience=self.patience, verbose=verbose, mode='auto')
        cbs = [early_stop]

        hist = self.model.fit(probs, true, epochs=self.max_epochs, callbacks=cbs, batch_size=batch_size, verbose=verbose)

        return hist
    
    def predict(self, probs, clip_preds = True):
        """
        Scales probabilities based on the model and returns calibrated probabilities
        
        Params:
            probs: probability values of data for each class (shape [samples, classes])
            clip_preds (bool): If the predictions should be clipped to be in region between 0 to 1.
            
        Returns:
            calibrated probabilities (nd.array with shape [samples, classes])
        """
        
        pred = self.model.predict(probs)
        
        return pred


def node_scores_xy_with_crossvalidation(method, p_hat, y, n_splits = 5, input = None, add_error = False, seed = 0, 
                                        max_nodes = 15, equal_size = False, monotonic = False, logit_scale = False,
                                        logit_input = False, logistic_out = False, use_ce_loss = False):
    
    node_scores = [0]*(max_nodes+1)
    node_ECEs_abs = [0]*(max_nodes+1)
    node_ECEs_square = [0]*(max_nodes+1)
    node_loss = [0]*(max_nodes+1)

    all_weights = []
    all_cv_scores = []
    
    if input is None:
        input = p_hat
    
    for n_nodes in range(0, max_nodes+1, 1):
    
        weights = []
        cv_scores = []
        cv_ECE_square = []
        cv_ECE_abs = []
        cv_loss = []
        
        print("Nodes:", n_nodes)
        start_cv = time.time()
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        
        ij = 0
        
        for train_index, test_index in kf.split(p_hat):
            try:           
                p_hat_train, p_hat_test = p_hat[train_index], p_hat[test_index]
                input_train, input_test = input[train_index], input[test_index]
                y_train, y_test = y[train_index], y[test_index]
                
                if logit_scale or logit_input or use_ce_loss:
                    model = method(k=n_nodes, max_epochs=1500, random_state = seed, equal_size = equal_size, monotonic = monotonic, 
                    logit_scale = logit_scale, logit_input = logit_input, logistic_out = logistic_out, use_ce_loss = use_ce_loss)
                else:
                    model = method(k=n_nodes, max_epochs=1500, random_state = seed, equal_size = equal_size, monotonic = monotonic)
                h = model.fit(input_train, y_train, verbose=False, batch_size=len(y_train)//4)
                print("Last epoch", len(h.history['loss']))
                c_hat_test = model.predict(input_test)
                c_hat_train = model.predict(input_train)
                
                weights.append(model.model.get_weights()) # Model weights
                cv_scores.append(np.mean((c_hat_test - y_test)**2)) # Squared error
                cv_ECE_square.append(np.mean((c_hat_test - p_hat_test)**2))
                cv_ECE_abs.append(np.mean(np.abs(c_hat_test - p_hat_test)))
                cv_loss.append(np.mean(np.square(c_hat_train - y_train)))  # Train loss

                
                # Garbage collection
                del model.model
                del model

                gc.collect()
                K.clear_session()
                tf.compat.v1.reset_default_graph()
                
                
                #print("Split:", ij)
                
                ij += 1
            except:
                print("error for " + str(n_nodes) + " with method " + method)
                node_scores[n_nodes] += 9999
            
        node_scores[n_nodes] += np.mean(cv_scores)
        node_ECEs_square[n_nodes] = np.mean(cv_ECE_square)
        node_ECEs_abs[n_nodes] = np.mean(cv_ECE_abs)
        all_cv_scores.append(cv_scores)
        all_weights.append(weights)
        node_loss[n_nodes] = cv_loss

        print("Time for %i-fold CV for %i nodes was %f" % (n_splits, n_nodes, (time.time()-start_cv)))
     
    if add_error:
        error = 1 / len(p_hat)**0.5
        errors = [0 for i in range(max_nodes+1)]
        for i in range(max_nodes+1):
            errors[i] = error * i**0.125 * node_scores[i]

        node_scores = node_scores + np.asarray(errors)
    
    
    return node_scores, all_weights, all_cv_scores, node_ECEs_square, node_ECEs_abs, node_loss
    
    


"""
Example:

One class example

pw_nn = Piecewise_NN(classes=1, units=3)  # Three break lines, and one class.
x = np.array([[0.5], [0.4], [0.2]])
y = np.array([[0], [1], [0]])
h = pw_nn.fit(x, y, verbose=True)
"""


"""
model = Piecewise_NN4(k=n_nodes, max_epochs=1500, random_state = seed, equal_size = True, use_ce_loss = False)
model = Piecewise_NN4(k=n_nodes, max_epochs=1500, random_state = seed, equal_size = True, use_ce_loss = True)

model = Piecewise_NN4(k=n_nodes, max_epochs=1500, random_state = seed, monotonic = True, equal_size = True, use_ce_loss = False)
model = Piecewise_NN4(k=n_nodes, max_epochs=1500, random_state = seed, monotonic = True, equal_size = True, use_ce_loss = True)

model = Piecewise_NN4(k=n_nodes, max_epochs=1500, random_state = seed, equal_size = True, monotonic = True, logit_scale = True, logistic_out = True, use_ce_loss = False)
model = Piecewise_NN4(k=n_nodes, max_epochs=1500, random_state = seed, equal_size = True, monotonic = True, logit_scale = True, logistic_out = True, use_ce_loss = True)


"""