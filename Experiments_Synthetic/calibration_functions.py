import numpy as np

def poly(x):
    return np.polynomial.polynomial.polyval(x,  [0, 4.09, -17.62, 35.17, -32.2 , 11.56])

def square(x):
    return x**2

def sqrt(x):
    return x**0.5

def three_piece(x):
    
    return np.piecewise(x, [(x <=0.57) & (x>=0), (x <=0.61) & (x>0.57), (x <=1) & (x>0.61)], 
                        [lambda y: y**2, 
                         lambda y: (0.61**0.5-0.57**2)/0.04*y+0.57**2-0.57*(0.61**0.5-0.57**2)/0.04, 
                         lambda y: y**0.5])

def two_piece(x):
    x = np.array(x, dtype=float)
    
    bools = (x <= 0.4)
    res = np.zeros_like(x)
    res[bools] = 1.7*x[bools]
    res[~bools] = 0.32/0.6*x[~bools] + 1 - 0.32/0.6
    
    return res

def beta1(x):
 
    m = 0.4#0.56 #0.5
    a = 0.4#0.49 #0.3
    b = 0.45#0.49 #0.3
    c = b*np.log(1 - m) - a*np.log(m)
    
    return 1 / (1 + 1 / (np.exp(c) * (x**a / (1-x)**b) ) )

def beta2(x):
    
    m = 0.48    
    a = 2
    b = 2.2
    c = b*np.log(1 - m) - a*np.log(m)
    
    return 1 / (1 + 1 / (np.exp(c) * (x**a / (1-x)**b) ) )


# https://math.stackexchange.com/questions/1671132/equation-for-a-smooth-staircase-function/2970318
def stairs(x):
    h = 2/3 # step height
    w = 2/3 # step width
    
    def stairs_helper(x):
        x = (x+w/2*0) * 2*np.pi/w
        step = lambda x: x-np.sin(x)
        return step(step(x)) * h/(2*np.pi)
    
    return stairs_helper(x+w*0.5) - stairs_helper(w*0.5)



calibration_functions = [square, sqrt, beta1, beta2, stairs]