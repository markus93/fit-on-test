import numpy as np

"""
def poly_p3(x):
    return 0.8*x**3 - 0.8*x**2 + 0.9*x

def logistic(x):
    return 1 / (1 + np.exp(-x*8 + 4))
    
def poly(x):
    return np.polynomial.polynomial.polyval(x,  [0, 0 , 29.3, -142.9, 192.3, 41.8, -232.8, 113.3])
    
    
x = [0.0,      0.2,   0.4,    0.6,     0.8,    0.85, 1.0]
y = [0.,       0.35,   0.35,    0.45,    0.55,    0.7,  0.99]
args = np.round(np.polyfit(x, y, deg=5), 3)
def poly(x):
    return np.polynomial.polynomial.polyval(x,  args[::-1])
"""

def poly(x):
    return np.polynomial.polynomial.polyval(x,  [0, 4.09, -17.62, 35.17, -32.2 , 11.56])

def square(x):
    return x**2

def sqrt(x):
    return x**0.5

def log_sqrt(x):
    return (np.log(x+1) + x**0.5) / 2.4

def poly_cos(x):
    return 0.05*np.cos(x*np.pi*5) + 2*x**2 - 1.1*x**3

def lin_sin(x):
    return x + 0.2 * np.sin(x*np.pi*6)

def two_piece(x):
    x = np.array(x, dtype=float)
    
    bools = (x <= 0.4)
    res = np.zeros_like(x)
    res[bools] = 1.3*x[bools] + 0.1
    res[~bools] = 0.45*x[~bools] + 0.44
    
    return res

def linear(x):
    return 0.25+0.62*x

def beta1(x):
 
    m = 0.56 #0.5
    a = 0.49 #0.3
    b = 0.49 #0.3
    c = b*np.log(1 - m) - a*np.log(m)
    
    return 1 / (1 + 1 / (np.exp(c) * (x**a / (1-x)**b) ) )

def beta2(x):
    
    m = 0.6    
    a = 3
    b = 3
    c = b*np.log(1 - m) - a*np.log(m)
    
    return 1 / (1 + 1 / (np.exp(c) * (x**a / (1-x)**b) ) )



calibration_functions = [poly, square, sqrt, log_sqrt, poly_cos, lin_sin, two_piece, linear, beta1, beta2]