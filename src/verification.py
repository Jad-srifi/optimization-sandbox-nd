import numpy as np
from .core_math import original_fct, derived_fct, fct_2d, derived_fct_2d

def numerical_gradient(f, x: float | np.ndarray, h: float =1e-5) -> float | np.ndarray:
    """Calculate the numerical derivate using limit laws for N dimensions\n
    f: function\n
    x: coordinates or array of coordinates\n
    h: value added or removed"""
    x = np.array(x, dtype=float)
    
    test_out = f(x)
    
    # If true implies that it is a 1-dimension function : [x1, x2] -> [y1, y2] 1d function ; [x1, x2] -> y1 2d function...
    if test_out.shape == x.shape:
        return (f(x+h) - f(x-h)) / (2*h)
    
    grad = np.zeros_like(x)
    
    # For each dimension of the function generate an array with h in the i-th element to calculate it's partial derivative
    for i in range(len(x)):
        h_vec = np.zeros_like(x)
        h_vec[i] = h
        
        dvec = (f(x+h_vec) - f(x-h_vec)) / (2*h)
        
        grad[i] = dvec
    return grad
    
def fct_eval(fct, analytical_der, x: float | np.ndarray, tolerance: float =1e-5) -> bool:
    """Checks the difference between analytical and numerical difference if it is below the tolerance this means that the analytical derivate is usable for this gradient descent\n
    For n dimensions: [x, y, ...] compares each component of the numerical and analytical derivate to the tolerance\n
    fct: function
    analytical_der: analytical derivate of the function
    x: coordinate | array of coordinates
    tolerance: tolerance value"""
    num_der = numerical_gradient(fct, x)
    analy_der = analytical_der(x)
    
    if np.allclose(num_der, analy_der, atol=tolerance):
        return True
    return False