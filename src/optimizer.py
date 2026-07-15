import numpy as np
from .core_math import original_fct, derived_fct
from .verification import fct_eval

def gradient_descent(f, f_prime, x_start: np.ndarray | float=5, alpha: float=0.5, eps: float=1e-5, max_steps: int=1e3) -> None | np.ndarray:
    """This is what lets us find the minimum value of a function\n
    f: Initial Function\n
    f_prime: Analytical derivate of first function\n
    x_start: Starting value of X or starting coordinates of a N-dimension function\n
    alpha: Step value (How much we move in the derivate direction)\n
    eps: Value that acts as a threshold to know if minimum value is obtained\n
    max_steps: The max number of steps to be run in case of a divergence or did not find minimum"""
    if not fct_eval(f, f_prime, x_start):
        return None
    
    steps = [x_start]
    count = 0

    x_old = x_start
    
    while count < max_steps:
        x_n = x_old - f_prime(x_old)*alpha
        steps.append(x_n)
        
        if np.allclose(x_old, x_n, atol=eps):
            break
        
        x_old = x_n
        count += 1
    
    formatted_steps = np.array(steps)
    return formatted_steps
    