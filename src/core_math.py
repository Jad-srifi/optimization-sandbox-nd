import numpy as np

def original_fct(x: float | np.ndarray) -> float | np.ndarray:
    # Origianl input function of the gradient descent
    return x**2 + 7*x - 9

def derived_fct(x: float | np.ndarray) -> float | np.ndarray:
    # Analytical derived function of the gradient descent 
    return 2*x + 7

def fct_2d(coo: np.ndarray) -> np.ndarray:
    # Original 2d function
    x, y = coo[0], coo[1]
    return x**2 + 2*y**2 - 7*x + 5*y - 4

def derived_fct_2d(coo: np.ndarray) -> np.ndarray:
    # Analytical derived 2d function
    x, y = coo[0], coo[1]
    return np.array([2*x - 7, 4*y + 5])