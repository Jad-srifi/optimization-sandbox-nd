import numpy as np
import matplotlib.pyplot as plt
from .core_math import original_fct, derived_fct, fct_2d, derived_fct_2d
from .optimizer import gradient_descent

def plot_divergence_2d(x_range: tuple=(-10, 10, 100), x_start: float=3, alpha: float=0.5, max_steps: int=1e4, margins: float=0.05):
    """Plot the 2D function with the history of the steps taken\n
    x_range: min_x_value, max_x_value, number_of_points_in_between\n
    x_start: starting position\n
    alpha: step value (How much we move in the derivate direction)\n
    max_step: number of max steps\n
    margins: margins to visualize plot"""
    x = np.linspace(x_range[0], x_range[1], x_range[2])
    y = original_fct(x)

    steps_x = gradient_descent(original_fct, derived_fct, x_start=x_start, alpha=alpha, max_steps=max_steps)
    steps_y = original_fct(steps_x)

    plt.plot(x, y, color='blue')
    plt.plot(steps_x, steps_y, color='green', marker='o', label='Alpha 0.1')

    
    min_x, max_x = min(x), max(x)
    min_y, max_y = min(y), max(y)

    margins = margins * max(max_y, max_x)

    plt.ylim(min_y - margins, max_y + margins)
    plt.xlim(min_x - margins, max_x + margins)

    plt.show()
    
def plot_hyperparameter_trough_3d(x_range: tuple=(-10, 10, 100), alpha_range: tuple=(0.1, 1.2, 100), number_alphas: int=5, x_start: float=10, max_steps: int=1e4):
    """Plot the 2D function with the history of the steps taken into a 3d graph visualizing the impact of alpha on the gradient descent\n
    x_range: min_x_value, max_x_value, number_of_points_in_between\n
    alpha_range: min_alpha_value, max_alpha_value, number_of_points_in_between\n
    number_alphas: number of alphas in the alpha range evenly spaced to visualize the impact
    x_start: starting position\n
    max_step: number of max steps\n"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    x_vals = np.linspace(x_range[0], x_range[1], x_range[2])
    z_vals = original_fct(x_vals)
    alpha_vals = np.linspace(alpha_range[0], alpha_range[1], alpha_range[2])
    
    x_matrix, alpha_matrix = np.meshgrid(x_vals, alpha_vals)
    
    z_matrix = original_fct(x_matrix)
    
    alphas_view = np.linspace(alpha_range[0], alpha_range[1], number_alphas)
    for test_alpha in alphas_view:
        steps_x = gradient_descent(original_fct, derived_fct, x_start, alpha=test_alpha, max_steps=max_steps)
        steps_y = np.full(steps_x.shape, test_alpha)
        steps_z = original_fct(steps_x)
        
        ax.plot(steps_x, steps_y, steps_z, color='red', marker='o')
        
    
    ax.plot_surface(x_matrix, alpha_matrix, z_matrix, alpha=0.3, color='blue')
    
    ax.set_xlabel('Position (x)')
    ax.set_ylabel('Learning rate (Alpha)')
    ax.set_zlabel('Position (Z)')
    
    min_x, max_x = min(x_vals), max(x_vals)
    min_z, max_z = min(z_vals), max(z_vals)
    min_alpha, max_alpha = min(alpha_vals), max(alpha_vals)
    
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_alpha, max_alpha)
    ax.set_zlim(min_z, max_z)
    
    plt.show()

def plot_optimization_bowl_3d(x_range: tuple=(-10, 10, 100), y_range: tuple=(-10, 10, 100), co_start: np.array=([10, 10]), alpha: float=0.5, max_steps: int=1e4):
    """Plot the 2D function with the history of the steps taken into a 3d graph visualizing the impact of alpha on the gradient descent\n
    x_range: min_x_value, max_x_value, number_of_points_in_between\n
    y_range: min_y_value, max_y_value, number_of_points_in_between\n
    co_start: starting position coordinates\n
    alpha: step value (How much we move in the derivate direction)\n
    max_step: number of max steps\n"""
    fig = plt.figure(figsize=[10, 8])
    ax = fig.add_subplot(111, projection='3d')
    
    x_vals = np.linspace(x_range[0], x_range[1], x_range[2])
    y_vals = np.linspace(y_range[0], y_range[1], y_range[2])
    z_vals = fct_2d(np.array([x_vals, y_vals]))
    
    x_matrix, y_matrix = np.meshgrid(x_vals, y_vals)
    z_matrix = fct_2d(np.array([x_matrix, y_matrix]))
    
    steps_xy = gradient_descent(fct_2d, derived_fct_2d, co_start, alpha=alpha, max_steps=max_steps)
    steps_x = steps_xy[:, 0]
    steps_y = steps_xy[:, 1]
    steps_z = fct_2d(np.array([steps_x, steps_y]))
    
    ax.plot(steps_x, steps_y, steps_z, alpha=0.5, color='blue', marker='o')
    ax.plot_surface(x_matrix, y_matrix, z_matrix, alpha=0.3, color='red')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    min_x, max_x = min(x_vals), max(x_vals)
    min_z, max_z = min(z_vals), max(z_vals)
    min_y, max_y = min(y_vals), max(y_vals)
    
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_zlim(min_z, max_z)

    plt.show()