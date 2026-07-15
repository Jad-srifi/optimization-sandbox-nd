from src.analysis import plot_divergence_2d, plot_hyperparameter_trough_3d, plot_optimization_bowl_3d
import numpy as np

# This main file once ran will open up 3 plots successively after you have closed the precedent one in this order:
# 1. 1D Divergence Map (Alpha Testing)
# 2. 3D Hyperparameter Trough (Learning Rate vs Cost)
# 3. 3D Optimization Bowl (2-Variable Descent)

# 1.
plot_divergence_2d(x_range=(-30, 30, 400), x_start=27, alpha=0.1, max_steps=1e4, margins=0.05)

# 2.
plot_hyperparameter_trough_3d(x_range=(-30, 30, 400), alpha_range=(0.1, 1.2, 100), number_alphas=6, x_start=27, max_steps=1e4)

# 3.
plot_optimization_bowl_3d(x_range=(-30, 30, 400), y_range=(-30, 30, 400), co_start=np.array([27, 27]), alpha=0.1, max_steps=1e4)