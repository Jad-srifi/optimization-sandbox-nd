# Optimization Sandbox: Calculus to Code 📐🚀

An N-dimensional, purely vectorized gradient descent engine and mathematical visualization sandbox built from first principles.

This repository strips away **black-box Machine Learning frameworks** (such as PyTorch or TensorFlow) to demonstrate the linear algebra, multivariable calculus, and optimization techniques that underpin modern quantitative finance, machine learning, and scientific computing.

![3D Optimization Bowl](https://github.com/user-attachments/assets/0b68bd5d-a6ba-4e8d-a144-728524d356f4)

---

# 🧠 Executive Summary

Optimization lies at the core of quantitative finance and artificial intelligence. Whether minimizing portfolio risk, calibrating financial models, or training neural networks, every optimization problem reduces to finding the minimum of an objective function.

This project implements a fully vectorized **N-dimensional Gradient Descent Optimizer** capable of operating on both scalar and multi-variable objective functions using the exact same optimization engine.

To guarantee mathematical correctness, the repository also includes an automated **Finite Difference Gradient Auditor** that numerically validates every analytical derivative before optimization begins.

The project was built entirely from first principles using only **NumPy** and **Matplotlib**, intentionally avoiding machine learning libraries to expose every mathematical component.

---

# ✨ Features

- Fully vectorized Gradient Descent implementation
- Native support for both 1D and N-dimensional optimization
- Zero Python loops inside the optimization engine
- Automatic finite-difference gradient verification
- Multiple mathematical visualizations
- 3D optimization surface rendering
- Learning-rate divergence analysis
- Modular architecture designed for future optimization algorithms

---

# 🏗️ Project Architecture

```text
optimization-sandbox-nd/
│
├── main.py
│
└── src/
    ├── core_math.py
    ├── optimizer.py
    ├── verification.py
    └── analysis.py
```

## `main.py`

Entry point of the project.

Provides a set of pre-made plots for test usage.

---

## `core_math.py`

Contains objective functions together with their analytical gradients.

Examples include:

- 1D quadratic functions
- Multi-variable convex bowls

---

## `optimizer.py`

Implements the polymorphic Gradient Descent engine.

The optimizer automatically handles vectors of arbitrary dimension using NumPy vectorization, allowing identical optimization logic for:

- One-dimensional functions
- Two-dimensional functions
- Higher-dimensional optimization problems

No explicit Python loops are used in the core numerical update routine.

---

## `verification.py`
Contains:
- Gradient definitions used by the optimizer

Implements the Finite Difference Gradient Auditor.

The auditor:

- Computes numerical gradients
- Automatically detects input dimensionality
- Compares analytical and numerical derivatives
- Reports approximation error

This serves as an automated mathematical verification layer before optimization.

---

## `analysis.py`

Contains visualization utilities built with Matplotlib.

Generates:

- 2D convergence plots
- Learning-rate divergence analysis
- 3D optimization surfaces

---

# 📊 Visual Diagnostics


## 1. Learning Rate Divergence Map

Illustrates how Gradient Descent behaves under different learning rates.

Safe learning rates converge smoothly toward the minimum, while excessive learning rates produce oscillation and divergence.

![1D Divergence Map](https://github.com/user-attachments/assets/1c8f6a89-9c46-4419-b191-8a09d6c0aefd)

---

## 2. 3D Hyperparameter Landscape

Extrudes the optimization surface to illustrate how different learning-rate values affect optimization trajectories.

The resulting visualization demonstrates how hyperparameter selection changes optimization behavior across the objective manifold.

![3D Hyperparameter Trough](https://github.com/user-attachments/assets/e8eb56ea-7b25-4c53-b992-1ef79176ecea)

---

## 3. 3D Optimization Bowl

Displays the optimization of a multi-variable convex function.

The trajectory demonstrates how gradients always point in the direction of steepest descent, often producing characteristic zig-zag paths before convergence.

![3D Optimization Bowl](https://github.com/user-attachments/assets/36e718ce-7710-4e1f-add3-0e83698346cb)

---

# ⚙️ Installation

## Requirements

- Python 3.10+
- NumPy
- Matplotlib

Install dependencies:

```bash
pip install numpy matplotlib
```

---

# 🚀 Usage

Clone the repository:

```bash
git clone https://github.com/Jad-srifi/optimization-sandbox-nd.git
```

Move into the project directory:

```bash
cd optimization-sandbox-nd
```

Launch the dashboard:

```bash
python main.py
```

---

# 🧮 Mathematical Foundations

The optimizer is based on the Gradient Descent update rule

\[
\theta_{t+1} = \theta_t - \alpha \nabla f(\theta_t)
\]

where

- **θ** represents the parameter vector
- **α** is the learning rate
- **∇f(θ)** is the analytical gradient of the objective function

Analytical gradients are verified numerically using finite differences:

\[
\frac{\partial f}{\partial x}
\approx
\frac{f(x+h)-f(x-h)}{2h}
\]

This numerical approximation serves as an independent validation of the implemented derivatives.

---

# 🛡️ Design Constraints

## Pure Vectorization

The optimization engine avoids explicit Python loops within numerical computation.

Instead, all updates rely on NumPy's optimized C-backed vectorized operations.

---

## Mathematical Verification

Every analytical gradient can be verified through numerical approximation before optimization begins.

This reduces implementation errors and improves confidence in the optimizer.

---

## Separation of Concerns

Each module has a single responsibility:

- Mathematical definitions
- Optimization
- Verification
- Visualization

This architecture simplifies future expansion.

---

# 📚 Learning Outcomes

This project demonstrates practical implementation of:

- Multivariable Calculus
- Partial Derivatives
- Gradient Descent
- Numerical Differentiation
- Linear Algebra
- NumPy Vectorization
- Scientific Visualization
- Modular Software Design
- Optimization Theory

---

# 🔮 Future Improvements

- Momentum Gradient Descent
- Nesterov Accelerated Gradient
- RMSProp
- Adam Optimizer
- Stochastic Gradient Descent
- Mini-Batch Gradient Descent
- Adaptive Learning Rate Scheduling
- Hessian Approximation
- Newton's Method
- Interactive GUI Dashboard
- Benchmarking against SciPy optimizers

---

# 📄 License

This project is released under the MIT License.

---

Developed as **Phase 1 of the Quantitative Architecture & Systems Baseline**, focusing on building optimization algorithms from first principles before progressing toward quantitative finance and machine learning systems.