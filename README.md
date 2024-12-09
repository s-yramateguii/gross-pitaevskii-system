# Gross-Pitaevskii System

This project simulates the evolution of a 3D wave function under the influence of a potential, and visualizes the results using isosurface animations. The simulation solves the Gross-Pitaevskii system (nonlinear Schrödinger with potential) using the **Split Step Fourier Method**. The wave function is evolved over time in a 3D spatial domain, and the final results are visualized as 3D isosurface plots using Plotly.

## Overview

The Gross-Pitaevskii system is given by:

$$
i\frac{1}{2}\nabla^2 \psi -|\psi|^2\psi+[A_1\sin^2(x)+B1][A_2\sin^2(y)+B_2][A_3\sin^2(z)+B_3]\psi=0
$$

Where:
- $$\psi$$ is the wave function, which describes the quantum state of a particle.
- $$\nabla^2=\partial_x^2+\partial_y^2+\partial_z^2$$
- $$A_i = -1$$ define the modulation strength along the each axis
- $$B_j=-A_j$$ define the constant offset to the potential along each axis

## Key Features

- **Wave Function Simulation:** This project simulates the time evolution of a 3D wave function in a box, subject to a specific potential. The initial conditions can be either a cosine or sine function.
- **Split Step Fourier Method:** The simulation uses the Split Step Fourier method to solve the Gross-Pitaevskii system in the Fourier domain. The approach provides an efficient way to handle nonlinear and time-dependent problems.
- **Visualization:** After the wave function is evolved over time, the results are visualized as interactive 3D isosurface plots using Plotly.
- **Initial Conditions:** Two different initial conditions are tested:
  - **Cosine Initial Condition:** $$\psi(x, y, z, t=0) = \cos(x) \cos(y) \cos(z)$$
  - **Sine Initial Condition:** $$\psi(x, y, z, t=0) = \sin(x) \sin(y) \sin(z)$$
