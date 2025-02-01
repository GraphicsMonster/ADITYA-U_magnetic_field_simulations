import numpy as np
import matplotlib.pyplot as plt

# Constants
mu_0 = 4 * np.pi * 1e-7  # Permeability of free space (T*m/A)
N = 100  # Number of turns
I = 1.0  # Current in Amperes
R = 1.0  # Major radius of the toroid (m)
r = 0.2  # Minor radius of the toroid (m)

# Create a grid of points
theta = np.linspace(0, 2 * np.pi, 100)
phi = np.linspace(0, 2 * np.pi, 100)
theta, phi = np.meshgrid(theta, phi)

# Calculate the magnetic field using Ampere's Law
B = (mu_0 * N * I) / (2 * np.pi * r)  # Magnetic field inside the toroid

# Magnetic field components
Bx = np.zeros_like(theta)  # No x-component
By = B * (R + r * np.cos(theta))  # y-component based on position

# Set up the quiver plot
plt.figure(figsize=(10, 8))
plt.quiver(R + r * np.cos(theta), r * np.sin(theta), Bx, By, color='r', scale=5)

# Plotting the toroid for context
toroid_x = (R + r * np.cos(theta)) * np.cos(phi)
toroid_y = (R + r * np.cos(theta)) * np.sin(phi)
plt.plot(toroid_x, toroid_y, color='b', alpha=0.5)

# Set labels and title
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('Magnetic Field in a Toroid')
plt.axis('equal')
plt.grid()
plt.show()
