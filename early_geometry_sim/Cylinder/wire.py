# Aim: To simulate the magnetic field due to a current carrying wire

import numpy as np
import matplotlib.pyplot as plt

# Constants
mu_0 = 4 * np.pi * 1e-7  # Permeability of free space

def biot_savart_cylinder(I, L, positions, N=100):
    """
    Calculate the magnetic field at specified positions around a cylinder
    using the Biot-Savart law.
    
    Parameters:
    I        --> Total current in the cylinder (Amps)
    L        --> Length of the cylinder (meters)
    positions --> Array of positions to calculate the field at (in meters)
    N        --> Number of segments to divide the cylinder for approximation
    
    Returns:
    Magnetic field vector B at each position.
    """
    B = np.zeros(positions.shape)  # Initialize magnetic field at each position

    # Define cylinder current elements
    z_cylinder = np.linspace(-L/2, L/2, N)  # Cylinder is along the z-axis
    dz = L / N  # Length of each small current element
    
    for z in z_cylinder:
        dl = np.array([0, 0, dz])  # Small current element in the z-direction
        for i, pos in enumerate(positions):
            r_vec = pos - np.array([0, 0, z])  # Vector from current element to position
            r_mag = np.linalg.norm(r_vec)  # Magnitude of r_vec
            
            if r_mag == 0:
                continue  # Avoid division by zero at the center of the cylinder
            
            # Biot-Savart law to calculate dB
            dB = (mu_0 * I / (4 * np.pi)) * np.cross(dl, r_vec) / (r_mag**3)
            B[i] += dB
    
    return B

# Parameters
I = 1.0  # Current (Amps)
L = 1.0  # Length of the cylinder (meters)

# Positions where we calculate the field (e.g., a grid of points around the cylinder)
X, Y, Z = np.meshgrid(np.linspace(-1, 1, 10), np.linspace(-1, 1, 10), [0])  # A plane at z=0
positions = np.array([X.flatten(), Y.flatten(), Z.flatten()]).T

# Compute the magnetic field
B = biot_savart_cylinder(I, L, positions)

# Plot the magnetic field in the x-y plane
plt.quiver(X, Y, B[:, 0], B[:, 1], color='b')  # Quiver plot for 2D vectors
plt.title("Magnetic Field around a wire in x-y Plane")
plt.xlabel('x (meters)')
plt.ylabel('y (meters)')
plt.show()
