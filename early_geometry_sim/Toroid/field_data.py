import numpy as np
import matplotlib.pyplot as plt

# Constants
mu0 = 4 * np.pi * 1e-7  # Permeability of free space
I = 1  # Current in the coil

# Torus parameters
R = 10  # Major radius
r = 3   # Minor radius
N_turns = 100     # Number of poloidal turns around the torus
points_per_turn = 200

# Generate points along the coil
phi = np.linspace(0, 2 * np.pi, N_turns * points_per_turn)
theta = N_turns * phi  # Poloidal angle based on toroidal angle

x_coil = (R + r * np.cos(theta)) * np.cos(phi)
y_coil = (R + r * np.cos(theta)) * np.sin(phi)
z_coil = r * np.sin(theta)

# Radial distance analysis
radial_distances = np.linspace(0, 3 * r, 50)  # Ranges from 0 to 3 times the minor radius
field_strengths = []

for d in radial_distances:
    # Observation point along radial distance
    obs_x, obs_y, obs_z = R + d, 0, 0  # Fixed at x-axis, moving radially outward
    r_obs = np.array([obs_x, obs_y, obs_z])
    B = np.zeros(3)
    
    # Calculate the magnetic field at the observation point
    for j in range(len(x_coil) - 1):
        r1 = np.array([x_coil[j], y_coil[j], z_coil[j]])
        r2 = np.array([x_coil[j + 1], y_coil[j + 1], z_coil[j + 1]])
        dl = r2 - r1
        r_dl = r_obs - 0.5 * (r1 + r2)
        r_mag3 = np.linalg.norm(r_dl) ** 3
        B += mu0 * I / (4 * np.pi) * np.cross(dl, r_dl) / r_mag3
    
    field_strengths.append(np.linalg.norm(B))

# Gradient of the field
field_gradients = np.gradient(field_strengths, radial_distances)

# Plot Field Strength vs Radial Distance
plt.figure(figsize=(12, 6))
plt.plot(radial_distances, field_strengths, label="Magnetic Field Strength")
plt.xlabel("Radial Distance(Minor)")
plt.ylabel("Field Strength (T)")
plt.title("Magnetic Field Strength vs Radial Distance")
plt.grid(True)
plt.legend()

# Plot Field Gradient vs Radial Distance
plt.figure(figsize=(12, 6))
plt.plot(radial_distances, field_gradients, color="orange", label="Field Gradient")
plt.xlabel("Radial Distance(Minor)")
plt.ylabel("Field Gradient (T/m)")
plt.title("Magnetic Field Gradient vs Radial Distance")
plt.grid(True)
plt.legend()

plt.show()
