import magpylib as magpy
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# Parameters
num_coils = 20  # Number of toroidal field coils
major_radius = 0.75  # Major radius of the torus (meters)
minor_radius = 0.25  # Minor radius of the torus (meters)
coil_current = 10_000  # Current through each coil (Amperes)
coil_size = [0.1, 0.1]  # Rectangular cross-section size [width, height] (meters)
coil_turns = 1  # Number of turns per coil

def distance_from_minor_axis(points, R):
    
    # Extract x, y, z components
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    # Calculate the projection onto the major radius
    major_distance = np.sqrt(x**2 + y**2)

    # Avoid division by zero for points at the origin
    major_distance[major_distance == 0] = np.nan  # Handle separately if needed

    # Center of the minor circle for each point
    X_c = R * x / major_distance
    Y_c = R * y / major_distance
    Z_c = 0

    # Calculate distances to the minor circle center
    distances = np.sqrt((x - X_c)**2 + (y - Y_c)**2 + (z - Z_c)**2)

    return distances

# Function to create a single TF coil at a given position
def create_tf_coil(position, rotation):
    # Define a rectangular current loop
    loop = magpy.current.Loop(current=coil_current, diameter=0.25)
    # Rotate and position the coil
    loop.rotate(rotation = rotation, anchor=0)
    loop.move(position)
    return loop

# Arrange TF coils symmetrically around the torus
tf_coils = []
for i in range(num_coils):
    angle = 2 * np.pi * i / num_coils  # Angular position of the coil
    x = major_radius * np.cos(angle)
    y = major_radius * np.sin(angle)
    z = 0  # All coils lie in the same plane for simplicity
    position = (x, y, z)
    unit_vector = position / np.linalg.norm(position)
    rotation_vector = 90 * np.pi / 180 * unit_vector
    rotation = R.from_rotvec(rotation_vector)
    tf_coils.append(create_tf_coil(position, rotation))

# Combine all coils into a single collection
toroidal_field = magpy.Collection(*tf_coils)

# Observation points inside the torus
num_points = 500  # Number of observation points
phi = np.random.uniform(0, 2 * np.pi, num_points)  # Uniformly distributed along toroidal direction
theta = np.random.uniform(0, 2 * np.pi, num_points)  # Uniformly distributed in poloidal direction
r = np.random.uniform(0.1 * minor_radius, 0.9 * minor_radius, num_points)  # Avoid coil boundary

# Compute Cartesian coordinates for points in vacuum
x_obs = (major_radius*0.9 + r * np.cos(theta)) * np.cos(phi)
y_obs = (major_radius*0.9 + r * np.cos(theta)) * np.sin(phi)
z_obs = r * np.sin(theta)


# Define a grid of points to sample the magnetic field
obs_points = np.array([x_obs.flatten(), y_obs.flatten(), z_obs.flatten()]).T

# Compute the magnetic field at all grid points
fields = toroidal_field.getB(obs_points)

r_obs = distance_from_minor_axis(obs_points, major_radius)

# Normalize field for quiver plot
fields_norm = fields / np.linalg.norm(fields, axis=1, keepdims=True)

# Reshape for visualization
Bx, By, Bz = fields_norm[:, 0], fields_norm[:, 1], fields_norm[:, 2]
x, y , z = np.meshgrid(x_obs, y_obs, z_obs)

# Visualizing the coils
magpy.show(toroidal_field, animation=True)

# Radial variation
sorted_indices = np.argsort(r_obs)
r_sorted = r_obs[sorted_indices]
Br = np.linalg.norm(fields, axis=1)
Br_sorted = Br[sorted_indices]
# Plot as a line
plt.figure(figsize=(8, 6))
plt.plot(r_sorted, Br_sorted, label="Magnetic Field Strength") # Changed to plt.plot
plt.xlabel("Radial Distance from Centerline (m)")
plt.ylabel("Magnetic Field Strength (T)")
plt.title("Radial Variation of Magnetic Field in Toroidal Configuration")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


plt.figure(num=2)
ax = plt.subplot(111 , projection='3d')
ax.quiver(x_obs, y_obs, z_obs, Bx, By, Bz, length=0.02, normalize=True)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.title("Toroidal magnetic field")
plt.tight_layout()
plt.show()