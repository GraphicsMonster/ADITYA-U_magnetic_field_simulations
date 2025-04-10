import magpylib as magpy
import numpy as np
from scipy.spatial.transform import Rotation as R

major_radius = 0.75
minor_radius = 0.25
plasma_current = 250e3

# Introducing plasma current distribution
grid_size = 10
phi_num = 30
R_plasma = np.linspace(0.6, 0.9, grid_size)
Z_plasma = np.linspace(-0.2, 0.2, grid_size)
phi_plasma = np.linspace(0, 2*np.pi, phi_num)
loop_diameter = 0.05
plasma_current_coils = []
# adding a loop distribution: J = 1 - (r/a)^2
for r in R_plasma:
    for z in Z_plasma:
        for phi in phi_plasma:
            position = (r*np.cos(phi), r*np.sin(phi), z)

            j_weight = max(0, 1 - (r/major_radius)**2)
            if j_weight > 0:
                loop = magpy.current.Circle(diameter=loop_diameter, current=1.0)
                loop.move(position)
                loop.rotate(R.from_rotvec([0, 0, 0]))
                plasma_current_coils.append((loop, j_weight))

total_weight = sum([weight for _, weight in plasma_current_coils])
if total_weight > 0:
    for loop, weight in plasma_current_coils:
        loop.current = plasma_current * weight / total_weight

else:
    raise ValueError("distribtion not valid")

plasma_current_coils = magpy.Collection(*[loop for loop, _ in plasma_current_coils])
plasma_current_coils.show()

def generate_points_in_chamber(major_radius, minor_radius, num_points):
    # Generate points inside the toroidal chamber
    phi = np.random.uniform(0, 2 * np.pi, num_points)  # Uniformly distributed along toroidal direction
    theta = np.random.uniform(0, 2 * np.pi, num_points)  # Uniformly distributed in poloidal direction
    r = np.random.uniform(0.1 * minor_radius, 0.9 * minor_radius, num_points)  # Avoid coil boundary

    # Compute Cartesian coordinates for points in vacuum
    x_obs = (major_radius*0.9 + r * np.cos(theta)) * np.cos(phi)
    y_obs = (major_radius*0.9 + r * np.cos(theta)) * np.sin(phi)
    z_obs = r * np.sin(theta)

    return x_obs, y_obs, z_obs

x_obs, y_obs, z_obs = generate_points_in_chamber(major_radius, minor_radius, 500)
obs_points = np.array([x_obs.flatten(), y_obs.flatten(), z_obs.flatten()]).T

field = plasma_current_coils.getB(obs_points)

# Visualize the field as a quiver plot
import matplotlib.pyplot as plt

field_norm = np.linalg.norm(field, axis=1)
field = field / field_norm[:, None]
Bx, By, Bz = field[:, 0], field[:, 1], field[:, 2]

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_box_aspect([1, 1, 0.4])
ax.quiver(x_obs, y_obs, z_obs, Bx, By, Bz, length=0.1, normalize=False)
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
plt.show()
# Compare this snippet from aditya-u/plasma_shaping/test.py: