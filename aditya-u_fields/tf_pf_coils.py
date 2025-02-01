import magpylib as magpy
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

# Forming the toroidal field coils first

def form_rectangular_coil(position, orientation, width, height, thickness, turns, current):
    
    loop = magpy.current.Polyline(vertices=[(-width/2, height/2, 0), (width/2, height/2, 0), (width/2, -height/2, 0), (-width/2, -height/2, 0), (-width/2, height/2, 0)], current=current)

    coils = []
    for i in range(turns):
        offset = thickness/turns
        loop_copy = loop.copy()
        loop_copy.move([0, 0, offset*i])
        angle = np.arctan(position[1]/position[0]) # Need to update the angle of the coil such that a 90 degree rotation makes it stand upright
        loop_copy.rotate(R.from_rotvec([0, 0, angle]))
        coils.append(loop_copy)

    coil = magpy.Collection(coils)
    coil.move(position)
    coil.rotate(orientation)

    return coil

def generate_points_minor_axis(major_radius, num_points):
    # Generate points on a cirlce passing through the center of all the tf coils inside the chamber
    theta = np.linspace(0, 2*np.pi, num_points)
    x = major_radius * np.cos(theta)
    y = major_radius * np.sin(theta)
    z = np.zeros(num_points)
    return x, y, z

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

# Parameters
num_coils = 20
major_radius = 0.75
minor_radius = 0.25
coil_current_tf_coils = 18750 # 18.75kA per loop --> 150kA for 8 loops
coil_current_pf_coils = 1e4 # 10kA per loop
coil_size = [0.5, 0.7] # --> [width, height]
turns_per_coil = 8
coil_thickness = 0.08

tf_coils = []
for i in range(num_coils):
    angle = 2 * np.pi * i / num_coils  # Angular position of the coil
    x = major_radius * np.cos(angle)
    y = major_radius * np.sin(angle)
    z = 0  # All coils lie in the same plane initially(for simplicity)
    position = (x, y, z)
    unit_vector = position / np.linalg.norm(position)
    rotation_vector = np.pi/2 * unit_vector
    rotation = R.from_rotvec(rotation_vector)
    tf_coils.append(form_rectangular_coil(position=position, orientation=rotation, width=coil_size[0], height=coil_size[1], thickness=coil_thickness, turns=turns_per_coil, current=coil_current_tf_coils))

toroidal_field_coils = magpy.Collection(*tf_coils)


# Forming the poloidal field coils and adding them to the system
plasma_current = magpy.current.Circle(position=[0, 0, 0], diameter=1.5, current=150e3) # Modeling the plasma current as a current loop carrying a 100kA current
pf_coils = [plasma_current]

poloidal_field_coils = magpy.Collection(*pf_coils)

# Final system
system = magpy.Collection(toroidal_field_coils, poloidal_field_coils)
magpy.show(system)

# Numpy array of observation points
x_obs, y_obs, z_obs = generate_points_in_chamber(major_radius, minor_radius, 500)
obs_points = np.array([x_obs.flatten(), y_obs.flatten(), z_obs.flatten()]).T

# Compute the magnetic field at all grid points
field = system.getB(obs_points)
field_norm = field / np.linalg.norm(field, axis=1, keepdims=True)
Bx, By, Bz = field_norm[:, 0], field_norm[:, 1], field_norm[:, 2]

# Plotting the magnetic field
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_box_aspect([1, 1, 0.4])
ax.quiver(x_obs, y_obs, z_obs, Bx, By, Bz, length=0.1, normalize=False)
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
plt.show()