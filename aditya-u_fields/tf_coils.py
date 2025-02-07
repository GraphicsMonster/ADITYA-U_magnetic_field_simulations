import magpylib as magpy
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

np.random.seed(42)

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

def form_coil_with_c_sections(num_sections, turns_per_section, position, orientation, width, height, thickness, current):
    '''
    Okay here I'm gonna define a function that forms a tf coil with sections like what's mentioned on the available papers online about how tf-coils are manufactured.
    There's gonna be a certain number of conducting filaments per section. And sections will be rectangular in shape but with decreasing width and height(depending on the total number of sections).
    '''
    coils = []
    for i in range(num_sections):
        num = i*0.08/num_sections
        coils.extend(form_rectangular_coil(position, orientation, width-(width*num), height-(height*num), thickness, turns_per_section, current))
    return magpy.Collection(*coils, override_parent=True)
        

# Parameters
num_coils = 20
major_radius = 0.75
minor_radius = 0.25
coil_current = 12500 # 4.166kA per loop --> 50kA for 12 loops
coil_size = [1.03, 1.26] # --> [width, height]
turns_per_section = 6
coil_thickness = 0.083
num_sections = 2

tf_coils = []
for i in range(num_coils):
    angle = 2 * np.pi * i / num_coils  # Angular position of the filament
    x = major_radius * np.cos(angle)
    y = major_radius * np.sin(angle)
    z = 0  # All coils lie in the same plane initially(for simplicity)
    position = (x, y, z)
    unit_vector = position / np.linalg.norm(position)
    rotation_vector = np.pi/2 * unit_vector
    rotation = R.from_rotvec(rotation_vector)
    tf_coils.append(form_coil_with_c_sections(num_sections, turns_per_section, position, rotation, coil_size[0], coil_size[1], coil_thickness, coil_current))

toroidal_field_coils = magpy.Collection(*tf_coils)

# All the loops visualized
magpy.show(toroidal_field_coils, animation=True)

# Observation points inside the torus
num_points = 500  # Number of observation points
phi = np.random.uniform(0, 2 * np.pi, num_points)  # Uniformly distributed along toroidal direction
theta = np.random.uniform(0, 2 * np.pi, num_points)  # Uniformly distributed in poloidal direction
r = np.random.uniform(minor_radius*0.1, minor_radius*0.9, num_points)  # Avoid coil boundary

# Compute Cartesian coordinates for points in vacuum
x_obs = (major_radius*0.9 + r * np.cos(theta)) * np.cos(phi)
y_obs = (major_radius*0.9 + r * np.cos(theta)) * np.sin(phi)
z_obs = r * np.sin(theta)


# Define a grid of points to sample the magnetic field
obs_points = np.array([x_obs.flatten(), y_obs.flatten(), z_obs.flatten()]).T

# Compute the magnetic field at all grid points
toroidal_field = toroidal_field_coils.getB(obs_points)

def distance_from_minor_axis(points, major_radius):
    radial_dist = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
    Xc = points[:, 0] * major_radius/radial_dist
    Yc = points[:, 1] * major_radius/radial_dist
    Zc = np.zeros(len(points))

    distances = np.sqrt((points[:, 0] - Xc)**2 + (points[:, 1] - Yc)**2 + (points[:, 2] - Zc)**2)
    return distances

def distance_from_machine_axis(points):
    return np.sqrt(points[:, 0]**2 + points[:, 1]**2)

r_obs = distance_from_machine_axis(obs_points) # Distance from machine's axis

# Normalize field for quiver plot
fields_norm = toroidal_field / np.linalg.norm(toroidal_field, axis=1, keepdims=True)

# Reshape for visualization
Bx, By, Bz = fields_norm[:, 0], fields_norm[:, 1], fields_norm[:, 2]
x, y , z = np.meshgrid(x_obs, y_obs, z_obs)

# Radial variation
sorted_indices = np.argsort(r_obs)
r_sorted = r_obs[sorted_indices]
Br = np.linalg.norm(toroidal_field, axis=1)
Br_sorted = Br[sorted_indices]
# Plot as a line
plt.figure(figsize=(8, 6))
plt.scatter(r_sorted, Br_sorted, label="Magnetic Field Strength")
plt.xlabel("Distance from the hollow center of the machine (m)")
plt.ylabel("Magnetic Field Strength (T)")
plt.title("Radial Variation of Magnetic Field in Toroidal Configuration")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(num=2)
ax = plt.subplot(111 , projection='3d')
ax.set_box_aspect([1,1,0.4])
ax.quiver(x_obs, y_obs, z_obs, Bx, By, Bz, length=0.08, normalize=True)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.tight_layout()
plt.show()