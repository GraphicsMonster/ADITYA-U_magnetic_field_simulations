import magpylib as magpy
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

# Forming the toroidal field coils first
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

def form_circular_coil_at_origin(radius, z, num_turns, dz, current):
    '''
    This function forms a circular coil at the origin with the given radius and number of turns
    '''
    coils = []
    offset = dz/num_turns
    current = current/num_turns
    loop = magpy.current.Circle(diameter=2*radius, current=current)
    for i in range(int(num_turns/2)):
        offset = i * dz/num_turns
        loop_copy1 = loop.copy()
        loop_copy2 = loop.copy()
        loop_copy1.move([0, 0, z+offset])
        loop_copy2.move([0, 0, z-offset])
        coils.append(loop_copy1)
        coils.append(loop_copy2)
    return magpy.Collection(coils)

def form_coil_with_c_sections(num_sections, turns_per_section, position, orientation, width, height, thickness, current):
    '''
    Okay here I'm gonna define a function that forms a tf coil with c-sections like what's mentioned on the available papers online about how tf-coils are manufactured.
    There's gonna be a certain number of conducting filaments per section
    '''
    coils = []
    for i in range(num_sections):
        num = i*0.01/num_sections
        coils.extend(form_rectangular_coil(position, orientation, width-(width*num), height-(height*num), thickness, turns_per_section, current))
    return magpy.Collection(*coils, override_parent=True)

# Parameters
num_coils = 20
major_radius = 0.75
minor_radius = 0.25
coil_current = 4166 # 4.166kA per loop --> 50kA for 12 loops
coil_size = [1.03, 1.26] # --> [width, height]
turns_per_section = 6
coil_thickness = 0.083
num_sections = 2
num_points = 1000
num_plasma_current_coils = 4 # --> 1 on each side. Total plasma current = 100kA
coil_current_pf_coils = 1e4 # 10kA per loop

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


# Forming the plasma current coils
plasma_current_coils = []
for i in range(int(num_plasma_current_coils/2)):
    position1  = (0, 0, i*0.05)
    position2 = (0, 0, -i*0.05)
    orientation = 0
    plasma_current_coils.append(magpy.current.Circle(position=position1, diameter=1.5, current=coil_current_pf_coils))
    plasma_current_coils.append(magpy.current.Circle(position=position2, diameter=1.5, current=coil_current_pf_coils))
plasma_current = magpy.Collection(*plasma_current_coils)

# Let's also add the plasma shaping coils -- Data provided by Dr. Joydeep Ghosh
plasma_shaping_coils = []
bv_1_up = form_circular_coil_at_origin(radius=0.37981, z=1.05270, num_turns=60, dz=0.12003, current=-12500)
bv_1_down = form_circular_coil_at_origin(radius=0.382, z=-1.051, num_turns=60, dz=0.12003, current=-12500)
bv_2_up = form_circular_coil_at_origin(radius=1.640, z=1.1895, num_turns=22, dz=0.00386, current=-12500)
bv_2_down = form_circular_coil_at_origin(radius=1.640, z=-1.189, num_turns=22, dz=0.0038, current=-12500)
plasma_shaping_coils.extend([bv_1_up, bv_1_down, bv_2_up, bv_2_down])

plasma_shaping_coils = magpy.Collection(*plasma_shaping_coils)



# Final system
system = magpy.Collection(toroidal_field_coils, plasma_current)


'''
time to visualize the flux surfaces. Some math ahead.
'''
# Define the grid in cylindrical coordinates:
Rmin, Rmax = 0.5, 1.0   # adjust based on your tokamak dimensions
Zmin, Zmax = -0.25, 0.25
NR, NZ = 200, 200

R = np.linspace(Rmin, Rmax, NR, dtype=np.float32)
Z = np.linspace(Zmin, Zmax, NZ, dtype=np.float32)
dR = R[1]-R[0]
dZ = Z[1]-Z[0]
RR, ZZ = np.meshgrid(R, Z, indexing='ij')

# Prepare the observation points at phi=0:
# (x = R, y = 0, z = Z)
points = np.vstack(tup=[RR.flatten(), np.zeros(RR.size), ZZ.flatten()], dtype=np.float32).T

field = system.getB(points).astype(np.float32)
Bx = field[:, 0].reshape(RR.shape)
Bz = field[:, 2].reshape(RR.shape)

# At phi = 0: assign
BR = Bx   # radial component
BZ = Bz   # vertical (Z) component

# Initialize psi array
psi = np.zeros(RR.shape)

# Function for RK4 step along R (using BZ component)
def rk4_step_R(R, psi, BZ, dR):
    k1 = R * BZ
    k2 = (R + dR/2) * BZ
    k3 = (R + dR/2) * BZ
    k4 = (R + dR) * BZ
    return psi + (dR/6) * (k1 + 2*k2 + 2*k3 + k4)

# Function for RK4 step along Z (using BR component)
def rk4_step_Z(R, psi, BR, dZ):
    k1 = -R * BR
    k2 = -R * BR
    k3 = -R * BR
    k4 = -R * BR
    return psi + (dZ/6) * (k1 + 2*k2 + 2*k3 + k4)

# Integrate along R at Z=Z[0]
for i in range(1, NR):
    psi[i, 0] = rk4_step_R(R[i-1], psi[i-1, 0], BZ[i-1, 0], dR)

# For each R, integrate along Z
for i in range(NR):
    for j in range(1, NZ):
        psi[i, j] = rk4_step_Z(R[i], psi[i, j-1], BR[i, j], dZ)


plt.figure(figsize=(8,6))
contours = plt.contour(RR, ZZ, psi, levels=50, cmap='viridis')
plt.clabel(contours, inline=True, fontsize=8)
plt.xlabel('R (m)')
plt.ylabel('Z (m)')
plt.title('Extracted Flux Surfaces (Contours of ψ)')
plt.show()