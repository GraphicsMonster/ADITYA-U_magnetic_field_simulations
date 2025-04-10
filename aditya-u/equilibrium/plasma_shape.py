# This model uses the same old "single-current-loop-around-the-torus" method to model the plasma but this time with a higher number of coils

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
    if num_turns == 1:
        loop.move([0, 0, z])
        coils.append(loop)
        return magpy.Collection(coils)
    
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
coil_current = 300e3/12 # 4.166kA per loop --> 50kA for 12 loops
coil_size = [1.03, 1.26] # --> [width, height]
turns_per_section = 6
coil_thickness = 0.083
num_sections = 2
num_points = 1000
num_plasma_current_coils = 2
current_per_plasma_current_coil = 250e3/num_plasma_current_coils # 100kA for shaped plasma, 250kA for circular plasma
divertor_coil_current = 20e3 # 20kA per coil

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
plasma_thickness = 0.2
for i in range(int(num_plasma_current_coils/2)):
    # Distributing the plasma current coils uniformly in the toroidal direction
    z1 = i * plasma_thickness/num_plasma_current_coils
    z2 = -z1
    position1  = (0, 0, z1)
    position2 = (0, 0, z2)
    orientation = 0
    plasma_current_coils.append(magpy.current.Circle(position=position1, diameter=1.5, current=current_per_plasma_current_coil))
    plasma_current_coils.append(magpy.current.Circle(position=position2, diameter=1.5, current=current_per_plasma_current_coil))
plasma_current = magpy.Collection(*plasma_current_coils)

# Let's also add the plasma shaping coils -- Data provided by Dr. Joydeep Ghosh
plasma_shaping_coils = []
bv_1_up = form_circular_coil_at_origin(radius=0.37981, z=1.05270, num_turns=60, dz=0.12003, current=-12500)
bv_1_down = form_circular_coil_at_origin(radius=0.382, z=-1.051, num_turns=60, dz=0.12003, current=-12500)
bv_2_up = form_circular_coil_at_origin(radius=1.640, z=1.1895, num_turns=22, dz=0.00386, current=-12500)
bv_2_down = form_circular_coil_at_origin(radius=1.640, z=-1.189, num_turns=22, dz=0.0038, current=-12500)

# Divertor coils
main_div_inner_top = form_circular_coil_at_origin(radius=0.4625, z=0.297, num_turns=6, dz=0.065, current=divertor_coil_current)
main_div_inner_bottom = form_circular_coil_at_origin(radius=0.462, z=-0.297, num_turns=6, dz=0.065, current=divertor_coil_current)
main_div_outer_top = form_circular_coil_at_origin(radius=1.06288, z=0.3375, num_turns=1, dz=0.013, current=divertor_coil_current)
main_div_outer_bottom = form_circular_coil_at_origin(radius=1.06288, z=-0.3375, num_turns=1, dz=0.013, current=divertor_coil_current)
aux_div_inner_top = form_circular_coil_at_origin(radius=0.470, z=0.430, num_turns=2, dz=0.021, current=divertor_coil_current)
aux_div_inner_bottom = form_circular_coil_at_origin(radius=0.4704, z=-0.430, num_turns=2, dz=0.021, current=divertor_coil_current)

plasma_shaping_coils.extend([bv_1_up, bv_1_down, bv_2_up, bv_2_down, main_div_inner_top, main_div_inner_bottom, main_div_outer_top, main_div_outer_bottom, aux_div_inner_top, aux_div_inner_bottom])

plasma_shaping_coils = magpy.Collection(*plasma_shaping_coils)



# Final system
system = magpy.Collection(toroidal_field_coils, plasma_current)
system.show()


'''
time to visualize the flux surfaces. Some math ahead. Runge kutta method employed.
=======================================================
'''
# Define the grid in cylindrical coordinates:
Rmin, Rmax = 0.5, 1.0   # adjust based on your tokamak dimensions
Zmin, Zmax = -0.25, 0.25
NR, NZ = 100, 100

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


'''
Let's now add some post-processing to add the conditions for the limiter and obtain a LCFS
==========================================================================================
'''
mag_axis_R = 0.75
mag_axis_Z = 0.0
limiter_radius = 0.22

theta = np.linspace(0, 2*np.pi, 300)
limiter_points_r = limiter_radius * np.cos(theta)
limiter_points_z = limiter_radius * np.sin(theta)

original_points = (R, Z)
limiter_points = np.vstack(tup=[mag_axis_R + limiter_points_r, mag_axis_Z + limiter_points_z], dtype=np.float32).T

# Interpolating psi values at the limiter points
from scipy.interpolate import interpn
psi_limiter = interpn(original_points, psi, limiter_points, bounds_error=False, fill_value=None)

psi_edge = np.min(psi_limiter)
print(f"psi at the lcfs: {psi_edge}")

psi_adjusted = psi - psi_edge
print(f"min(psi_adjusted): {np.min(psi_adjusted)}")

# Let's see the elongation and triangularity of the obtained plasma
cs = plt.contour(RR, ZZ, psi_adjusted, levels=100, colors='r')
plt.close()

contours = cs.allsegs

vessel_Rmin, vessel_Rmax = Rmin, Rmax
vessel_Zmin, vessel_Zmax = Zmin, Zmax

selected_path = None
selected_area = 0

for level_id, segs in enumerate(cs.allsegs):
    for segment in segs:
        path = np.array(segment)
        Rs = path[:, 0]
        Zs = path[:, 1]

        if not (np.all((Rs >= vessel_Rmin) & (Rs <= vessel_Rmax)) and 
                np.all((Zs >= vessel_Zmin) & (Zs <= vessel_Zmax))):
            print(Rs, Zs)
            continue  # Skip this contour if any point is outside
        
        if not np.all((Rs - mag_axis_R)**2 + (Zs - mag_axis_Z)**2 <= limiter_radius**2):
            continue  # Skip this contour if any point is outside the limiter

        # Calculate the area of the path
        area = 0.5 * np.abs(np.dot(Rs, np.roll(Zs, 1)) - np.dot(Zs, np.roll(Rs, 1)))
        if area > selected_area:
            selected_area = area
            selected_path = path

# Calculate the elongation and triangularity
if selected_path is None:
    print("No valid contour found inside the limiter and vessel boundaries")
else:
    # --- 2. Compute Elongation and Triangularity ---
    R_contour = selected_path[:, 0]
    Z_contour = selected_path[:, 1]
    
    # Compute extremes from the contour:
    R_min_contour = np.min(R_contour)
    R_max_contour = np.max(R_contour)
    Z_min_contour = np.min(Z_contour)
    Z_max_contour = np.max(Z_contour)
    
    # Define horizontal and vertical half-widths (minor radii)
    a = (R_max_contour - R_min_contour) / 2.0  # horizontal half-width
    b = (Z_max_contour - Z_min_contour) / 2.0  # vertical half-width
    
    # Elongation is the ratio b/a:
    elongation = b / a

    # For triangularity, one common definition is using the top point.
    # Find the point with maximum Z (the "nose" at the top)
    max_Z_index = np.argmax(Z_contour)
    R_top = R_contour[max_Z_index]
    # Triangularity is then measured as the horizontal shift of the top relative to the magnetic axis, normalized by a.
    triangularity = (R_top - mag_axis_R) / a

    print("Elongation (kappa):", elongation)
    print("Triangularity (delta):", triangularity)


# plotting new contours
plt.figure(figsize=(8,8))
contours = plt.contour(RR, ZZ, psi_adjusted, levels=50, cmap='viridis')
plt.clabel(contours, inline=True, fontsize=8)
plt.plot(mag_axis_R, mag_axis_Z, 'ro', label='Magnetic Axis')
plt.plot(mag_axis_R + limiter_points_r, mag_axis_Z + limiter_points_z, 'k--', label='Limiter')
plt.plot(selected_path[:, 0], selected_path[:, 1], 'r-', label='LCFS')
plt.xlabel('R (m)')
plt.ylabel('Z (m)')
plt.title('Extracted Flux Surfaces (Contours of ψ)')
plt.legend()
plt.show()