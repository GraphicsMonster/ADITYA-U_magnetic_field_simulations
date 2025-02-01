'''
This is the most recent version of the code that I wrote for this purpose. Do not entertain any other files in
this directory as they are not relevant to the task at hand.
'''
import numpy as np
import matplotlib.pyplot as plt

# Torus parameters
R = 0.75          # Major radius
r = 0.25         # Minor radius (coil thickness)
N_turns = 20     # Number of poloidal turns around the torus
points_per_turn = 400

# Generate angles
phi = np.linspace(0, 2 * np.pi, N_turns * points_per_turn)
theta = N_turns * phi  # Poloidal angle based on toroidal angle

# Coil points based on toroidal and poloidal angles
x_coil = (R + r * np.cos(theta)) * np.cos(phi)
y_coil = (R + r * np.cos(theta)) * np.sin(phi)
z_coil = r * np.sin(theta)

# Plotting
fig1 = plt.figure(0)
ax1 = fig1.add_subplot(111, projection='3d')
ax1.set_box_aspect([1, 1, 0.4])

# Draw the torus surface
phi_torus, theta_torus = np.meshgrid(np.linspace(0, 2 * np.pi, 50), np.linspace(0, 2 * np.pi, 50))
x_torus = (R + r * np.cos(theta_torus)) * np.cos(phi_torus)
y_torus = (R + r * np.cos(theta_torus)) * np.sin(phi_torus)
z_torus = r * np.sin(theta_torus)
ax1.plot_surface(x_torus, y_torus, z_torus, color='lightgray', alpha=0.3)

# Plot the coil path
ax1.plot(x_coil, y_coil, z_coil, color='blue', linewidth=1.5)

# Labels and show
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

# Now using the same wire points as current elements, we can calculate the magnetic field at various points inside the torus.
# First we must choose a certain number of observation points inside the torus.

# Observation points inside the torus
N_obs = 100
phi_obs = np.random.uniform(0, 2 * np.pi, N_obs)
theta_obs = np.random.uniform(0, 2 * np.pi, N_obs)
x_obs = (R + 0.9 * r * np.cos(theta_obs)) * np.cos(phi_obs)
y_obs = (R + 0.9 * r * np.cos(theta_obs)) * np.sin(phi_obs)
z_obs = 0.9 * r * np.sin(theta_obs)

# Calculate the magnetic field at each observation point due to the current elements
mu0 = 4 * np.pi * 1e-7  # Permeability of free space
I = 1  # Current in the coil 1 A
Bx = np.zeros(N_obs)
By = np.zeros(N_obs)
Bz = np.zeros(N_obs)

for i in range(N_obs):
    r_obs = np.array([x_obs[i], y_obs[i], z_obs[i]])
    B = np.zeros(3)
    for j in range(len(x_coil) - 1):
        r1 = np.array([x_coil[j], y_coil[j], z_coil[j]])
        r2 = np.array([x_coil[j + 1], y_coil[j + 1], z_coil[j + 1]])
        dl = r2 - r1
        r_mid = 0.5 * (r1 + r2)
        r_dl = r_obs - r_mid
        r_diff = r_obs - r1
        r_cross = np.cross(r_dl, dl)
        r_mag = np.linalg.norm(r_dl)
        r_mag3 = r_mag ** 3 # r_mag cubed
        B += mu0 * I / (4 * np.pi) * np.cross(dl, r_dl) / r_mag3

    Bx[i] = B[0]
    By[i] = B[1]
    Bz[i] = B[2]

# Plotting the magnetic field vectors
ax1.quiver(x_obs, y_obs, z_obs, Bx, By, Bz, length=0.1, normalize=True, color='red')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
plt.show()



# Now we shall obtain some data from the generated field
radial_distances = np.sqrt(x_obs**2 + y_obs**2)

B_mag = np.linalg.norm(np.array([Bx, By, Bz]), axis=0)

# Sorting the data for better visualization
sorted_indices = np.argsort(radial_distances)
radial_distances = radial_distances[sorted_indices]
B_mag = B_mag[sorted_indices]

plt.figure(1)
plt.plot(radial_distances, B_mag)
plt.xlabel('Radial Distance from the center of the torus (m)')
plt.ylabel('Magnetic Field Strength (T)')
plt.title('Radial Variation of Magnetic Field Strength in Toroidal Configuration')
plt.grid(True)
plt.show()