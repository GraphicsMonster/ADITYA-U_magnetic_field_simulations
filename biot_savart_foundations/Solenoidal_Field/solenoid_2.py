import numpy as np
import matplotlib.pyplot as plt

# Constants
mu_0 = 4 * np.pi * 1e-7  # Permeability of free space (T·m/A)

def small_wire_field(current, dl, r):
    """Calculate the magnetic field at a point due to a small current element using the Biot-Savart law."""
    r_magnitude = np.linalg.norm(r)
    if r_magnitude == 0:
        return np.zeros(3)
    dB = (mu_0 * current / (4 * np.pi)) * (np.cross(dl, r)) / (r_magnitude ** 3)
    return dB

def magnetic_field_wire(current, wire_positions, observation_positions):
    """Calculate the magnetic field at each observation point due to a current-carrying wire."""
    B = np.zeros((len(observation_positions), 3))
    for i, pos in enumerate(observation_positions):
        for j in range(len(wire_positions) - 1):
            dl = wire_positions[j + 1] - wire_positions[j]
            r = pos - wire_positions[j]
            dB = small_wire_field(current, dl, r)
            B[i] += dB
    return B

def generate_helix_points(radius, pitch, turns, num_points_per_turn):
    """Generate points along a helical path as a NumPy array."""
    theta = np.linspace(0, 2 * np.pi * turns, num_points_per_turn * turns)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z = np.linspace(-pitch * turns / 2, pitch * turns / 2, num_points_per_turn * turns)
    points = np.column_stack((x, y, z))
    return points

def generate_symmetrical_points_inside_cylinder(radius, height, n_points_radius=5, n_points_height=10):
    """Generate symmetrical observation points within a cylinder."""
    z_positions = np.linspace(-height / 2, height / 2, n_points_height)
    radial_positions = np.linspace(0, radius, n_points_radius)
    theta_positions = np.linspace(0, 2 * np.pi, n_points_radius * 2)

    points = []
    for z in z_positions:
        for r in radial_positions:
            for theta in theta_positions:
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                points.append([x, y, z])
    return np.array(points)

# Parameters
current = 10  # Current through the solenoid (A)
n_turns = 50  # Number of turns in the solenoid
length = 1  # Length of the solenoid (m)
radius = 0.05  # Radius of the solenoid

# Generate points along the wire of the solenoid
wire_positions = generate_helix_points(radius=radius, pitch=length/n_turns, turns=n_turns, num_points_per_turn=50)

# Generate symmetrical observation points inside the cylinder
observation_positions = generate_symmetrical_points_inside_cylinder(radius * 0.8, length)

# Calculate the magnetic field at these points
B = magnetic_field_wire(current, wire_positions, observation_positions)

# Plot the solenoid and magnetic field vectors
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the solenoid helix at the center with a bright color
ax.plot(wire_positions[:, 0], wire_positions[:, 1], wire_positions[:, 2], color='red', linewidth=2, linestyle='dashed', label='Solenoid Coil')

# Scale factor for visualizing small field vectors
scale_factor = 50

# Plot the magnetic field vectors
ax.quiver(observation_positions[:, 0], observation_positions[:, 1], observation_positions[:, 2],
          B[:, 0] * scale_factor, B[:, 1] * scale_factor, B[:, 2] * scale_factor, color='blue', length=0.05, normalize=True)

# Labels and settings
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.show()

# Now we would also like to plot the field stregth relative to the distance from the center of the solenoid

# Calculate the magnitude of the magnetic field at each observation point
B_magnitude = np.linalg.norm(B, axis=1)

# Calculate radial distances from the center (z-axis) for each observation point
obs_points_at_first_turn = observation_positions[:len(wire_positions)]
radial_distances_first_turn = np.sqrt(obs_points_at_first_turn[:, 0]**2 + obs_points_at_first_turn[:, 1]**2) # radial distance of point in the first turn

# find the field magnitudes at the points in the first turn
B_magnitude_first_turn = B_magnitude[:len(radial_distances_first_turn)]

# Plot the radial variation of the magnetic field strength
plt.figure(figsize=(8, 6))
plt.plot(radial_distances_first_turn, B_magnitude_first_turn, label="Magnetic Field Strength")
plt.xlabel("Radial Distance from the center of the solenoid (m)")
plt.ylabel("Magnetic Field Strength (T)")
plt.title("Radial Variation of Magnetic Field in a Solenoid")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

