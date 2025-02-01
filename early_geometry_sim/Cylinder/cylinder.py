import numpy as np

mu_0 = 4 * np.pi * 1e-7  # Permeability of free space

def biot_savart_cylinder(I, R, L, positions, N=100):
    '''
    I --> Total current in the cylinder
    R --> Radius of the cylinder
    L --> Length of the cylinder
    positions --> Positions to calculate the field at (shape: Nx3)
    N --> Number of segments for discretization
    '''

    B = np.zeros((len(positions), 3))  # To store magnetic field at each position
    z_cylinder = np.linspace(-L/2, L/2, N)  # Discretize along the length
    dz = L / N  # Length of each segment
    dr = R / N  # Radial step for discretization

    for z in z_cylinder:
        for r in np.linspace(0, R, N):  # Discretize radial component
            for theta in np.linspace(0, 2*np.pi, N):  # Discretize angular component
                dl = np.array([0, 0, dz])  # Current element in the z direction
                current_position = np.array([r * np.cos(theta), r * np.sin(theta), z])  # Current element position
                
                for i, pos in enumerate(positions):  # Loop over positions where we want B-field
                    r_vec = pos - current_position  # Vector from current element to position
                    r_mag = np.linalg.norm(r_vec)  # Distance magnitude
                    
                    if r_mag == 0:
                        continue  # Avoid singularity
                    
                    dB = (mu_0 * I / (4 * np.pi)) * np.cross(dl, r_vec) / (r_mag**3)
                    B[i] += dB

    return B

# Example usage
I = 1  # Current in Amperes
R = 0.5  # Radius of the cylinder in meters
L = 2  # Length of the cylinder in meters
positions = np.array([[0.1, 0.1, 0.1], [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]])  # Points to calculate B-field

B_field = biot_savart_cylinder(I, R, L, positions)
print(B_field)
