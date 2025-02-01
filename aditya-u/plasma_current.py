import numpy as np
import magpylib as magpy
from scipy.spatial.transform import Rotation as R
import plotly.graph_objects as go

# ========================================================================
# COIL GENERATION FUNCTIONS
# ========================================================================

def form_rectangular_coil(position, orientation, width, height, thickness, turns, current):
    """Create a rectangular coil with multiple turns."""
    loop = magpy.current.Polyline(
        vertices=[
            (-width/2, height/2, 0),
            (width/2, height/2, 0),
            (width/2, -height/2, 0),
            (-width/2, -height/2, 0),
            (-width/2, height/2, 0)
        ],
        current=current
    )

    coils = []
    for i in range(turns):
        loop_copy = loop.copy()
        loop_copy.position = (0, 0, thickness * i / turns)
        angle = np.arctan2(position[1], position[0])
        loop_copy.rotate(R.from_rotvec([0, 0, angle]))
        coils.append(loop_copy)

    coil = magpy.Collection(coils)
    coil.position = position
    coil.rotate(orientation)
    return coil

def generate_structured_grid(major_radius, minor_radius, resolution=20):
    """Create structured grid in cylindrical coordinates (R, φ, Z)."""
    R_vals = np.linspace(major_radius - 0.8*minor_radius, major_radius + 0.8*minor_radius, resolution)
    phi_vals = np.linspace(0, 2*np.pi, resolution)
    Z_vals = np.linspace(-minor_radius, minor_radius, resolution)

    R_grid, phi_grid, Z_grid = np.meshgrid(R_vals, phi_vals, Z_vals, indexing='ij')
    
    # Convert to Cartesian coordinates
    x = R_grid * np.cos(phi_grid)
    y = R_grid * np.sin(phi_grid)
    z = Z_grid
    
    return x, y, z, R_grid, phi_grid, Z_grid

# ========================================================================
# SYSTEM SETUP
# ========================================================================

# Parameters
major_radius = 0.75  # Major radius of tokamak [m]
minor_radius = 0.25  # Minor radius [m]
num_tf_coils = 18     # Number of toroidal field coils
coil_current_tf = 1e5  # Current in TF coils [A]
coil_current_pf = 5e4  # Current in PF coils [A]

# Generate structured grid
x, y, z, R_grid, phi_grid, Z_grid = generate_structured_grid(major_radius, minor_radius, resolution=15)
obs_points = np.array([x.flatten(), y.flatten(), z.flatten()]).T

# ========================================================================
# COIL CONFIGURATION
# ========================================================================

# Toroidal Field (TF) Coils
tf_coils = []
for i in range(num_tf_coils):
    angle = 2*np.pi*i/num_tf_coils
    position = (major_radius*np.cos(angle), major_radius*np.sin(angle), 0)
    orientation = R.from_rotvec([0, 0, angle])
    tf_coil = form_rectangular_coil(
        position=position,
        orientation=orientation,
        width=0.4,
        height=0.6,
        thickness=0.1,
        turns=8,
        current=coil_current_tf
    )
    tf_coils.append(tf_coil)

# Poloidal Field (PF) Coils
pf_coils = [
    magpy.current.Circle(position=(0,0,0.4), diameter=1.2, current=-coil_current_pf),
    magpy.current.Circle(position=(0,0,-0.4), diameter=1.2, current=-coil_current_pf),
    magpy.current.Circle(position=(0,0,0.7), diameter=1.5, current=coil_current_pf),
    magpy.current.Circle(position=(0,0,-0.7), diameter=1.5, current=coil_current_pf)
]

# Plasma Current (Simplified)
plasma_current = [
    magpy.current.Circle(position=(0,0,z), diameter=1.0, current=2e5)
    for z in np.linspace(-0.2, 0.2, 5)
]

# Combine all components
system = magpy.Collection(*tf_coils, *pf_coils, *plasma_current)

# ========================================================================
# FIELD COMPUTATION AND VISUALIZATION
# ========================================================================

# Compute magnetic field at grid points
B = system.getB(obs_points)
B_normalized = B / np.linalg.norm(B, axis=1, keepdims=True)

# Reshape to 3D grid structure
Bx = B[:,0].reshape(x.shape)
By = B[:,1].reshape(y.shape)
Bz = B[:,2].reshape(z.shape)

# Create streamtube plot
fig = go.Figure(data=go.Streamtube(
    x=x.flatten(),
    y=y.flatten(),
    z=z.flatten(),
    u=Bx.flatten(),
    v=By.flatten(),
    w=Bz.flatten(),
    sizeref=0.3,          # Adjust tube thickness
    maxdisplayed=3000,    # Reduce number of tubes for clarity
    colorscale='Viridis',
    showscale=True
))

# Add coils to plot
for coil in tf_coils + pf_coils:
    if isinstance(coil, magpy.Collection):
        # Handle Collections (e.g., multi-turn coils)
        for child in coil.children:
            if isinstance(child, (magpy.current.Polyline, magpy.current.Circle)):
                vertices = child.vertices
                fig.add_trace(go.Scatter3d(
                    x=vertices[:,0], 
                    y=vertices[:,1], 
                    z=vertices[:,2],
                    mode='lines',
                    line=dict(color='red', width=4),
                    name='TF/PF Coils'
                ))
    elif isinstance(coil, (magpy.current.Polyline, magpy.current.Circle)):
        # Handle standalone coils
        vertices = coil.vertices
        fig.add_trace(go.Scatter3d(
            x=vertices[:,0], 
            y=vertices[:,1], 
            z=vertices[:,2],
            mode='lines',
            line=dict(color='red', width=4),
            name='PF Coils'
        ))

fig.update_layout(
    scene=dict(
        aspectmode='data',
        camera=dict(eye=dict(x=1.5, y=1.5, z=0.8)),
        xaxis_title='X [m]',
        yaxis_title='Y [m]',
        zaxis_title='Z [m]'
    ),
    margin=dict(l=0, r=0, b=0, t=0)
)

fig.show()