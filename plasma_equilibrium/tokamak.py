import numpy as np
import matplotlib.pyplot as plt
from freegs.machine import Machine
from freegs.equilibrium import Equilibrium
from freegs.jtor import ProfilesPprimeFfprime
from freegs.plotting import plotEquilibrium
from freegs.machine import Coil

# Defining the coils
coils = [("first", Coil(1, -1)), ("second", Coil(2, 0)), ("third", Coil(1, 1))]

# Machine setup
tokamak = Machine(coils=coils)

# Define the profiles
def pressure(psi, psi_edge=5, max_pressure=1.5, n=2):
    return max_pressure * (1 - psi/psi_edge)**n * (psi<psi_edge)

def pprime_profile(psi, psi_edge=5, max_pressure=1.5, n=2):
    return -n * max_pressure / psi_edge * (1 - psi/psi_edge)**(n-1) * (psi<psi_edge)

def fvac(psi):
    return 1

def ffprime_profile(psi, psi_edge=5, max_pressure=1.5, n=2):
    return np.full_like(psi, fvac(psi))

# Finally the profile creation
profiles = ProfilesPprimeFfprime(pprime_profile, ffprime_profile, fvac)

# Solve the equilibrium
eq = Equilibrium(tokamak=tokamak, Rmin=0.1, Rmax=3.9, Zmin=-3.0, Zmax=3.0, nx=129, ny=129)
eq.solve(profiles=profiles)

# Extract the flux function and domain
psi = eq.psi  # Assuming eq.psi stores the flux function
R = np.linspace(eq.Rmin, eq.Rmax, eq.nx)
Z = np.linspace(eq.Zmin, eq.Zmax, eq.ny)
R, Z = np.meshgrid(R, Z)

# Define the plasma edge as a specific psi level (e.g., psi_edge)
psi_edge = 5  # This should match the one in your pressure definition

# Plot the results
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlabel('R (m)')
ax.set_ylabel('Z (m)')
ax.set_title('Field in R-Z Plane')

# Plot the equilibrium contours
plotEquilibrium(eq, axis=ax)

# Add plasma boundary (LCFS)
boundary = ax.contour(R, Z, psi, levels=[psi_edge], colors='red', linewidths=2, linestyles='dashed')

# Add a legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='red', lw=2, linestyle='dashed', label='Plasma Boundary')
]
ax.legend(handles=legend_elements, loc='upper right')

plt.show()





