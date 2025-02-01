from freegs.machine import MAST
from freegs.equilibrium import Equilibrium
import numpy as np
import matplotlib.pyplot as plt

# Create a tokamak machine
tokamak = MAST()

# Defining the domain
Rmin = 0.1
Rmax = 3.9
Zmin = -3.0
Zmax = 3.0
nx = 65
ny = 65

# Define equilibrium instance
eq = Equilibrium(tokamak=tokamak, Rmin=Rmin, Rmax=Rmax, Zmin=Zmin, Zmax=Zmax, nx=nx, ny=ny)

# Define required profiles
from freegs.jtor import ConstrainPaxisIp

profiles = ConstrainPaxisIp(eq,
                            3e3, # Plasma current in A
                            7e5, # Plasma pressure in Pa
                            0.4) # Plasma elongation

from freegs import control

x_points = [(0.7, -1,1), (0.7, 1.1)] # R, Z location of x-points

isoflux = [(0.7,-1.0, 1.4, 0.0),(0.7,1.0, 1.4, 0.0), (0.7,-1.0, 0.3, 0.0)]

constrain = control.constrain(gamma=1e-12, xpoints=x_points, isoflux=isoflux)
constrain(eq)

from freegs import picard
picard.solve(eq, profiles, constrain)

print("Plasma current: %e Amps" % (eq.plasmaCurrent()))

tokamak.printCurrents()

from freegs.plotting import plotEquilibrium
plotEquilibrium(eq)

# Define a grid for R and Z
R = np.linspace(Rmin, Rmax, nx)
Z = np.linspace(Zmin, Zmax, ny)
RR, ZZ = np.meshgrid(R, Z)

# Define fixed Z and R for 1D slices
Z_fixed = Z[ny // 2]  # Middle of the Z grid
R_fixed = R[nx // 2]  # Middle of the R grid

# Extract 1D data slices
Btor_vals_R = eq.Btor(R, Z_fixed)  # Toroidal field vs R at fixed Z
Btor_vals_Z = eq.Btor(R_fixed, Z)  # Toroidal field vs Z at fixed R

Bpol_vals_R = eq.Bpol(R, Z_fixed)  # Poloidal field vs R at fixed Z
Bpol_vals_Z = eq.Bpol(R_fixed, Z)  # Poloidal field vs Z at fixed R

# Plot Toroidal and Poloidal Fields
plt.figure(figsize=(12, 6))

# Toroidal field
plt.subplot(1, 2, 1)
plt.plot(R, Btor_vals_R, label=r'$B_{\mathrm{tor}}$ vs R', color='blue')
plt.plot(Z, Btor_vals_Z, label=r'$B_{\mathrm{tor}}$ vs Z', color='orange')
plt.xlabel('Position (m)')
plt.ylabel('Toroidal Field (T)')
plt.title('Toroidal Field Variation')
plt.legend()
plt.grid()

# Poloidal field
plt.subplot(1, 2, 2)
plt.plot(R, Bpol_vals_R, label=r'$B_{\mathrm{pol}}$ vs R', color='blue')
plt.plot(Z, Bpol_vals_Z, label=r'$B_{\mathrm{pol}}$ vs Z', color='orange')
plt.xlabel('Position (m)')
plt.ylabel('Poloidal Field (T)')
plt.title('Poloidal Field Variation')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()



# Extract 1D data slices
Br_R = eq.Br(R, Z_fixed)  # Toroidal field vs R at fixed Z
Br_Z = eq.Br(R_fixed, Z)  # Toroidal field vs Z at fixed R

Bz_R = eq.Bz(R, Z_fixed)  # Poloidal field vs R at fixed Z
Bz_Z = eq.Bpol(R_fixed, Z)  # Poloidal field vs Z at fixed R

# Plot Toroidal and Poloidal Fields
plt.figure(figsize=(12, 6))

# Toroidal field
plt.subplot(1, 2, 1)
plt.plot(R, Br_R, label=r'$B_{\mathrm{r}}$ vs R', color='blue')
plt.plot(Z, Br_Z, label=r'$B_{\mathrm{r}}$ vs Z', color='orange')
plt.xlabel('Position (m)')
plt.ylabel('Br (T)')
plt.title('Radial field component')
plt.legend()
plt.grid()

# Poloidal field
plt.subplot(1, 2, 2)
plt.plot(R, Bz_R, label=r'$B_{\mathrm{z}}$ vs R', color='blue')
plt.plot(Z, Bz_Z, label=r'$B_{\mathrm{z}}$ vs Z', color='orange')
plt.xlabel('Position (m)')
plt.ylabel('Bz (T)')
plt.title('Vertical field component')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()





