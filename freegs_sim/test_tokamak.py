import numpy as np
import matplotlib.pyplot as plt
from freegs.machine import TestTokamak
from freegs.jtor import ProfilesPprimeFfprime
from freegs.plotting import plotEquilibrium
from freegs.plotting import plotConstraints
from freegs.equilibrium import Equilibrium

# Create a tokamak machine
tokamak = TestTokamak()

# Step 2 create a pprime and ffprime profile
def pprime(psi):
    return -2 * (1-psi)

def ffprime(psi):
    return 1-psi

def fvac(psi):
    return 1

# Step 3 create plasma profile to pass along to the equilibrium solver
profile = ProfilesPprimeFfprime(pprime, ffprime, fvac)

# Step 4 create an equilibrium instance
equilibrium = Equilibrium(tokamak=tokamak)

# Step 5 solve for the equilibrium psi
equilibrium.solve(profiles=profile)

# Step 6 plot the equilibrium
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlabel('R (m)')
ax.set_ylabel('Z (m)')
ax.set_title('Plasma Shape in R-Z Plane')
plotEquilibrium(equilibrium, axis=ax)
plt.show()

