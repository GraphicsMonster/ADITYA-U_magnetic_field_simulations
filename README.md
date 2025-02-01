
# ADITYA-U Magnetic Field Simulations

## Overview

This repository contains my work on simulating plasma behaviour and magnetic field in the ADITYA-U tokamak at IPR, Gandhinagar.

## Organization
The repository contains several folders as given below with brief descriptions about their function and code.

- **early_geometry:** I started by trying to simulate the magnetic fields due to 3 simple geometries. The code inclues implementation of the biot-savart law for magnetic field calculation at a pre-defined set of points in 3d space.
    - **Cylinder:** Field due to a straight wire
    - **Solenoidal_Field:** Field due to a solenoid
    - **Toroid:** Field due to a toroidal shaped winding of wires
- **aditya-u:** Contains code simulating the toroidal magnetic field, poloidal magnetic field and the net magnetic field in ADITYA-U with accurate physical parameters and coil positions.
- **freegs_sim:** Contains code using FreeGS library to solve the Grad-Shafranov equation for realistic coil positions and geometries.



