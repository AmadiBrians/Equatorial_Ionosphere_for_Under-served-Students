import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import pi
from pathlib import Path

# Define reusable constants
d2r = np.deg2rad
r2d = np.rad2deg
fcalc = lambda x: np.sqrt(np.dot(x, x))

# Load IGRF coefficients once
BASE_DIR = Path(__file__).resolve().parent
IGRF13_FILE = BASE_DIR / "dependencies" / "igrf13coeffs.txt"
igrf13 = pd.read_csv(IGRF13_FILE, delim_whitespace=True, header=3)
gh2020 = np.append(0., igrf13['2020.0'])
NMAX = 13

def trace_field_line(theta_0, phi_0=0, step_size=10):
    """
    Trace a magnetic field line using IGRF coefficients.
    Returns arrays of (r, theta, phi) coordinates in radians.
    """
    import sha as sha  # Import your spherical harmonic module
    
    r0 = 6371.2  # Earth’s mean radius (km)
    thrd = d2r(theta_0)
    phrd = d2r(phi_0)
    track = [(r0, thrd, phrd)]
    
    bxyz = sha.shm_calculator(gh2020, NMAX, r0, theta_0, phi_0, 'Geocentric')
    eff = fcalc(bxyz)
    lamb = step_size / eff
    newrad = r0 + 0.001
    maxstep = 10000
    step = 0

    while step <= maxstep and newrad >= r0:
        rad, th, ph = track[step]
        lx, ly, lz = tuple(el * lamb for el in bxyz)
        newrad = rad + lz
        newth = th + lx / rad
        newph = ph - ly / (rad * np.sin(th))
        track += [(newrad, newth, newph)]
        bxyz = sha.shm_calculator(gh2020, NMAX, newrad, r2d(newth), r2d(newph), 'Geocentric')
        lamb = step_size / fcalc(bxyz)
        step += 1
    
    rads = np.array([r[0] for r in track])
    ths = np.array([t[1] for t in track])
    phs = np.array([p[2] for p in track])
    
    return rads, ths, phs

def plot_field_lines(colat_range=(60, 80, 2), lon=125, step_size=10):
    """
    Plot magnetic field lines for a range of colatitudes.
    """
    rads_a, ths_a, phs_a = [], [], []
    
    for tt in np.arange(*colat_range):
        rads, ths, phs = trace_field_line(tt, lon, step_size)
        rads_a.append(rads)
        ths_a.append(ths)
        phs_a.append(phs)
    
    plt.figure(figsize=(10, 5))
    for p in range(len(rads_a)):
        if p > 3:  # skip first few if too short
            plt.plot(r2d(ths_a[p]) - 90, rads_a[p] - 6400)
    
    plt.ylim(0, 500)
    plt.title('Magnetic Field Lines', fontsize=16)
    plt.ylabel('Apex Height (km)', fontsize=16)
    plt.axhline(y=351, color='blue', linestyle='-')
    plt.axvline(x=-7, color='red', linestyle='--')
    plt.xticks([])
    plt.figtext(0.15, 0.05, "N", ha='center', fontsize=14)
    plt.figtext(0.87, 0.05, "S", ha='center', fontsize=14)
    plt.figtext(0.5, 0.01, "Geomagnetic Latitude", ha='center', fontsize=14)
    plt.show()

