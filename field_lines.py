import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import pi
from pathlib import Path
from matplotlib.patches import FancyArrowPatch

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
    import src.sha as sha  # Import your spherical harmonic module
    
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

def plot_field_lines_num(
    start_colat: float = 60,
    end_colat: float = 80,
    lon: float = 125,
    num_lines: int = 5,
    step_size: float = 10,
    show_guides: bool = True
):
    """
    Plot magnetic field lines for a given range of colatitudes.

    Parameters
    ----------
    start_colat : float
        Starting colatitude in degrees (e.g., 60).
    end_colat : float
        Ending colatitude in degrees (e.g., 80).
    lon : float
        Fixed longitude in degrees.
    num_lines : int
        Number of field lines to plot between start_colat and end_colat.
    step_size : float
        Integration step size for field line tracing (km).
    show_guides : bool
        Whether to show horizontal/vertical reference lines and labels.
    """

    # Import helper dependencies
    from field_lines import trace_field_line, r2d  # assumes you defined these earlier

    # Compute the colatitudes automatically
    colats = np.linspace(start_colat, end_colat, num_lines)
    rads_a, ths_a, phs_a = [], [], []

    for tt in colats:
        rads, ths, phs = trace_field_line(tt, lon, step_size)
        rads_a.append(rads)
        ths_a.append(ths)
        phs_a.append(phs)

    # --- Plot ---
    plt.figure(figsize=(10, 5))
    for p in range(len(rads_a)):
        plt.plot(r2d(ths_a[p]) - 90, rads_a[p] - 6400, label=f"{colats[p]:.1f}°")

    plt.ylim(0, 500)
    plt.ylabel('Apex Height (km)', fontsize=16)
    plt.title('Magnetic Field Lines', fontsize=16)
    plt.legend(title="Colatitude")
    
    # add curved double-headed arrow between lines
    arrow = FancyArrowPatch(
    (-14, 355),   # start point (x, y)
    (-11.8, 300),    # end point (x, y)
    arrowstyle="<->,head_length=6,head_width=3",
    connectionstyle="arc3,rad=0.3",  # curvature
    color="green", linewidth=2
    )
    plt.gca().add_patch(arrow)

    # Add the label at the middle of the arrow
    plt.text(-14.1, 310, "Inclination", 
	 ha='center', fontsize=12, color="green",
	 rotation=0)

    if show_guides:
        plt.axhline(y=351, color='blue', linestyle='-')
        plt.axvline(x=-7, color='red', linestyle='--')
        plt.xticks([])
        plt.figtext(0.15, 0.05, "N", ha='center', fontsize=14)
        plt.figtext(0.87, 0.05, "S", ha='center', fontsize=14)
        plt.figtext(0.5, 0.015, "Geomagnetic Latitude", ha='center', fontsize=14)

    plt.show()
    

def var_orth_to_magfldline(
    start_colat: float = 60,
    end_colat: float = 80,
    lon: float = 125,
    num_lines: int = 5,
    step_size: float = 10,
    show_guides: bool = True,
    plot_index: int = 4  # default: plot the 5th line (index = 4)
):
    """
    Plot magnetic field lines for a given range of colatitudes,
    highlighting the apex (maximum height) of the selected line.
    """

    from field_lines import trace_field_line, r2d  # assumes you have this module

    # Generate field lines
    colats = np.linspace(start_colat, end_colat, num_lines)
    rads_a, ths_a, phs_a = [], [], []

    for tt in colats:
        rads, ths, phs = trace_field_line(tt, lon, step_size)
        rads_a.append(rads)
        ths_a.append(ths)
        phs_a.append(phs)

    # Select the chosen field line
    idx = min(plot_index, len(rads_a) - 1)
    x = r2d(ths_a[idx]) - 90
    y = rads_a[idx] - 6400

    # Find the apex (maximum height) and its latitude
    apex_idx = np.argmax(y)
    apex_height = y[apex_idx]
    apex_lat = x[apex_idx]

    # ====== Plot ======
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, color="black", linewidth=2)
    plt.ylim(0, 500)
    plt.title('Figure 3.2: Variables Orthogonal to Magnetic Field Lines', fontsize=16)
    plt.ylabel('Apex Height (km)', fontsize=16)

    if show_guides:
        # Horizontal line at the apex
        plt.axhline(y=apex_height, color='blue', linestyle='-', linewidth=1.8)
        # Vertical line through the apex
        plt.axvline(x=apex_lat, color='red', linestyle='--', linewidth=1.5)

        # Annotate the apex
        plt.plot(apex_lat, apex_height, 'bo', markersize=10)
        plt.text(apex_lat + 2, apex_height + 15,
                 f"Apex ≈ {apex_height:.0f} km",
                 color='blue', fontsize=12, fontweight='bold')

        # Curved arrow (inclination)
        arrow = FancyArrowPatch(
            (apex_lat - 9, apex_height + 10),
            (apex_lat - 4, apex_height - 40),
            arrowstyle="<->,head_length=6,head_width=3",
            connectionstyle="arc3,rad=0.3",
            color="green", linewidth=2
        )
        plt.gca().add_patch(arrow)
        plt.text(apex_lat - 13, apex_height - 25, "Inclination", color="green", fontsize=13)

        # Add Pressure Gradient and Gravity arrows
        plt.annotate("Pressure \nGradient", xy=(apex_lat - 4, 100), xytext=(apex_lat - 4, 250),
                     arrowprops=dict(facecolor='black', width=2, headwidth=8),
                     fontsize=15, ha='center', va='center')
        plt.annotate("Gravity", xy=(apex_lat + 4, 100), xytext=(apex_lat + 4, 250),
                     arrowprops=dict(facecolor='black', width=2, headwidth=8),
                     fontsize=15, ha='center', va='center')

        # Eastward electric field mark
        plt.plot(apex_lat, apex_height, marker='o', markersize=12, markerfacecolor='blue', markeredgecolor='black')
        plt.plot(apex_lat, apex_height, marker='x', markersize=8, color='white', markeredgewidth=2)
        plt.text(apex_lat, apex_height + 20, "E", ha='center', fontsize=15, color="blue", fontweight='bold')

        # Axis labels
        plt.xticks([])
        plt.figtext(0.15, 0.05, "N", ha='center', fontsize=14)
        plt.figtext(0.87, 0.05, "S", ha='center', fontsize=14)
        plt.figtext(0.5, 0.015, "Geomagnetic Latitude", ha='center', fontsize=14)

    plt.show()

