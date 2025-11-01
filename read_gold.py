from pathlib import Path
import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import matplotlib
import cartopy.crs as ccrs
import datetime as dt

def get_gold_paths(base_dir="/media/amadi/amadi_gate1/GOLD_2021/SEL"):
    """
    Build and return a list of GOLD Level 1C NetCDF file paths for 2021.
    Handles month- and version-specific differences.
    """
    base_dir = Path(base_dir)

    config = {
        "NOVSEL": {"day": 329, "version": "v04"},
        "AUGSEL": {"day": 240, "version": "v05"},
        "MAYSEL": {"day": 128, "version": "v05"},
    }

    channels = ["CHA", "CHB"]

    sell4 = [
        base_dir / month / f"GOLD_L1C_{ch}_NI1_2021_{cfg['day']}_23_40_{cfg['version']}_r01_c01.nc"
        for month, cfg in config.items()
        for ch in channels
    ]

    return [str(f) for f in sell4]


def read_gold_files_v2(sell4, year='2021'):
    """
    Reads GOLD Level-1C .nc files and extracts OI 135.6 nm radiance data.

    Parameters
    ----------
    sell4 : list
        List of full file paths to GOLD .nc files.
    year : str, optional
        Year of observation (default: '2021').

    Returns
    -------
    cordas : list
        [latz, lonz, dataz] — latitude, longitude, and radiance data arrays.
    fylz : list
        List of corresponding file paths.
    alph : list
        Alphabetical labels ['e', 'f', 'g', 'h'] for referencing plots.
    """
    latz, lonz, dataz, fylz = [], [], [], []
    alph = ['e', 'f', 'g', 'h']

    folds = sell4

    for i in np.arange(len(folds)):
        g1A = Dataset(folds[i], 'r')

        # Extract variables
        w1A = g1A.variables['WAVELENGTH'][:]
        lat1A = g1A.variables['REFERENCE_POINT_LAT'][:]   # Latitude grid
        lon1A = g1A.variables['REFERENCE_POINT_LON'][:]   # Longitude grid
        r1A = g1A.variables['RADIANCE'][:]
        g1A.close()

        # Select OI 135.6 nm band
        O5s_ids1A = np.argwhere((134.3 <= w1A[40, 17, :]) & (w1A[40, 17, :] <= 137.7))
        O5s1A = np.array(np.nansum(r1A[:, :, O5s_ids1A], axis=2)) * 0.04  # integrate area under peak
        O5s1A = np.transpose(O5s1A[:, :, 0])

        # Crop borders for uniform grids
        lats1A = lat1A[9:-7, 9:-7]
        lons1A = lon1A[9:-7, 9:-7]
        data1A = O5s1A[9:-7, 9:-7].T

        # Append results
        latz.append(lats1A)
        lonz.append(lons1A)
        dataz.append(data1A)
        fylz.append(folds[i])

    cordas = [latz, lonz, dataz]
    return cordas, fylz, alph

    
def plot_edens(wacx, wac_time, i=1, lat_range=(63, 129), z_idx=(6, 50, 252), cmap='jet'):
    """
    Plot electron density (EDens) from WAC data for a given time index.

    Parameters
    ----------
    wacx : dict-like
        Data structure containing 'lat', 'Z3', 'EDens', and 'date' arrays.
    wac_time : list or array
        Time labels corresponding to EDens time indices.
    i : int, optional
        Time index to plot (default: 1).
    lat_range : tuple, optional
        Latitude slice (start, stop) for plotting.
    z_idx : tuple, optional
        Indices for selecting Z3 and EDens dimensions (default: (6, 50, 252)).
    cmap : str, optional
        Colormap for the pcolormesh (default: 'jet').
    """
    SMALL_SIZE = 15
    matplotlib.rc('font', size=SMALL_SIZE)

    lat_start, lat_stop = lat_range
    z1, z2, z3 = z_idx

    # Extract data slices
    lat = wacx['lat'][lat_start:lat_stop]
    height = wacx['Z3'][z1, :, z2, z3] / 1000
    edens = wacx['EDens'][12 * i, :, lat_start:lat_stop, z3]

    # Plot setup
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.set(xlim=(-30, 30), ylim=(200, 500))
    
    # Main pcolormesh
    im = ax.pcolormesh(lat, height, edens, cmap=cmap, vmin=0, vmax=1.5e6)
    
    # Contour overlay
    cax = ax.contour(lat, height, edens, 2, colors='black')
    ax.clabel(cax, inline=True, fontsize=SMALL_SIZE)

    # Title
    date_str = str(wacx['date'][1])
    ax.set_title(f"{date_str[:4]}:{date_str[4:6]}:{date_str[6:8]} {wac_time[12 * i][:5]}", fontsize=20)

    # Colorbar
    cb = fig.colorbar(im, ax=ax)
    cb.formatter.set_powerlimits((0, 0))
    cb.ax.yaxis.set_offset_position('right')
    cb.update_ticks()
    cb.ax.set_ylabel(r"$m^{-3}$", rotation=270, labelpad=30)

    # Labels
    ax.set_xlabel("Latitude (°)", fontsize=15)
    ax.set_ylabel("Apex Height (km)", fontsize=15)

    plt.tight_layout()
    plt.show()


# ====================================================
# 1️⃣ LOAD GOLD FILES
# ====================================================
def read_gold_files2(source, year):
    """
    Reads GOLD .nc files and returns (cordas, fylz)
    
    Parameters
    ----------
    source : str | list
        Either a directory path (string) OR a list of .nc file paths (e.g., sell4)
    year : int
        The data year (used for labeling)
    
    Returns
    -------
    cordas : list of [latz, lonz, dataz]
    fylz   : list of file paths
    """
    latz, lonz, dataz, fylz = [], [], [], []

    # --- If user passed a directory ---
    if isinstance(source, str):
        path = Path(source)
        files = sorted(path.rglob("*.nc"))
    # --- If user passed a pre-made list (like sell4) ---
    elif isinstance(source, list):
        files = [Path(f) for f in source]
    else:
        raise TypeError("source must be a folder path or list of file paths")

    # --- Loop through all .nc files ---
    for fpath in files:
        try:
            g = Dataset(fpath, "r")

            w = g.variables["WAVELENGTH"][:]
            lat = g.variables["REFERENCE_POINT_LAT"][:]
            lon = g.variables["REFERENCE_POINT_LON"][:]
            rad = g.variables["RADIANCE"][:]
            g.close()

            # Extract OI 135.6 nm emission band
            O5s_idx = np.argwhere((134.3 <= w[40, 17, :]) & (w[40, 17, :] <= 137.7))
            O5s = np.array(np.nansum(rad[:, :, O5s_idx], axis=2)) * 0.04
            O5s = np.transpose(O5s[:, :, 0])

            # Crop edges
            lat_crop = lat[9:-7, 9:-7]
            lon_crop = lon[9:-7, 9:-7]
            data_crop = O5s[9:-7, 9:-7].T

            latz.append(lat_crop)
            lonz.append(lon_crop)
            dataz.append(data_crop)
            fylz.append(str(fpath))

        except Exception as e:
            print(f"⚠️ Skipped {fpath.name}: {e}")

    return [latz, lonz, dataz], fylz


# ====================================================
# 2️⃣ PLOT GOLD MAPS
# ====================================================
def plot_gold_maps2(cordas, fylz, year, alph=None, dfind=None,
                   vmin=0, vmax=1.5e6, cmap="plasma", figsize=(15, 6)):
    """
    Plot GOLD OI 135.6 nm Radiance for 3 frames.
    """
    if alph is None:
        alph = ["a", "b", "c"]

    fig = plt.figure(figsize=figsize)

    for i in range(3):
        ax = fig.add_subplot(1, 3, i + 1,
                             projection=ccrs.Orthographic(central_longitude=-40))

        # Overlay two frames (i and i+3)
        for j in (i, i + 3):
            im = ax.pcolormesh(
                cordas[1][j],
                cordas[0][j],
                cordas[2][j],
                transform=ccrs.PlateCarree(),
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )

        # Coastlines and labels
        ax.coastlines()
        ax.set_global()
        ax.set_xlabel("Longitude (°)")
        ax.set_ylabel("Latitude (°)")

        # Extract time from filename
        fname = Path(fylz[i]).name
        try:
            day_num = fname.split("_")[5]
            hour = fname.split("_")[6]
            minute = fname.split("_")[7]
            local_time = f"{int(hour)-3}:{minute}"
            date_str = dt.datetime.strptime(f"{year}-{day_num}", "%Y-%j").strftime("%Y-%m-%d")
        except Exception:
            date_str, local_time = "Unknown", ""

        ax.set_title(f"({alph[i]}) {date_str} {local_time} LT", fontsize=12, weight="bold")

        # Optional: Magnetic equator
        if dfind is not None:
            ax.scatter(
                dfind["con_lon"], dfind["con_lat"],
                s=10, transform=ccrs.PlateCarree(),
                color="red", label="Mag. Equator"
            )

        # Optional markers (example)
        if i == 0:
            ax.scatter([-45.5, -57, -49.5], [-18, 18, 0],
                       s=15, transform=ccrs.PlateCarree(), color="white")

    # Colorbar
    cbar_ax = fig.add_axes([0.1, 0.35, 0.8, 0.015])
    fig.colorbar(im, cax=cbar_ax, orientation="horizontal",
                 label="OI 135.6 nm Radiance (R)")

    plt.subplots_adjust(wspace=0.025, hspace=0.12)
    plt.show()

def plot_edens2(wacx, wac_time=None, i=1, lat_range=(63, 129), cmap='jet'):
    """
    Plot electron density (EDens) from WAC data for a given time index.

    Works with numpy arrays or xarray-like data.

    Parameters
    ----------
    wacx : dict-like
        Contains 'lat', 'Z3', 'EDens', and 'date' arrays.
    wac_time : list or array, optional
        Time labels corresponding to EDens indices.
    i : int, optional
        Time index to plot (default: 1).
    lat_range : tuple of int
        Latitude index range to slice (default: (63, 129)).
    cmap : str
        Colormap for the pcolormesh.
    """
    SMALL_SIZE = 15
    matplotlib.rc('font', size=SMALL_SIZE)
    lat_start, lat_stop = lat_range

    # Extract arrays
    lat = np.array(wacx['lat'])[lat_start:lat_stop]
    Z3 = np.array(wacx['Z3'])
    EDens = np.array(wacx['EDens'])
    
    # Handle Z3 shape flexibly
    if Z3.ndim == 4:
        height = Z3[6, :, 50, 252] / 1000
    elif Z3.ndim == 3:
        height = Z3[6, :, 50] / 1000
    elif Z3.ndim == 2:
        height = Z3[6, :] / 1000
    else:
        raise ValueError(f"Unexpected Z3 shape: {Z3.shape}")
    
    # Handle EDens shape flexibly
    if EDens.ndim == 4:
        edens = EDens[12 * i, :, lat_start:lat_stop, 252]
    elif EDens.ndim == 3:
        edens = EDens[12 * i, :, lat_start:lat_stop]
    else:
        raise ValueError(f"Unexpected EDens shape: {EDens.shape}")

    # Plot setup
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.set(xlim=(-30, 30), ylim=(200, 500))

    im = ax.pcolormesh(lat, height, edens, cmap=cmap, vmin=0, vmax=1.5e6)
    cax = ax.contour(lat, height, edens, 2, colors='black')
    ax.clabel(cax, inline=True, fontsize=SMALL_SIZE)

    # Build title safely
    date_val = str(np.array(wacx['date'])[1])
    time_str = f" {wac_time[12 * i][:5]}" if wac_time is not None else ""
    ax.set_title(f"Figure 6.1: EIA CRESTS for {date_val[:4]}:{date_val[4:6]}:{date_val[6:8]}{time_str}", fontsize=20)

    # Colorbar
    cb = fig.colorbar(im, ax=ax)
    cb.formatter.set_powerlimits((0, 0))
    cb.ax.yaxis.set_offset_position('right')
    cb.update_ticks()
    cb.ax.set_ylabel(r"$m^{-3}$", rotation=270, labelpad=30)

    # Labels
    ax.set_xlabel("Latitude (°)", fontsize=15)
    ax.set_ylabel("Apex Height (km)", fontsize=15)

    plt.tight_layout()
    plt.show()

