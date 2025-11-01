"""
get_dependencies.py
--------------------

Utility for automatically downloading required data and code dependencies
from Zenodo into a local "dependencies/" folder.

Usage:
    from get_dependencies import ensure_dependencies
    ensure_dependencies()
"""

import os
import requests

def ensure_dependencies():
    """Download all required files from Zenodo if they don't exist locally."""
    os.makedirs("dependencies", exist_ok=True)

    # ✅ Direct file download links from Zenodo (replace if record is updated)
    zenodo_files = {
    "convert_waccmx_datesec.py": "https://zenodo.org/records/17466062/files/convert_waccmx_datesec.py?download=1",
    "sha.py": "https://zenodo.org/records/17466062/files/sha.py?download=1",
    "igrf13coeffs.txt": "https://zenodo.org/records/17466062/files/igrf13coeffs.txt?download=1",
    "mag2geo_all.csv": "https://zenodo.org/records/17466062/files/mag2geo_all.csv?download=1"
    }

    for name, url in zenodo_files.items():
        local_path = os.path.join("dependencies", name)
        if not os.path.exists(local_path):
            print(f"⬇️  Downloading {name} ...")
            r = requests.get(url)
            if r.status_code == 200:
                with open(local_path, "wb") as f:
                    f.write(r.content)
                print(f"✅ Saved {name}")
            else:
                raise RuntimeError(f"❌ Download failed ({r.status_code}) for {name}")
        else:
            print(f"✔ {name} already exists. Skipping download.")


if __name__ == "__main__":
    ensure_dependencies()
