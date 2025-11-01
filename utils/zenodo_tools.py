# zenodo_tools.py

import os
import requests

def get_from_zenodo(filename, url, local_dir="dependencies"):
    """Download a file from Zenodo and save it locally."""
    os.makedirs(local_dir, exist_ok=True)
    local_path = os.path.join(local_dir, filename)

    # Skip if already downloaded
    if os.path.exists(local_path):
        print(f"✔ {filename} already exists. Skipping download.")
        return local_path

    print(f"↓ Downloading {filename} from Zenodo...")
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        with open(local_path, "wb") as f:
            f.write(r.content)
        print(f"✅ Saved: {local_path}")
    else:
        print(f"❌ Failed to download {filename} (status {r.status_code})")
    return local_path
