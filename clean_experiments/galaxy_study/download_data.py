"""
Download Galaxy Zoo 2 dataset (from ppi_py package).

Downloads the .npz via ppi_py's load_dataset, then extracts and saves
Y (binary spiral labels) and Yhat (model predictions) as .npy files.

Dataset: 16,743 galaxies from Galaxy Zoo 2.
Estimand: fraction of spiral galaxies (θ* ≈ 0.2593).
"""
import numpy as np
import os
import subprocess

dataset_folder = "data"
os.makedirs(dataset_folder, exist_ok=True)

npz_path = os.path.join(dataset_folder, "galaxies.npz")

# Download from ppi_py's Google Drive
if not os.path.exists(npz_path):
    print("Downloading Galaxy Zoo 2 dataset via ppi_py...")
    try:
        from ppi_py.datasets import load_dataset
        data = load_dataset(dataset_folder, "galaxies")
    except ImportError:
        print("ppi_py not installed. Installing...")
        subprocess.check_call(["pip", "install", "ppi-python"])
        from ppi_py.datasets import load_dataset
        data = load_dataset(dataset_folder, "galaxies")
else:
    print(f"Dataset already exists at {npz_path}")
    data = np.load(npz_path)

print(f"Keys in npz: {list(data.keys())}")

for key in data.keys():
    arr = data[key]
    print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}, "
          f"min={arr.min():.4f}, max={arr.max():.4f}, mean={arr.mean():.4f}")

# Extract arrays
Y_total = data["Y"].flatten().astype(np.float32)
Yhat_total = data["Yhat"].flatten().astype(np.float32)

# Basic checks
assert np.isin(Y_total, [0, 1]).all(), "Y must be binary"
assert (Yhat_total >= 0).all() and (Yhat_total <= 1).all(), "Yhat must be in [0, 1]"

# Save as individual .npy files
np.save(os.path.join(dataset_folder, "Y.npy"), Y_total)
np.save(os.path.join(dataset_folder, "Yhat.npy"), Yhat_total)

# Print statistics
print(f"\nY shape:    {Y_total.shape}, mean: {Y_total.mean():.4f}")
print(f"Yhat shape: {Yhat_total.shape}, mean: {Yhat_total.mean():.4f}")

n_spiral = Y_total.sum()
n_not_spiral = len(Y_total) - n_spiral
print(f"\nSpiral (Y=1):     {int(n_spiral)}")
print(f"Not spiral (Y=0): {int(n_not_spiral)}")
print(f"Total:            {len(Y_total)}")
print(f"\nTrue spiral fraction θ* = {Y_total.mean():.4f}")
