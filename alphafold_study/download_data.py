"""
Download AlphaFold proteomics dataset (from ppi_py's Google Drive).

Downloads the .npz directly via gdown, then extracts and saves
Y (binary disorder labels), Yhat (AlphaFold predictions),
and Z (phosphorylation indicator) as .npy files.
"""
import numpy as np
import os
import subprocess

dataset_folder = "alphafold_study/data"
os.makedirs(dataset_folder, exist_ok=True)

npz_path = os.path.join(dataset_folder, "alphafold.npz")

# Download from Google Drive (same ID used by ppi_py)
if not os.path.exists(npz_path):
    print("Downloading AlphaFold dataset...")
    subprocess.check_call([
        "gdown", "1lOhdSJEcFbZmcIoqmlLxo3LgLG1KqPho", "-O", npz_path
    ])
else:
    print(f"Dataset already exists at {npz_path}")

# Load and inspect
data = np.load(npz_path)
print(f"Keys in npz: {list(data.keys())}")

# Extract arrays — names may vary; print to discover
for key in data.keys():
    arr = data[key]
    print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}, "
          f"min={arr.min():.4f}, max={arr.max():.4f}, mean={arr.mean():.4f}")

# The AlphaFold dataset from ppi_py contains:
#   Y: binary disorder labels
#   Yhat: AlphaFold predicted disorder probabilities
#   Z: phosphorylation indicator (covariate)
Y_total = data["Y"].flatten().astype(np.float32)
Yhat_total = data["Yhat"].flatten().astype(np.float32)
Z_total = data["phosphorylated"].flatten().astype(np.float32)

# Basic checks
assert np.isin(Y_total, [0, 1]).all(), "Y must be binary"
assert (Yhat_total >= 0).all() and (Yhat_total <= 1).all(), "Yhat must be in [0, 1]"
assert np.isin(Z_total, [0, 1]).all(), "Z must be binary"

# Save as individual .npy files
np.save(os.path.join(dataset_folder, "Y.npy"), Y_total)
np.save(os.path.join(dataset_folder, "Yhat.npy"), Yhat_total)
np.save(os.path.join(dataset_folder, "Z.npy"), Z_total)

# Print statistics
print(f"\nY shape:    {Y_total.shape}, mean: {Y_total.mean():.4f}")
print(f"Yhat shape: {Yhat_total.shape}, mean: {Yhat_total.mean():.4f}")
print(f"Z shape:    {Z_total.shape}, mean: {Z_total.mean():.4f}")

n_nonphos = (Z_total == 0).sum()
n_phos = (Z_total == 1).sum()
print(f"\nNon-phosphorylated (Z=0): {n_nonphos}")
print(f"Phosphorylated (Z=1):     {n_phos}")
print(f"Total:                    {len(Y_total)}")

for z_val, label in [(0, "Non-phosphorylated"), (1, "Phosphorylated")]:
    mask = Z_total == z_val
    print(f"\n{label}:")
    print(f"  Y mean:    {Y_total[mask].mean():.4f}")
    print(f"  Yhat mean: {Yhat_total[mask].mean():.4f}")
