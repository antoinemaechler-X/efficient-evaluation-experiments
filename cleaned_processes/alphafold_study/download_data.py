import numpy as np
import os
import subprocess

dataset_folder = "data"
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

data = np.load(npz_path)
print(f"Keys: {list(data.keys())}")

Y_total = data["Y"].flatten().astype(np.float32)
Yhat_total = data["Yhat"].flatten().astype(np.float32)
Z_total = data["phosphorylated"].flatten().astype(np.float32)

assert np.isin(Y_total, [0, 1]).all()
assert (Yhat_total >= 0).all() and (Yhat_total <= 1).all()
assert np.isin(Z_total, [0, 1]).all()

np.save(os.path.join(dataset_folder, "Y.npy"), Y_total)
np.save(os.path.join(dataset_folder, "Yhat.npy"), Yhat_total)
np.save(os.path.join(dataset_folder, "Z.npy"), Z_total)

print(f"Y:    shape={Y_total.shape}, mean={Y_total.mean():.4f}")
print(f"Yhat: shape={Yhat_total.shape}, mean={Yhat_total.mean():.4f}")
print(f"Z:    shape={Z_total.shape}, mean={Z_total.mean():.4f}")

for z_val, label in [(0, "Non-phosphorylated"), (1, "Phosphorylated")]:
    mask = Z_total == z_val
    print(f"\n{label} (N={mask.sum()}):")
    print(f"  Y mean:    {Y_total[mask].mean():.4f}")
    print(f"  Yhat mean: {Yhat_total[mask].mean():.4f}")
