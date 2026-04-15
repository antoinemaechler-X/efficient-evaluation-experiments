"""Quick check that the downloaded CSV is complete (no full load into memory)."""
import subprocess, os

csv_path = os.path.join("data", "2019", "1-Year", "psam_p06.csv")
if not os.path.exists(csv_path):
    print(f"File not found: {csv_path}")
    exit(1)

# File size
size_mb = os.path.getsize(csv_path) / 1e6
print(f"File: {csv_path}")
print(f"Size: {size_mb:.1f} MB")

# Count lines without loading into memory
result = subprocess.run(["wc", "-l", csv_path], capture_output=True, text=True)
n_lines = int(result.stdout.strip().split()[0])
print(f"Lines: {n_lines} (expect ~195,666 = header + 195,665 rows)")

# Print header
with open(csv_path) as f:
    header = f.readline().strip()
cols = header.split(",")
print(f"Columns: {len(cols)}")

# Check our needed columns are present
needed = ['AGEP', 'SCHL', 'MAR', 'DIS', 'ESP', 'CIT', 'MIG', 'MIL',
          'ANC1P', 'NATIVITY', 'DEAR', 'DEYE', 'DREM', 'SEX', 'RAC1P',
          'SOCP', 'COW', 'PINCP']
missing = [c for c in needed if c not in cols]
if missing:
    print(f"MISSING columns: {missing}")
else:
    print("All needed columns present.")
