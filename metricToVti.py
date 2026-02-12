import sys
import pandas as pd
import numpy as np
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: python make_vti.py <input.csv>")
    sys.exit(1)

csv_path = Path(sys.argv[1])
out_path = csv_path.with_suffix(".vti")
coords_path = csv_path.with_name("axis_values.csv")

df = pd.read_csv(csv_path)

# --- Identify axis columns ---
cols_lower = {c.lower(): c for c in df.columns}

def find_col(candidates):
    for c in df.columns:
        if c in candidates:
            return c
    for c in df.columns:
        if any(k in c.lower() for k in candidates):
            return c
    return None

az_col = find_col({"az"})
el_col = find_col({"el"})
iso_col = find_col({"isovalue", "iso", "iso_value", "threshold"})
if not (az_col and el_col and iso_col):
    raise ValueError("Could not detect AZ, EL, or isovalue columns")

metric_cols = [c for c in df.columns if c not in [az_col, el_col, iso_col]]
preferred = ["chamfer", "hausdorff", "L2_diff1", "L2_diff2", "L2_diff"]
arrays_to_write = [c for c in metric_cols if c.lower() in [p.lower() for p in preferred]][:5]
if len(arrays_to_write) < 5:
    arrays_to_write += [c for c in metric_cols if c not in arrays_to_write]
arrays_to_write = arrays_to_write[:5]

az_vals = np.sort(df[az_col].unique())
el_vals = np.sort(df[el_col].unique())
iso_vals = np.sort(df[iso_col].unique())
nx, ny, nz = len(az_vals), len(el_vals), len(iso_vals)

ix = {v: i for i, v in enumerate(az_vals)}
iy = {v: j for j, v in enumerate(el_vals)}
iz = {v: k for k, v in enumerate(iso_vals)}

volumes = {name: np.full((nx, ny, nz), np.nan, dtype=float) for name in arrays_to_write}

for _, row in df.iterrows():
    i, j, k = ix[row[az_col]], iy[row[el_col]], iz[row[iso_col]]
    for name in arrays_to_write:
        volumes[name][i, j, k] = float(row[name])

for name, vol in volumes.items():
    if np.isnan(vol).any():
        vol[np.isnan(vol)] = np.nanmean(vol)

def array_to_ascii(arr):
    return " ".join(f"{x:.9g}" for x in arr.flatten(order="C"))

xml = [
    '<?xml version="1.0"?>',
    '<VTKFile type="ImageData" version="0.1" byte_order="LittleEndian">',
    f'  <ImageData WholeExtent="0 {nx-1} 0 {ny-1} 0 {nz-1}" Origin="0 0 0" Spacing="1 1 1">',
    f'    <Piece Extent="0 {nx-1} 0 {ny-1} 0 {nz-1}">',
    '      <PointData>',
]

for name in arrays_to_write:
    xml.append(f'        <DataArray type="Float32" Name="{name}" format="ascii">')
    xml.append("          " + array_to_ascii(volumes[name]))
    xml.append('        </DataArray>')

xml += [
    '      </PointData>',
    '      <CellData/>',
    '    </Piece>',
    '  </ImageData>',
    '</VTKFile>',
]

out_path.write_text("\n".join(xml))

pd.DataFrame({
    "axis": ["AZ"] * nx + ["EL"] * ny + ["isovalue"] * nz,
    "index": list(range(nx)) + list(range(ny)) + list(range(nz)),
    "value": list(az_vals) + list(el_vals) + list(iso_vals),
}).to_csv(coords_path, index=False)

print(f"Saved {out_path} and {coords_path}")
