import numpy as np
import matplotlib.pyplot as plt
import re
from scipy.interpolate import griddata
from nerf_imdb.fibonacci_lattice import fibonacci_lattice
import sys
from scipy.interpolate import RectBivariateSpline
from mpl_toolkits.axes_grid1 import make_axes_locatable



if len(sys.argv) == 2:
    L2_file = sys.argv[1]

else:
    print("Enter folder path: ")
    sys.exit()
    
# Parse L2 file
# Split into lines
with open(L2_file) as f:
    lines = f.readlines()

# Mode A: direct CSV rows azimuth,elevation,...,value (uses first 2 columns and last column)
csv_rows = []
if lines:
    header_parts = [p.strip() for p in lines[0].strip().split(",")]
    header_lower = [h.lower() for h in header_parts]
    has_header = ("az" in header_lower and "el" in header_lower)
else:
    has_header = False
    header_parts = []
    header_lower = []

if has_header:
    az_idx = header_lower.index("az")
    el_idx = header_lower.index("el")
    val_idx = len(header_parts) - 1  # default: last column
    for line in lines[1:]:
        parts = [p.strip() for p in line.strip().split(",")]
        if len(parts) <= max(az_idx, el_idx, val_idx):
            continue
        try:
            az = float(parts[az_idx])
            el = float(parts[el_idx])
            val = float(parts[val_idx])
        except ValueError:
            continue
        csv_rows.append((az, el, val))
else:
    for line in lines:
        parts = [p.strip() for p in line.strip().split(",")]
        if len(parts) < 3:
            continue
        try:
            az = float(parts[0])
            el = float(parts[1])
            val = float(parts[-1])
        except ValueError:
            continue
        csv_rows.append((az, el, val))

if csv_rows:
    arr = np.asarray(csv_rows, dtype=np.float64)
    azim = arr[:, 0]
    elev = arr[:, 1]
    vals = arr[:, 2]
    order = np.argsort(vals)
    for i in order:
        print(f"{azim[i]},{elev[i]},{vals[i]}")
else:
    # Mode B: legacy lines containing checkpoint/lattice index -> map to fibonacci lattice
    parsed = []
    for line in lines:
        match = re.findall(r"[-+]?\d*\.\d+|\d+", line)
        if match:
            last_val = float(match[-1])
            parsed.append((line.strip(), last_val))

    sorted_lines = sorted(parsed, key=lambda x: x[1])
    for line, _ in sorted_lines:
        print(line)

    numberOfAngles = len(sorted_lines)
    elevations, azimuths = fibonacci_lattice(numberOfAngles)
    L2_vals = np.full(numberOfAngles, np.nan)

    for line in lines:
        match = re.search(r'(?:checkpoint_(\d+)|lattice:\s+(\d+))', line)
        if match:
            number = match.group(1) or match.group(2) if match else None
            idx = int(number)
            if idx < numberOfAngles:
                try:
                    val = float(line.strip().split()[-1])
                    L2_vals[idx] = val
                except Exception:
                    continue

    mask = ~np.isnan(L2_vals)
    elev = elevations[mask]
    azim = azimuths[mask]
    vals = L2_vals[mask]

if vals.size == 0:
    raise RuntimeError(
        "No numeric values parsed. Expected either CSV rows like 'az,el,...,value' "
        "or legacy lines containing checkpoint/lattice indices."
    )

# Remove upper outliers (e.g. > 95th percentile)
q95 = np.percentile(vals, 100)
valid = vals <= q95
elev_filt = elev[valid]
azim_filt = azim[valid]
vals_filt = vals[valid]

## Interpolate
#azim_grid, elev_grid = np.meshgrid(
#    np.linspace(0, 90, 300),
#    np.linspace(min(elev), max(elev), 300)
#)
#vals_grid = griddata(
#    (azim_filt, elev_filt), vals_filt, (azim_grid, elev_grid), method='linear'
#)
#
## Smooth
#vals_smooth = gaussian_filter(vals_grid, sigma=5)


## Threshold near-zero values to treat as missing
#bad_mask = vals_filt < 1e-6
#valid_mask = ~bad_mask

# Interpolate directly in the same plotting domain (no angle remapping).
x_min, x_max = float(np.min(azim_filt)), float(np.max(azim_filt))
y_min, y_max = float(np.min(elev_filt)), float(np.max(elev_filt))
azim_lin = np.linspace(x_min, x_max, 360)
elev_lin = np.linspace(y_min, y_max, 180)
AZIM, ELEV = np.meshgrid(azim_lin, elev_lin)

points = np.stack((azim_filt, elev_filt), axis=1)
query = np.stack((AZIM.ravel(), ELEV.ravel()), axis=1)

# Robust interpolation:
# - full 2D: smooth RBF, then linear/nearest fallback
# - rank-deficient (effectively 1D): interpolate along varying axis and tile
ptp_x = float(np.ptp(azim_filt))
ptp_y = float(np.ptp(elev_filt))
rank = int(np.linalg.matrix_rank(points - points.mean(axis=0)))
if rank < 2 or ptp_x < 1e-12 or ptp_y < 1e-12:
    print("[warn] Input samples are effectively 1D; using 1D interpolation fallback.")
    if ptp_x >= ptp_y:
        order = np.argsort(azim_filt)
        x_sorted = azim_filt[order]
        v_sorted = vals_filt[order]
        # Collapse duplicate x by averaging
        ux, inv = np.unique(x_sorted, return_inverse=True)
        uv = np.zeros_like(ux, dtype=np.float64)
        cnt = np.zeros_like(ux, dtype=np.int64)
        np.add.at(uv, inv, v_sorted)
        np.add.at(cnt, inv, 1)
        uv = uv / np.maximum(cnt, 1)
        line = np.interp(azim_lin, ux, uv)
        grid_vals = np.tile(line[None, :], (elev_lin.size, 1))
    else:
        order = np.argsort(elev_filt)
        y_sorted = elev_filt[order]
        v_sorted = vals_filt[order]
        uy, inv = np.unique(y_sorted, return_inverse=True)
        uv = np.zeros_like(uy, dtype=np.float64)
        cnt = np.zeros_like(uy, dtype=np.int64)
        np.add.at(uv, inv, v_sorted)
        np.add.at(cnt, inv, 1)
        uv = uv / np.maximum(cnt, 1)
        line = np.interp(elev_lin, uy, uv)
        grid_vals = np.tile(line[:, None], (1, azim_lin.size))
else:
    try:
        from scipy.interpolate import RBFInterpolator
        rbf = RBFInterpolator(points, vals_filt, kernel="thin_plate_spline", smoothing=1e-10)
        grid_vals = rbf(query).reshape(AZIM.shape)
    except Exception:
        # Fallback path: linear + nearest fill.
        try:
            grid_vals = griddata(points, vals_filt, (AZIM, ELEV), method='linear')
        except Exception:
            grid_vals = np.full_like(AZIM, np.nan, dtype=np.float64)
        if np.any(~np.isfinite(grid_vals)):
            try:
                grid_nn = griddata(points, vals_filt, (AZIM, ELEV), method='nearest')
                grid_vals = np.where(np.isfinite(grid_vals), grid_vals, grid_nn)
            except Exception:
                # last resort: fill with median value
                med = float(np.nanmedian(vals_filt))
                grid_vals = np.where(np.isfinite(grid_vals), grid_vals, med)


# Automatic shared color scale from actual linear-value range
linear_vals = np.concatenate([vals_filt.ravel(), grid_vals[np.isfinite(grid_vals)].ravel()])
if linear_vals.size > 0:
    vmin = float(np.min(linear_vals))
    vmax = float(np.max(linear_vals))
else:
    vmin, vmax = 0.0, 1.0
if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
    vmin, vmax = 0.0, 1.0

# Log plot data (base-10) from interpolated grid
eps = max(vmax * 1e-12, 1e-20)
log_grid_vals = np.log10(np.clip(grid_vals, eps, None))
finite_log = np.isfinite(log_grid_vals)
if np.any(finite_log):
    log_vmin = float(np.nanmin(log_grid_vals[finite_log]))
    log_vmax = float(np.nanmax(log_grid_vals[finite_log]))
else:
    log_vmin, log_vmax = -12.0, 0.0
# Plot
fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

# Compute common axis limits based on actual scalar range
def representative_ticks(values: np.ndarray, max_ticks: int) -> np.ndarray:
    uniq = np.unique(np.round(values.astype(np.float64), 8))
    if uniq.size <= max_ticks:
        return uniq
    idx = np.linspace(0, uniq.size - 1, max_ticks).astype(int)
    return uniq[idx]

xticks = representative_ticks(azim_filt, max_ticks=9)
yticks = representative_ticks(elev_filt, max_ticks=7)

# 1. Raw scatter
sc = axs[0].scatter(azim_filt, elev_filt, c=vals_filt, cmap='viridis', s=50)
axs[0].set_title('Scatter (Filtered)')
axs[0].set_xlabel('Azimuth (°)')
axs[0].set_ylabel('Elevation (°)')
axs[0].set_xlim(x_min, x_max)
axs[0].set_ylim(y_min, y_max)
axs[0].grid(True)

# 2. Interpolated
im1 = axs[1].imshow(
    grid_vals,
#    extent=(0, 90, min(elev), max(elev)),
    extent=(x_min, x_max, y_min, y_max),
    origin='lower',
    aspect='auto',
    cmap='viridis',
    vmin=vmin, vmax=vmax
)
axs[1].set_title('Interpolated')
axs[1].set_xlabel('Azimuth (°)')
axs[1].set_xlim(x_min, x_max)
axs[1].set_ylim(y_min, y_max)
axs[1].grid(True)

# 3. Log Interpolated
im2 = axs[2].imshow(
    log_grid_vals,
#    extent=(0, 90, min(elev), max(elev)),
    extent=(x_min, x_max, y_min, y_max),
    origin='lower',
    aspect='auto',
    cmap='viridis'
    ,vmin=log_vmin, vmax=log_vmax
)
axs[2].set_title('Interpolated (log10)')
axs[2].set_xlabel('Azimuth (°)')
axs[2].set_xlim(x_min, x_max)
axs[2].set_ylim(y_min, y_max)
axs[2].grid(True)

# Force identical tick positions on all panels.
for ax in axs:
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

# Colorbar for panel 1 (scatter)
divider0 = make_axes_locatable(axs[0])
cax0 = divider0.append_axes("right", size="4%", pad=0.06)
fig.colorbar(sc, cax=cax0, label='L2 Value')

# Put colorbar in a dedicated axis so subplot 3 keeps the same size as the others.
divider = make_axes_locatable(axs[2])
cax = divider.append_axes("right", size="4%", pad=0.06)
fig.colorbar(im2, cax=cax, label='log10(L2 Value)')
plt.tight_layout()
plt.show()
