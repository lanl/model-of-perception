import numpy as np
import matplotlib.pyplot as plt
import re
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from nerf_imdb.fibonacci_lattice import fibonacci_lattice
import sys
from scipy.interpolate import RectBivariateSpline



if len(sys.argv) == 2:
    L2_file = sys.argv[1]

else:
    print("Enter folder path: ")
    sys.exit()
    
# Parse L2 file
# Split into lines
with open(L2_file) as f:
    lines = f.readlines()

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
# Generate spherical coordinates
elevations, azimuths = fibonacci_lattice(numberOfAngles)
    
L2_vals = np.full(numberOfAngles, np.nan)

with open(L2_file) as f:
    for line in f:
        match = re.search(r'(?:checkpoint_(\d+)|lattice:\s+(\d+))', line)
        if match:
            number = match.group(1) or match.group(2) if match else None
            idx = int(number)
            if idx < numberOfAngles:
                try:
                    val = float(line.strip().split()[-1])
                    L2_vals[idx] = val
                except:
                    continue

# Filter valid entries
mask = ~np.isnan(L2_vals)
elev = elevations[mask]
azim = azimuths[mask]
#azim = azimuths[mask] % 90
vals = L2_vals[mask]

# Remove upper outliers (e.g. > 95th percentile)
q95 = np.percentile(vals, 99)
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

# Define regular grid with your specified ranges
azim_lin = np.linspace(0, 90, 90, endpoint=False)         # periodic
#azim_lin = np.linspace(0, 360, 360, endpoint=False)         # periodic
elev_lin = np.linspace(-90, 90, 180)                      # periodic
AZIM, ELEV = np.meshgrid(azim_lin, elev_lin)

## Original valid values
#az = azim_filt[valid_mask]
#el = elev_filt[valid_mask]
#va = vals_filt[valid_mask]

# Original valid values
az = azim_filt
el = elev_filt
va = vals_filt

# Wrap azimuth (0–90)
az_wrapped = np.concatenate([
    az,
    (az + 90) % 90,
    (az - 90) % 90
])
## Wrap azimuth (0–90)
#az_wrapped = np.concatenate([
#    az,
#    (az + 360) % 360,
#    (az - 360) % 360
#])

el_wrapped = np.concatenate([
    el,
    el,
    el
])

val_wrapped = np.concatenate([
    va,
    va,
    va
])

# Wrap elevation (−90 to 90)
az_final = np.concatenate([az_wrapped, az_wrapped, az_wrapped])
el_final = np.concatenate([
    el_wrapped,
    np.clip(el_wrapped + 180, -90, 90),
    np.clip(el_wrapped - 180, -90, 90)
])
val_final = np.concatenate([val_wrapped, val_wrapped, val_wrapped])

# Final shape check
assert az_final.shape == el_final.shape == val_final.shape

# Interpolate to grid
points = np.stack((az_final, el_final), axis=1)
grid_vals = griddata(points, val_final, (AZIM, ELEV), method='linear')


# Final smooth to clean up
smoothed_vals = gaussian_filter(grid_vals, sigma=4, mode=['wrap', 'wrap'])

# Plot
fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

# 1. Raw scatter
sc = axs[0].scatter(azim_filt, elev_filt, c=vals_filt, cmap='viridis', s=50)
axs[0].set_title('Scatter (Filtered)')
axs[0].set_xlabel('Azimuth (°)')
axs[0].set_ylabel('Elevation (°)')
axs[0].grid(True)

# 2. Interpolated
im1 = axs[1].imshow(
    grid_vals,
#    extent=(0, 90, min(elev), max(elev)),
    extent=(0, 360, min(elev), max(elev)),
    origin='lower',
    aspect='auto',
    cmap='viridis',
    vmin=0, vmax=0.2
)
axs[1].set_title('Interpolated')
axs[1].set_xlabel('Azimuth (°)')
axs[1].grid(True)

# 3. Smoothed
im2 = axs[2].imshow(
    smoothed_vals,
#    extent=(0, 90, min(elev), max(elev)),
    extent=(0, 360, min(elev), max(elev)),
    origin='lower',
    aspect='auto',
    cmap='viridis',
    vmin=0, vmax=0.2
)
axs[2].set_title('Smoothed')
axs[2].set_xlabel('Azimuth (°)')
axs[2].grid(True)

fig.colorbar(im2, ax=axs[2], label='L2 Value', shrink=0.9)
plt.tight_layout()
plt.show()
