import os
import glob
import h5py
import pandas as pd
import matplotlib.pyplot as plt
import sys

# Get folder path
folder = sys.argv[1] if len(sys.argv) > 1 else input("Enter folder path: ")

h5_files = glob.glob(os.path.join(folder, "*.h5"))
n_files = len(h5_files)

fig, axes = plt.subplots(n_files, 2, figsize=(10, 5*n_files))
if n_files == 1:
    axes = [axes]

for i, h5_path in enumerate(h5_files):
    with h5py.File(h5_path, 'r') as f:
        depth = f['channels']['Depth'][:]
#        result = f['channels']['Result'][:]
        result = f['channels']['RTData'][:]
#        result = f['channels'][next(key for key in channels if key != 'Depth')]
    ax1, ax2 = axes[i]
    im1 = ax1.imshow(depth, cmap='viridis')
    ax1.set_title(f"{os.path.basename(h5_path)} - Depth")
    fig.colorbar(im1, ax=ax1)
    
    im2 = ax2.imshow(result, cmap='plasma')
    ax2.set_title(f"{os.path.basename(h5_path)} - Result")
    fig.colorbar(im2, ax=ax2)

plt.tight_layout()
plt.savefig(os.path.join(folder, "gt"))
#plt.show()
