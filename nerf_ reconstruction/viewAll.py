import os
import glob
import h5py
import pandas as pd
import matplotlib.pyplot as plt
import sys

# Get folder path
folder = sys.argv[1] if len(sys.argv) > 1 else input("Enter folder path: ")

# Read CSV and get last image file
csv_file = glob.glob(os.path.join(folder, "*.csv"))[0]
df = pd.read_csv(csv_file)

for i in range(len(df)):
    filename = df.iloc[i]['FILE']

    # Load image from HDF5
    h5_path = os.path.join(folder, filename)
    with h5py.File(h5_path, 'r') as f:
        img_data = f['channels/rgba'][:]

    # Get image dimensions
    height, width, _ = img_data.shape

    # Create figure matching image dimensions (using a chosen dpi)
    dpi = 100
    fig = plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])  # Fill the whole figure
    ax.imshow(img_data)
    ax.axis('off')  # No coordinate system
    plt.savefig(os.path.join(folder, "gt"+ str(i)))
    plt.close()
    #plt.show()
