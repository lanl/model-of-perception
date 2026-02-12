import os
import glob
import h5py
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import sys

if len(sys.argv) > 1:
    folder = sys.argv[1]
else:
    folder = input("Enter folder path: ")



csv_file = glob.glob(os.path.join(folder, "*.csv"))[0]
df = pd.read_csv(csv_file)
#df = df[(df['Phi'] >= 0) & (df['Phi'] <= 90)]
#df['Theta'] = df['Theta'] % 90


fig, ax = plt.subplots()

for _, row in df.iterrows():
    phi, theta, filename = row['Phi'], row['Theta'], row['FILE']
    h5_path = os.path.join(folder, filename)
    with h5py.File(h5_path, 'r') as f:
        img_data = f['channels/rgba'][:]
    im = OffsetImage(img_data, zoom=0.5)
    ab = AnnotationBbox(im, (theta, phi), frameon=False)
    ax.add_artist(ab)

ax.set_xlabel("Azimuth (°)")
ax.set_ylabel("Elevation (°)")
ax.set_title("Image Montage")

# Adjust axis limits based on your data range
ax.set_xlim(df['Theta'].min()-10, df['Theta'].max()+10)
ax.set_ylim(df['Phi'].min()-10, df['Phi'].max()+10)
plt.savefig(os.path.join(folder, "gt"))

plt.show()
