import numpy as np
from scipy.ndimage import gaussian_filter
from skimage import color
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt


def fillLinear(f_original, iso_val):
    # Create a binary mask where the original function is above (or equal) to the iso value.
    mask = f_original >= iso_val

    # Compute the distance from each voxel to the boundary of the mask.
    # dist_out: distance for voxels outside (f_original >= iso_val)
    # dist_in: distance for voxels inside (f_original < iso_val)
    dist_out = ndimage.distance_transform_edt(mask)
    dist_in = ndimage.distance_transform_edt(~mask)

    # The signed distance: positive outside, negative inside.
    signed_distance = dist_out - dist_in

    # Now, assign new data values by linear interpolation:
    #   - At signed_distance == 0, value = iso_val.
    #   - For positive distances: linearly from iso_val to max_val.
    #   - For negative distances: linearly from iso_val to min_val.
    new_data = np.empty_like(f_original)

    pos = signed_distance >= 0
    new_data[pos] = iso_val + (max_val - iso_val) * (
        signed_distance[pos] / np.max(signed_distance)
    )
    neg = signed_distance < 0
    new_data[neg] = iso_val + (min_val - iso_val) * (
        signed_distance[neg] / np.min(signed_distance)
    )
    return new_data


def get_scalar_from_rgb(rgb_array, scalarRange, cmap_name="rainbow", num_samples=21):
    """
    Convert an array of RGB colors (values in [0,1]) to CIELAB.
    For colors with chromaticity (a, b) magnitude greater than threshold,
    average their a and b values, and find the closest scalar in the colormap.

    Parameters:
        rgb_array : numpy.ndarray
            Array of shape (n, 3) with RGB values.
        cmap_name : str
            Name of the matplotlib colormap to invert (default: "rainbow").
        num_samples : int
            Number of samples to use from the colormap.
        threshold : float
            Minimum magnitude of (a, b) to consider a pixel "colored".

    Returns:
        best_scalar : float
            Scalar between 0 and 1 corresponding to the closest colormap entry.
    """
    # Convert the RGB values to CIELAB.
    lab1 = color.rgb2lab(rgb_array.reshape(1, -1, 3)).reshape(-1, 3)
    # print("lab1: ", lab1)
    lab = color.lab2lch(color.rgb2lab(rgb_array.reshape(1, -1, 3))).reshape(-1, 3)
    # deltas = np.diff(lab1, axis=0)
    # # deltas = np.linalg.norm(np.diff(lab1, axis=0), axis=1)
    # print("delta: ", deltas)
    # arc = np.cumsum(deltas)
    # print("arc length: ", arc)
    euclidean_arc_length = {
        "rainbow": 349.1361,
        "Spectral": 280.8456,
        "coolwarm": 189.3216,
        "viridis": 221.0219,
    }

    deltaE2000_arc_length = {
        "rainbow": 170.8986,
        "Spectral": 174.6924,
        "coolwarm": 121.7846,
        "viridis": 120.5467,
    }

    new_num_samples = int(np.ceil(euclidean_arc_length[cmap_name] / 2.9))

    # Cluster into 2 groups to separate the black-ish background
    kmeans = KMeans(n_clusters=2, random_state=0)
    labels = kmeans.fit_predict(lab)
    centers = kmeans.cluster_centers_

    # Return the cluster center with higher brightness (less like black)
    avg_lab = centers[np.argmax(centers[:, 0])]

    # Sample the colormap.
    cmap = plt.get_cmap(cmap_name)
    sample_scalars = np.linspace(0, 1, new_num_samples)

    sample_rgbs = np.array([cmap(s)[:3] for s in sample_scalars])

    # Convert colormap RGBs to CIELAB.
    sample_lab = color.lab2lch(color.rgb2lab(sample_rgbs.reshape(1, -1, 3))).reshape(
        -1, 3
    )

    # Find the colormap scalar whose hue is closest to the average.
    dists = abs(sample_lab[:, 2] - avg_lab[2])
    best_idx = np.argmin(dists)
    best_scalar = sample_scalars[best_idx]

    return best_scalar * (scalarRange[1] - scalarRange[0]) + scalarRange[0]
