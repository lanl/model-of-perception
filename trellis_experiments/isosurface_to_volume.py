import numpy as np
from scipy.ndimage import gaussian_filter
from skimage import color
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle


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


def _lch_to_rgb01(lch_array):
    lab = color.lch2lab(np.asarray(lch_array, dtype=np.float64).reshape(1, -1, 3)).reshape(-1, 3)
    rgb = color.lab2rgb(lab.reshape(1, -1, 3)).reshape(-1, 3)
    return np.clip(rgb, 0.0, 1.0)


def _plot_cluster_patches(ax, centers, avg_lab, chosen_rgb, cmap_rgbs):
    center_rgbs = _lch_to_rgb01(centers)
    avg_rgb = _lch_to_rgb01(avg_lab.reshape(1, 3))[0]
    patch_rgbs = [
        avg_rgb,
        center_rgbs[0],
        center_rgbs[1],
        np.clip(chosen_rgb, 0.0, 1.0),
    ]
    patch_labels = [
        "average of all patches",
        "cluster center 0",
        "cluster center 1",
        "closest colormap color",
    ]
    for i, (rgb, label) in enumerate(zip(patch_rgbs, patch_labels)):
        x = i * 1.15
        ax.add_patch(Rectangle((x, 0.0), 1.0, 1.0, facecolor=rgb, edgecolor="black"))
        ax.text(x + 0.5, 1.08, label, ha="center", va="bottom", fontsize=9)

    for i, rgb in enumerate(cmap_rgbs):
        ax.add_patch(Rectangle((0.02 + i * 0.12, 1.55), 0.12, 0.28, facecolor=rgb, edgecolor="none"))
    ax.text(0.5 * (len(cmap_rgbs) * 0.12), 1.9, "sampled colormap", ha="center", va="bottom", fontsize=9)

    ax.set_xlim(-0.05, max(4.7, len(cmap_rgbs) * 0.12 + 0.1))
    ax.set_ylim(0, 2.05)
    ax.set_title("Representative colors")
    ax.set_yticks([])
    ax.set_xticks([])


def _plot_scalar_debug(rgb_array, labels, centers, avg_lab, sample_scalars, sample_lab, best_idx, sample_rgbs):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    _plot_cluster_patches(
        axes[0],
        centers,
        avg_lab,
        sample_rgbs[best_idx],
        sample_rgbs,
    )

    axes[1].plot(sample_scalars, sample_lab[:, 2], label="colormap hue", color="steelblue")
    axes[1].axhline(avg_lab[2], color="darkorange", linestyle="--", label="reference hue")
    axes[1].scatter(
        sample_scalars[best_idx],
        sample_lab[best_idx, 2],
        color="crimson",
        label="chosen scalar",
        zorder=3,
    )
    axes[1].set_title("Hue Match in LCH")
    axes[1].set_xlabel("scalar")
    axes[1].set_ylabel("hue")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def get_scalar_from_rgb(
    rgb_array,
    scalarRange,
    cmap_name="rainbow",
    num_samples=21,
    plot_debug=False,
    use_clusters=False,
):
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

    # Default behavior: use the average over all samples.
    # Optional cluster mode: use the brighter cluster center.
    if use_clusters:
        avg_lab = centers[np.argmax(centers[:, 0])]
    else:
        avg_lab = np.mean(lab, axis=0)

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

    if plot_debug:
        _plot_scalar_debug(
            rgb_array,
            labels,
            centers,
            avg_lab,
            sample_scalars,
            sample_lab,
            best_idx,
            sample_rgbs,
        )

    return best_scalar * (scalarRange[1] - scalarRange[0]) + scalarRange[0]
