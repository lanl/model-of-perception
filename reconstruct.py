import csv
import numpy as np

#import h5py
import jax
import jax.numpy as jnp
import numpy
import optax
from flax.training.train_state import TrainState
from matplotlib import pyplot as plt
from tqdm import tqdm

from synema.models.cinema import CinemaRGBAImage
from synema.renderers.ray_gen import Parallel
from synema.renderers.rays import RayBundle
from synema.renderers.volume import Hierarchical
from synema.samplers.pixel import Dense, UniformRandom
import sys
import glob
import os
import shutil
import orbax.checkpoint
from flax.training import orbax_utils
import pyvista as pv
from scipy.ndimage import gaussian_filter
import scipy.ndimage as ndimage
from scipy.interpolate import RBFInterpolator

from skimage import color
from sklearn.cluster import KMeans
import vtk
from colour import delta_E
from scipy.interpolate import interp1d

from skimage.measure import marching_cubes

def extract_extrema_and_contour(f, iso_val):
    # 3D neighborhood for extrema
    neighborhood = ndimage.generate_binary_structure(3, 2)
    print(f.shape, neighborhood.shape)
    local_max = (f == ndimage.maximum_filter(f, footprint=neighborhood, mode='reflect'))
    local_min = (f == ndimage.minimum_filter(f, footprint=neighborhood, mode='reflect'))

    maxima_coords = np.argwhere(local_max & (f > iso_val))
    minima_coords = np.argwhere(local_min & (f < iso_val))

    # Extract iso-surface points using marching cubes
    verts, _,_,_ = marching_cubes(f, level=iso_val)
    contour_coords = verts  # already in float coordinates

    # Combine all coordinates and values
    extrema_coords = np.vstack((maxima_coords, minima_coords))
    extrema_values = np.hstack((
        f[maxima_coords[:, 0], maxima_coords[:, 1], maxima_coords[:, 2]],
        f[minima_coords[:, 0], minima_coords[:, 1], minima_coords[:, 2]]
    ))

    # Interpolate values at iso-contour points (all iso_val)
    contour_values = np.full(len(contour_coords), iso_val)

    coords = np.vstack((extrema_coords, contour_coords))
    values = np.hstack((extrema_values, contour_values))
    return coords, values


def rbf_from_extrema_and_contour(coords, values, shape):
    # Remove duplicate points
    unique_coords, idx = np.unique(coords, axis=0, return_index=True)
    values = values[idx]

    # Fit RBFInterpolator (fast alternative to Rbf)
    rbf_interp = RBFInterpolator(unique_coords, values, kernel='thin_plate_spline')

    # Create a grid for interpolation
    grid_x, grid_y, grid_z = np.meshgrid(
        np.arange(shape[0]),
        np.arange(shape[1]),
        np.arange(shape[2]),
        indexing='ij'
    )
    grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel(), grid_z.ravel()))

    # Interpolate and reshape
    result = rbf_interp(grid_points).reshape(shape)
    return result
    
def plot_lch_patches(lch_array):
    # Original RGB
    lab = color.lch2lab(lch_array.reshape(1, -1, 3)).reshape(-1, 3)
    rgb = color.lab2rgb(lab.reshape(1, -1, 3)).reshape(-1, 3)
    rgb = np.clip(rgb, 0, 1)

    # Modified LCH with L=50, C=max(C)
    lch_fixed = lch_array.copy()
    lch_fixed[:, 0] = 50
    lch_fixed[:, 1] = lch_array[:, 1].max()
    lab_fixed = color.lch2lab(lch_fixed.reshape(1, -1, 3)).reshape(-1, 3)
    rgb_fixed = color.lab2rgb(lab_fixed.reshape(1, -1, 3)).reshape(-1, 3)
    rgb_fixed = np.clip(rgb_fixed, 0, 1)

    n = len(rgb)
    fig, ax = plt.subplots(figsize=(n, 2))
    for i in range(n):
        ax.add_patch(plt.Rectangle((i, 1), 1, 1, color=rgb[i]))        # original
        ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=rgb_fixed[i]))  # fixed
    ax.set_xlim(0, n)
    ax.set_ylim(0, 2)
    ax.axis('off')
    plt.show()
   
def compute_deltaE2000_path(lab_sorted):
    lab1 = lab_sorted[:-1]
    lab2 = lab_sorted[1:]

    dE = delta_E(lab1, lab2, method='CIE 2000')
    return np.concatenate([[0], np.cumsum(dE)])
    
def sample_by_arclength(colormap_rgb, lab_sorted, arc, lut_samples):
    s = np.linspace(0, arc[-1], lut_samples)
    s_norm = s / arc[-1]
    
    # Interpolate RGB and LAB along arc
    f_rgb = interp1d(arc, colormap_rgb, axis=0)
    f_lab = interp1d(arc, lab_sorted, axis=0)

    samp_rgb = f_rgb(s)
    samp_lab = f_lab(s)

    return s_norm, samp_rgb, samp_lab
    
def get_scalar_from_rgb(rgb_array, scalarRange, cmap_name="rainbow", num_samples=256):
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
#    lab = color.lab2lch(color.rgb2lab(rgb_array.reshape(1, -1, 3))).reshape(-1, 3)
    lab = (color.rgb2lab(rgb_array.reshape(1, -1, 3))).reshape(-1, 3)

    # Cluster into 2 groups to separate the black-ish background
    kmeans = KMeans(n_clusters=2, random_state=0)
    labels = kmeans.fit_predict(lab)
    centers = kmeans.cluster_centers_
    avg_lab = centers[np.argmax(centers[:, 0])]
    avg_lch = color.lab2lch(avg_lab)
#    print('centers', centers)
#    print('avg_lab', avg_lab)
#    print('avg_lch', avg_lch)

    # Sample the colormap.
    cmap = plt.get_cmap(cmap_name)
    sample_scalars = np.linspace(0, 1, num_samples)
    sample_rgbs = np.array([cmap(s)[:3] for s in sample_scalars])
    
    # Convert colormap RGBs to CIELAB.
    sample_lab = color.rgb2lab(sample_rgbs.reshape(1, -1, 3)).reshape(-1, 3)

    arc = compute_deltaE2000_path(sample_lab)
    lut_samples = int(np.ceil(arc[-1]) / 2.9)
    sample_scalars, sample_rgbs, sample_lab = sample_by_arclength(sample_rgbs, sample_lab, arc, lut_samples)
    sample_lch = color.lab2lch(color.rgb2lab(sample_rgbs.reshape(1, -1, 3))).reshape(-1, 3)

    # Find the colormap scalar whose hue is closest to the average.
    dists = abs(sample_lch[:, 2] - avg_lch[2])
    best_idx = np.argmin(dists)
    best_scalar = sample_scalars[best_idx]
    print('best_scalar', best_scalar)
#    plot_lch_patches(np.concatenate([centers, [avg_lch], [sample_lch[best_idx]],[sample_lch[int(0.5*lut_samples)]]], axis=0))

    return best_scalar * (scalarRange[1] - scalarRange[0]) + scalarRange[0]
    
def create_train_steps(key, model, optimizer):
    init_state = TrainState.create(apply_fn=model.apply,
                                   params=model.init(key, jnp.empty((1024, 3)),
                                                     jnp.empty((1024, 3))),
                                   tx=optimizer)
    train_renderer = Hierarchical()

    def loss_fn(params, ray_bundle: RayBundle, targets, key: jax.random.PRNGKey):
        _, rgb, alpha, depth = train_renderer(coarse_field=model.bind(params),
                                              fine_field=model.bind(params),
                                              ray_bundle=ray_bundle,
                                              rng_key=key).values()
        return jnp.mean(optax.l2_loss(rgb, targets['rgb']))
        # return (jnp.mean(optax.l2_loss(scalar, targets['scalar'])) +
        #         1.e-3 * jnp.mean(jnp.abs(depth - jnp.nan_to_num(targets['depth']))))

    @jax.jit
    def train_step(state, ray_bundle: RayBundle, targets, key: jax.random.PRNGKey):
        loss_val, grads = jax.value_and_grad(loss_fn)(state.params, ray_bundle, targets, key)
        new_state = state.apply_gradients(grads=grads)
        return new_state, loss_val

    return train_step, init_state

def readCinemaDatabase():
    with open(csv_file, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')

        poses = []
        images = []

        for row in reader:
            h5file = h5py.File(os.path.join(cinema_folder, row['FILE']), 'r')
            meta = h5file.get('meta')
            for key, value in meta.items():
                print(key, value[...])  # value[...] reads the full dataset            sys.exit()
            camera_height = numpy.array(meta['CameraHeight'])  # not used by color image exporter
            camera_dir = numpy.array(meta['CameraDir'])
            camera_pos = numpy.array(meta['CameraPos'])
            camera_near_far = numpy.array(meta['CameraNearFar'])  # not used by color image exporter
            camera_up = numpy.array(meta['CameraUp'])

            # construct camera orientation matrix
            camera_w = -camera_dir / numpy.linalg.norm(camera_dir)
            camera_u = numpy.cross(camera_up, camera_w)
            camera_u = camera_u / numpy.linalg.norm(camera_u)
            camera_v = numpy.cross(camera_w, camera_u)
            camera_v = camera_v / numpy.linalg.norm(camera_v)

            # normalize the bbox to [-0.5, 0.5]^3 to prevent vanishing gradient.
            camera_pos_normalized = 0.5 * camera_w

            pose = numpy.zeros((4, 4))
            pose[:3, 0] = camera_u
            pose[:3, 1] = camera_v
            pose[:3, 2] = camera_w
            pose[:3, 3] = camera_pos_normalized
            pose[3, 3] = 1

            poses.append(pose)

            channels = h5file.get('channels')
            image = numpy.array(channels['rgba'], dtype=numpy.float32) / 255.
            images.append(image)

        poses = numpy.stack(poses, axis=0)
        images = numpy.stack(images, axis=0)

        return poses, images






if len(sys.argv) > 3:
    data_file = sys.argv[1]
    checkpoint_folder = sys.argv[2]
    cmap_name = sys.argv[3]
elif len(sys.argv) > 2:
    data_file = sys.argv[1]
    checkpoint_folder = sys.argv[2]
    cmap_name = 'Spectral'
else:
    print("Enter folder path: ")
    
## read cinema database
#csv_file = glob.glob(os.path.join(cinema_folder, "*.csv"))[0]
#
#poses, images = readCinemaDatabase()
#height, width = images.shape[1], images.shape[2]
#
#plt.imshow(images[-1])
#plt.savefig("rgb_gt")
#plt.close()

## read model
#t_near = 0.
#t_far = 1.
#viewport_height = 1.
#key = jax.random.PRNGKey(0)
#model = CinemaRGBAImage()
#
#schedule_fn = optax.exponential_decay(init_value=1e-3, transition_begin=600,
#                                      transition_steps=200, decay_rate=0.5)
#optimizer = optax.adam(learning_rate=schedule_fn)
#
#train_step, state = create_train_steps(key, model, optimizer)
#
#pixel_sampler = UniformRandom(width=width,
#                              height=height,
#                              n_samples=4096)
#
#ray_generator = Parallel(width=width, height=height, viewport_height=viewport_height)
#renderer = Hierarchical()
#    
#target = {'state': state}
#
#checkpointer = orbax.checkpoint.PyTreeCheckpointer()
#raw_restored = checkpointer.restore(checkpoint_folder, item=target)
#
#state = raw_restored['state']
#
#
#pixel_coordinates_infer = Dense(width=width, height=height)()
#ray_bundle = Parallel(width, height, viewport_height)(pixel_coordinates_infer, poses[-1], t_near, t_far)
#
#key, _ = jax.random.split(key)
#_, image_recon, alpha_recon, depth_recon = renderer(model.bind(state.params),
#                                                    model.bind(state.params),
#                                                    ray_bundle,
#                                                    key).values()
#
#plt.imshow(image_recon.reshape((width, height, 3)))
#plt.savefig(os.path.join(checkpoint_folder, "rgb"))
#plt.close()
#plt.imshow(depth_recon.reshape((width, height, 1)))
#plt.colorbar()
#plt.savefig(os.path.join(checkpoint_folder, "depth"))
#plt.close()


# load the vti
mesh_orig = pv.read(data_file)
points = mesh_orig.points  # Now points is a nx3 numpy array
mesh_orig.GetPointData().SetScalars(mesh_orig.GetPointData().GetArray(0))
scalarRange = mesh_orig.GetPointData().GetScalars().GetRange()

#mins = points.min(axis=0)
#maxs = points.max(axis=0)
#points = (points - mins) / (maxs - mins) - 0.5
#print(points.shape, points.min(axis=0), points.max(axis=0))
#
#mesh_reconstruct = mesh_orig.copy(deep=True)
#
#field_fun = model.bind(state.params)
#array_of_rgb, array_of_density = field_fun(points, points)
##print(array_of_density)
#mesh_reconstruct["density"] = array_of_density
#mesh_reconstruct["rgb"] = array_of_rgb

mesh_reconstruct = pv.read(os.path.join(checkpoint_folder, "density.vti"))

# get average rgb
iso_val = get_scalar_from_rgb(mesh_reconstruct["rgb"], scalarRange, cmap_name=cmap_name)
iso_val = 0.15
print('iso_val', iso_val)
#sys.exit()
#print(mesh_reconstruct["density"].min(), mesh_reconstruct["density"].max(), (mesh_reconstruct["density"].min() + mesh_reconstruct["density"].max())/2,np.median(mesh_reconstruct["density"]), np.mean(mesh_reconstruct["density"]))
#avg_density = (mesh_reconstruct["density"].min() + mesh_reconstruct["density"].max())/2

# Apply Gaussian smoothing
density_3d = mesh_reconstruct["density"].reshape(mesh_orig.dimensions)
smoothed_density = gaussian_filter(density_3d, sigma=0.5)
mesh_reconstruct["density"] = smoothed_density.ravel()

# compute the contour at mid density
#print(mesh_reconstruct["density"].min(), mesh_reconstruct["density"].max(), (mesh_reconstruct["density"].min() + mesh_reconstruct["density"].max())/2,np.median(mesh_reconstruct["density"]), np.mean(mesh_reconstruct["density"]))
#if avg_density < mesh_reconstruct["density"].min() or avg_density > mesh_reconstruct["density"].max():
avg_density = (mesh_reconstruct["density"].min() + mesh_reconstruct["density"].max())/2

clipped = mesh_reconstruct.clip_scalar(scalars="density", invert=False, value=avg_density)
surface = clipped.extract_surface()
surface.save(os.path.join(checkpoint_folder, "surface.vtp"))

# Create an implicit distance function from the surface.
distance_func = vtk.vtkImplicitPolyDataDistance()
distance_func.SetInput(surface)

# Evaluate the signed distance for each point in the mesh.
points = mesh_reconstruct.points
signed_distances = np.array([distance_func.EvaluateFunction(p) for p in points])

# Optionally, add the distances to the mesh's point data.
mesh_reconstruct.point_data["signed_distance"] = signed_distances

# interpolate with min and max
new_data_lin1 = np.empty_like(signed_distances)
new_data_lin2 = np.empty_like(signed_distances)
new_data1 = np.empty_like(signed_distances)
new_data2 = np.empty_like(signed_distances)
pos = signed_distances >= 0
new_data_lin1[pos] = iso_val + (scalarRange[1] - iso_val) * (signed_distances[pos] / np.max(signed_distances))
new_data_lin2[pos] = iso_val + (scalarRange[0] - iso_val) * (signed_distances[pos] / np.max(signed_distances))
neg = signed_distances < 0
new_data_lin1[neg] = iso_val + (scalarRange[0] - iso_val) * (signed_distances[neg] / np.min(signed_distances))
new_data_lin2[neg] = iso_val + (scalarRange[1] - iso_val) * (signed_distances[neg] / np.min(signed_distances))
new_data_lin1 = new_data_lin1.reshape(mesh_orig.dimensions, order="F")
new_data_lin2 = new_data_lin2.reshape(mesh_orig.dimensions, order="F")
#coords, values = extract_extrema_and_contour(new_data_lin1 , iso_val)
#new_data1 = rbf_from_extrema_and_contour(coords, values, new_data_lin1.shape)
#coords, values = extract_extrema_and_contour(new_data_lin2, iso_val)
#new_data2 = rbf_from_extrema_and_contour(coords, values, new_data_lin2.shape)
new_data1 = new_data_lin1
new_data2 = new_data_lin2
# Compute L2 difference between original and reconstructed data.

L2_diff1 = np.sqrt(np.mean((mesh_orig.GetPointData().GetScalars() - new_data1.flatten())**2))
L2_diff2 = np.sqrt(np.mean((mesh_orig.GetPointData().GetScalars() - new_data2.flatten())**2))
L2_diff = min(L2_diff1, L2_diff2)
new_data = new_data1 if L2_diff1 < L2_diff2 else new_data2
print('iso_val', iso_val, "L2 difference:", L2_diff1, L2_diff2, L2_diff)

#mesh_reconstruct["scalar"] = new_data
mesh_reconstruct.point_data["scalar"] = new_data.ravel(order="F")   # or .flatten("F")

#sys.exit()

# Save as .vti file
mesh_reconstruct.save(os.path.join(checkpoint_folder, "reconstructed.vti"))
