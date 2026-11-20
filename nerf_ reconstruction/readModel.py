import csv
import numpy as np

import h5py
from jax.sharding import SingleDeviceSharding
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
from skimage import color
from sklearn.cluster import KMeans
import vtk


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
    new_data[pos] = iso_val + (max_val - iso_val) * (signed_distance[pos] / np.max(signed_distance))
    neg = signed_distance < 0
    new_data[neg] = iso_val + (min_val - iso_val) * (signed_distance[neg] / np.min(signed_distance))
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
    lab = color.lab2lch(color.rgb2lab(rgb_array.reshape(1, -1, 3))).reshape(-1, 3)
    
    # Cluster into 2 groups to separate the black-ish background
    kmeans = KMeans(n_clusters=2, random_state=0)
    labels = kmeans.fit_predict(lab)
    centers = kmeans.cluster_centers_

    # Return the cluster center with higher brightness (less like black)
    avg_lab = centers[np.argmax(centers[0])]
#    print(avg_lab)
    
    # Sample the colormap.
    cmap = plt.get_cmap(cmap_name)
    sample_scalars = np.linspace(0, 1, num_samples)

    sample_rgbs = np.array([cmap(s)[:3] for s in sample_scalars])
    
    # Convert colormap RGBs to CIELAB.
    sample_lab = color.lab2lch(color.rgb2lab(sample_rgbs.reshape(1, -1, 3))).reshape(-1, 3)
#    print(sample_lab)
    
    # Find the colormap scalar whose hue is closest to the average.
    dists = abs(sample_lab[:, 2] - avg_lab[2])
    best_idx = np.argmin(dists)
    best_scalar = sample_scalars[best_idx]
#    print(best_scalar)
    
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
    cinema_folder = sys.argv[2]
    checkpoint_folder = sys.argv[3]
else:
    print("Enter folder path: ")
    sys.exit()
    
# read cinema database
csv_file = glob.glob(os.path.join(cinema_folder, "*.csv"))[0]

poses, images = readCinemaDatabase()
height, width = images.shape[1], images.shape[2]

plt.imshow(images[-1])
plt.savefig("rgb_gt")
plt.close()

# read model
t_near = 0.
t_far = 1.
viewport_height = 1.
key = jax.random.PRNGKey(0)
model = CinemaRGBAImage()

schedule_fn = optax.exponential_decay(init_value=1e-3, transition_begin=600,
                                      transition_steps=200, decay_rate=0.5)
optimizer = optax.adam(learning_rate=schedule_fn)

train_step, state = create_train_steps(key, model, optimizer)

pixel_sampler = UniformRandom(width=width,
                              height=height,
                              n_samples=4096)

ray_generator = Parallel(width=width, height=height, viewport_height=viewport_height)
renderer = Hierarchical()
    
target = {'state': state}

checkpointer = orbax.checkpoint.PyTreeCheckpointer()

cpu_device = jax.devices("cpu")[0]
sharding = SingleDeviceSharding(cpu_device)

raw_restored = checkpointer.restore(checkpoint_folder, item=target)

state = raw_restored['state']


pixel_coordinates_infer = Dense(width=width, height=height)()
ray_bundle = Parallel(width, height, viewport_height)(pixel_coordinates_infer, poses[-1], t_near, t_far)

key, _ = jax.random.split(key)
_, image_recon, alpha_recon, depth_recon = renderer(model.bind(state.params),
                                                    model.bind(state.params),
                                                    ray_bundle,
                                                    key).values()

plt.imshow(image_recon.reshape((width, height, 3)))
plt.savefig(os.path.join(checkpoint_folder, "rgb"))
plt.close()
plt.imshow(depth_recon.reshape((width, height, 1)))
plt.colorbar()
plt.savefig(os.path.join(checkpoint_folder, "depth"))
plt.close()


# load the vti
mesh_orig = pv.read(data_file)
points = mesh_orig.points  # Now points is a nx3 numpy array
scalarRange = mesh_orig.GetPointData().GetScalars().GetRange()

mins = points.min(axis=0)
maxs = points.max(axis=0)
points = ((points - mins) / (maxs - mins) - 0.5) / 3**0.5
print(points.shape, points.min(axis=0), points.max(axis=0))

mesh_reconstruct = mesh_orig.copy(deep=True)

field_fun = model.bind(state.params)
array_of_rgb, array_of_density = field_fun(points, points)
#print(array_of_density)
mesh_reconstruct["density"] = array_of_density
mesh_reconstruct["rgb"] = array_of_rgb

mesh_reconstruct.save(os.path.join(checkpoint_folder, "density.vti"))
