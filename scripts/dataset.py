import gzip
import pickle
import random

import numpy as np
import lie_learn.spaces.S2 as S2
from tqdm import tqdm
from torchvision.datasets import MNIST
from torch.utils.data import TensorDataset
import torch
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
from icoCNN.tools import icosahedral_grid_coordinates, random_icosahedral_rotation_matrix, rotate_signal

# Much of this comes from This comes immediately from https://github.com/jonkhler/s2cnn/tree/master/examples/mnist.
# Some parts have been tweaked for icosahedral things later on.

# --- SPHERICAL DATASETS ---


def random_rotation_matrix():
    theta, phi, z = np.random.uniform(size=(3,))

    theta *= 2. * np.pi
    phi *= 2. * np.pi
    z *= 2.

    r = np.sqrt(z)
    V = (
        r * np.sin(phi),
        r * np.cos(phi),
        np.sqrt(2. - z)
    )

    st = np.sin(theta)
    ct = np.cos(theta)

    R = np.array(
        ((ct,  st,  0),
         (-st, ct,  0),
         (0,   0,   1))
    )

    # Define rotation matrix by (V @ V^T - I) @ R.
    return (np.outer(V, V) - np.eye(3)).dot(R)


def rotate_grid(rotation_matrix, grid):
    return np.einsum('ij,jab->iab', rotation_matrix, np.array((grid)))


def get_projection_grid(b, grid_type='Driscoll-Healy'):
    theta, phi = S2.meshgrid(b=b, grid_type=grid_type)

    x_ = np.sin(theta) * np.cos(phi)
    y_ = np.sin(theta) * np.sin(phi)
    z_ = np.cos(theta)

    return x_, y_, z_


def project_sphere_on_xy_plane(grid, projection_origin):
    sx, sy, sz = projection_origin
    x, y, z = grid
    z = z.copy() + 1

    t = -z / (z - sz)
    qx = t * (x - sx) + x
    qy = t * (y - sy) + y

    x_min = 1 / 2 * (-1 - sx) + -1
    y_min = 1 / 2 * (-1 - sy) + -1

    rx = (qx - x_min) / (2 * np.abs(x_min))
    ry = (qy - y_min) / (2 * np.abs(y_min))

    return rx, ry


def sample_within_bounds(signal, x, y, bounds):
    x_min, x_max, y_min, y_max = bounds

    idxs = (x_min <= x) & (x < x_max) & (y_min <= y) & (y < y_max)

    if len(signal.shape) > 2:
        sample = np.zeros((signal.shape[0], x.shape[0], x.shape[1]))
        sample[:, idxs] = signal[:, x[idxs], y[idxs]]

    else:
        sample = np.zeros((x.shape[0], x.shape[1]))
        sample[idxs] = signal[x[idxs], y[idxs]]

    return sample


def sample_bilinear(signal, rx, ry):

    signal_dim_x = signal.shape[1]
    signal_dim_y = signal.shape[2]

    rx *= signal_dim_x
    ry *= signal_dim_y

    # Discretise sample position.
    ix = rx.astype(int)
    iy = ry.astype(int)

    # Obtain four sample coordinates.
    ix0 = ix
    iy0 = iy
    ix1 = ix + 1
    iy1 = iy + 1

    bounds = (0, signal_dim_x, 0, signal_dim_y)

    # Sample signal at each four positions.
    signal_00 = sample_within_bounds(signal, ix0, iy0, bounds)
    signal_10 = sample_within_bounds(signal, ix1, iy0, bounds)
    signal_01 = sample_within_bounds(signal, ix0, iy1, bounds)
    signal_11 = sample_within_bounds(signal, ix1, iy1, bounds)

    # Linear interpolation in x-direction.
    fx1 = (ix1-rx) * signal_00 + (rx-ix0) * signal_10
    fx2 = (ix1-rx) * signal_01 + (rx-ix0) * signal_11

    # Linear interpolation in y-direction.
    return (iy1 - ry) * fx1 + (ry - iy0) * fx2


def project_2d_on_sphere(signal, grid, projection_origin=None):
    if projection_origin is None:
        # Add a little bit of error onto the pole.
        projection_origin = (0, 0, 2 + 1e-3)

    rx, ry = project_sphere_on_xy_plane(grid, projection_origin)
    sample = sample_bilinear(signal, rx, ry)

    # Ensure that only south hemisphere gets projected.
    sample *= (grid[2] <= 1).astype(np.float64)

    # Rescale signal to [0,1].S
    sample_min = sample.min(axis=(1, 2)).reshape(-1, 1, 1)
    sample_max = sample.max(axis=(1, 2)).reshape(-1, 1, 1)

    sample = (sample - sample_min) / (sample_max - sample_min)
    sample *= 255
    sample = sample.astype(np.uint8)

    return sample


# Original.
def generate_dataset(b, rotate_train, rotate_test, chunk_size, save_path=None):
    # Download/load data.
    train_dataset = MNIST(root='./data', train=True, download=True)
    test_dataset = MNIST(root='./data', train=False, download=True)

    # Split data from labels, and put into a dictionary.
    mnist_train = {'images': train_dataset.train_data.numpy(), 'labels': train_dataset.train_labels.numpy()}
    mnist_test = {'images': test_dataset.test_data.numpy(), 'labels': test_dataset.test_labels.numpy()}

    # Grid of given bandwidth.
    grid = get_projection_grid(b)

    dataset = {}
    no_rotate = {'train': rotate_train, 'test': rotate_test}

    for split, data in zip(['train', 'test'], [mnist_train, mnist_test]):
        signals = data['images'].reshape(-1, 28, 28).astype(np.float64)
        n_signals = signals.shape[0]

        projections = np.ndarray(
            # Dataset shape: [No. samples, 2 * bandwidth, 2 * bandwidth].
            (n_signals, 2 * b, 2 * b),
            dtype=np.uint8
        )

        for current in tqdm(np.arange(n_signals, step=chunk_size), desc=f'Processing {split} dataset'):
            # Do (or do not -- there is no try) rotation.
            if not no_rotate[split]:
                rotation_matrix = random_rotation_matrix()
                rotated_grid = rotate_grid(rotation_matrix, grid)
            else:
                rotated_grid = grid

            # Get the signals (for the current chunk) and compute projections onto the sphere.
            idxs = np.arange(current, min(n_signals, current + chunk_size))
            chunk = signals[idxs]
            projections[idxs] = project_2d_on_sphere(chunk, rotated_grid)

        dataset[split] = {'images': projections, 'labels': data['labels']}

    if save_path:
        with gzip.open(save_path, mode='wb') as file:
            pickle.dump(dataset, file)

    return dataset


def generate_s2_dataset(b, train, augment, chunk_size, save_path=None):
    assert augment in ['none', 'ico', 's2']

    # Download/load data.
    dataset = MNIST(root='./data', train=train, download=True)

    # Split data from labels, and put into a dictionary.
    mnist = {'images': dataset.train_data.numpy(), 'labels': dataset.train_labels.numpy()}

    # Grid of given bandwidth.
    grid = get_projection_grid(b)

    signals = mnist['images'].reshape(-1, 28, 28).astype(np.float64)
    n_signals = signals.shape[0]

    projections = np.ndarray(
        # Dataset shape: [No. samples, 2 * bandwidth, 2 * bandwidth].
        (n_signals, 2 * b, 2 * b),
        dtype=np.uint8
    )

    for current in tqdm(np.arange(n_signals, step=chunk_size), desc=f'Processing dataset'):
        # Do (or do not -- there is no try) rotation.
        if augment == 's2':
            # Random continuous rotation.
            rotation_matrix = random_rotation_matrix()
            rotated_grid = rotate_grid(rotation_matrix, grid)
        elif augment == 'ico':
            # Random discrete rotation from the icosahedral symmetry group.
            icosahedral_group = R.create_group('I')
            icosahedral_rotations = icosahedral_group.as_matrix()
            rotation_matrix = random.choice(icosahedral_rotations)
            rotated_grid = rotate_grid(rotation_matrix, grid)
        else:
            rotated_grid = grid

        # Get the signals (for the current chunk) and compute projections onto the sphere.
        idxs = np.arange(current, min(n_signals, current + chunk_size))
        chunk = signals[idxs]
        projections[idxs] = project_2d_on_sphere(chunk, rotated_grid)

    if save_path:
        with gzip.open(save_path, mode='wb') as file:
            pickle.dump({'images': projections, 'labels': mnist['labels']}, file)

    return {'images': projections, 'labels': mnist['labels']}


def generate_augment_s2_dataset(b, train, augment, save_path=None, save_folder=None):
    assert augment in ['ico', 's2']

    # Download/load data.
    dataset = MNIST(root='./data', train=train, download=True)

    # Split data from labels, and put into a dictionary.
    mnist = {'images': dataset.train_data.numpy(), 'labels': dataset.train_labels.numpy()}

    # Grid of given bandwidth.
    grid = get_projection_grid(b)

    signals = mnist['images'].reshape(-1, 28, 28).astype(np.float64)
    n_signals = signals.shape[0] * 60

    projections = np.ndarray(
        # Dataset shape: [60 * No. samples, 2 * bandwidth, 2 * bandwidth].
        (n_signals, 2 * b, 2 * b),
        dtype=np.uint8
    )

    for current in tqdm(range(signals.shape[0]), desc=f'Processing dataset'):
        if augment == 's2':
            rotation_matrices = [random_rotation_matrix() for _ in range(60)]
        else:
            icosahedral_group = R.create_group('I')
            rotation_matrices = icosahedral_group.as_matrix()

        for i in range(60):
            rotation_matrix = rotation_matrices[i]
            rotated_grid = rotate_grid(rotation_matrix, grid)

            # Get the signals and compute projections onto the sphere.
            idxs = np.arange(60 * current, 60 * (current + 1))
            chunk = np.tile(signals[current], (60, 1, 1))
            projections[idxs] = project_2d_on_sphere(chunk, rotated_grid)

        if save_folder and current % 1000:
            with gzip.open(save_folder + f'./{current}.gz', mode='wb') as file:
                pickle.dump({'images': projections, 'labels': mnist['labels']}, file)

    if save_path:
        with gzip.open(save_path, mode='wb') as file:
            pickle.dump({'images': projections, 'labels': mnist['labels']}, file)

    return {'images': projections, 'labels': mnist['labels']}


# --- ICOSAHEDRAL DATASETS ---


def s2_signal_to_cartesian(signal):
    b = signal.shape[1] // 2

    theta = torch.linspace(0, torch.pi, 2 * b)
    phi = torch.linspace(0, 2 * torch.pi, 2 * b)
    theta, phi = torch.meshgrid(theta, phi)

    x = torch.sin(theta) * torch.cos(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(theta)

    return x.numpy(), y.numpy(), z.numpy(), signal.squeeze().numpy()


def s2_to_ico(s2_signal, ico_grid_coords):
    # To cartesian
    x, y, z, value = s2_signal_to_cartesian(s2_signal)

    # Flatten.
    coords = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T
    value = value.ravel()

    # KD-tree for nearest neighbour lookup.
    tree = cKDTree(coords)
    _, nearest_idxs = tree.query(ico_grid_coords.reshape(-1, 3))
    interpolated_value = value[nearest_idxs]

    # Reshape.
    output_array = interpolated_value.reshape(ico_grid_coords.shape[:-1])
    return np.expand_dims(np.expand_dims(output_array, axis=0), axis=0)  # [1, 1, 5, 2**r, 2**(r+1)].


# Old version.
def generate_ico_dataset(r, b, rotate_train, rotate_test, chunk_size, load_path=None, save_path=None):
    assert rotate_train in ['none', 'ico', 's2']
    assert rotate_test in ['none', 'ico', 's2']

    # Spherical dataset.
    if load_path is not None:
        with gzip.open(load_path, 'rb') as file:
            s2_dataset = pickle.load(file)
    else:
        s2_dataset = generate_dataset(b, rotate_train=='s2', rotate_test=='s2', chunk_size)

    # Icosahedral grid, in the Cartesian coordinate system.
    ico_grid_coords = icosahedral_grid_coordinates(r)

    dataset = {}

    # Project onto the icosahedron.
    for split in ['train', 'test']:
        # xs into tensors.
        xs = torch.from_numpy(
        s2_dataset[split]['images'][:, None, :, :].astype(np.float32))
        ys = s2_dataset[split]['labels'].astype(np.int64)

        ico_xs = []
        for x in tqdm(xs, desc=f'Projecting {split} onto Icosahedron'):
            # Project.
            ico_x = s2_to_ico(x, ico_grid_coords)

            # Rotate.
            if (split == 'train' and rotate_train == 'ico') or (split == 'test' and rotate_test == 'ico'):
                rotation_matrix = random_icosahedral_rotation_matrix()
                ico_x = rotate_signal(torch.tensor(ico_x), rotation_matrix, ico_grid_coords)

                # Back into numpy array.
                ico_x = ico_x.numpy()


            ico_xs.append(ico_x)

        dataset[split] = {'images': ico_xs, 'labels': ys}

    if save_path:
        with gzip.open(save_path, mode='wb') as file:
            pickle.dump(dataset, file)

    return dataset


def generate_ico_dataset(r, b, train, augment, chunk_size, load_path=None, save_path=None):
    assert augment in ['none', 'ico', 's2']

    # Spherical dataset.
    if load_path is not None:
        with gzip.open(load_path, 'rb') as file:
            s2_dataset = pickle.load(file)
    else:
        s2_dataset = generate_s2_dataset(b, train, augment, chunk_size)

    # Icosahedral grid, in the Cartesian coordinate system.
    ico_grid_coords = icosahedral_grid_coordinates(r)

    # xs into tensors.
    xs = torch.from_numpy(
        s2_dataset['images'][:, None, :, :].astype(np.float32)
    )

    ico_xs = []
    for x in tqdm(xs, desc=f'Projecting dataset'):
        # Project.
        ico_x = s2_to_ico(x, ico_grid_coords)
        ico_xs.append(ico_x)

    if save_path:
        with gzip.open(save_path, mode='wb') as file:
            pickle.dump({'images': ico_xs, 'labels': s2_dataset['labels'].astype(np.int64)}, file)

    return {'images': ico_xs, 'labels': s2_dataset['labels'].astype(np.int64)}


def load_datasets(path):
    # Read in data.
    with gzip.open(path, mode='rb') as file:
        dataset = pickle.load(file)

    # Extract training data.
    train_data = torch.from_numpy(
        dataset['train']['images'][:, None, :, :].astype(np.float32)
    )
    train_labels = torch.from_numpy(
        dataset['train']['labels'].astype(np.int64)
    )
    train_dataset = TensorDataset(train_data, train_labels)

    # Likewise for test data.
    test_data = torch.from_numpy(
        dataset['test']['images'][:, None, :, :].astype(np.float32)
    )
    test_labels = torch.from_numpy(
        dataset['test']['labels'].astype(np.int64)
    )
    test_dataset = TensorDataset(test_data, test_labels)

    return train_dataset, test_dataset
