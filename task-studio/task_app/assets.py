"""Session assets: finite square scalar grids and explicit color lists."""
import hashlib
import io
import json
from pathlib import Path
import re

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from .resampling import downsample

BUILTIN = Path(__file__).resolve().parent / "builtin"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def validate_grid(value):
    raw = np.asarray(value)
    if raw.dtype.kind not in 'fiu' or raw.ndim != 2 or raw.shape[0] != raw.shape[1] or not 3 <= raw.shape[0] <= 128:
        raise ValueError('Data must be a square numeric grid, 3–128 cells per side.')
    a = raw.astype(float)
    if not np.isfinite(a).all() or np.max(np.abs(a)) > 1e100:
        raise ValueError('Grid values must be finite and have magnitude at most 1e100.')
    return a


def load_grid(filename, data):
    suffix = Path(filename).suffix.lower()
    if suffix == '.npy':
        # Validate dimensions before np.load can allocate from an untrusted header.
        stream = io.BytesIO(data)
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError('Use an NPY version 1 or 2 numeric array.')
        if len(shape) != 2 or shape[0] != shape[1] or not 3 <= shape[0] <= 128 or dtype.kind not in 'fiu':
            raise ValueError('Data must be a square numeric grid, 3–128 cells per side.')
        a = np.load(io.BytesIO(data), allow_pickle=False)
    elif suffix == '.csv': a = np.loadtxt(io.StringIO(data.decode('utf-8-sig')), delimiter=',')
    else: raise ValueError('Upload a .csv numeric matrix (no header) or a .npy array.')
    return validate_grid(a)


def load_colors(filename, data):
    suffix = Path(filename).suffix.lower()
    if suffix == '.json':
        value = json.loads(data)
        if isinstance(value, dict): value = value.get('colors')
    elif suffix == '.csv':
        value = np.loadtxt(io.StringIO(data.decode('utf-8-sig')), delimiter=',').tolist()
    else: raise ValueError('Upload .json colors or a numeric RGB/RGBA .csv (no header).')
    if not isinstance(value, list) or not 2 <= len(value) <= 4096:
        raise ValueError('Provide between 2 and 4096 colors.')
    if not all(isinstance(v, str) for v in value):
        rgb = np.asarray(value, dtype=float)
        if rgb.ndim != 2 or rgb.shape[1] not in (3, 4) or not np.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 255:
            raise ValueError('Colors must be RGB/RGBA rows in [0, 1] or [0, 255].')
        if rgb.max() > 1: rgb /= 255
        value = rgb.tolist()
    rgba = [list(to_rgba(v)) for v in value]
    if any(v[3] != 1 for v in rgba): raise ValueError('Use opaque colors (alpha 1 or 255).')
    return rgba


class Assets:
    def __init__(self, dataset_dir=None):
        self.datasets = {}
        self.maps = {}
        self.dataset_warnings = []
        source = BUILTIN
        coordinates = json.loads((source/'coordinates.json').read_text())
        x = np.asarray(coordinates['x_km'], dtype=float)
        y = np.asarray(coordinates['y_km'], dtype=float)
        spacing = float(x[1]-x[0])
        if not np.allclose(x,y) or not np.allclose(np.diff(x),spacing):
            raise ValueError('Built-in terrain requires matching regular x/y coordinates.')
        # The stored samples span 0..3; the display covers half a sample beyond
        # each edge, so existing sample-centered coordinate math stays exact.
        original = (source/'scalars.npy').read_bytes()
        values = downsample(np.load(io.BytesIO(original), allow_pickle=False))
        reduced = io.BytesIO(); np.save(reduced, values, allow_pickle=False)
        self.add_grid('Terrain (100 × 100)', reduced.getvalue(), 'scalars.npy',
                      [float(x[0]-spacing/2),float(x[-1]+spacing/2)], 'terrain_100',
                      provenance=dict(source_file='code/retrieve_value/runs/pilot_20261014/private/scalars.npy',
                                      source_shape=[101,101], prepared_shape=[100,100], source_sha256=digest(original),
                                      resampling='area-weighted cell averaging before color mapping; no value normalization'))
        for scene, name in [('s01', 'Monotone control'), ('s02', 'Plateaus and tied extrema')]:
            path = BUILTIN / f'{scene}.npy'
            self.add_grid(name, path.read_bytes(), path.name, [0, 3], scene)
        self.load_prepared(Path(dataset_dir) if dataset_dir is not None else Path(__file__).parent/'datasets')
        for name in ('gray', 'viridis', 'cividis', 'coolwarm', 'RdBu', 'turbo',
                     'jet', 'twilight', 'prism', 'gist_ncar', 'plasma', 'inferno', 'magma'):
            colors = colormaps[name](np.linspace(0, 1, 256)).tolist()
            self.maps[name] = dict(id=name, name=name, colors=colors, sha256=digest(json.dumps(colors).encode()))

    def load_prepared(self, folder):
        for path in sorted(folder.glob('*.json')):
            if path.name == 'download-report.json':
                continue
            try:
                meta = json.loads(path.read_text())
                ident = meta['id']
                if not re.fullmatch(r'[a-z][a-z0-9_]*', ident) or path.name != f'{ident}.json' or meta['file'] != f'{ident}.npy':
                    raise ValueError('Invalid prepared dataset filename or ID.')
                if ident in self.datasets:
                    raise ValueError('Prepared dataset ID collides with a built-in.')
                data = (folder/meta['file']).read_bytes()
                if digest(data) != meta['file_sha256']:
                    raise ValueError('Prepared dataset checksum mismatch.')
                if max(load_grid(meta['file'], data).shape) > 100:
                    raise ValueError('Prepared datasets must be at most 100 × 100.')
                self.add_grid(meta['name'], data, meta['file'], meta['extent'], ident,
                              provenance=meta['provenance'])
            except (OSError, ValueError, KeyError, TypeError) as exc:
                self.dataset_warnings.append(f'{path.name}: {exc}')

    def add_grid(self, name, data, filename, extent, ident=None, provenance=None):
        a = load_grid(filename, data)
        if len(extent) != 2 or not np.isfinite(extent).all() or not 1e-9 <= extent[1] - extent[0] <= 1e9 or max(abs(v) for v in extent) > 1e12:
            raise ValueError('Axes require finite increasing bounds and a span between 1e-9 and 1e9.')
        fingerprint = digest(a.tobytes() + json.dumps(extent).encode())
        ident = ident or 'data_' + fingerprint[:12]
        self.datasets[ident] = dict(id=ident, name=name[:100], values=a, extent=extent, sha256=fingerprint)
        if provenance is not None:
            self.datasets[ident]['source'] = provenance
        return ident

    def add_map(self, name, data, filename):
        colors = load_colors(filename, data)
        fingerprint = digest(json.dumps(colors).encode())
        ident = 'map_' + fingerprint[:12]
        self.maps[ident] = dict(id=ident, name=name[:100], colors=colors, sha256=fingerprint)
        return ident

    def cmap(self, ident):
        return LinearSegmentedColormap.from_list(ident, self.maps[ident]['colors'], N=256)

    def catalog(self):
        return dict(datasets=[{k: v for k, v in a.items() if k != 'values'} | {'shape': list(a['values'].shape)} for a in self.datasets.values()],
                    colormaps=list(self.maps.values()), dataset_warnings=self.dataset_warnings)
