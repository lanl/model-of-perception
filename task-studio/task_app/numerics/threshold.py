"""Scalar definitions shared by ground truth and image-only threshold tools."""
import math

import numpy as np

from .extrema import offsets
from .search import snap

VARIANTS = (
    ('area', 'Threshold area', 'Retrieve'),
    ('fraction', 'Threshold area fraction', 'Retrieve'),
    ('largest', 'Find the largest component', 'Find'),
    ('component_count', 'Count threshold components', 'Count'),
    ('path_exists', 'Threshold path existence', 'Count'),
)


def boolean_mask(mask):
    result = np.asarray(mask)
    if result.ndim != 2 or not result.size or result.dtype != np.dtype(bool):
        raise ValueError('Expected a nonempty 2D Boolean mask')
    return result


def threshold(values, level, valid=None):
    a = np.asarray(values, float)
    if isinstance(level, (bool, np.bool_)) or not np.isscalar(level) or not math.isfinite(level):
        raise ValueError('Threshold must be finite')
    m = np.ones(a.shape, bool) if valid is None else boolean_mask(valid)
    if a.ndim != 2 or not a.size or a.shape != m.shape or not np.isfinite(a[m]).all():
        raise ValueError('Expected a finite 2D scalar support')
    return m & (a > level)


def cell_areas(shape, areas):
    a = np.asarray(areas, float)
    if a.ndim == 0:
        a = np.full(shape, float(a))
    if a.shape != shape or not np.isfinite(a).all() or np.any(a <= 0):
        raise ValueError('Cell areas must be finite, positive, and match the grid')
    return a


def area(mask, areas=1., reference=None):
    m = boolean_mask(mask)
    weights = cell_areas(m.shape, areas)
    result = math.fsum(weights[m])
    if reference is not None:
        ref = boolean_mask(reference)
        if ref.shape != m.shape or not ref.any() or np.any(m & ~ref):
            raise ValueError('Reference must contain the selection and have positive area')
        result /= math.fsum(weights[ref])
    return result


def components(mask, connectivity=4):
    """Label selected cells; background is zero, including an empty selection."""
    m = boolean_mask(mask)
    neighbors = offsets(connectivity)
    labels = np.zeros(m.shape, dtype=np.int32)
    count = 0
    for row, col in np.argwhere(m):
        if labels[row, col]:
            continue
        count += 1
        labels[row, col] = count
        todo = [(int(row), int(col))]
        while todo:
            y, x = todo.pop()
            for dy, dx in neighbors:
                yy, xx = y + dy, x + dx
                if (0 <= yy < m.shape[0] and 0 <= xx < m.shape[1]
                        and m[yy, xx] and not labels[yy, xx]):
                    labels[yy, xx] = count
                    todo.append((yy, xx))
    return labels


def component_areas(labels, areas=1.):
    a = np.asarray(labels)
    if a.ndim != 2 or not a.size or not np.issubdtype(a.dtype, np.integer) or np.any(a < 0):
        raise ValueError('Expected a nonnegative component-label grid')
    weights = cell_areas(a.shape, areas)
    return {int(i): math.fsum(weights[a == i]) for i in np.unique(a) if i}


def largest_components(labels, areas=1.):
    sizes = component_areas(labels, areas)
    if not sizes:
        return np.zeros_like(labels, dtype=bool)
    best = max(sizes.values())
    return np.isin(labels, [i for i, size in sizes.items() if size == best])


def endpoint(xy, n, extent):
    if (not isinstance(xy, (list, tuple, np.ndarray)) or len(xy) != 2
            or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.number)) for v in xy)):
        raise ValueError('Endpoint must be a finite coordinate pair')
    return snap(xy, n, extent)


def connected(labels, p, q):
    return bool(labels[p] != 0 and labels[p] == labels[q])


def ground_truth(field, specification, calibration):
    n = calibration['grid_cells']
    lo, hi = calibration['extent']
    if (type(n) is not int or n < 1 or np.shape(field) != (n, n)
            or not np.isfinite([lo, hi]).all() or hi <= lo):
        raise ValueError('Invalid calibrated square grid')
    weights = calibration.get('cell_areas', ((hi - lo) / n) ** 2)
    mask = threshold(field, specification['threshold'])
    kind = specification['kind']
    if kind in ('area', 'fraction'):
        ref = np.ones_like(mask) if kind == 'fraction' else None
        return dict(truth=area(mask, weights, ref))
    labels = components(mask, specification['connectivity'])
    if kind == 'component_count':
        return dict(truth=int(labels.max()))
    if kind == 'path_exists':
        p, q = (endpoint(specification[k], n, [lo, hi]) for k in ('p', 'q'))
        return dict(truth=connected(labels, p, q))
    if kind == 'largest':
        acceptable = largest_components(labels, weights)
        if not acceptable.any():
            raise ValueError('Largest-component tasks require a nonempty selection')
        return dict(acceptable_cells=np.argwhere(acceptable).tolist())
    raise ValueError('Unknown threshold task')


def prompt(kind, level, connectivity=4):
    offsets(connectivity)
    selection = f'cells with scalar value strictly greater than {level:g}'
    moves = ('horizontal or vertical edges, not diagonal contact' if connectivity == 4
             else 'horizontal, vertical, or diagonal contact')
    rule = f'Cells connect through {moves}. Stay inside the field; include components touching its boundary.'
    if kind == 'area':
        return (f'What is the total area occupied by {selection}? Sum their cell areas using the axes. '
                'Use squared coordinate units. Type one number; return 0 if no cells pass.')
    if kind == 'fraction':
        return (f'What fraction of the whole field area is occupied by {selection}? '
                'Divide their total cell area by the whole field area. Type a number from 0 to 1; return 0 if no cells pass.')
    if kind == 'largest':
        return (f'Click the center of any cell in the largest connected component of {selection}. '
                f'Largest means greatest total cell area. {rule} Any tied largest component is accepted. '
                'At least one cell is guaranteed to pass the threshold.')
    if kind == 'component_count':
        return (f'How many connected components consist of {selection}? {rule} '
                'Count each component once. Type an integer; return 0 if no cells pass.')
    if kind == 'path_exists':
        return (f'Is there a path from the cell marked P to the cell marked Q using only {selection}? '
                f'Use the cells containing the open centers of the markers. {rule} '
                'Answer true or false. If either endpoint fails the threshold, answer false.')
    raise ValueError('Unknown threshold task')
