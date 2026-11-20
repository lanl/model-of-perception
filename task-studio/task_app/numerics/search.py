"""Shared public search definitions. This module never opens an answer key."""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# Gradient searches remain available for archived specifications.
# The old single-cell gradient task
# remains supported as 'gradient' when reading archived specifications.
KINDS = ('value', 'rightmost', 'maximum', 'minimum', 'flattest', 'range', 'deviation', 'gradient_maximum',
         'leftmost', 'topmost', 'bottommost', 'range_minimum', 'gradient_minimum',
         'mean_gradient_minimum', 'mean_gradient_maximum', 'mean_minimum', 'mean_maximum',
         'min_gradient_minimum', 'min_gradient_maximum')
LABELS = ('Find value', 'Rightmost value', 'Maximum', 'Minimum', 'Minimum-SD window',
          'Maximum-range window', 'Maximum-SD window', 'Largest window-maximum gradient',
          'Leftmost value', 'Topmost value', 'Bottommost value', 'Minimum-range window', 'Smallest window-maximum gradient',
          'Smallest window-mean gradient', 'Largest window-mean gradient', 'Minimum-mean window', 'Maximum-mean window',
          'Smallest window-minimum gradient', 'Largest window-minimum gradient')
LEGACY_LABELS = {'gradient': 'Largest local gradient'}
CURRENT_KINDS = tuple(kind for kind in KINDS if 'gradient' not in kind)
WINDOW_SEARCHES = {'flattest': ('std', 'min'), 'deviation': ('std', 'max'),
                   'range_minimum': ('range', 'min'), 'range': ('range', 'max'),
                   'gradient_minimum': ('window_gradient', 'min'), 'gradient_maximum': ('window_gradient', 'max'),
                   'mean_gradient_minimum': ('window_gradient_mean', 'min'), 'mean_gradient_maximum': ('window_gradient_mean', 'max'),
                   'mean_minimum': ('mean', 'min'), 'mean_maximum': ('mean', 'max'),
                   'min_gradient_minimum': ('window_gradient_min', 'min'), 'min_gradient_maximum': ('window_gradient_min', 'max')}
# Array row zero is the smallest y coordinate (rendered with origin='lower').
# Values are (spatial axis: row=0/column=1, maximize=1/minimize=-1).
DIRECTIONS = {'rightmost': (1, 1), 'leftmost': (1, -1),
              'topmost': (0, 1), 'bottommost': (0, -1)}
TARGET_KINDS = ('value', *DIRECTIONS)


def coordinates(n, extent):
    lo, hi = extent
    axis = lo + (np.arange(n) + .5) * (hi-lo)/n
    x, y = np.meshgrid(axis, axis)
    return np.stack((x, y), axis=-1)


def snap(xy, n, extent):
    """Half-open cells, with outer upper border included. No clamping invalid replies."""
    xy = np.asarray(xy, float)
    lo, hi = extent
    if xy.shape != (2,) or not np.isfinite(xy).all() or np.any(xy < lo) or np.any(xy > hi):
        raise ValueError('Coordinates must be finite and inside the visible axes')
    scaled=np.nextafter((xy-lo)/((hi-lo)/n),np.inf)
    col, row = np.minimum(np.floor(scaled).astype(int), n-1)
    return int(row), int(col)


def derived_map(values, statistic, spacing, window_cells=5):
    """Leading dimensions represent joint observer integration nodes.

    Gradient at a cell = largest |difference|/spacing on its incident horizontal
    or vertical edges. Both endpoints of a strongest edge are valid answers.
    Window centers are cells; partial windows are excluded.
    """
    a = np.asarray(values, float)
    if a.ndim < 2 or a.shape[-1] != a.shape[-2] or not np.isfinite(a).all():
        raise ValueError('Expected a finite square scalar field')
    if a.ndim == 3 and len(a) > 256:
        chunks=[derived_map(a[i:i+256],statistic,spacing,window_cells) for i in range(0,len(a),256)]
        return np.concatenate([v for v,_ in chunks]),chunks[0][1]
    n = a.shape[-1]
    valid = np.ones((n,n), bool)
    if statistic == 'value':
        return a, valid
    if statistic == 'gradient':
        out = np.zeros_like(a)
        dx = np.abs(np.diff(a, axis=-1))/spacing
        dy = np.abs(np.diff(a, axis=-2))/spacing
        out[..., :, :-1] = np.maximum(out[..., :, :-1], dx)
        out[..., :, 1:] = np.maximum(out[..., :, 1:], dx)
        out[..., :-1, :] = np.maximum(out[..., :-1, :], dy)
        out[..., 1:, :] = np.maximum(out[..., 1:, :], dy)
        return out, valid
    w = window_cells
    if type(w) is not int or w < 3 or w % 2 != 1 or w > n:
        raise ValueError('Window must have an odd integer number of cells, from 3 to grid size')
    if statistic in ('window_gradient','window_gradient_mean','window_gradient_min'):
        # Composition: first derive the cell gradient field on the full support,
        # then pool that field. A gradient at a window edge may use a neighbor
        # outside the window; the window selects gradient values, not edges.
        gradients,_ = derived_map(a,'gradient',spacing)
        windows = sliding_window_view(gradients,(w,w),axis=(-2,-1))
    else:
        windows = sliding_window_view(a, (w,w), axis=(-2,-1))
    if statistic == 'std':
        out_small = windows.std(axis=(-2,-1), ddof=0)
    elif statistic == 'range':
        out_small = windows.max(axis=(-2,-1)) - windows.min(axis=(-2,-1))
    elif statistic == 'mean':
        out_small = windows.mean(axis=(-2,-1))
    elif statistic == 'min':
        out_small = windows.min(axis=(-2,-1))
    elif statistic == 'max':
        out_small = windows.max(axis=(-2,-1))
    elif statistic == 'window_gradient':
        out_small = windows.max(axis=(-2,-1))
    elif statistic == 'window_gradient_mean':
        out_small = windows.mean(axis=(-2,-1))
    elif statistic == 'window_gradient_min':
        out_small = windows.min(axis=(-2,-1))
    else:
        raise ValueError('Unknown derived statistic')
    r = w//2
    valid[:] = False
    valid[r:n-r,r:n-r] = True
    out = np.zeros_like(a)
    out[..., r:n-r,r:n-r] = out_small
    return out, valid


def directional_mask(matches, operation):
    """Retain all coordinate extrema in each matching set, including ties.

    Leading axes may represent joint observer fields. Empty matching sets stay
    empty so the caller can apply its documented no-match policy.
    """
    axis, sign = DIRECTIONS[operation]
    m = np.asarray(matches, bool)
    coordinate = np.indices(m.shape[-2:])[axis]
    score = sign * coordinate
    best = np.max(np.where(m, score, -np.inf), axis=(-2, -1), keepdims=True)
    return m & (score == best)


def search_weights(values, valid, operation, target=None, tolerance=0):
    """Distribution over tied winners for each input field, without truth access.

    Target search picks the closest reading. Directional searches first apply the
    tolerance, then maximize/minimize x or y. Among coordinate ties, select the
    closest scalar readings, preserving the existing rightmost policy. If no
    cells qualify, return closest readings over the selected support.
    All ties within floating-point precision share mass, with no index bias.
    """
    a = np.asarray(values, float)
    m = np.asarray(valid, bool)
    if m.shape != a.shape[-2:] or not m.any():
        raise ValueError('No valid search candidates')
    if operation == 'target' or operation in DIRECTIONS:
        if target is None or not np.isfinite(target) or not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError('Finite target and nonnegative tolerance required')
        loss = np.abs(a-target)
        if operation in DIRECTIONS:
            matches = m & (loss <= tolerance)
            qualifying = directional_mask(matches, operation)
            has_match = matches.any(axis=(-2,-1), keepdims=True)
            loss = np.where(has_match & ~qualifying, np.inf, loss)
    elif operation == 'max':
        loss = -a
    elif operation == 'min':
        loss = a
    else:
        raise ValueError('Unknown search operation')
    loss = np.where(m, loss, np.inf)
    best = np.min(loss, axis=(-2,-1), keepdims=True)
    finite_magnitude=np.max(np.where(np.isfinite(loss),np.abs(loss),0),axis=(-2,-1),keepdims=True)
    numerical_slack=16*np.finfo(float).eps*np.maximum(finite_magnitude,np.finfo(float).tiny)
    winners = np.isclose(loss, best, rtol=0, atol=numerical_slack)
    return winners / winners.sum(axis=(-2,-1), keepdims=True)


def specification(kind, target, window_cells=5, *, tolerance=0.):
    if kind not in KINDS and kind not in LEGACY_LABELS:
        raise ValueError('Unknown task kind')
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError('Tolerance must be finite and nonnegative')
    stat, op = WINDOW_SEARCHES.get(kind, ('gradient' if kind=='gradient' else 'value',
                              kind if kind in DIRECTIONS else {'value':'target','minimum':'min'}.get(kind,'max')))
    return dict(statistic=stat, operation=op, target=target if kind in TARGET_KINDS else None,
                tolerance=tolerance if kind in TARGET_KINDS else 0., window_cells=window_cells)


def ground_truth(values, spec, calibration):
    """Reference map, acceptable set and errors, evaluated only by generator/scorer."""
    n = values.shape[0]
    spacing = (calibration['extent'][1]-calibration['extent'][0])/n
    field, valid = derived_map(values, spec['statistic'], spacing, spec['window_cells'])
    op = spec['operation']
    horizontal = np.zeros_like(field)
    vertical = np.zeros_like(field)
    if op == 'target' or op in DIRECTIONS:
        regret = np.abs(field-spec['target'])
        acceptable = valid & (regret <= spec['tolerance'])
        if not acceptable.any():
            raise ValueError('Generated target has no acceptable solution')
        if op in DIRECTIONS:
            acceptable = directional_mask(acceptable, op)
            axis, _ = DIRECTIONS[op]
            coordinate = np.indices(field.shape)[axis]
            error = np.abs(coordinate - coordinate[acceptable][0])
            if axis == 1: horizontal = error
            else: vertical = error
        scale = float(np.ptp(values))
    else:
        optimum = float(field[valid].max() if op == 'max' else field[valid].min())
        regret = optimum-field if op == 'max' else field-optimum
        numerical_slack=16*np.finfo(float).eps*max(float(np.max(np.abs(field[valid]))),np.finfo(float).tiny)
        acceptable = valid & np.isclose(regret, 0, rtol=0, atol=numerical_slack)
        scale = float(np.ptp(field[valid]))
    points = np.argwhere(acceptable)
    yy,xx = np.indices(field.shape)
    distance = np.min(np.hypot(yy[...,None]-points[:,0],xx[...,None]-points[:,1]),axis=-1)
    return dict(field=field, valid=valid, acceptable=acceptable, distance=distance,
                regret=np.maximum(regret,0), normalized_regret=np.maximum(regret,0)/(scale if scale>0 else 1.),
                horizontal_error=horizontal, vertical_error=vertical, scale=scale)


def evaluate_distribution(probabilities, truth):
    p = np.asarray(probabilities,float)
    if p.shape != truth['field'].shape or not np.isfinite(p).all() or np.any(p<0) or not np.isclose(p.sum(),1):
        raise ValueError('Selection probabilities must be nonnegative and sum to one')
    valid_mass = float(p[truth['valid']].sum())
    # Invalid window centers remain failures; metrics are not silently clipped to a valid center.
    return dict(success=float(p[truth['acceptable']].sum()), valid_mass=valid_mass,
                distance_cells=float(np.sum(p*truth['distance'])),
                task_error=float(np.sum(p*truth['regret'])) if valid_mass > 1-1e-9 else None,
                normalized_task_error=float(np.sum(p*truth['normalized_regret'])) if valid_mass > 1-1e-9 else None,
                horizontal_error_cells=float(np.sum(p*truth['horizontal_error'])),
                vertical_error_cells=float(np.sum(p*truth.get('vertical_error',0))),
                directional_error_cells=float(np.sum(p*(truth['horizontal_error']+truth.get('vertical_error',0)))))
