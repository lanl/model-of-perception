"""Bilinear scalar interpolation on the app's regular, sample-centered grids.

Interpolate BEFORE applying the color lookup table. This follows PyVista's
interpolate_before_map principle; it is not a replacement for VTK's general
mesh-specific interpolation (e.g. barycentric interpolation on triangles).
"""
import numpy as np

INTERPOLATION = {
    'method': 'bilinear',
    'stage': 'scalar_before_colormap',
    'sample_locations': 'grid_cell_centers',
    'boundary': 'clamp to nearest sample in the outer half-cell',
    'render_samples_per_axis': 768,
}


def scalar_at(values, extent, xy):
    """Evaluate scalar values at (..., 2) x/y locations inside the visible axes."""
    values = np.asarray(values, dtype=float)
    xy = np.asarray(xy, dtype=float)
    lo, hi = extent
    if xy.shape[-1:] != (2,) or not np.isfinite(xy).all() or np.any(xy < lo) or np.any(xy > hi):
        raise ValueError('Locations must be finite [x, y] coordinates inside the field.')
    h, w = values.shape
    x = np.clip((xy[..., 0]-lo)/(hi-lo)*w-.5, 0, w-1)
    y = np.clip((xy[..., 1]-lo)/(hi-lo)*h-.5, 0, h-1)
    # Preserve exactly sampled nodes despite floating-point coordinate roundoff.
    x = np.where(np.abs(x-np.rint(x)) < 1e-12, np.rint(x), x)
    y = np.where(np.abs(y-np.rint(y)) < 1e-12, np.rint(y), y)
    c0, r0 = np.floor(x).astype(int), np.floor(y).astype(int)
    c1, r1 = np.minimum(c0+1,w-1), np.minimum(r0+1,h-1)
    u, v = x-c0, y-r0
    bottom = (1-u)*values[r0,c0] + u*values[r0,c1]
    top = (1-u)*values[r1,c0] + u*values[r1,c1]
    return (1-v)*bottom + v*top


def scalar_raster(values, extent, resolution=INTERPOLATION['render_samples_per_axis']):
    """A display-only scalar raster; never use its samples for region statistics."""
    lo, hi = extent
    axis = lo+(np.arange(resolution)+.5)*(hi-lo)/resolution
    x,y = np.meshgrid(axis,axis)
    return scalar_at(values,extent,np.stack((x,y),axis=-1))
