"""Nondegenerate saddles strictly inside regular-grid bilinear patches."""
import numpy as np


def bilinear_saddles(values, extent):
    """For f(u,v)=a+b*u+c*v+d*u*v, solve f_u=f_v=0.

    If d != 0, the Hessian determinant is -d**2 < 0. A solution with
    0 < u,v < 1 is an interior saddle. Grid-edge/vertex and degenerate
    stationary sets are intentionally excluded. Coordinates follow the same
    sample-centered geometry as task_app.interpolation.scalar_at.
    """
    field = np.asarray(values,dtype=float)
    lo,hi = extent
    if field.ndim != 2 or min(field.shape) < 2 or not np.isfinite(field).all():
        raise ValueError('Saddle detection requires a finite 2D scalar grid.')
    if not np.isfinite([lo,hi]).all() or hi <= lo:
        raise ValueError('Saddle detection requires increasing finite axis bounds.')
    # Scale the field to avoid unnecessary overflow in the patch coefficients.
    scale = float(np.max(np.abs(field)))
    if scale == 0: return []
    f = field/scale
    a = f[:-1,:-1]
    b = f[:-1,1:] - a
    c = f[1:,:-1] - a
    d = (f[1:,1:] - f[1:,:-1]) - b
    with np.errstate(divide='ignore',invalid='ignore',over='ignore'):
        u = -c/d; v = -b/d
    valid = (d != 0) & (u > 0) & (u < 1) & (v > 0) & (v < 1)
    dx,dy = (hi-lo)/field.shape[1],(hi-lo)/field.shape[0]
    result=[]
    for row,col in np.argwhere(valid):
        r,c0 = int(row),int(col)
        uu,vv = float(u[r,c0]),float(v[r,c0])
        result.append(dict(patch_row_col=[r,c0],patch_uv=[uu,vv],
                           xy=[lo+(c0+.5+uu)*dx,lo+(r+.5+vv)*dy],
                           value=float((a[r,c0]+b[r,c0]*uu+c[r,c0]*vv+d[r,c0]*uu*vv)*scale)))
    return result
