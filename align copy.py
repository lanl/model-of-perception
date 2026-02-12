#!/usr/bin/env python3
"""
Randomly rotate+translate a VTP mesh, then re-align it to the original using SVD (Kabsch).
Plots:
  (left)  original vs misaligned
  (right) original vs re-aligned
Also prints:
  - true rotation/translation
  - estimated rotation/translation
  - Chamfer distance before/after alignment

Usage:
  python align_vtp_svd.py --vtp mesh.vtp [--vtp2 mesh2.vtp] --seed 0 --tmax 10 --deg 45 --noise 0.0

Notes:
  - Assumes point-to-point correspondence (same mesh points/order). This is true if you
    transform the same mesh and then try to align it back.
  - Plotting uses pyvista if available. If not installed, script will still run but skip plots.
"""

import argparse
import math
import sys
import numpy as np

def load_vtp_points(vtp_path: str) -> np.ndarray:
    try:
        import vtk  # type: ignore
    except ImportError as e:
        raise RuntimeError("vtk is required to read .vtp files. Install with: pip install vtk") from e

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(vtp_path)
    reader.Update()
    poly = reader.GetOutput()
    n = poly.GetNumberOfPoints()
    pts = np.empty((n, 3), dtype=np.float64)
    for i in range(n):
        pts[i] = poly.GetPoint(i)
    return pts

def random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    # Uniform random rotation via random unit quaternion
    u1, u2, u3 = rng.random(3)
    q1 = math.sqrt(1 - u1) * math.sin(2 * math.pi * u2)
    q2 = math.sqrt(1 - u1) * math.cos(2 * math.pi * u2)
    q3 = math.sqrt(u1) * math.sin(2 * math.pi * u3)
    q4 = math.sqrt(u1) * math.cos(2 * math.pi * u3)
    x, y, z, w = q1, q2, q3, q4

    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),       2*(x*z + y*w)],
        [2*(x*y + z*w),         1 - 2*(x*x + z*z),   2*(y*z - x*w)],
        [2*(x*z - y*w),         2*(y*z + x*w),       1 - 2*(x*x + y*y)],
    ], dtype=np.float64)
    return R

def apply_rigid(P: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (P @ R.T) + t  # P is Nx3 row-vectors


def apply_similarity(P: np.ndarray, R: np.ndarray, t: np.ndarray, s: float) -> np.ndarray:
    """Uniform scale + rotation + translation: P' = s * P @ R.T + t."""
    return (s * (P @ R.T)) + t

def kabsch_svd(P: np.ndarray, Q: np.ndarray, allow_scale: bool = False):
    """
    Find R,t (and optional scale s) such that: P ≈ s * Q @ R^T + t
    (same convention as apply_similarity with s=1 when allow_scale=False).
    P: target, Q: source to be aligned.
    Returns (R, t, s) if allow_scale else (R, t, 1.0).
    """
    cP = P.mean(axis=0)
    cQ = Q.mean(axis=0)
    X = P - cP
    Y = Q - cQ
    H = Y.T @ X
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    # Ensure a proper rotation (det=+1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    if allow_scale:
        varY = np.sum(np.sum(Y * Y, axis=1))
        s = np.sum(S) / max(varY, 1e-12)
    else:
        s = 1.0
    t = cP - s * (cQ @ R.T)
    return R, t, s

def chamfer_distance(A: np.ndarray, B: np.ndarray) -> float:
    try:
        from scipy.spatial import cKDTree  # type: ignore
    except ImportError as e:
        raise RuntimeError("scipy is required for Chamfer distance. Install with: pip install scipy") from e

    treeA = cKDTree(A)
    treeB = cKDTree(B)
    dBA, _ = treeA.query(B, k=1)  # dist from each B to nearest A
    dAB, _ = treeB.query(A, k=1)  # dist from each A to nearest B
    # mean squared distances, symmetric
    return float(np.mean(dAB**2) + np.mean(dBA**2))

def rotation_angle_deg(R: np.ndarray) -> float:
    # angle from trace, numerically safe
    tr = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(tr)))

def fmt_mat(M: np.ndarray) -> str:
    return np.array2string(M, precision=6, suppress_small=False)

def plot_meshes(vtp_path: str, P: np.ndarray, Q_mis: np.ndarray, Q_aligned: np.ndarray):
    # Optional plotting with pyvista (best for meshes)
    try:
        import pyvista as pv  # type: ignore
    except ImportError:
        print("\n[plot] pyvista not installed -> skipping plots. Install with: pip install pyvista\n")
        return

    mesh = pv.read(vtp_path)
    mesh_orig = mesh.copy(deep=True)
    mesh_mis = mesh.copy(deep=True)
    mesh_aln = mesh.copy(deep=True)

    mesh_orig.points = P
    mesh_mis.points = Q_mis
    mesh_aln.points = Q_aligned

    pl = pv.Plotter(shape=(1, 2), window_size=(1400, 700))

    # Left: original vs misaligned
    pl.subplot(0, 0)
    pl.add_text("Original vs Misaligned", font_size=12)
    pl.add_mesh(mesh_orig, opacity=0.3, color="dodgerblue")
    pl.add_mesh(mesh_mis, opacity=0.3, color="tomato")
    pl.show_grid()

    # Right: original vs re-aligned (zoomed in for detail)
    pl.subplot(0, 1)
    pl.add_text("Original vs Re-aligned", font_size=12)
    pl.add_mesh(mesh_orig, opacity=0.3, color="dodgerblue")
    pl.add_mesh(mesh_aln, opacity=0.1, color="tomato")
    pl.show_grid()
    pl.camera.zoom(1.6)

    # Views are independent so we can zoom the right panel without affecting the left.
    pl.show()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vtp", required=True, help="Path to reference .vtp mesh (aligned target)")
    ap.add_argument("--vtp2", default=None, help="Optional second .vtp mesh to align to the first (skips synthetic transform)")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed")
    ap.add_argument("--tmax", type=float, default=10.0, help="Max abs translation per axis (uniform in [-tmax, tmax])")
    ap.add_argument("--deg", type=float, default=45.0, help="Max rotation angle in degrees (applied by slerp from identity)")
    ap.add_argument("--noise", type=float, default=0.0, help="Stddev of Gaussian noise added to misaligned points")
    ap.add_argument("--allow_scale", action="store_true", help="Estimate a uniform scale along with R,t")
    ap.add_argument("--no_plot", action="store_true", help="Disable plotting")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    P = load_vtp_points(args.vtp)

    if args.vtp2:
        # Use provided second mesh as the misaligned source
        R_true = None
        t_true = None
        Q_mis = load_vtp_points(args.vtp2)
        if args.noise > 0:
            Q_mis = Q_mis + rng.normal(scale=args.noise, size=Q_mis.shape)
    else:
        # Random rotation: draw uniform SO(3), then limit angle by interpolating from identity
        R_full = random_rotation_matrix(rng)
        ang = rotation_angle_deg(R_full)
        if ang < 1e-12:
            R_true = np.eye(3)
        else:
            # scale angle down to <= args.deg by moving toward identity
            alpha = min(1.0, args.deg / ang)
            # Rodrigues from axis-angle of R_full
            # Compute axis from skew-symmetric part
            w = np.array([
                R_full[2, 1] - R_full[1, 2],
                R_full[0, 2] - R_full[2, 0],
                R_full[1, 0] - R_full[0, 1],
            ]) / 2.0
            axis_norm = np.linalg.norm(w)
            if axis_norm < 1e-12:
                R_true = np.eye(3)
            else:
                axis = w / axis_norm
                theta = np.radians(ang * alpha)
                K = np.array([
                    [0, -axis[2], axis[1]],
                    [axis[2], 0, -axis[0]],
                    [-axis[1], axis[0], 0],
                ], dtype=np.float64)
                R_true = np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)

        t_true = rng.uniform(-args.tmax, args.tmax, size=3).astype(np.float64)

        Q_mis = apply_rigid(P, R_true, t_true)
        if args.noise > 0:
            Q_mis = Q_mis + rng.normal(scale=args.noise, size=Q_mis.shape)

    # Estimate alignment (align misaligned back to original)
    R_est_align, t_est_align, s_est_align = kabsch_svd(P, Q_mis, allow_scale=args.allow_scale)  # maps misaligned -> original
    Q_aligned = apply_similarity(Q_mis, R_est_align, t_est_align, s_est_align)

    # For comparison with the ground-truth forward transform (original -> misaligned),
    # take the inverse of the estimated alignment.
    s_est_fwd = 1.0 / s_est_align if abs(s_est_align) > 1e-12 else 0.0
    R_est_fwd = R_est_align.T
    t_est_fwd = (-t_est_align @ R_est_align) * s_est_fwd

    cd_before = chamfer_distance(P, Q_mis)
    cd_after = chamfer_distance(P, Q_aligned)

    # Print results
    if R_true is not None:
        print("=== True transform (original -> misaligned) ===")
        print(f"t_true = {t_true}")
        print(f"R_true =\n{fmt_mat(R_true)}")
        print(f"rotation_angle_true_deg ~ {rotation_angle_deg(R_true):.6f}")
    else:
        print("=== True transform ===")
        print("external pair provided; true transform unknown (only estimate reported)")

    print("\n=== Estimated transform (misaligned -> aligned-to-original) ===")
    print(f"t_est_align  = {t_est_align}")
    print(f"R_est_align  =\n{fmt_mat(R_est_align)}")
    print(f"s_est_align  = {s_est_align}")
    print(f"rotation_angle_est_deg ~ {rotation_angle_deg(R_est_align):.6f}")

    if args.vtp2 is None:
        print("\n=== Estimated transform (original -> misaligned, inverse of estimate) ===")
        print(f"t_est_fwd  = {t_est_fwd}")
        print(f"R_est_fwd  =\n{fmt_mat(R_est_fwd)}")
        print(f"s_est_fwd  = {s_est_fwd}")
        print(f"rotation_angle_est_fwd_deg ~ {rotation_angle_deg(R_est_fwd):.6f}")

    print("\n=== Chamfer distance (mean squared, symmetric) ===")
    print(f"before alignment: {cd_before:.12g}")
    print(f"after  alignment: {cd_after:.12g}")

    if not args.no_plot:
        plot_meshes(args.vtp, P, Q_mis, Q_aligned)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
