#!/usr/bin/env python3
"""
Randomly rotate+translate a mesh, then re-align it to the original using SVD (Kabsch).
Plots:
  (left)  original vs misaligned
  (right) original vs re-aligned
Also prints:
  - true rotation/translation
  - estimated rotation/translation
  - Chamfer distance before/after alignment

Usage:
  python align_vtp_svd.py --vtp mesh.vtp [--vtp2 mesh2.glb] --seed 0 --tmax 10 --deg 45 --noise 0.0

Notes:
  - Assumes point-to-point correspondence (same mesh points/order). This is true if you
    transform the same mesh and then try to align it back.
  - Plotting uses pyvista if available. If not installed, script will still run but skip plots.
"""

import argparse
import math
import os
import sys
import time
from itertools import permutations
import numpy as np

def _read_mesh_as_polydata(mesh_path: str):
    try:
        import pyvista as pv  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "pyvista is required to read mesh files (.vtp/.glb). Install with: pip install pyvista"
        ) from e

    if mesh_path.lower().endswith(".vtp"):
        # Keep .vtp handling identical to the point-loading path to avoid ordering drift.
        try:
            import vtk  # type: ignore
        except ImportError as e:
            raise RuntimeError("vtk is required to read .vtp files. Install with: pip install vtk") from e
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(mesh_path)
        reader.Update()
        mesh = pv.wrap(reader.GetOutput())
        return mesh

    mesh = pv.read(mesh_path)

    # Preserve original surface connectivity from each block when possible.
    if isinstance(mesh, pv.MultiBlock):
        polys = []
        for block in mesh:
            if block is None:
                continue
            if isinstance(block, pv.PolyData):
                pd = block
            else:
                pd = block.extract_surface()
            if pd is None or pd.n_points == 0:
                continue
            polys.append(pd)
        if not polys:
            raise RuntimeError(f"No surface geometry found in {mesh_path}")
        if len(polys) == 1:
            mesh = polys[0].copy(deep=True)
        else:
            mesh = pv.append_polydata(polys)
    elif not isinstance(mesh, pv.PolyData):
        mesh = mesh.extract_surface()

    # Keep polygons as-is; avoid forced triangulation to prevent topology changes.
    if not isinstance(mesh, pv.PolyData) or mesh.n_points == 0:
        raise RuntimeError(f"Unsupported or empty surface mesh from {mesh_path}: {type(mesh)}")
    return mesh


def load_vtp_points(vtp_path: str) -> np.ndarray:
    # Use direct VTK path for .vtp to preserve exact point ordering/connectivity semantics.
    if vtp_path.lower().endswith(".vtp"):
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

    mesh = _read_mesh_as_polydata(vtp_path)
    pts = np.asarray(mesh.points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise RuntimeError(f"Unexpected points shape {pts.shape} from {vtp_path}")
    return pts


def _path_stem(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    stem = stem.strip()
    return stem if stem else "mesh"


def save_aligned_vtp(
    template_mesh_path: str,
    aligned_points: np.ndarray,
    out_dir_or_prefix: str,
    output_stem: str,
) -> str:
    """Save aligned points into a VTP mesh, reusing topology from template when possible."""
    # Direct VTK copy path for .vtp template preserves connectivity exactly.
    if template_mesh_path.lower().endswith(".vtp"):
        try:
            import vtk  # type: ignore
            from vtk.util import numpy_support  # type: ignore
        except ImportError as e:
            raise RuntimeError("vtk is required to write .vtp files. Install with: pip install vtk") from e

        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(template_mesh_path)
        reader.Update()
        template_poly = reader.GetOutput()

        pts = np.asarray(aligned_points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise RuntimeError(f"Expected aligned_points shape (N,3), got {pts.shape}")
        if template_poly.GetNumberOfPoints() != pts.shape[0]:
            raise RuntimeError(
                f"Template point count ({template_poly.GetNumberOfPoints()}) does not match "
                f"aligned points ({pts.shape[0]}). Cannot preserve connectivity."
            )

        out_poly = vtk.vtkPolyData()
        out_poly.DeepCopy(template_poly)
        vtk_pts = vtk.vtkPoints()
        vtk_pts.SetData(numpy_support.numpy_to_vtk(pts, deep=True))
        out_poly.SetPoints(vtk_pts)

        normalized = out_dir_or_prefix.rstrip(" .")
        if not normalized:
            normalized = "."
        if normalized.lower().endswith(".vtp"):
            normalized = normalized[:-4]
        out_dir = os.path.abspath(normalized)
        os.makedirs(out_dir, exist_ok=True)
        final_out = os.path.join(out_dir, f"{output_stem}_aligned.vtp")

        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(final_out)
        writer.SetInputData(out_poly)
        writer.SetDataModeToBinary()
        if writer.Write() != 1:
            raise RuntimeError(f"Failed to write {final_out}")
        return final_out

    try:
        import pyvista as pv  # type: ignore
    except ImportError as e:
        raise RuntimeError("pyvista is required to write .vtp files. Install with: pip install pyvista") from e

    mesh = _read_mesh_as_polydata(template_mesh_path)

    pts = np.asarray(aligned_points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise RuntimeError(f"Expected aligned_points shape (N,3), got {pts.shape}")

    if np.asarray(mesh.points).shape[0] != pts.shape[0]:
        raise RuntimeError(
            f"Template point count ({np.asarray(mesh.points).shape[0]}) does not match "
            f"aligned points ({pts.shape[0]}). Cannot preserve connectivity."
        )
    out_mesh = mesh.copy(deep=True)
    out_mesh.points = pts

    normalized = out_dir_or_prefix.rstrip(" .")
    if not normalized:
        normalized = "."
    # Treat argument as a directory/prefix and always generate filename + extension.
    # If user passes something ending in ".vtp", interpret it as a folder-like prefix.
    if normalized.lower().endswith(".vtp"):
        normalized = normalized[:-4]
    out_dir = os.path.abspath(normalized)
    os.makedirs(out_dir, exist_ok=True)
    final_out = os.path.join(out_dir, f"{output_stem}_aligned.vtp")
    out_mesh.save(final_out)
    return final_out

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


def _pca_axes(points: np.ndarray) -> np.ndarray:
    """Return (axes, singular values) via SVD of centered points."""
    centered = points - points.mean(axis=0)
    n = centered.shape[0]
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    S_norm = S / math.sqrt(max(n, 1))
    return Vt.T, S_norm


def pca_init_candidates(P: np.ndarray, Q: np.ndarray, allow_scale: bool):
    """
    Centroid alignment + PCA axes with sign disambiguation.
    Builds all 8 sign-flip x 6 axis-permutation initializations and returns them.
    """
    cP = P.mean(axis=0)
    cQ = Q.mean(axis=0)
    Qc = Q - cQ
    Pc = P - cP
    axes_P, sv_P = _pca_axes(Pc)
    axes_Q, sv_Q = _pca_axes(Qc)
    print("[init] singular values normalized by sqrt(N) to remove point-count dependence.")
    print("[init] centroid P:", cP)
    print("[init] centroid Q:", cQ)
    print("[init] PCA axes P (cols):\n", axes_P)
    print("[init] PCA axes Q (cols):\n", axes_Q)
    print("[init] PCA singular values P:", sv_P)
    print("[init] PCA singular values Q:", sv_Q)

    varQ = float(np.sum(sv_Q ** 2))  # normalized variance (independent of point count)
    varP = float(np.sum(sv_P ** 2))
    if allow_scale and varQ > 1e-12:
        s_base = math.sqrt(varP / varQ)  # scale maps Q -> P; sqrt of variance ratio
    else:
        s_base = 1.0
    print(f"[init] base scale guess (sqrt variance ratio): {s_base:.6g}")

    candidates = []

    for perm in permutations((0, 1, 2)):
        axes_Qp = axes_Q[:, perm]
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    S = np.diag([sx, sy, sz]).astype(np.float64)
                    R = axes_P @ S @ axes_Qp.T
                    if np.linalg.det(R) < 0:
                        # Keep correction in PCA/sign space (not world-space columns),
                        # otherwise permutations can produce inconsistent transforms.
                        S[2, 2] *= -1.0
                        R = axes_P @ S @ axes_Qp.T
                    s = s_base if allow_scale else 1.0
                    t = cP - s * (cQ @ R.T)
                    Q_guess = apply_similarity(Q, R, t, s)
                    cost = chamfer_distance(P, Q_guess)
                    print(
                        f"[init] perm={perm} signs=({sx},{sy},{sz}) "
                        f"det={np.linalg.det(R):.3f} chamfer={cost:.6g}"
                    )
                    candidates.append({
                        "perm": perm,
                        "signs": (sx, sy, sz),
                        "R0": R,
                        "t0": t,
                        "s0": s,
                        "init_cost": cost,
                    })
    return candidates


def icp_align(P: np.ndarray, Q: np.ndarray, R0: np.ndarray, t0: np.ndarray, s0: float, allow_scale: bool = False, max_iter: int = 30, tol: float = 1e-6):
    """
    Point-to-point ICP: iteratively find nearest neighbors from current transformed Q to P,
    then solve for best similarity (or rigid) transform.
    Returns R, t, s mapping Q -> P.
    """
    try:
        from scipy.spatial import cKDTree  # type: ignore
    except ImportError as e:
        raise RuntimeError("scipy is required for ICP (cKDTree). Install with: pip install scipy") from e

    R_tot = R0
    t_tot = t0
    s_tot = s0
    Q_init_vis = apply_similarity(Q, R0, t0, s0)

    treeP = cKDTree(P)
    prev_err = None

    n_iter = 0
    for it in range(max_iter):
        Q_curr = apply_similarity(Q, R_tot, t_tot, s_tot)
        dists, idx = treeP.query(Q_curr, k=1)
        P_match = P[idx]

        R_delta, t_delta, s_delta = kabsch_svd(P_match, Q_curr, allow_scale=allow_scale)
        if allow_scale:
            s_delta = float(max(s_delta, 1e-9))
        else:
            s_delta = 1.0

        # compose transforms: new = delta ∘ current
        R_tot = R_delta @ R_tot
        t_tot = s_delta * (t_tot @ R_delta.T) + t_delta
        s_tot = s_delta * s_tot

        err = float(np.mean(dists ** 2))
        if prev_err is not None and abs(prev_err - err) < tol:
            n_iter = it + 1
            break
        prev_err = err
        n_iter = it + 1

    return R_tot, t_tot, s_tot, Q_init_vis, n_iter

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
        print(f"[kabsch] scale from S={S} varY={varY:.6g} -> s={s:.6g}")
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

def plot_meshes(
    ref_path: str,
    src_path: str,
    P: np.ndarray,
    Q_mis: np.ndarray,
    trials: list[dict],
    save_dir: str | None = None,
    output_stem: str = "mesh",
):
    # Optional plotting with pyvista (best for meshes)
    try:
        import pyvista as pv  # type: ignore
    except ImportError:
        print("\n[plot] pyvista not installed -> skipping plots. Install with: pip install pyvista\n")
        return

    mesh_ref = _read_mesh_as_polydata(ref_path)
    mesh_src = _read_mesh_as_polydata(src_path)
    if len(trials) == 0:
        raise RuntimeError("No ICP trials to plot")
    if mesh_ref.n_points != P.shape[0]:
        raise RuntimeError(
            f"Reference mesh points ({mesh_ref.n_points}) != loaded reference points ({P.shape[0]})."
        )
    if mesh_src.n_points != Q_mis.shape[0]:
        raise RuntimeError(
            f"Source mesh points ({mesh_src.n_points}) != loaded source points ({Q_mis.shape[0]})."
        )

    # First plot: both inputs with PCA/SVD principal vectors.
    # Also show Q's tripod mapped by the best transform so alignment is visible.
    best_tr = min(trials, key=lambda x: x["cd_after"])
    cP = P.mean(axis=0)
    cQ = Q_mis.mean(axis=0)
    axes_P, sv_P = _pca_axes(P)
    axes_Q, sv_Q = _pca_axes(Q_mis)
    R_best = best_tr["R_est_align"]
    t_best = best_tr["t_est_align"]
    s_best = best_tr["s_est_align"]
    cQ_map = s_best * (cQ @ R_best.T) + t_best
    axes_Q_map = R_best @ axes_Q
    bounds = np.vstack([P, Q_mis])
    diag = float(np.linalg.norm(bounds.max(axis=0) - bounds.min(axis=0)))
    base_len = max(diag * 0.18, 1e-6)
    len_P = base_len * (sv_P / max(float(np.max(sv_P)), 1e-12))
    len_Q = base_len * (sv_Q / max(float(np.max(sv_Q)), 1e-12))

    save_mode = save_dir is not None
    if save_mode:
        save_dir = os.path.abspath(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        print(f"[plot] output dir: {save_dir}")

    pl0 = pv.Plotter(window_size=(1400, 900), off_screen=save_mode)
    pl0.add_text("Inputs + PCA/SVD vectors (raw + mapped Q)", font_size=14)
    mesh_orig = mesh_ref.copy(deep=True)
    mesh_src_vis = mesh_src.copy(deep=True)
    mesh_orig.points = P
    mesh_src_vis.points = Q_mis
    # pl0.add_mesh(mesh_orig, opacity=0.08, color="dodgerblue")
    # pl0.add_mesh(mesh_src, opacity=0.08, color="tomato")

    p_colors = ["royalblue", "deepskyblue", "navy"]
    q_colors = ["orangered", "gold", "firebrick"]
    q_map_colors = ["limegreen", "springgreen", "seagreen"]
    for i in range(3):
        arr_p = pv.Arrow(start=cP, direction=axes_P[:, i], scale=float(len_P[i]))
        arr_q = pv.Arrow(start=cQ, direction=axes_Q[:, i], scale=float(len_Q[i]))
        arr_qm = pv.Arrow(start=cQ_map, direction=axes_Q_map[:, i], scale=float(len_Q[i]))
        pl0.add_mesh(arr_p, color=p_colors[i])
        pl0.add_mesh(arr_q, color=q_colors[i])
        pl0.add_mesh(arr_qm, color=q_map_colors[i])
    pl0.show_grid()
    if save_mode:
        fig1_path = os.path.join(save_dir, f"{output_stem}_01_inputs_pca_vectors.png")
        pl0.show(screenshot=fig1_path, auto_close=True)
        print(f"[plot] saved: {fig1_path}")
    else:
        pl0.show()
    # sys.exit()

    # Second plot: only the best run.
    best_tr = min(trials, key=lambda x: x["cd_after"])
    signs = best_tr["signs"]
    perm = best_tr.get("perm", (0, 1, 2))
    label = f"best perm={perm} signs={signs}"
    pl = pv.Plotter(shape=(3, 1), window_size=(1200, 1400), off_screen=save_mode)
    cP = P.mean(axis=0)
    mis_cent_err = float(np.linalg.norm(Q_mis.mean(axis=0) - cP))
    init_cent_err = float(np.linalg.norm(best_tr["Q_init_vis"].mean(axis=0) - cP))
    aln_cent_err = float(np.linalg.norm(best_tr["Q_aligned"].mean(axis=0) - cP))

    # Row 0: original vs misaligned.
    pl.subplot(0, 0)
    mesh_orig = mesh_ref.copy(deep=True)
    mesh_mis = mesh_src.copy(deep=True)
    mesh_orig.points = P
    mesh_mis.points = Q_mis
    pl.add_text(f"{label}\nmisaligned cent_err={mis_cent_err:.3g}", font_size=11)
    pl.add_mesh(mesh_orig, opacity=0.25, color="dodgerblue")
    pl.add_mesh(mesh_mis, opacity=0.25, color="tomato")
    pl.show_grid()

    # Row 1: original vs initialization of best run.
    pl.subplot(1, 0)
    mesh_orig = mesh_ref.copy(deep=True)
    mesh_init = mesh_src.copy(deep=True)
    mesh_orig.points = P
    mesh_init.points = best_tr["Q_init_vis"]
    pl.add_text(f"{label}\ninitial cent_err={init_cent_err:.3g}", font_size=11)
    pl.add_mesh(mesh_orig, opacity=0.25, color="dodgerblue")
    pl.add_mesh(mesh_init, opacity=0.25, color="goldenrod")
    pl.show_grid()

    # Row 2: original vs final aligned of best run.
    pl.subplot(2, 0)
    mesh_orig = mesh_ref.copy(deep=True)
    mesh_aln = mesh_src.copy(deep=True)
    mesh_orig.points = P
    mesh_aln.points = best_tr["Q_aligned"]
    pl.add_text(
        f"{label}\naligned cd={best_tr['cd_after']:.3g} cent_err={aln_cent_err:.3g}",
        font_size=11,
    )
    pl.add_mesh(mesh_orig, opacity=0.25, color="dodgerblue")
    pl.add_mesh(mesh_aln, opacity=0.25, color="tomato")
    pl.show_grid()
    pl.camera.zoom(1.4)
    pl.link_views()

    if save_mode:
        fig2_path = os.path.join(save_dir, f"{output_stem}_02_best_alignment_panels.png")
        pl.show(screenshot=fig2_path, auto_close=True)
        print(f"[plot] saved: {fig2_path}")
    else:
        pl.show()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vtp", required=True, help="Path to reference mesh (.vtp/.glb, aligned target)")
    ap.add_argument("--vtp2", default=None, help="Optional second mesh (.vtp/.glb) to align to the first (skips synthetic transform)")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed")
    ap.add_argument("--tmax", type=float, default=10.0, help="Max abs translation per axis (uniform in [-tmax, tmax])")
    ap.add_argument("--deg", type=float, default=45.0, help="Max rotation angle in degrees (applied by slerp from identity)")
    ap.add_argument("--noise", type=float, default=0.0, help="Stddev of Gaussian noise added to misaligned points")
    ap.add_argument("--allow_scale", action="store_true", help="Estimate a uniform scale along with R,t")
    ap.add_argument(
        "--icp_mode",
        choices=["all", "best_init"],
        default="all",
        help="Run ICP from all PCA/SVD initializations, or only from the best initial Chamfer one",
    )
    ap.add_argument("--plot", dest="plot", action="store_true", help="Enable plotting")
    ap.add_argument("--no_plot", dest="plot", action="store_false", help="Disable plotting (alias)")
    ap.add_argument(
        "--save_plots_dir",
        default=None,
        help="If set, save plot images to this directory instead of showing plot windows",
    )
    ap.add_argument(
        "--save_aligned_vtp",
        default=None,
        help="If set, save the best aligned source mesh to <this_path>/aligned_best.vtp",
    )
    ap.set_defaults(plot=False)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    P = load_vtp_points(args.vtp)

    if args.vtp2:
        # Use provided second mesh as the misaligned source
        R_true = None
        t_true = None
        s_true = None
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
        s_true = 1.0

        Q_mis = apply_similarity(P, R_true, t_true, s_true)
        if args.noise > 0:
            Q_mis = Q_mis + rng.normal(scale=args.noise, size=Q_mis.shape)

    unaligned_source_path = args.vtp2 if args.vtp2 else args.vtp
    output_stem = _path_stem(unaligned_source_path)

    # Build all PCA/SVD initializations and run ICP from each
    candidates = pca_init_candidates(P, Q_mis, args.allow_scale)
    if args.icp_mode == "best_init":
        candidates_to_run = [min(candidates, key=lambda c: c["init_cost"])]
        print(
            "\n[icp] mode=best_init -> running only "
            f"perm={candidates_to_run[0]['perm']} signs={candidates_to_run[0]['signs']} "
            f"init_chamfer={candidates_to_run[0]['init_cost']:.12g}"
        )
    else:
        candidates_to_run = candidates
        print(f"\n[icp] mode=all -> running {len(candidates_to_run)} initializations")

    trials = []
    for i, cand in enumerate(candidates_to_run):
        t_start = time.time()
        R_est_align_i, t_est_align_i, s_est_align_i, Q_init_vis_i, n_iter_i = icp_align(
            P,
            Q_mis,
            cand["R0"],
            cand["t0"],
            cand["s0"],
            allow_scale=args.allow_scale,
        )
        t_icp_i = time.time() - t_start
        Q_aligned_i = apply_similarity(Q_mis, R_est_align_i, t_est_align_i, s_est_align_i)
        cd_after_i = chamfer_distance(P, Q_aligned_i)
        trials.append({
            "idx": i,
            "perm": cand["perm"],
            "signs": cand["signs"],
            "init_cost": cand["init_cost"],
            "R_est_align": R_est_align_i,
            "t_est_align": t_est_align_i,
            "s_est_align": s_est_align_i,
            "Q_init_vis": Q_init_vis_i,
            "Q_aligned": Q_aligned_i,
            "cd_after": cd_after_i,
            "icp_time": t_icp_i,
            "n_iter": n_iter_i,
        })

    best_trial = min(trials, key=lambda x: x["cd_after"])
    R_est_align = best_trial["R_est_align"]
    t_est_align = best_trial["t_est_align"]
    s_est_align = best_trial["s_est_align"]
    Q_init_vis = best_trial["Q_init_vis"]
    Q_aligned = best_trial["Q_aligned"]
    best_signs = best_trial["signs"]
    t_icp = best_trial["icp_time"]

    # For comparison with the ground-truth forward transform (original -> misaligned),
    # take the inverse of the estimated alignment.
    if args.vtp2 is None:
        s_est_fwd = 1.0 / s_est_align if abs(s_est_align) > 1e-12 else 0.0
        R_est_fwd = R_est_align.T
        t_est_fwd = (-t_est_align @ R_est_align) * s_est_fwd

    cd_before = chamfer_distance(P, Q_mis)
    cd_after = best_trial["cd_after"]

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
    print(f"best PCA sign combo: {best_signs}")
    print(f"rotation_angle_est_deg ~ {rotation_angle_deg(R_est_align):.6f}")

    print("\n=== ICP runs executed ===")
    for tr in trials:
        print(
            f"run {tr['idx']}: perm={tr.get('perm', (0, 1, 2))} signs={tr['signs']} "
            f"init_chamfer={tr['init_cost']:.12g} "
            f"final_chamfer={tr['cd_after']:.12g} "
            f"iters={tr['n_iter']} time={tr['icp_time']:.3f}s"
        )
    print(
        f"best run: {best_trial['idx']} "
        f"perm={best_trial.get('perm', (0, 1, 2))} "
        f"signs={best_trial['signs']} "
        f"final_chamfer={best_trial['cd_after']:.12g}"
    )

    if args.vtp2 is None:
        print("\n=== Estimated transform (original -> misaligned, inverse of estimate) ===")
        print(f"t_est_fwd  = {t_est_fwd}")
        print(f"R_est_fwd  =\n{fmt_mat(R_est_fwd)}")
        print(f"s_est_fwd  = {s_est_fwd}")
        print(f"rotation_angle_est_fwd_deg ~ {rotation_angle_deg(R_est_fwd):.6f}")

    print("\n=== Chamfer distance (mean squared, symmetric) ===")
    print(f"before alignment: {cd_before:.12g}")
    print(f"after  alignment: {cd_after:.12g}")
    print(f"ICP time: {t_icp:.3f} s")

    if args.save_aligned_vtp:
        template_path = args.vtp2 if args.vtp2 else args.vtp
        saved_path = save_aligned_vtp(template_path, Q_aligned, args.save_aligned_vtp, output_stem)
        print(f"saved aligned mesh: {saved_path}")

    if args.plot or args.save_plots_dir:
        source_for_plot = args.vtp2 if args.vtp2 else args.vtp
        plot_meshes(args.vtp, source_for_plot, P, Q_mis, trials, save_dir=args.save_plots_dir, output_stem=output_stem)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
