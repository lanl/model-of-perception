import argparse
import os
import numpy as np
import pyvista as pv
import vtk


def compute_signed_distance(mesh, scalar_name, iso_val):
    surface = mesh.contour(isosurfaces=[iso_val], scalars=scalar_name)
    if surface.n_points == 0:
        raise RuntimeError(f"Isosurface at {iso_val} produced no points.")
    distance_func = vtk.vtkImplicitPolyDataDistance()
    distance_func.SetInput(surface)
    points = mesh.points
    return np.array([distance_func.EvaluateFunction(p) for p in points])


def compute_contours_and_signed_fields(mesh, scalar_name, iso_vals):
    contours = []
    signed_fields = []
    distance_funcs = []
    for iso_val in iso_vals:
        contour = mesh.contour(isosurfaces=[float(iso_val)], scalars=scalar_name)
        if contour.n_points == 0:
            raise RuntimeError(f"Isosurface at {iso_val} produced no points.")
        contours.append(contour)
        distance_func = vtk.vtkImplicitPolyDataDistance()
        distance_func.SetInput(contour)
        distance_funcs.append(distance_func)
        points = mesh.points
        signed = np.array([distance_func.EvaluateFunction(p) for p in points])
        signed_fields.append(signed)

    signed_fields = np.stack(signed_fields, axis=0)
    n_isos = len(contours)
    for i in range(n_isos):
        score = 0.0
        count = 0
        if i > 0 and contours[i - 1].n_points > 0:
            pts = contours[i - 1].points
            signs = np.sign([distance_funcs[i].EvaluateFunction(p) for p in pts])
            signs = signs[signs != 0]
            if signs.size > 0:
                score += float(np.mean(signs))
                count += 1
        if i < n_isos - 1 and contours[i + 1].n_points > 0:
            pts = contours[i + 1].points
            signs = np.sign([distance_funcs[i].EvaluateFunction(p) for p in pts])
            signs = signs[signs != 0]
            if signs.size > 0:
                score -= float(np.mean(signs))
                count += 1
        if count > 0 and score > 0:
            signed_fields[i] = -signed_fields[i]

    return contours, signed_fields


def extrapolate_from_iso(signed_distances, iso_val, scalar_range):
    out = np.empty_like(signed_distances)
    pos = signed_distances >= 0
    neg = ~pos
    if np.any(pos):
        max_pos = np.max(signed_distances[pos])
        if max_pos == 0:
            out[pos] = iso_val
        else:
            out[pos] = iso_val + (scalar_range[1] - iso_val) * (signed_distances[pos] / max_pos)
    if np.any(neg):
        min_neg = np.min(signed_distances[neg])
        if min_neg == 0:
            out[neg] = iso_val
        else:
            out[neg] = iso_val + (scalar_range[0] - iso_val) * (signed_distances[neg] / min_neg)
    return out


def compute_iso_vals(scalar_range, n_isos):
    if n_isos < 1:
        raise ValueError("n_isos must be >= 1")

    if n_isos == 1:
        return np.array([(scalar_range[0] + scalar_range[1]) / 2.0])
    else:
        min_val, max_val = scalar_range
        # return np.array((min_val + 0.01, max_val - 0.01))

        d = (max_val - min_val) / n_isos
        return min_val + 0.5 * d + np.arange(n_isos) * d


def reconstruct_from_multiple_isos(
    mesh,
    scalar_name,
    n_isos=None,
    iso_vals=None,
    contours=None,
    signed_fields=None,
    eps=1e-6,
):
    scalars = mesh.point_data[scalar_name]
    scalar_range = (float(np.min(scalars)), float(np.max(scalars)))
    if iso_vals is None:
        if n_isos is None:
            raise ValueError("Either n_isos or iso_vals must be provided.")
        iso_vals = compute_iso_vals(scalar_range, n_isos)
    iso_vals = np.asarray(iso_vals, dtype=float)
    n_isos = int(iso_vals.size)
    print(f"iso_vals: {iso_vals}")
    if signed_fields is None or contours is None:
        contours, signed_fields = compute_contours_and_signed_fields(mesh, scalar_name, iso_vals)

    if n_isos == 1:
        return extrapolate_from_iso(signed_fields[0], iso_vals[0], scalar_range)

    n_points = signed_fields.shape[1]
    neg_vals = np.where(signed_fields < 0, signed_fields, -np.inf)
    pos_vals = np.where(signed_fields > 0, signed_fields, np.inf)

    neg_idx = np.argmax(neg_vals, axis=0)
    pos_idx = np.argmin(pos_vals, axis=0)
    neg_val = neg_vals[neg_idx, np.arange(n_points)]
    pos_val = pos_vals[pos_idx, np.arange(n_points)]

    has_neg = neg_val != -np.inf
    has_pos = pos_val != np.inf
    bracket = has_neg & has_pos

    interp_vals = np.empty(n_points, dtype=float)

    # Exact hits on an isosurface
    zero_mask = np.any(signed_fields == 0, axis=0)
    if np.any(zero_mask):
        zero_idx = np.argmax(signed_fields == 0, axis=0)
        interp_vals[zero_mask] = iso_vals[zero_idx[zero_mask]]

    # Interpolate between nearest negative and positive fields
    if np.any(bracket):
        dn = -neg_val[bracket]
        dp = pos_val[bracket]
        denom = dn + dp + eps
        t = dn / denom
        v0 = iso_vals[neg_idx[bracket]]
        v1 = iso_vals[pos_idx[bracket]]
        interp_vals[bracket] = v0 * (1.0 - t) + v1 * t

    # Extrapolate outside range using outermost isosurfaces.
    min_iso = iso_vals[0]
    max_iso = iso_vals[-1]
    min_field = signed_fields[0]
    max_field = signed_fields[-1]
    extrap_min = extrapolate_from_iso(min_field, min_iso, scalar_range)
    extrap_max = extrapolate_from_iso(max_field, max_iso, scalar_range)

    below = (~has_pos) & (~zero_mask)
    above = (~has_neg) & (~zero_mask)
    if np.any(below):
        interp_vals[below] = extrap_min[below]
    if np.any(above):
        interp_vals[above] = extrap_max[above]

    interp_vals = np.clip(interp_vals, scalar_range[0], scalar_range[1])
    return interp_vals


def l2_objective(mesh, scalar_name, iso_vals):
    scalars = mesh.point_data[scalar_name]
    reconstructed = reconstruct_from_multiple_isos(mesh, scalar_name, iso_vals=iso_vals)
    return float(np.sqrt(np.mean((scalars - reconstructed) ** 2)))


def optimize_iso_vals_blackbox(mesh, scalar_name, n_isos, max_iters=8, samples_per_iter=12, seed=0):
    scalars = mesh.point_data[scalar_name]
    scalar_range = (float(np.min(scalars)), float(np.max(scalars)))
    iso_vals = compute_iso_vals(scalar_range, n_isos)
    min_val, max_val = scalar_range
    rng = np.random.default_rng(seed)

    step = (max_val - min_val) / max(n_isos, 1)
    best_iso = iso_vals.copy()
    best_l2 = l2_objective(mesh, scalar_name, best_iso)

    for it in range(max_iters):
        print(f"opt iter {it + 1}/{max_iters}: best L2 so far = {best_l2}")
        for s in range(samples_per_iter):
            proposal = best_iso + rng.normal(0.0, step, size=best_iso.shape)
            proposal = np.clip(proposal, min_val + 1e-9, max_val - 1e-9)
            proposal = np.sort(proposal)
            l2 = l2_objective(mesh, scalar_name, proposal)
            print(f"  sample {s + 1}/{samples_per_iter}: L2 = {l2}")
            if l2 < best_l2:
                best_l2 = l2
                best_iso = proposal
        print(f"opt iter {it + 1}/{max_iters}: done {samples_per_iter} samples, best L2 = {best_l2}")
        step *= 0.5

    return best_iso, best_l2


def main():
    parser = argparse.ArgumentParser(description="Reconstruct scalar field from multiple isocontours.")
    parser.add_argument("input_vti", help="Input VTI file")
    parser.add_argument("--scalar", default=None, help="Scalar array name (default: first array)")
    parser.add_argument("-n", "--num-isos", type=int, default=2, help="Number of isocontours")
    parser.add_argument("-o", "--output", default=None, help="Output VTI file")
    parser.add_argument("--contour-dir", default=None, help="Directory to save isocontours (.vtp)")
    parser.add_argument("--optimize-isos", action="store_true", help="Optimize iso values to minimize L2")
    parser.add_argument("--opt-iters", type=int, default=2, help="Optimization iterations")
    parser.add_argument("--opt-samples", type=int, default=5, help="Samples per iteration")
    parser.add_argument("--opt-seed", type=int, default=0, help="Random seed for optimization")
    args = parser.parse_args()

    mesh = pv.read(args.input_vti)
    if args.scalar is None:
        if mesh.point_data is None or len(mesh.point_data) == 0:
            raise RuntimeError("No point data arrays found in input VTI.")
        scalar_name = mesh.point_data.keys()[0]
    else:
        scalar_name = args.scalar

    out_dir = os.path.dirname(os.path.abspath(args.input_vti))
    if args.output is None:
        args.output = os.path.join(out_dir, "reconstructed_multiple.vti")
    if args.contour_dir is None:
        args.contour_dir = os.path.join(out_dir, "contours_vtp")

    scalars = mesh.point_data[scalar_name]
    scalar_range = (float(np.min(scalars)), float(np.max(scalars)))
    iso_vals = compute_iso_vals(scalar_range, args.num_isos)
    if args.optimize_isos:
        iso_vals, best_l2 = optimize_iso_vals_blackbox(
            mesh,
            scalar_name,
            args.num_isos,
            max_iters=args.opt_iters,
            samples_per_iter=args.opt_samples,
            seed=args.opt_seed,
        )
        print(f"optimized iso_vals: {iso_vals}")
        print(f"optimized L2: {best_l2}")

    print("computing contours and signed distance fields")
    contours, signed_fields = compute_contours_and_signed_fields(mesh, scalar_name, iso_vals)
    for i, iso_val in enumerate(iso_vals):
        if contours[i].n_points > 0:
            contours[i].point_data["iso_value"] = np.full(contours[i].n_points, float(iso_val))

    if args.contour_dir is not None:
        os.makedirs(args.contour_dir, exist_ok=True)
        print(f"contour output dir: {args.contour_dir}")
        merged = pv.merge(contours, merge_points=False)
        merged_path = os.path.join(args.contour_dir, "contours_merged.vtp")
        merged.save(merged_path)
        print(f"saved merged contours: {merged_path}")

    reconstructed = reconstruct_from_multiple_isos(
        mesh,
        scalar_name,
        iso_vals=iso_vals,
        contours=contours,
        signed_fields=signed_fields,
    )
    mesh.point_data["scalar_reconstructed"] = reconstructed
    mesh.save(args.output)
    print(f"saved reconstructed: {args.output}")
    l2_diff = np.sqrt(np.mean((scalars - reconstructed) ** 2))
    print(f"L2 difference: {l2_diff}")


if __name__ == "__main__":
    main()
