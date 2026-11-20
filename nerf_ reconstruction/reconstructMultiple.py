import argparse
import os
import time
import numpy as np
import pyvista as pv
import torch
import torch.nn.functional as F
import vtk


def softmin_stack(values, tau):
    # values: torch tensor (K, D, H, W)
    return -tau * torch.logsumexp(-values / tau, dim=0)


def bellman_soft_distance(cost, dims, tau=0.25, iterations=120):
    shape = tuple(int(d) for d in dims)
    stride_f = (1, shape[0], shape[0] * shape[1])
    cost_grid = torch.as_strided(cost, size=shape, stride=stride_f)
    D = cost_grid.clone()
    for _ in range(iterations):
        neighbors = [cost_grid]
        for axis in range(3):
            plus = torch.empty_like(D)
            minus = torch.empty_like(D)
            if axis == 0:
                plus[:-1, :, :] = D[1:, :, :] + 1.0
                plus[-1, :, :] = D[-1, :, :] + 1.0
                minus[1:, :, :] = D[:-1, :, :] + 1.0
                minus[0, :, :] = D[0, :, :] + 1.0
            elif axis == 1:
                plus[:, :-1, :] = D[:, 1:, :] + 1.0
                plus[:, -1, :] = D[:, -1, :] + 1.0
                minus[:, 1:, :] = D[:, :-1, :] + 1.0
                minus[:, 0, :] = D[:, 0, :] + 1.0
            else:
                plus[:, :, :-1] = D[:, :, 1:] + 1.0
                plus[:, :, -1] = D[:, :, -1] + 1.0
                minus[:, :, 1:] = D[:, :, :-1] + 1.0
                minus[:, :, 0] = D[:, :, 0] + 1.0
            neighbors.append(plus)
            neighbors.append(minus)
        stacked = torch.stack(neighbors, dim=0)
        D = softmin_stack(stacked, tau)
    return torch.clamp(D.reshape(-1), min=0.0)


def _neighbor_average(field, dims):
    shape = tuple(int(d) for d in dims)
    # Fortran-order reshape using as_strided on torch
    if isinstance(field, np.ndarray):
        field_t = torch.from_numpy(field.astype(np.float32))
    else:
        field_t = field
    stride_f = (1, shape[0], shape[0] * shape[1])
    grid = torch.as_strided(field_t, size=shape, stride=stride_f)
    tensor = grid.unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
    padded = F.pad(tensor, (1, 1, 1, 1, 1, 1), mode="replicate")
    kernel = torch.zeros((1, 1, 3, 3, 3), dtype=torch.float32)
    kernel[0, 0, 1, 1, 1] = 1.0
    kernel[0, 0, 0, 1, 1] = 1.0
    kernel[0, 0, 2, 1, 1] = 1.0
    kernel[0, 0, 1, 0, 1] = 1.0
    kernel[0, 0, 1, 2, 1] = 1.0
    kernel[0, 0, 1, 1, 0] = 1.0
    kernel[0, 0, 1, 1, 2] = 1.0
    ones = torch.ones_like(tensor)
    ones_padded = F.pad(ones, (1, 1, 1, 1, 1, 1), mode="replicate")
    denominator = F.conv3d(ones_padded, kernel)
    numerator = F.conv3d(padded, kernel)
    smoothed = numerator / denominator
    return smoothed.squeeze(0).squeeze(0).reshape(-1)


def compute_iso_vals(scalar_range, n_isos):
    if n_isos < 1:
        raise ValueError("n_isos must be >= 1")
    if n_isos == 1:
        return np.array([(scalar_range[0] + scalar_range[1]) / 2.0])
    min_val, max_val = scalar_range
    d = (max_val - min_val) / n_isos
    return min_val + 0.5 * d + np.arange(n_isos) * d


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


def extrapolate_from_iso_torch(signed_distances, iso_val, scalar_range, eps=1e-6):
    out = torch.empty_like(signed_distances)
    pos = signed_distances >= 0
    neg = ~pos
    if pos.any():
        max_pos = torch.max(signed_distances[pos])
        out[pos] = torch.where(
            max_pos == 0,
            iso_val,
            iso_val + (scalar_range[1] - iso_val) * (signed_distances[pos] / (max_pos + eps)),
        )
    if neg.any():
        min_neg = torch.min(signed_distances[neg])
        out[neg] = torch.where(
            min_neg == 0,
            iso_val,
            iso_val + (scalar_range[0] - iso_val) * (signed_distances[neg] / (min_neg - eps)),
        )
    return out


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
    # Orient signed fields: negative points toward lower iso value
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


def reconstruct_from_multiple_isos_torch(signed_fields, iso_vals, scalar_range, eps=1e-6):
    k, n_points = signed_fields.shape
    device = signed_fields.device
    neg_inf = torch.tensor(-1e20, device=device, dtype=signed_fields.dtype)
    pos_inf = torch.tensor(1e20, device=device, dtype=signed_fields.dtype)

    neg_vals = torch.where(signed_fields < 0, signed_fields, neg_inf)
    pos_vals = torch.where(signed_fields > 0, signed_fields, pos_inf)

    neg_idx = torch.argmax(neg_vals, dim=0)
    pos_idx = torch.argmin(pos_vals, dim=0)
    neg_val = neg_vals.gather(0, neg_idx.unsqueeze(0)).squeeze(0)
    pos_val = pos_vals.gather(0, pos_idx.unsqueeze(0)).squeeze(0)

    has_neg = neg_val != neg_inf
    has_pos = pos_val != pos_inf
    bracket = has_neg & has_pos

    interp = torch.zeros(n_points, device=device, dtype=signed_fields.dtype)

    zero_mask = (signed_fields == 0).any(dim=0)
    if zero_mask.any():
        zero_idx = (signed_fields == 0).float().argmax(dim=0)
        interp[zero_mask] = iso_vals[zero_idx[zero_mask]]

    if bracket.any():
        dn = -neg_val[bracket]
        dp = pos_val[bracket]
        denom = dn + dp + eps
        t = dn / denom
        v0 = iso_vals[neg_idx[bracket]]
        v1 = iso_vals[pos_idx[bracket]]
        interp[bracket] = v0 * (1.0 - t) + v1 * t

    min_iso = iso_vals[0]
    max_iso = iso_vals[-1]
    min_field = signed_fields[0]
    max_field = signed_fields[-1]
    extrap_min = extrapolate_from_iso_torch(min_field, min_iso, scalar_range, eps)
    extrap_max = extrapolate_from_iso_torch(max_field, max_iso, scalar_range, eps)

    below = (~has_pos) & (~zero_mask)
    above = (~has_neg) & (~zero_mask)
    interp[below] = extrap_min[below]
    interp[above] = extrap_max[above]

    return torch.clamp(interp, scalar_range[0], scalar_range[1])


def reconstruct_from_multiple_isos_np(signed_fields, iso_vals, scalar_range, eps=1e-6):
    iso_vals = np.asarray(iso_vals, dtype=float)
    n_isos, n_points = signed_fields.shape

    neg_vals = np.where(signed_fields < 0, signed_fields, -np.inf)
    pos_vals = np.where(signed_fields > 0, signed_fields, np.inf)

    neg_idx = np.argmax(neg_vals, axis=0)
    pos_idx = np.argmin(pos_vals, axis=0)
    neg_val = neg_vals[neg_idx, np.arange(n_points)]
    pos_val = pos_vals[pos_idx, np.arange(n_points)]

    has_neg = neg_val != -np.inf
    has_pos = pos_val != np.inf
    bracket = has_neg & has_pos

    interp = np.empty(n_points, dtype=float)

    zero_mask = np.any(signed_fields == 0, axis=0)
    if np.any(zero_mask):
        zero_idx = np.argmax(signed_fields == 0, axis=0)
        interp[zero_mask] = iso_vals[zero_idx[zero_mask]]

    if np.any(bracket):
        dn = -neg_val[bracket]
        dp = pos_val[bracket]
        t = dn / (dn + dp + eps)
        v0 = iso_vals[neg_idx[bracket]]
        v1 = iso_vals[pos_idx[bracket]]
        interp[bracket] = v0 * (1.0 - t) + v1 * t

    extrap_min = extrapolate_from_iso(signed_fields[0], iso_vals[0], scalar_range)
    extrap_max = extrapolate_from_iso(signed_fields[-1], iso_vals[-1], scalar_range)
    below = (~has_pos) & (~zero_mask)
    above = (~has_neg) & (~zero_mask)
    if np.any(below):
        interp[below] = extrap_min[below]
    if np.any(above):
        interp[above] = extrap_max[above]

    return np.clip(interp, scalar_range[0], scalar_range[1])

# Alias for compatibility
def reconstruct_from_multiple_isos(signed_fields, iso_vals, scalar_range, eps=1e-6):
    return reconstruct_from_multiple_isos_np(signed_fields, iso_vals, scalar_range, eps)


import numpy as np
import torch

import numpy as np
import torch

def optimize_iso_vals_gd_4anchor_kernel_3d(
    rho, dims, iso_init, *,
    rho_min_anchor=None, rho_max_anchor=None,
    steps=0, lr=0.1,
    tau=0.5, power=2, eps=1e-8,
):
    """
    3D differentiable reconstruction using ONE consistent kernel over 4 anchors:
      [outer_min, iso0, iso1, outer_max]  (in rho/value space)

    Weights (per voxel) via softmax over distance to anchors:
        w_i = softmax( -|rho - a_i|^p / tau )   over i=0..3
    Reconstruction:
        recon = sum_i w_i * v_i

    Default "anchor values" v_i are set equal to anchor positions a_i
    (i.e. assumes f(a)=a, like your 1D demo). If you want different function
    values at the outer anchors, pass them by changing v0/v3 below.

    Args:
        rho: np.ndarray, shape (nx,ny,nz) scalar field samples
        dims: unused (kept for API compatibility)
        iso_init: array-like length 2 (two inner contours / iso values)
        rho_min_anchor: float, outer min anchor position (default=min(rho))
        rho_max_anchor: float, outer max anchor position (default=max(rho))
        steps: if >0, gradient-descent the two iso values (optional)
        lr: learning rate for iso optimization (only if steps>0)
        tau: kernel temperature/width (smaller -> sharper)
        power: 1 uses |d|, 2 uses d^2 (usually smoother)
        eps: numerical stability

    Returns:
        iso_vals_np: (2,) optimized (or initial if steps=0)
        recon_np: reconstructed field, same shape as rho
        weights_np: weights, shape (4,nx,ny,nz) in anchor order
                    [w_min, w_iso0, w_iso1, w_max]
    """
    rho_np = np.asarray(rho)
    rho_t = torch.tensor(rho_np, dtype=torch.float32)

    iso_t = torch.tensor(np.asarray(iso_init, dtype=np.float32), requires_grad=(steps > 0))

    # Outer anchors in rho-space
    a_min = float(rho_t.min()) if rho_min_anchor is None else float(rho_min_anchor)
    a_max = float(rho_t.max()) if rho_max_anchor is None else float(rho_max_anchor)

    def build_recon_and_weights(iso_vals):
        # anchors in rho-space (positions): [min, iso0, iso1, max]
        # enforce ordering for stability
        iso_sorted, _ = torch.sort(iso_vals)
        a = torch.stack([
            torch.tensor(a_min, dtype=rho_t.dtype, device=rho_t.device),
            iso_sorted[0],
            iso_sorted[1],
            torch.tensor(a_max, dtype=rho_t.dtype, device=rho_t.device),
        ], dim=0)  # (4,)

        # anchor "values" (what you blend)
        # For your 1D demo, these equal positions; keep that default here.
        v = a  # (4,)

        # distances (4, N)
        r = rho_t.reshape(1, -1)
        d = torch.abs(r - a.view(-1, 1))
        if power == 2:
            d = d * d

        logits = -d / (float(tau) + eps)
        W = torch.softmax(logits, dim=0)  # (4, N)

        recon = (W * v.view(-1, 1)).sum(dim=0)  # (N,)
        return recon, W, iso_sorted, a, v

    recon_out = None
    W_out = None

    if steps <= 0:
        with torch.no_grad():
            recon, W, iso_sorted, a, v = build_recon_and_weights(iso_t.detach())
        recon_out = recon.cpu().numpy().reshape(rho_np.shape)
        W_out = W.cpu().numpy().reshape((4,) + rho_np.shape)
        return iso_t.detach().cpu().numpy(), recon_out, W_out

    # Optional: GD optimize iso values to match rho (like your original function)
    for step in range(steps):
        recon, W, iso_sorted, a, v = build_recon_and_weights(iso_t)

        loss = torch.mean((rho_t.reshape(-1) - recon) ** 2)
        loss.backward()

        with torch.no_grad():
            iso_t -= lr * iso_t.grad
            # keep iso inside [a_min, a_max]
            iso_t.clamp_(min=a_min, max=a_max)
            iso_t.grad.zero_()

        print(f"GD step {step+1}/{steps}: loss={loss.item():.6g}, iso_vals={iso_t.detach().cpu().numpy()}")

        recon_out = recon.detach().cpu().numpy().reshape(rho_np.shape)
        W_out = W.detach().cpu().numpy().reshape((4,) + rho_np.shape)

    return iso_t.detach().cpu().numpy(), recon_out



def optimize_iso_vals_gd_simple(rho, dims, iso_init, steps=20, lr=0.1):
    """
    Simple differentiable proxy: build soft masks per iso, blend iso values,
    and gradient‑descent the iso positions to minimize L2 to the ground truth rho.
    """
    rho_t = torch.tensor(rho, dtype=torch.float32)
    iso_t = torch.tensor(iso_init, dtype=torch.float32, requires_grad=True)
    recon_out = None

    for step in range(steps):
        # Scale each iso's sigmoid so it spans the local value range.
        # scales shape: (k,)
        per_iso_scale = 10.0 / torch.clamp(torch.max(torch.abs(rho_t[:, None] - iso_t), dim=0).values, min=1e-6)

        soft_masks = []
        for j, iso_val in enumerate(iso_t):
            offset = rho_t - iso_val                           # signed offset to iso
            sig = torch.sigmoid(offset * per_iso_scale[j])     # in (0,1)

            # Symmetrize around 0.5 to avoid saturation issues.
            dev = sig - 0.5
            max_dev = torch.clamp(torch.max(torch.abs(dev)), min=1e-6)
            sym = 0.5 + torch.clamp(dev / max_dev, -1.0, 1.0)

            # Clamp to avoid zero gradients; smooth to spread support to neighbors.
            clipped = torch.clamp(sym, 1e-6, 1.0 - 1e-6)
            soft_masks.append(clipped)

        W = torch.stack(soft_masks, dim=0)                     # (k, N)
        weight_sum = torch.clamp(W.sum(dim=0, keepdim=True), min=1e-6)
        recon = (W * iso_t.view(-1, 1)).sum(dim=0) / weight_sum

        loss = torch.mean((rho_t - recon) ** 2)
        loss.backward()
        with torch.no_grad():
            iso_t -= lr * iso_t.grad
            iso_t.clamp_(min=float(torch.min(rho_t)), max=float(torch.max(rho_t)))
            iso_t.grad.zero_()

        print(f"GD step {step+1}/{steps}: loss={loss.item():.6g}, iso_vals={iso_t.detach().cpu().numpy()}")
        recon_out = recon.detach().cpu().numpy().reshape(rho.shape)

    return iso_t.detach().cpu().numpy(), recon_out


def optimize_iso_vals_gd_signed(rho, dims, iso_init, steps=20, lr=0.05):
    """GD variant that uses torch SDF reconstruction (reconstruct_from_multiple_isos_torch)."""
    rho_t = torch.tensor(rho, dtype=torch.float32)
    iso_t = torch.tensor(iso_init, dtype=torch.float32, requires_grad=True)
    recon_out = None
    scalar_range = (float(torch.min(rho_t)), float(torch.max(rho_t)))

    for step in range(steps):
        signed_fields = []
        for iso_val in iso_t:
            offset_t = rho_t - iso_val
            max_abs = torch.clamp(torch.max(torch.abs(offset_t)), min=1e-6)
            scale = 10.0 / max_abs
            sig_t = torch.sigmoid(offset_t * scale)
            dev = sig_t - 0.5
            max_dev = torch.clamp(torch.max(torch.abs(dev)), min=1e-6)
            sym_t = 0.5 + torch.clamp(dev / max_dev, -1.0, 1.0)
            clipped_t = torch.clamp(sym_t, 1e-6, 1.0 - 1e-6)
            smoothed_t = _neighbor_average(clipped_t, dims)
            smoothed_clipped_t = torch.clamp(smoothed_t, torch.min(clipped_t), torch.max(clipped_t))
            b_t = 4.0 * smoothed_clipped_t * (1.0 - smoothed_clipped_t)
            cost_t = 1000.0 * (1.0 - b_t)
            soft_dist_t = bellman_soft_distance(cost_t, dims, tau=0.25, iterations=120)
            sign_t = torch.tanh((smoothed_clipped_t - 0.5) * 20.0)
            signed_fields.append(sign_t * soft_dist_t)

        signed_stack = torch.stack(signed_fields, dim=0)  # (k, N)
        recon_t = reconstruct_from_multiple_isos_torch(signed_stack, iso_t, scalar_range)

        loss = torch.mean((rho_t - recon_t) ** 2)
        loss.backward()
        with torch.no_grad():
            iso_t -= lr * iso_t.grad
            iso_t.clamp_(min=float(torch.min(rho_t)), max=float(torch.max(rho_t)))
            iso_t.grad.zero_()
        print(f"GD signed step {step+1}/{steps}: loss={loss.item():.6g}, iso_vals={iso_t.detach().cpu().numpy()}")
        recon_out = recon_t.detach().cpu().numpy().reshape(rho.shape)

    return iso_t.detach().cpu().numpy(), recon_out


def main():
    parser = argparse.ArgumentParser(description="Produce sigmoid diagnostics and soft distances.")
    parser.add_argument("input_vti", help="Input VTI file")
    parser.add_argument("--scalar", default=None, help="Scalar array name (default: first array)")
    parser.add_argument("-n", "--num-isos", type=int, default=2, help="Number of isocontours")
    parser.add_argument("-o", "--output", default=None, help="Output VTI file")
    parser.add_argument("--contour-dir", default=None, help="Directory to save merged isocontours (.vtp)")
    parser.add_argument("--optimize-gd", action="store_true", help="Gradient-descent optimize iso values (differentiable proxy)")
    parser.add_argument("--gd-steps", type=int, default=100, help="GD steps for iso optimization")
    parser.add_argument("--gd-lr", type=float, default=0.1, help="GD learning rate for iso optimization")
    args = parser.parse_args()

    mesh = pv.read(args.input_vti)
    if args.scalar is None:
        if mesh.point_data is None or not mesh.point_data:
            raise RuntimeError("No scalar field found in the input VTI.")
        scalar_name = list(mesh.point_data.keys())[0]
    else:
        scalar_name = args.scalar

    out_dir = os.path.dirname(os.path.abspath(args.input_vti))
    if args.output is None:
        args.output = os.path.join(out_dir, "reconstructed_multiple.vti")
    if args.contour_dir is None:
        args.contour_dir = os.path.join(out_dir, "contours_vtp")

    scalars = mesh.point_data[scalar_name]
    rho = scalars.copy()
    mesh.point_data["rho"] = rho
    scalar_range = (float(np.min(rho)), float(np.max(rho)))
    print(f"scalar range: {scalar_range}")
    # iso_vals = compute_iso_vals(scalar_range, args.num_isos)
    iso_vals = scalar_range
    recon_field = None

    if args.optimize_gd:
        dims = mesh.dimensions
        if dims is None or len(dims) != 3:
            raise RuntimeError("Mesh dimensions are required for smoothing.")
        dims = tuple(int(d) for d in dims)
        t0 = time.time()
        # iso_vals, recon_field = optimize_iso_vals_gd_simple(rho, dims, iso_vals, steps=args.gd_steps, lr=args.gd_lr)
        # iso_vals, recon_field = optimize_iso_vals_gd_4anchor_kernel_3d(rho, dims, iso_vals, steps=args.gd_steps, lr=args.gd_lr)
        iso_vals, recon_field = optimize_iso_vals_gd_signed(rho, dims, iso_vals, steps=10, lr=args.gd_lr)
        print(f"optimized iso_vals (gd): {iso_vals} (elapsed {time.time()-t0:.3f}s)")
        if np.any(iso_vals <= scalar_range[0]) or np.any(iso_vals >= scalar_range[1]):
            print(f"Warning: optimized iso_vals {iso_vals} outside of scalar range {scalar_range}, clamping.")
            iso_vals = np.clip(iso_vals, scalar_range[0], scalar_range[1])
        if recon_field is not None and recon_field.size == rho.size:
            mesh.point_data["reconstructed_gd"] = recon_field

    # Nudge iso values away from exact endpoints to avoid empty contours
    lo, hi = scalar_range
    margin = 1e-6 * max(hi - lo, 1.0)
    iso_vals = np.clip(iso_vals, lo + margin, hi - margin)
    if np.any((iso_vals <= lo) | (iso_vals >= hi)):
        print(f"Warning: iso_vals {iso_vals} were clamped inside scalar range {scalar_range}")

    # Non-differentiable contours and signed distances (vtk-based)
    print(f"using iso_vals for contours/SDF: {iso_vals}")
    try:
        contours, signed_fields = compute_contours_and_signed_fields(mesh, scalar_name, iso_vals)
    except RuntimeError as e:
        iso_vals = np.clip(iso_vals, lo + 5 * margin, hi - 5 * margin)
        print(f"Retrying contours with iso_vals {iso_vals} because: {e}")
        contours, signed_fields = compute_contours_and_signed_fields(mesh, scalar_name, iso_vals)
    for i, iso_val in enumerate(iso_vals):
        mesh.point_data[f"signed_distance_{i:03d}"] = signed_fields[i]
        if contours[i].n_points > 0:
            contours[i].point_data["iso_value"] = np.full(contours[i].n_points, float(iso_val))

    reconstructed_gd_sdf = reconstruct_from_multiple_isos(signed_fields, iso_vals, scalar_range)
    mesh.point_data["reconstructed_gd_sdf"] = reconstructed_gd_sdf

    # Save merged contours
    os.makedirs(args.contour_dir, exist_ok=True)
    merged = pv.merge(contours, merge_points=False)
    merged_path = os.path.join(args.contour_dir, "contours_merged.vtp")
    merged.save(merged_path)
    print(f"saved merged contours: {merged_path}")

    # L2 of vtk-based reconstruction
    l2_vtk = float(np.sqrt(np.mean((rho - reconstructed_gd_sdf) ** 2)))
    print(f"L2 difference (vtk reconstruction): {l2_vtk}")

    mesh.save(args.output)
    arrays = list(mesh.point_data.keys())
    print(f"stored arrays: {arrays}")
    for name in arrays:
        arr = mesh.point_data[name]
        if arr is None or len(arr) == 0:
            continue
        print(f"  {name}: range=({np.min(arr):.6g}, {np.max(arr):.6g})")
    print(f"saved diagnostics: {args.output}")


if __name__ == "__main__":
    main()
