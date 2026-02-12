import argparse
import os
import numpy as np
import pyvista as pv
import torch
import torch.nn.functional as F


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


def optimize_iso_vals_gd_simple(rho, dims, iso_init, steps=20, lr=0.1):
    rho_t = torch.tensor(rho, dtype=torch.float32)
    iso_t = torch.tensor(iso_init, dtype=torch.float32, requires_grad=True)
    recon_out = None
    for step in range(steps):
        # compute sigmoids for each iso
        scales = 10.0 / torch.clamp(torch.max(torch.abs(rho_t[:, None] - iso_t), dim=0).values, min=1e-6)
        weights = []
        for j, iso_val in enumerate(iso_t):
            offset_t = rho_t - iso_val
            sig_t = torch.sigmoid(offset_t * scales[j])
            dev = sig_t - 0.5
            max_dev = torch.clamp(torch.max(torch.abs(dev)), min=1e-6)
            sym_t = 0.5 + torch.clamp(dev / max_dev, -1.0, 1.0)
            clipped_t = torch.clamp(sym_t, 1e-6, 1.0 - 1e-6)
            smoothed_t = _neighbor_average(clipped_t, dims)
            smoothed_clipped_t = torch.clamp(smoothed_t, torch.min(clipped_t), torch.max(clipped_t))
            weights.append(smoothed_clipped_t)
        W = torch.stack(weights, dim=0)  # (k, N)
        norm = torch.clamp(W.sum(dim=0, keepdim=True), min=1e-6)
        recon = (W * iso_t.view(-1, 1)).sum(dim=0) / norm
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
    rho_t = torch.tensor(rho, dtype=torch.float32)
    iso_t = torch.tensor(iso_init, dtype=torch.float32, requires_grad=True)
    stride_f = (1, dims[0], dims[0] * dims[1])
    for step in range(steps):
        signed_fields = []
        # build signed distance proxies per iso
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
            b_t = torch.minimum(b_t, torch.tensor(0.5, dtype=b_t.dtype))
            cost_t = 1000.0 * (1.0 - b_t)
            soft_dist_t = bellman_soft_distance(cost_t, dims, tau=0.25, iterations=120)
            sign_t = torch.tanh((smoothed_clipped_t - 0.5) * 20.0)
            signed_np = (sign_t * soft_dist_t).numpy()
            signed_fields.append(signed_np)

        signed_stack = np.stack(signed_fields, axis=0)
        scalar_range = (float(torch.min(rho_t)), float(torch.max(rho_t)))
        recon_np = reconstruct_from_multiple_isos(signed_stack, iso_t.detach().cpu().numpy(), scalar_range)
        recon_t = torch.tensor(recon_np, dtype=torch.float32)
        loss = torch.mean((rho_t - recon_t) ** 2)
        loss.backward()
        with torch.no_grad():
            iso_t -= lr * iso_t.grad
            iso_t.clamp_(min=float(torch.min(rho_t)), max=float(torch.max(rho_t)))
            iso_t.grad.zero_()
        print(f"GD signed step {step+1}/{steps}: loss={loss.item():.6g}, iso_vals={iso_t.detach().cpu().numpy()}")
    return iso_t.detach().cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="Produce sigmoid diagnostics and soft distances.")
    parser.add_argument("input_vti", help="Input VTI file")
    parser.add_argument("--scalar", default=None, help="Scalar array name (default: first array)")
    parser.add_argument("-n", "--num-isos", type=int, default=2, help="Number of isocontours")
    parser.add_argument("-o", "--output", default=None, help="Output VTI file")
    parser.add_argument("--optimize-gd", action="store_true", help="Gradient-descent optimize iso values (differentiable proxy)")
    parser.add_argument("--gd-steps", type=int, default=0, help="GD steps for iso optimization")
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

    scalars = mesh.point_data[scalar_name]
    rho = scalars.copy()
    mesh.point_data["rho"] = rho

    if args.optimize_gd:
        scalar_range = (float(np.min(rho)), float(np.max(rho)))
        iso_vals = compute_iso_vals(scalar_range, args.num_isos)
        dims = mesh.dimensions
        if dims is None or len(dims) != 3:
            raise RuntimeError("Mesh dimensions are required for smoothing.")
        dims = tuple(int(d) for d in dims)
        # iso_vals, recon_field = optimize_iso_vals_gd_simple(rho, dims, iso_vals, steps=args.gd_steps, lr=args.gd_lr)
        iso_vals, recon_field = optimize_iso_vals_gd_simple(rho, dims, iso_vals, steps=args.gd_steps, lr=args.gd_lr)
        print(f"optimized iso_vals (gd): {iso_vals}")
        gd_signed_fields = []
        if recon_field is not None and recon_field.size == rho.size:
            mesh.point_data["reconstructed_gd"] = recon_field
        # Compute diagnostics with updated iso_vals
        for i, iso_val in enumerate(iso_vals):
            offset_np = rho - float(iso_val)
            offset_t = torch.tensor(offset_np, dtype=torch.float32)
            max_abs = torch.clamp(torch.max(torch.abs(offset_t)), min=1e-6)
            scale = 10.0 / max_abs
            sig_t = torch.sigmoid(offset_t * scale)
            dev = sig_t - 0.5
            max_dev = torch.clamp(torch.max(torch.abs(dev)), min=1e-6)
            sym_t = 0.5 + torch.clamp(dev / max_dev, -1.0, 1.0)
            clipped_t = torch.clamp(sym_t, 1e-6, 1.0 - 1e-6)
            smoothed_t = _neighbor_average(clipped_t, dims)
            smoothed_clipped_t = torch.clamp(smoothed_t, torch.min(clipped_t), torch.max(clipped_t))

            mesh.point_data[f"sigmoid_{i:03d}"] = clipped_t.numpy()
            mesh.point_data[f"sigmoid_{i:03d}_smoothed"] = smoothed_clipped_t.numpy()

            b_t = 4.0 * smoothed_clipped_t * (1.0 - smoothed_clipped_t)
            b_t = 10 * torch.minimum(b_t, torch.tensor(0.1, dtype=b_t.dtype))
            mesh.point_data[f"boundary_weight_{i:03d}"] = b_t.numpy()

            cost_t = 1000.0 * (1.0 - b_t)
            mesh.point_data[f"cost_{i:03d}"] = cost_t.numpy()

            soft_dist_t = bellman_soft_distance(cost_t, dims, tau=0.25, iterations=120)
            sign_t = torch.tanh((smoothed_clipped_t - 0.5) * 20.0)
            soft_np = soft_dist_t.numpy()
            signed_np = (sign_t * soft_dist_t).numpy()
            mesh.point_data[f"soft_distance_{i:03d}"] = soft_np
            mesh.point_data[f"signed_distance_{i:03d}"] = signed_np
            gd_signed_fields.append(signed_np)

        # Reconstruct from the signed distance fields using the classic iso interpolation
        if gd_signed_fields:
            gd_signed_stack = np.stack(gd_signed_fields, axis=0)
            scalar_range = (float(np.min(rho)), float(np.max(rho)))
            reconstructed_gd_sdf = reconstruct_from_multiple_isos(gd_signed_stack, iso_vals, scalar_range)
            mesh.point_data["reconstructed_gd_sdf"] = reconstructed_gd_sdf

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
