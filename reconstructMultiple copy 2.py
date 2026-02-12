import argparse
import os
import numpy as np
import pyvista as pv
import torch
import torch.nn.functional as F


def softmin_stack(values, tau):
    v = -values / tau
    vmax = np.max(v, axis=0)
    return -tau * (vmax + np.log(np.sum(np.exp(v - vmax), axis=0)))


def bellman_soft_distance(cost, dims, tau=0.25, iterations=120):
    shape = tuple(int(d) for d in dims)
    cost_grid = cost.reshape(shape, order="F")
    D = cost_grid.copy()
    for _ in range(iterations):
        neighbors = [cost_grid]
        for axis in range(3):
            plus = np.empty_like(D)
            minus = np.empty_like(D)
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
        shapes = [arr.shape for arr in neighbors]
        if any(sh != shape for sh in shapes):
            raise ValueError(f"shape mismatch {shapes}")
        stacked = np.stack(neighbors, axis=0)
        D = softmin_stack(stacked, tau)
    return D.ravel(order="F")


def _neighbor_average(field, dims):
    shape = tuple(int(d) for d in dims)
    grid = field.reshape(shape, order="F")
    tensor = torch.from_numpy(grid.astype(np.float32)).unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
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
    return smoothed.squeeze(0).squeeze(0).numpy().ravel(order="F")


def compute_iso_vals(scalar_range, n_isos):
    if n_isos < 1:
        raise ValueError("n_isos must be >= 1")
    if n_isos == 1:
        return np.array([(scalar_range[0] + scalar_range[1]) / 2.0])
    min_val, max_val = scalar_range
    d = (max_val - min_val) / n_isos
    return min_val + 0.5 * d + np.arange(n_isos) * d


def main():
    parser = argparse.ArgumentParser(description="Produce sigmoid diagnostics and soft distances.")
    parser.add_argument("input_vti", help="Input VTI file")
    parser.add_argument("--scalar", default=None, help="Scalar array name (default: first array)")
    parser.add_argument("-n", "--num-isos", type=int, default=2, help="Number of isocontours")
    parser.add_argument("-o", "--output", default=None, help="Output VTI file")
    parser.add_argument("--optimize-differentiable", action="store_true", help="Compute diff diagnostics")
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

    if args.optimize_differentiable:
        scalar_range = (float(np.min(rho)), float(np.max(rho)))
        iso_vals = compute_iso_vals(scalar_range, args.num_isos)
        dims = mesh.dimensions
        if dims is None or len(dims) != 3:
            raise RuntimeError("Mesh dimensions are required for smoothing.")
        dims = tuple(int(d) for d in dims)
        for i, iso_val in enumerate(iso_vals):
            offset = rho - float(iso_val)
            max_abs = float(np.max(np.abs(offset))) if offset.size > 0 else 1.0
            scale = 10.0 / max(max_abs, 1e-6)
            raw_sig = torch.sigmoid(torch.tensor(offset * scale, dtype=torch.float32)).numpy()
            dev = raw_sig - 0.5
            max_dev = max(abs(np.min(dev)), abs(np.max(dev)), 1e-6)
            sym = 0.5 + np.clip(dev / max_dev, -1.0, 1.0)
            clipped = np.clip(sym, 1e-6, 1.0 - 1e-6)
            smoothed = _neighbor_average(clipped, dims)
            smoothed_clipped = np.clip(smoothed, clipped.min(), clipped.max())
            mesh.point_data[f"sigmoid_{i:03d}"] = clipped
            mesh.point_data[f"sigmoid_{i:03d}_smoothed"] = smoothed_clipped
            b = np.clip(4.0 * smoothed_clipped * (1.0 - smoothed_clipped),0,0.1)*10
            mesh.point_data[f"boundary_weight_{i:03d}"] = b
            cost = 1000.0 * (1.0 - b)
            mesh.point_data[f"cost_{i:03d}"] = cost
            soft_dist = bellman_soft_distance(cost, dims, tau=0.25, iterations=120)
            sign = np.sign(offset)
            sign[sign == 0.0] = 1.0
            mesh.point_data[f"soft_distance_{i:03d}"] = soft_dist
            mesh.point_data[f"signed_distance_{i:03d}"] = sign * soft_dist

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
