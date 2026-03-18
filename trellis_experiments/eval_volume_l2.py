import argparse
from pathlib import Path

import numpy as np
import pyvista as pv
import vtk

from isosurface_to_volume import get_scalar_from_rgb


def eval2(
    recon_surface,
    orig_volume=None,
    orig_surface=None,
    cmap_name="rainbow",
    plot_scalar_debug=False,
    use_clusters=False,
    out_path=None,
):
    array_of_rgb = recon_surface["RGBA"][:, :3]
    scalar_range = orig_volume.get_data_range()

    avg_scalar = get_scalar_from_rgb(
        array_of_rgb,
        scalar_range,
        cmap_name=cmap_name,
        num_samples=256,
        plot_debug=plot_scalar_debug,
        use_clusters=use_clusters,
    )

    # Create an implicit distance function from the surface.
    distance_func = vtk.vtkImplicitPolyDataDistance()
    distance_func.SetInput(recon_surface)

    # Evaluate the signed distance for each point in the mesh.
    points = orig_volume.points
    signed_distances = np.array([distance_func.EvaluateFunction(p) for p in points])

    # interpolate with min and max
    new_data_lin1 = np.empty_like(signed_distances)
    new_data_lin2 = np.empty_like(signed_distances)
    new_data1 = np.empty_like(signed_distances)
    new_data2 = np.empty_like(signed_distances)

    pos = signed_distances >= 0
    new_data_lin1[pos] = avg_scalar + (scalar_range[1] - avg_scalar) * (
        signed_distances[pos] / np.max(signed_distances)
    )
    new_data_lin2[pos] = avg_scalar + (scalar_range[0] - avg_scalar) * (
        signed_distances[pos] / np.max(signed_distances)
    )
    neg = signed_distances < 0
    new_data_lin1[neg] = avg_scalar + (scalar_range[0] - avg_scalar) * (
        signed_distances[neg] / np.min(signed_distances)
    )
    new_data_lin2[neg] = avg_scalar + (scalar_range[1] - avg_scalar) * (
        signed_distances[neg] / np.min(signed_distances)
    )
    new_data_lin1 = new_data_lin1.flatten()
    new_data_lin2 = new_data_lin2.flatten()
    # coords, values = extract_extrema_and_contour(new_data_lin1 , iso_val)
    # new_data1 = rbf_from_extrema_and_contour(coords, values, new_data_lin1.shape)
    # coords, values = extract_extrema_and_contour(new_data_lin2, iso_val)
    # new_data2 = rbf_from_extrema_and_contour(coords, values, new_data_lin2.shape)
    new_data1 = new_data_lin1
    new_data2 = new_data_lin2

    # Compute L2 difference between original and reconstructed data.
    L2_diff1 = np.sqrt(np.mean((orig_volume.active_scalars - new_data1.flatten()) ** 2))
    L2_diff2 = np.sqrt(np.mean((orig_volume.active_scalars - new_data2.flatten()) ** 2))
    L2_diff = min(L2_diff1, L2_diff2)
    new_data = new_data1 if L2_diff1 < L2_diff2 else new_data2
    print(
        "estimated_iso_val", avg_scalar, "L2 difference:", L2_diff1, L2_diff2, L2_diff
    )
    # recon_volume["recons_scalar_sd"] = new_data.flatten()
    orig_volume["recon_scalar_sd"] = new_data.flatten()
    if out_path is None:
        raise RuntimeError("out_path is required")
    out_path = Path(out_path).resolve()
    orig_volume.save(str(out_path))
    print("saved reconstructed volume:", out_path)

    # Calculate distance of two surfaces.
    from scipy.spatial import cKDTree

    tree1 = cKDTree(orig_surface.points)
    tree2 = cKDTree(recon_surface.points)

    d1, _ = tree1.query(recon_surface.points)
    d2, _ = tree2.query(orig_surface.points)

    chamfer = np.mean(d1**2) + np.mean(d2**2)
    hausdorff = max(np.max(d1), np.max(d2))

    print("chamfer", chamfer, "hausdorff", hausdorff)
    return chamfer, hausdorff, L2_diff1, L2_diff2, L2_diff


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--orig-volume", required=True, help="Path to original volume .vti")
    parser.add_argument("--orig-surface", required=True, help="Path to original surface .vtp")
    parser.add_argument("--recon-surface", required=True, help="Path to reconstructed/aligned surface .vtp")
    parser.add_argument("--scalar-name", default="tev", help="Active scalar name to use from the original volume")
    parser.add_argument("--cmap-name", default="rainbow", help="Colormap name used to decode RGB values")
    parser.add_argument("--plot-scalar-debug", action="store_true", help="Show debug plots for get_scalar_from_rgb")
    parser.add_argument("--use-clusters", action="store_true", help="Use cluster centers instead of the overall average color")
    args = parser.parse_args()

    orig_volume_path = Path(args.orig_volume).resolve()
    orig_surface_path = Path(args.orig_surface).resolve()
    recon_surface_path = Path(args.recon_surface).resolve()

    orig_volume = pv.read(orig_volume_path)
    orig_volume.set_active_scalars(args.scalar_name)
    orig_surface = pv.read(orig_surface_path)
    recon_surface = pv.read(recon_surface_path)
    out_path = recon_surface_path.with_name(f"{recon_surface_path.stem}_reconstructed.vti")

    print(args.cmap_name, recon_surface_path.stem)
    eval2(
        recon_surface=recon_surface,
        orig_volume=orig_volume,
        orig_surface=orig_surface,
        cmap_name=args.cmap_name,
        plot_scalar_debug=args.plot_scalar_debug,
        use_clusters=args.use_clusters,
        out_path=out_path,
    )
