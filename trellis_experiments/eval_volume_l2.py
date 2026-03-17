import numpy as np
from isosurface_to_volume import get_scalar_from_rgb
import vtk
import pyvista as pv
from pathlib import Path


def eval2(recon_surface, orig_volume=None, orig_surface=None, cmap_name="rainbow"):
    array_of_rgb = recon_surface["RGBA"][:, :3]
    scalar_range = orig_volume.get_data_range()

    avg_scalar = get_scalar_from_rgb(
        array_of_rgb,
        scalar_range,
        cmap_name=cmap_name,
        num_samples=256,
        # threshold=5,
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
    orig_volume.save("fully_reconstructed.vti")

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
    asteroid = pv.read("../TheTruthPaper/imdb/inputs/asteroid_scaled.vti")
    asteroid.set_active_scalars("tev")
    isovalues = [
        "0.08200450018048287",
        # "0.2086136944591999",
        # "0.33522288873791695",
        # "0.461832083016634",
        # "0.588441277295351",
    ]
    colormaps = ["rainbow"]  # ["coolwarm", "Spectral", "viridis"]

    for colormap in colormaps:
        for isovalue in isovalues:
            # print("Procing colormap: ", colormap)
            contour = pv.read(
                "/home/ollie/PycharmProjects/TheTruthPaper/imdb/outputs/asteroid_isosurface/tev_iso_"
                + isovalue
                + ".vtp"
            )

            dir_path = Path(
                "/home/ollie/PycharmProjects/TheTruthTrellis/outputs/asteroid/"
                + colormap
                + "/aligned/"
            )
            for vtp_path in dir_path.glob(
                "tev_iso_" + isovalue + "_AZ_180_EL_45.0_aligned.vtp"
            ):
                print(colormap, vtp_path.stem)
                recon_surface = pv.read(vtp_path)
                (
                    chamfer,
                    hausdorff,
                    L2_diff1,
                    L2_diff2,
                    L2_diff,
                ) = eval2(
                    recon_surface=recon_surface,
                    orig_volume=asteroid,
                    orig_surface=contour,
                    cmap_name=colormap,
                )
