import os
import numpy as np
import pyvista as pv
import reconstructMultiple as rm


def test_reconstruct_multiple_asteroid_vti(tmp_path):
    data_path = "/Users/bujack/Documents/doeAscrVis/nerf/data/asteroid.vti"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing test data: {data_path}")

    mesh = pv.read(data_path)
    if mesh.point_data is None or len(mesh.point_data) == 0:
        raise RuntimeError("No point data arrays found in input VTI.")

    scalar_name = mesh.point_data.keys()[0]
    scalars = mesh.point_data[scalar_name]
    scalar_range = (float(np.min(scalars)), float(np.max(scalars)))

    iso_vals = rm.compute_iso_vals(scalar_range, n_isos=2)
    contours, signed_fields = rm.compute_contours_and_signed_fields(mesh, scalar_name, iso_vals)

    # Save merged contours
    merged = pv.merge(contours, merge_points=False)
    merged_path = tmp_path / "contours_merged.vtp"
    merged.save(str(merged_path))
    assert merged_path.exists()
    assert merged.n_points > 0

    reconstructed = rm.reconstruct_from_multiple_isos(
        mesh,
        scalar_name,
        iso_vals=iso_vals,
        contours=contours,
        signed_fields=signed_fields,
    )

    # Range should match original scalar range (tight tolerance)
    assert np.isclose(np.min(reconstructed), scalar_range[0], rtol=0, atol=1e-9)
    assert np.isclose(np.max(reconstructed), scalar_range[1], rtol=0, atol=1e-9)

    # L2 should be stable (tight tolerance)
    l2 = float(np.sqrt(np.mean((scalars - reconstructed) ** 2)))
    expected_l2 = 0.14669417486146316
    assert np.isclose(l2, expected_l2, rtol=0, atol=1e-9)


def test_estimate_differentiable_iso_vals():
    data_path = "/Users/bujack/Documents/doeAscrVis/nerf/data/asteroid.vti"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing test data: {data_path}")

    mesh = pv.read(data_path)
    if mesh.point_data is None or len(mesh.point_data) == 0:
        raise RuntimeError("No point data arrays found in input VTI.")

    scalar_name = mesh.point_data.keys()[0]
    scalars = mesh.point_data[scalar_name]
    scalar_range = (float(np.min(scalars)), float(np.max(scalars)))

    iso_vals = rm.estimate_iso_vals_differentiable(
        scalars,
        scalar_range,
        n_isos=2,
        steps=1,
        lr=0.1,
        print_every=0,
        mesh=mesh,
        scalar_name=scalar_name,
    )

    assert iso_vals.shape == (2,)
    assert np.all(np.diff(iso_vals) > 0)
    assert iso_vals[0] > scalar_range[0]
    assert iso_vals[-1] < scalar_range[1]
