#!/usr/bin/env python3
"""
Display a GLB with VTK's native glTF/GLB importer.

This renders the GLB through VTK's importer instead of baking colors into VTK
point data first, so embedded textures and material colors can be displayed
directly by the viewer.

Usage:
  python view_glb_color.py mesh.glb
"""

import argparse
import os
import sys

import numpy as np

from convertGlb2Vtk import (
    bake_uv_to_rgba,
    extract_basecolor_png_from_glb,
    get_uv,
    load_first_mesh,
)


def _parse_bg(bg: str) -> tuple[float, float, float]:
    try:
        values = tuple(float(part.strip()) for part in bg.split(","))
    except ValueError as e:
        raise RuntimeError("--bg must contain three comma-separated floats") from e
    if len(values) != 3:
        raise RuntimeError("--bg must contain three comma-separated floats")
    return values


def _print_point_rgb_summary(glb_path: str) -> None:
    mesh = load_first_mesh(glb_path)
    uv = get_uv(mesh)
    image = extract_basecolor_png_from_glb(glb_path)
    rgba = bake_uv_to_rgba(uv, image)
    rgb = rgba[:, :3].astype(np.float64)

    print("=== Point RGB values ===")
    for idx, color in enumerate(rgb.astype(np.uint8)):
        print(f"point {idx}: ({color[0]}, {color[1]}, {color[2]})")

    mean_rgb = rgb.mean(axis=0)
    print("\n=== Average RGB ===")
    print(f"({mean_rgb[0]:.6f}, {mean_rgb[1]:.6f}, {mean_rgb[2]:.6f})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("glb", help="Input .glb file")
    parser.add_argument("--title", default=None, help="Optional window title")
    parser.add_argument(
        "--bg",
        default="1,1,1",
        help="Viewer background RGB floats in [0,1], for example 1,1,1",
    )
    args = parser.parse_args()

    try:
        import vtk  # type: ignore
        from vtkmodules.vtkIOImport import vtkGLTFImporter  # type: ignore
        from vtkmodules.vtkRenderingCore import (  # type: ignore
            vtkRenderWindow,
            vtkRenderWindowInteractor,
        )
    except ImportError as e:
        raise RuntimeError("vtk is required. Install with: pip install vtk") from e

    glb_path = os.path.abspath(args.glb)
    if not os.path.exists(glb_path):
        raise RuntimeError(f"File not found: {glb_path}")

    bg = _parse_bg(args.bg)
    _print_point_rgb_summary(glb_path)

    render_window = vtkRenderWindow()
    render_window.SetSize(1280, 900)
    render_window.SetWindowName(args.title or os.path.basename(glb_path))

    importer = vtkGLTFImporter()
    importer.SetFileName(glb_path)
    importer.SetRenderWindow(render_window)
    importer.Update()

    renderers = render_window.GetRenderers()
    renderer = renderers.GetFirstRenderer()
    if renderer is None:
        raise RuntimeError(f"No renderer created for {glb_path}")

    renderer.SetBackground(*bg)
    renderer.ResetCamera()

    interactor = vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    interactor.Initialize()
    render_window.Render()
    interactor.Start()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
