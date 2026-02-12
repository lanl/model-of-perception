#!/usr/bin/env python3
"""
convertGlb2Vtk.py

Bake GLB baseColorTexture (embedded PNG) using TEXCOORD_0 into per-point colors and write VTP.
- Output path defaults to: <input_basename>_color.vtp (same directory)
- Writes:
    * RGBA (unsigned char, 4 components)
  and sets active scalars to RGBA.

Usage:
  python convertGlb2Vtk.py /path/to/in.glb
  python convertGlb2Vtk.py /path/to/in.glb --out /path/to/out.vtp

Requires:
  pip install trimesh pillow vtk numpy
"""

import argparse
import io
import os
import struct
import sys

import numpy as np
import trimesh
from PIL import Image

import vtk
from vtk.util import numpy_support


def read_glb_chunks(glb_path: str):
    with open(glb_path, "rb") as f:
        data = f.read()

    if len(data) < 12:
        raise ValueError("File too small to be a GLB")

    magic, version, length = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF":
        raise ValueError("Not a GLB (missing glTF magic)")
    if length != len(data):
        # some writers may mismatch, but usually it's correct
        pass

    off = 12
    json_bytes = None
    bin_bytes = None

    while off + 8 <= len(data):
        chunk_len, chunk_type = struct.unpack_from("<I4s", data, off)
        off += 8
        chunk = data[off : off + chunk_len]
        off += chunk_len

        if chunk_type == b"JSON":
            json_bytes = chunk
        elif chunk_type == b"BIN\x00":
            bin_bytes = chunk

    if json_bytes is None:
        raise ValueError("GLB missing JSON chunk")
    if bin_bytes is None:
        raise ValueError("GLB missing BIN chunk")

    return json_bytes, bin_bytes


def extract_basecolor_png_from_glb(glb_path: str) -> Image.Image:
    import json

    json_bytes, bin_bytes = read_glb_chunks(glb_path)
    gltf = json.loads(json_bytes.decode("utf-8"))

    # Follow: materials[0].pbrMetallicRoughness.baseColorTexture.index -> textures[i].source -> images[j]
    mats = gltf.get("materials", [])
    if not mats:
        raise ValueError("No materials in GLB")

    pbr = mats[0].get("pbrMetallicRoughness", {})
    bct = pbr.get("baseColorTexture", None)
    if bct is None:
        raise ValueError("No pbrMetallicRoughness.baseColorTexture in material[0]")

    tex_index = int(bct.get("index", 0))
    textures = gltf.get("textures", [])
    if tex_index >= len(textures):
        raise ValueError("baseColorTexture.index out of range")

    img_index = int(textures[tex_index].get("source", 0))
    images = gltf.get("images", [])
    if img_index >= len(images):
        raise ValueError("textures[source] image index out of range")

    img = images[img_index]

    # Embedded image via bufferView
    if "bufferView" not in img:
        raise ValueError("Image is not embedded via bufferView (unexpected for your header)")

    bv_index = int(img["bufferView"])
    bvs = gltf.get("bufferViews", [])
    if bv_index >= len(bvs):
        raise ValueError("images[...].bufferView out of range")

    bv = bvs[bv_index]
    byte_offset = int(bv.get("byteOffset", 0))
    byte_length = int(bv["byteLength"])

    blob = bin_bytes[byte_offset : byte_offset + byte_length]

    # PNG bytes -> PIL
    try:
        im = Image.open(io.BytesIO(blob)).convert("RGBA")
    except Exception as e:
        raise ValueError(f"Failed to decode embedded image bytes: {e}")

    return im


def load_first_mesh(glb_path: str) -> trimesh.Trimesh:
    obj = trimesh.load(glb_path, force="scene")
    if isinstance(obj, trimesh.Scene):
        if not obj.geometry:
            raise ValueError("No geometry found in GLB.")
        return next(iter(obj.geometry.values()))
    return obj


def get_uv(mesh: trimesh.Trimesh) -> np.ndarray:
    uv = getattr(mesh.visual, "uv", None)
    if uv is None:
        raise ValueError("Mesh has no UVs (TEXCOORD_0).")
    uv = np.asarray(uv, dtype=np.float64)
    if uv.ndim != 2 or uv.shape[1] != 2:
        raise ValueError("Unexpected UV array shape.")
    return uv


def bake_uv_to_rgba(uv: np.ndarray, im: Image.Image) -> np.ndarray:
    W, H = im.size
    pix = np.array(im, dtype=np.uint8)  # (H, W, 4)

    u = np.mod(uv[:, 0], 1.0)
    v = np.mod(uv[:, 1], 1.0)

    # nearest sampling; flip V for image indexing
    x = np.clip(np.rint(u * (W - 1)).astype(np.int64), 0, W - 1)
    y = np.clip(np.rint((1.0 - v) * (H - 1)).astype(np.int64), 0, H - 1)

    rgba = pix[y, x, :]  # (n_points, 4) uint8
    return rgba


def to_vtk_polydata(mesh: trimesh.Trimesh, rgba: np.ndarray) -> vtk.vtkPolyData:
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)

    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError("Unexpected vertex array shape.")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("Expected triangular faces.")
    if rgba.shape != (verts.shape[0], 4):
        raise ValueError("RGBA must be (n_points, 4).")

    poly = vtk.vtkPolyData()

    pts = vtk.vtkPoints()
    pts.SetData(numpy_support.numpy_to_vtk(verts, deep=True))
    poly.SetPoints(pts)

    ntri = faces.shape[0]
    cell = np.empty((ntri, 4), dtype=np.int64)
    cell[:, 0] = 3
    cell[:, 1:] = faces
    cell_flat = cell.ravel()

    cells = vtk.vtkCellArray()
    cells.SetCells(ntri, numpy_support.numpy_to_vtkIdTypeArray(cell_flat, deep=True))
    poly.SetPolys(cells)

    # uchar4 RGBA
    rgba_vtk = numpy_support.numpy_to_vtk(rgba, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    rgba_vtk.SetName("RGBA")
    rgba_vtk.SetNumberOfComponents(4)
    poly.GetPointData().AddArray(rgba_vtk)

    poly.GetPointData().SetActiveScalars("RGBA")
    return poly


def write_vtp(poly: vtk.vtkPolyData, out_path: str) -> None:
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(out_path)
    writer.SetInputData(poly)
    writer.SetDataModeToBinary()
    if writer.Write() != 1:
        raise RuntimeError(f"Failed to write {out_path}")


def default_out_path(in_path: str) -> str:
    d = os.path.dirname(in_path)
    base = os.path.splitext(os.path.basename(in_path))[0]
    return os.path.join(d, f"{base}_color.vtp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("glb", help="Input .glb")
    ap.add_argument("--out", default=None, help="Output .vtp (default: <input>_color.vtp next to input)")
    args = ap.parse_args()

    in_path = args.glb
    out_path = args.out or default_out_path(in_path)

    mesh = load_first_mesh(in_path)
    uv = get_uv(mesh)

    # robust: extract embedded PNG directly from GLB (works with your header structure)
    im = extract_basecolor_png_from_glb(in_path)

    rgba = bake_uv_to_rgba(uv, im)
    poly = to_vtk_polydata(mesh, rgba)
    write_vtp(poly, out_path)

    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
