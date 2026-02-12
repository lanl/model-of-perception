#!/usr/bin/env python3
# make_slices_with_overlays_centered_tol.py
#
# Example:
# python make_slices_with_overlays_centered_tol.py \
#   --csv "/Users/bujack/Documents/doeAscrVis/nerf/data/results.csv" \
#   --images_dir "/Users/bujack/Documents/doeAscrVis/nerf/data/asteroid_xy_center/rainbow" \
#   --out_dir "./out" --thumb_zoom 0.25

import argparse
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image

# ---------- Helpers ----------

def detect_axes(df):
    def find_col(cands):
        for c in df.columns:
            lc = c.lower()
            if lc in cands or any(k in lc for k in cands):
                return c
        return None

    az = find_col({"az"})
    el = find_col({"el"})
    iso = find_col({"isovalue", "iso", "iso_value", "threshold"})
    if not (az and el and iso):
        raise ValueError(f"Could not detect axes. Found: AZ={az}, EL={el}, ISO={iso}")
    return az, el, iso

def pick_l2_col(df):
    prefs = ["L2_diff", "L2", "L2_distance", "l2", "l2diff", "l2_distance"]
    for p in prefs:
        for c in df.columns:
            if c.lower() == p.lower():
                return c
    for c in df.columns:
        if "l2" in c.lower() and pd.api.types.is_numeric_dtype(df[c]):
            return c
    raise ValueError("No L2 metric column found (looked for columns like L2_diff, L2, etc.).")

def build_index_pivot(plane, x_col, y_col, val_col):
    x_vals = np.sort(plane[x_col].unique())
    y_vals = np.sort(plane[y_col].unique())
    nx, ny = len(x_vals), len(y_vals)

    x_map = {v: i for i, v in enumerate(x_vals)}
    y_map = {v: j for j, v in enumerate(y_vals)}

    z = np.full((ny, nx), np.nan, dtype=float)

    for _, row in plane.iterrows():
        xv = row[x_col]
        yv = row[y_col]
        if xv in x_map and yv in y_map:
            i = x_map[xv]
            j = y_map[yv]
            z[j, i] = float(row[val_col])

    if np.isnan(z).any():
        finite_vals = z[np.isfinite(z)]
        fill = float(np.nanmean(finite_vals)) if finite_vals.size else 0.0
        z[np.isnan(z)] = fill

    return z, x_vals, y_vals, x_map, y_map

def parse_image_triplets(images_dir):
    """
    tev_iso_0.08200450018048287_AZ_135_EL_-45.0.png
    -> key: (az, el, iso) as raw floats
    """
    pat = re.compile(
        r".*?iso_(?P<iso>-?\d+(?:\.\d+)?)_AZ_(?P<az>-?\d+(?:\.\d+)?)_EL_(?P<el>-?\d+(?:\.\d+)?).*\.(png|jpg|jpeg)$",
        re.IGNORECASE
    )
    index = {}
    for p in Path(images_dir).glob("*"):
        if not p.is_file():
            continue
        m = pat.match(p.name)
        if not m:
            continue
        iso = float(m.group("iso"))
        az  = float(m.group("az"))
        el  = float(m.group("el"))
        key = (az, el, iso)
        index[key] = p
    return index

def find_nearest(value, grid, tol=1e-3):
    """Return index in grid whose value is closest to `value`, if within tol; otherwise None."""
    grid = np.asarray(grid, dtype=float)
    diffs = np.abs(grid - float(value))
    i = diffs.argmin()
    if diffs[i] <= tol:
        return int(i)
    return None

def add_overlays_index_space(ax,
                             fixed_axis, fixed_value,
                             images_index,
                             az_col, el_col, iso_col,
                             x_vals, y_vals,
                             thumb_zoom=0.25,
                             coord_tol=1e-3):
    """
    Overlays in index space with nearest-neighbour matching onto (x_vals, y_vals).
    """

    for (az, el, iso), img_path in images_index.items():
        # plane membership
        if fixed_axis == iso_col and abs(iso - fixed_value) > coord_tol:
            continue
        if fixed_axis == az_col and abs(az - fixed_value) > coord_tol:
            continue
        if fixed_axis == el_col and abs(el - fixed_value) > coord_tol:
            continue

        # map to slice coordinates
        if fixed_axis == iso_col:
            xv, yv = az, el        # iso=const -> x=AZ, y=EL
        elif fixed_axis == az_col:
            xv, yv = iso, el       # AZ=const -> x=iso, y=EL
        else:  # fixed_axis == el_col
            xv, yv = az, iso       # EL=const -> x=AZ, y=iso

        ix = find_nearest(xv, x_vals, tol=coord_tol)
        iy = find_nearest(yv, y_vals, tol=coord_tol)
        if ix is None or iy is None:
            continue

        try:
            arr = Image.open(img_path)
        except Exception:
            continue

        img = OffsetImage(arr, zoom=thumb_zoom)
        ab = AnnotationBbox(img, (ix, iy),
                            frameon=False,
                            box_alignment=(0.5, 0.5))
        ax.add_artist(ab)

def plot_slice(df, az_col, el_col, iso_col, l2_col,
               fixed_axis, fixed_value,
               images_index, out_path,
               thumb_zoom=0.25, cmap="gray"):

    if fixed_axis == iso_col:
        x_col, y_col = az_col, el_col
        title = f"{l2_col} | {iso_col}={fixed_value}"
    elif fixed_axis == az_col:
        x_col, y_col = iso_col, el_col
        title = f"{l2_col} | {az_col}={fixed_value}"
    elif fixed_axis == el_col:
        x_col, y_col = az_col, iso_col
        title = f"{l2_col} | {el_col}={fixed_value}"
    else:
        raise ValueError("fixed_axis must be one of the detected axis columns.")

    plane = df[np.isclose(df[fixed_axis].astype(float), float(fixed_value))]

    z, x_vals, y_vals, x_map, y_map = build_index_pivot(plane, x_col, y_col, l2_col)
    print("test")
    fig, ax = plt.subplots()
    # Use log scale, ensuring all values are positive
    z_log = np.maximum(z, np.finfo(float).eps)  # Replace zeros/negatives with smallest positive value
    z_min = np.nanmin(z_log[z_log > 0])
    z_max = np.nanmax(z_log)
    im = ax.imshow(z_log, origin="lower", aspect="auto", cmap=cmap, 
                   norm=LogNorm(vmin=z_min, vmax=z_max))

    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)

    def sparse_indices(vals, max_ticks=10):
        vals = list(vals)
        n = len(vals)
        if n <= max_ticks:
            return list(range(n))
        step = int(np.ceil(n / max_ticks))
        return list(range(0, n, step))

    xticks = sparse_indices(x_vals)
    yticks = sparse_indices(y_vals)

    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels([f"{x_vals[i]:.3g}" for i in xticks])
    ax.set_yticklabels([f"{y_vals[j]:.3g}" for j in yticks])

    cbar = plt.colorbar(im)
    cbar.set_label(l2_col)

    add_overlays_index_space(
        ax,
        fixed_axis=fixed_axis,
        fixed_value=float(fixed_value),
        images_index=images_index,
        az_col=az_col,
        el_col=el_col,
        iso_col=iso_col,
        x_vals=x_vals,
        y_vals=y_vals,
        thumb_zoom=thumb_zoom,
        coord_tol=1e-3,
    )

    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser(
        description="Make 3 heatmap slices with image overlays centered in cells through min L2."
    )
    ap.add_argument("--csv", required=True, help="Path to results.csv")
    ap.add_argument("--images_dir", required=True, help="Directory with PNGs (tev_iso_..._AZ_..._EL_...).")
    ap.add_argument("--out_dir", default="./out", help="Output directory for PNGs")
    ap.add_argument("--thumb_zoom", type=float, default=0.25, help="Overlay thumbnail zoom (default 0.25)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    az_col, el_col, iso_col = detect_axes(df)
    l2_col = pick_l2_col(df)

    for c in (az_col, el_col, iso_col, l2_col):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    min_idx = df[l2_col].idxmin()
    min_row = df.loc[min_idx, [az_col, el_col, iso_col, l2_col]]
    az0 = float(min_row[az_col])
    el0 = float(min_row[el_col])
    iso0 = float(min_row[iso_col])

    print(f"Global min {l2_col}: {min_row[l2_col]:.6g} at {az_col}={az0}, {el_col}={el0}, {iso_col}={iso0}")

    images_index = parse_image_triplets(args.images_dir)

    plot_slice(
        df, az_col, el_col, iso_col, l2_col,
        fixed_axis=iso_col, fixed_value=iso0,
        images_index=images_index,
        out_path=out_dir / f"slice_iso_{iso0:.6f}.png",
        thumb_zoom=args.thumb_zoom,
    )

    plot_slice(
        df, az_col, el_col, iso_col, l2_col,
        fixed_axis=az_col, fixed_value=az0,
        images_index=images_index,
        out_path=out_dir / f"slice_az_{az0:.6f}.png",
        thumb_zoom=args.thumb_zoom,
    )

    plot_slice(
        df, az_col, el_col, iso_col, l2_col,
        fixed_axis=el_col, fixed_value=el0,
        images_index=images_index,
        out_path=out_dir / f"slice_el_{el0:.6f}.png",
        thumb_zoom=args.thumb_zoom,
    )

    print(f"Saved slices to: {out_dir.resolve()}")

if __name__ == "__main__":
    main()
