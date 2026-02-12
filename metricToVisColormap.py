#!/usr/bin/env python3
# Extended version that handles colormaps as columns and creates visualizations
# across all dimensions including colormap choice

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

def detect_colormap_columns(df):
    """Detect which columns represent colormaps."""
    colormap_names = ['rainbow', 'coolwarm', 'spectral', 'viridis']
    found_cmaps = []
    for cmap in colormap_names:
        for c in df.columns:
            if c.lower() == cmap.lower():
                found_cmaps.append(c)
                break
    return found_cmaps

def reshape_to_long_format(df, az_col, el_col, iso_col, cmap_cols):
    """
    Reshape dataframe from wide format (colormaps as columns) to long format
    (colormap as a variable).
    """
    # Keep only the relevant columns
    id_cols = [az_col, el_col, iso_col]
    df_long = df[id_cols + cmap_cols].copy()
    
    # Melt to long format
    df_melted = df_long.melt(id_vars=id_cols, 
                              value_vars=cmap_cols,
                              var_name='colormap',
                              value_name='L2_diff')
    
    return df_melted

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

def parse_image_triplets(images_dir, colormap=None):
    """
    Parse images from directory, optionally filtering by colormap subdirectory.
    Format: [colormap_folder/]tev_iso_..._AZ_..._EL_....png
    """
    pat = re.compile(
        r".*?iso_(?P<iso>-?\d+(?:\.\d+)?)_AZ_(?P<az>-?\d+(?:\.\d+)?)_EL_(?P<el>-?\d+(?:\.\d+)?).*\.(png|jpg|jpeg)$",
        re.IGNORECASE
    )
    index = {}
    
    # If colormap specified, look in subdirectory
    # Try both exact name and capitalized versions
    if colormap:
        search_dirs = [
            Path(images_dir) / colormap,
            Path(images_dir) / colormap.lower(),
            Path(images_dir) / colormap.capitalize(),
            Path(images_dir) / colormap.title(),
        ]
        search_dir = None
        for d in search_dirs:
            if d.exists():
                search_dir = d
                break
        if search_dir is None:
            print(f"Warning: No directory found for colormap '{colormap}' in {images_dir}")
            print(f"  Tried: {[str(d) for d in search_dirs]}")
            return index
    else:
        search_dir = Path(images_dir)
    
    if not search_dir.exists():
        print(f"Warning: Directory {search_dir} does not exist")
        return index
    
    print(f"Loading images from: {search_dir}")
    
    for p in search_dir.glob("*"):
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
    
    print(f"  Found {len(index)} images")
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
    overlays_added = 0
    
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
        except Exception as e:
            print(f"  Warning: Could not load image {img_path}: {e}")
            continue

        img = OffsetImage(arr, zoom=thumb_zoom)
        ab = AnnotationBbox(img, (ix, iy),
                            frameon=False,
                            box_alignment=(0.5, 0.5))
        ax.add_artist(ab)
        overlays_added += 1
    
    print(f"  Successfully added {overlays_added} image overlays to plot")

def plot_slice(df, az_col, el_col, iso_col, l2_col,
               fixed_axis, fixed_value,
               images_index, out_path,
               thumb_zoom=0.25, cmap="gray",
               colormap_col=None, fixed_colormap=None):

    if fixed_axis == iso_col:
        x_col, y_col = az_col, el_col
        title = f"{l2_col} | {iso_col}={float(fixed_value):.3f}"
    elif fixed_axis == az_col:
        x_col, y_col = iso_col, el_col
        title = f"{l2_col} | {az_col}={fixed_value}"
    elif fixed_axis == el_col:
        x_col, y_col = az_col, iso_col
        title = f"{l2_col} | {el_col}={fixed_value}"
    elif fixed_axis == colormap_col:
        x_col, y_col = az_col, el_col
        title = f"{l2_col} | colormap={fixed_value}"
    else:
        raise ValueError("fixed_axis must be one of the detected axis columns or colormap.")

    # Filter data for this slice
    if fixed_axis == colormap_col:
        # Slicing through colormap dimension
        plane = df[df[colormap_col] == fixed_value]
    else:
        # Slicing through numeric dimension
        plane = df[np.isclose(df[fixed_axis].astype(float), float(fixed_value))]
        # If colormap column exists, filter by fixed_colormap
        if colormap_col and fixed_colormap:
            plane = plane[plane[colormap_col] == fixed_colormap]

    if plane.empty:
        print(f"Warning: No data for {fixed_axis}={fixed_value}")
        return

    z, x_vals, y_vals, x_map, y_map = build_index_pivot(plane, x_col, y_col, l2_col)

    fig, ax = plt.subplots()
    # Use linear scale
    im = ax.imshow(z, origin="lower", aspect="auto", cmap=cmap)
    z_valid = z[np.isfinite(z)]
    if len(z_valid) > 0:
        z_min = np.min(z_valid)
        z_max = np.max(z_valid)
        print(f"Linear scale: data range [{z_min:.6e}, {z_max:.6e}]")

    if fixed_colormap:
        title += f" | {fixed_colormap}"
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

    # Only add overlays for traditional slices (not colormap slices)
    if images_index and fixed_axis != colormap_col:
        print(f"Adding {len(images_index)} image overlays for {fixed_axis}={fixed_value}")
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

def plot_colormap_slice(df, az_col, el_col, iso_col, colormap_col, l2_col,
                        fixed_axis, fixed_value,
                        az_fixed=None, el_fixed=None, iso_fixed=None,
                        out_path=None, images_dir=None, thumb_zoom=0.25, cmap="gray"):
    """
    Plot a slice where one axis is the colormap and the other is a parameter.
    Adds image overlays from each colormap's subdirectory.
    All other parameters are fixed at specified values.
    """
    # Design three different views:
    # 1. Colormap vs AZ (fix ISO, EL)
    # 2. Colormap vs ISO (fix AZ, EL) 
    # 3. Colormap vs EL (fix ISO, AZ)
    
    # Initialize filter variables
    filter_col = None
    filter_value = None
    filter_col1 = None
    filter_value1 = None
    filter_col2 = None
    filter_value2 = None
    
    if fixed_axis == iso_col:
        # Fix ISO and EL, vary AZ → colormap vs AZ
        x_col = colormap_col
        y_col = az_col
        title = f"{l2_col} | {iso_col}={float(fixed_value):.3f}, {el_col}={el_fixed:.1f}"
        filter_col = el_col
        filter_value = el_fixed
    elif fixed_axis == az_col:
        # Fix AZ and EL, vary ISO → colormap vs ISO
        x_col = colormap_col
        y_col = iso_col
        title = f"{l2_col} | {az_col}={fixed_value:.1f}, {el_col}={el_fixed:.1f}"
        filter_col = el_col
        filter_value = el_fixed
    elif fixed_axis == el_col:
        # Fix ISO and AZ, vary EL → colormap vs EL
        x_col = colormap_col
        y_col = el_col
        title = f"{l2_col} | {iso_col}={float(iso_fixed):.3f}, {az_col}={az_fixed:.1f}"
        filter_col1 = iso_col
        filter_value1 = iso_fixed
        filter_col2 = az_col
        filter_value2 = az_fixed
    else:
        raise ValueError("fixed_axis must be one of the detected axis columns.")

    # Filter data based on fixed parameters
    if fixed_axis == el_col:
        # Special case: fix ISO and AZ, vary EL
        plane = df.copy()
        plane = plane[np.isclose(plane[filter_col1].astype(float), float(filter_value1))]
        plane = plane[np.isclose(plane[filter_col2].astype(float), float(filter_value2))]
        print(f"Creating colormap slice: {filter_col1}={filter_value1}, {filter_col2}={filter_value2}, varying {y_col}")
    else:
        # Standard case: one fixed axis, one filter parameter
        plane = df[np.isclose(df[fixed_axis].astype(float), float(fixed_value))]
        if filter_value is not None:
            plane = plane[np.isclose(plane[filter_col].astype(float), float(filter_value))]
        print(f"Creating colormap slice: {fixed_axis}={fixed_value}, {filter_col}={filter_value}, varying {y_col}")

    if plane.empty:
        print(f"Warning: No data found")
        return
    
    print(f"  Found {len(plane)} data points")
    
    # Build pivot with colormap as one dimension
    x_vals_numeric = np.sort(plane[y_col].unique())
    colormap_vals = sorted(plane[x_col].unique())
    
    x_map = {v: i for i, v in enumerate(x_vals_numeric)}
    y_map = {v: j for j, v in enumerate(colormap_vals)}
    
    ny, nx = len(colormap_vals), len(x_vals_numeric)
    z = np.full((ny, nx), np.nan, dtype=float)

    for _, row in plane.iterrows():
        xv = row[y_col]
        yv = row[x_col]
        if xv in x_map and yv in y_map:
            i = x_map[xv]
            j = y_map[yv]
            z[j, i] = float(row[l2_col])

    if np.isnan(z).any():
        finite_vals = z[np.isfinite(z)]
        fill = float(np.nanmean(finite_vals)) if finite_vals.size else 0.0
        z[np.isnan(z)] = fill

    fig, ax = plt.subplots()
    
    # Use linear scale
    im = ax.imshow(z, origin="lower", aspect="auto", cmap=cmap)

    ax.set_title(title)
    ax.set_xlabel(y_col)
    ax.set_ylabel("Colormap")

    ax.set_xticks(list(range(len(x_vals_numeric))))
    ax.set_xticklabels([f"{v:.3g}" for v in x_vals_numeric], rotation=45)
    
    ax.set_yticks(list(range(len(colormap_vals))))
    ax.set_yticklabels(colormap_vals)

    cbar = plt.colorbar(im)
    cbar.set_label(l2_col)

    # Add image overlays if images_dir provided
    if images_dir:
        overlays_added = 0
        print(f"Adding image overlays for colormap slice...")
        
        # For each colormap, load its images and place them on the corresponding row
        for j, colormap_name in enumerate(colormap_vals):
            colormap_images = parse_image_triplets(images_dir, colormap_name)
            if not colormap_images:
                continue
            
            # For each position along the x-axis (numeric parameter)
            for i, x_val in enumerate(x_vals_numeric):
                # Find an image that matches this position
                # Use the fixed values passed to the function
                if fixed_axis == iso_col:
                    # x_val is azimuth, iso is fixed, el is fixed
                    target_iso = fixed_value
                    target_az = x_val
                    target_el = el_fixed
                elif fixed_axis == az_col:
                    # x_val is isovalue, az is fixed, el is fixed
                    target_az = fixed_value
                    target_iso = x_val
                    target_el = el_fixed
                elif fixed_axis == el_col:
                    # x_val is elevation, az is fixed, iso is fixed
                    target_el = x_val
                    target_az = az_fixed
                    target_iso = iso_fixed
                
                # Find matching image
                key = (target_az, target_el, target_iso)
                if key not in colormap_images:
                    # Try to find closest match
                    found = False
                    for (az, el, iso), img_path in colormap_images.items():
                        if (abs(az - target_az) < 1e-3 and 
                            abs(el - target_el) < 1e-3 and 
                            abs(iso - target_iso) < 1e-3):
                            key = (az, el, iso)
                            found = True
                            break
                    if not found:
                        continue
                
                img_path = colormap_images[key]
                
                try:
                    arr = Image.open(img_path)
                    img = OffsetImage(arr, zoom=thumb_zoom)
                    ab = AnnotationBbox(img, (i, j),
                                        frameon=False,
                                        box_alignment=(0.5, 0.5))
                    ax.add_artist(ab)
                    overlays_added += 1
                except Exception as e:
                    continue
        
        print(f"  Successfully added {overlays_added} image overlays to colormap slice")

    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser(
        description="Make heatmap slices with image overlays including colormap dimension."
    )
    ap.add_argument("--csv", required=True, help="Path to results.csv with colormap columns")
    ap.add_argument("--images_dir", required=True, help="Directory with PNGs in colormap subdirectories")
    ap.add_argument("--out_dir", default="./out", help="Output directory for PNGs")
    ap.add_argument("--thumb_zoom", type=float, default=0.25, help="Overlay thumbnail zoom (default 0.25)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    az_col, el_col, iso_col = detect_axes(df)
    cmap_cols = detect_colormap_columns(df)
    
    if not cmap_cols:
        raise ValueError("No colormap columns found (expected: rainbow, coolwarm, spectral, viridis)")
    
    print(f"Found colormap columns: {cmap_cols}")
    
    # Convert to long format
    df_long = reshape_to_long_format(df, az_col, el_col, iso_col, cmap_cols)
    l2_col = 'L2_diff'
    colormap_col = 'colormap'
    
    for c in (az_col, el_col, iso_col, l2_col):
        df_long[c] = pd.to_numeric(df_long[c], errors="coerce")

    # Find global minimum across all dimensions
    min_idx = df_long[l2_col].idxmin()
    min_row = df_long.loc[min_idx, [az_col, el_col, iso_col, colormap_col, l2_col]]
    az0 = float(min_row[az_col])
    el0 = float(min_row[el_col])
    iso0 = float(min_row[iso_col])
    cmap0 = min_row[colormap_col]

    print(f"\nGlobal min {l2_col}: {min_row[l2_col]:.6g} at {az_col}={az0}, {el_col}={el0}, {iso_col}={iso0}, colormap={cmap0}\n")

    # Check what subdirectories exist
    images_base = Path(args.images_dir)
    print(f"Images directory: {images_base}")
    if images_base.exists():
        subdirs = [d.name for d in images_base.iterdir() if d.is_dir()]
        print(f"Available subdirectories: {subdirs}")
    else:
        print(f"WARNING: Images directory does not exist!")

    # Plot traditional slices for best colormap
    images_index = parse_image_triplets(args.images_dir, cmap0)
    print(f"Loaded {len(images_index)} images for colormap {cmap0}")

    plot_slice(
        df_long, az_col, el_col, iso_col, l2_col,
        fixed_axis=iso_col, fixed_value=iso0,
        images_index=images_index,
        out_path=out_dir / f"slice_iso_{iso0:.6f}_{cmap0}.png",
        thumb_zoom=args.thumb_zoom,
        colormap_col=colormap_col,
        fixed_colormap=cmap0,
    )

    plot_slice(
        df_long, az_col, el_col, iso_col, l2_col,
        fixed_axis=az_col, fixed_value=az0,
        images_index=images_index,
        out_path=out_dir / f"slice_az_{az0:.6f}_{cmap0}.png",
        thumb_zoom=args.thumb_zoom,
        colormap_col=colormap_col,
        fixed_colormap=cmap0,
    )

    plot_slice(
        df_long, az_col, el_col, iso_col, l2_col,
        fixed_axis=el_col, fixed_value=el0,
        images_index=images_index,
        out_path=out_dir / f"slice_el_{el0:.6f}_{cmap0}.png",
        thumb_zoom=args.thumb_zoom,
        colormap_col=colormap_col,
        fixed_colormap=cmap0,
    )

    # Plot colormap slices at optimal values
    # 1. Colormap vs AZ at optimal iso and el
    plot_colormap_slice(
        df_long, az_col, el_col, iso_col, colormap_col, l2_col,
        fixed_axis=iso_col, fixed_value=iso0,
        el_fixed=el0,
        out_path=out_dir / f"slice_colormap_vs_az.png",
        images_dir=args.images_dir,
        thumb_zoom=args.thumb_zoom,
    )

    # 2. Colormap vs ISO at optimal az and el
    plot_colormap_slice(
        df_long, az_col, el_col, iso_col, colormap_col, l2_col,
        fixed_axis=az_col, fixed_value=az0,
        el_fixed=el0,
        out_path=out_dir / f"slice_colormap_vs_iso.png",
        images_dir=args.images_dir,
        thumb_zoom=args.thumb_zoom,
    )

    # 3. Colormap vs EL at optimal az and iso
    plot_colormap_slice(
        df_long, az_col, el_col, iso_col, colormap_col, l2_col,
        fixed_axis=el_col, fixed_value=el0,
        az_fixed=az0,
        iso_fixed=iso0,
        out_path=out_dir / f"slice_colormap_vs_el.png",
        images_dir=args.images_dir,
        thumb_zoom=args.thumb_zoom,
    )

    print(f"\nSaved slices to: {out_dir.resolve()}")

if __name__ == "__main__":
    main()

