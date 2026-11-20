#!/usr/bin/env python3
"""
Compute and display arclengths of colormaps in LAB space.
Uses both Euclidean distance and deltaE2000 metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import color
from colour import delta_E


def compute_colormap_arclength(cmap_name, num_samples=256):
    """
    Compute the arclength of a colormap in LAB space using both Euclidean distance
    and deltaE2000.
    
    Parameters:
        cmap_name : str
            Name of the matplotlib colormap.
        num_samples : int
            Number of samples to use from the colormap.
    
    Returns:
        arclength_euclidean : float
            Total arclength using Euclidean distance in LAB space.
        arclength_deltaE2000 : float
            Total arclength using deltaE2000 metric.
    """
    # Sample the colormap
    cmap = plt.get_cmap(cmap_name)
    sample_scalars = np.linspace(0, 1, num_samples)
    sample_rgbs = np.array([cmap(s)[:3] for s in sample_scalars])
    
    # Convert to LAB
    lab = color.rgb2lab(sample_rgbs.reshape(1, -1, 3)).reshape(-1, 3)
    
    # Compute arclength using Euclidean distance in LAB space
    deltas_euclidean = np.linalg.norm(np.diff(lab, axis=0), axis=1)
    arclength_euclidean = np.sum(deltas_euclidean)
    
    # Compute arclength using deltaE2000
    lab1 = lab[:-1]
    lab2 = lab[1:]
    dE = delta_E(lab1, lab2, method='CIE 2000')
    arclength_deltaE2000 = np.sum(dE)
    
    return arclength_euclidean, arclength_deltaE2000


def print_colormap_arclengths(colormaps=None):
    """
    Compute and print arclengths for specified colormaps.
    Shows both Euclidean and deltaE2000 metrics.
    
    Parameters:
        colormaps : list of str, optional
            List of colormap names. Defaults to ['rainbow', 'Spectral', 'coolwarm', 'viridis'].
    """
    if colormaps is None:
        colormaps = ['rainbow', 'Spectral', 'coolwarm', 'viridis']
    
    print("\n" + "="*80)
    print("Colormap Arclengths in LAB Space")
    print("="*80)
    print(f"{'Colormap':<15s} {'Euclidean':>15s} {'deltaE2000':>15s} {'Ratio (E/dE)':>15s}")
    print("-"*80)
    
    for cmap_name in colormaps:
        try:
            arc_euclidean, arc_deltaE2000 = compute_colormap_arclength(cmap_name, num_samples=256)
            ratio = arc_euclidean / arc_deltaE2000 if arc_deltaE2000 > 0 else 0
            print(f"{cmap_name:<15s} {arc_euclidean:15.4f} {arc_deltaE2000:15.4f} {ratio:15.4f}")
        except Exception as e:
            print(f"{cmap_name:<15s} ERROR: {str(e)}")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    # Print arclengths for default colormaps
    print_colormap_arclengths()
    
    # You can also specify custom colormaps
    # print_colormap_arclengths(['plasma', 'inferno', 'magma', 'cividis'])


