import numpy as np
import matplotlib.pyplot as plt

# 2D grid: coordinates in [-50, 50] x [-50, 50]
coords = np.arange(-50, 51, 1, dtype=float)
X, Y = np.meshgrid(coords, coords, indexing="xy")  # shape (H,W)

# Soft "contour" along x=y via a steep sigmoid on (x-y)
k_sig = 1.0
s = 1.0 / (1.0 + np.exp(-k_sig * (X - Y)))  # s=0.5 on x=y

# Soft boundary weight and barrier seed cost
b = 4.0 * s * (1.0 - s)      # peaks at contour
M = 1e3
c = M * (1.0 - b)            # ~0 on contour, ~M elsewhere

# Soft Bellman / distance transform on 4-neighborhood (graph distance)
tau = 0.25
T = 120  # >= grid diameter (~100)

D = c.copy()

def softmin_stack(vals, tau):
    # vals: (K,H,W)
    v = -vals / tau
    vmax = np.max(v, axis=0, keepdims=True)
    return -tau * (vmax[0] + np.log(np.sum(np.exp(v - vmax), axis=0)))

for _ in range(T):
    # replicate padding shifts (no wrap-around)
    up = np.empty_like(D); down = np.empty_like(D); left = np.empty_like(D); right = np.empty_like(D)
    up[0, :] = D[0, :] + 1.0
    up[1:, :] = D[:-1, :] + 1.0
    down[-1, :] = D[-1, :] + 1.0
    down[:-1, :] = D[1:, :] + 1.0
    left[:, 0] = D[:, 0] + 1.0
    left[:, 1:] = D[:, :-1] + 1.0
    right[:, -1] = D[:, -1] + 1.0
    right[:, :-1] = D[:, 1:] + 1.0

    vals = np.stack([c, up, down, left, right], axis=0)  # stop here or come from neighbors
    D = softmin_stack(vals, tau)

D = D - D.min()

# True 4-neighbor graph distance to diagonal x=y on this grid is |i-j| in index space
# (since one step changes x or y by 1, and reducing |x-y| by 1 per step is optimal)
D_true = np.abs(X - Y)

# Plot
plt.figure()
plt.imshow(D_true, origin="lower", extent=[coords[0], coords[-1], coords[0], coords[-1]])
plt.colorbar()
plt.title("True 4-neighbor graph distance to contour x=y : |x-y|")
plt.xlabel("x"); plt.ylabel("y")
plt.show()

plt.figure()
plt.imshow(D, origin="lower", extent=[coords[0], coords[-1], coords[0], coords[-1]])
plt.colorbar()
plt.title(f"Soft Bellman + barrier (tau={tau}, M={M}, T={T})")
plt.xlabel("x"); plt.ylabel("y")
plt.show()

# Difference
plt.figure()
plt.imshow(D - D_true, origin="lower", extent=[coords[0], coords[-1], coords[0], coords[-1]])
plt.colorbar()
plt.title("Difference: (soft Bellman) - (true)")
plt.xlabel("x"); plt.ylabel("y")
plt.show()
