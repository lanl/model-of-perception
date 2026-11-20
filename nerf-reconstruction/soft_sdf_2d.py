import argparse
import os

import numpy as np
import matplotlib.pyplot as plt


def soft_signed_distance_2d(
    x,
    k=200.0,
    k_source=200.0,
    tau=0.25,
    kappa=20.0,
    eps=None,
    iterations=120,
):
    tau = max(float(tau), 1e-12)
    if eps is None:
        eps = float(np.exp(-float(max(x.shape))))

    x_sign = np.clip(k * x, -100.0, 100.0)
    m_sign = 1.0 / (1.0 + np.exp(-x_sign))

    x_src = np.clip(k_source * x, -50.0, 50.0)
    m_src = 1.0 / (1.0 + np.exp(-x_src))

    b = 4.0 * m_src * (1.0 - m_src)
    c_scale = max(1.0, float(max(x.shape)) / 30.0)
    c = -np.log(b + eps) * c_scale
    d = c.copy()

    pad_val = float(d.max()) + 1.0
    for _ in range(iterations):
        d_pad = np.pad(d, 1, mode="constant", constant_values=pad_val)
        d_l = d_pad[1:-1, :-2]
        d_r = d_pad[1:-1, 2:]
        d_u = d_pad[:-2, 1:-1]
        d_d = d_pad[2:, 1:-1]

        a0 = c
        a1 = d_l + 1.0
        a2 = d_r + 1.0
        a3 = d_u + 1.0
        a4 = d_d + 1.0

        min_a = np.minimum.reduce((a0, a1, a2, a3, a4))
        sum_exp = (
            np.exp(-(a0 - min_a) / tau)
            + np.exp(-(a1 - min_a) / tau)
            + np.exp(-(a2 - min_a) / tau)
            + np.exp(-(a3 - min_a) / tau)
            + np.exp(-(a4 - min_a) / tau)
        )
        d = min_a - tau * np.log(sum_exp + 1e-30)

    sign = np.tanh(kappa * (m_sign - 0.5))
    return sign * d


def main():
    parser = argparse.ArgumentParser(description="2D soft SDF demo (x=0 line).")
    parser.add_argument("--radius", type=int, default=50, help="Half-width of grid ([-r, r])")
    parser.add_argument("--iterations", type=int, default=120, help="Soft marching iterations")
    parser.add_argument("--tau", type=float, default=0.25, help="Softmin temperature")
    parser.add_argument("--k", type=float, default=200.0, help="Sign sigmoid sharpness")
    parser.add_argument("--k-source", type=float, default=200.0, help="Boundary sigmoid sharpness")
    parser.add_argument("--kappa", type=float, default=20.0, help="Sign tanh sharpness")
    parser.add_argument("--output", default="soft_sdf_2d.png", help="Output plot filename")
    args = parser.parse_args()

    axis = np.arange(-args.radius, args.radius + 1, dtype=float)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")

    real = xx.copy()

    x_sign = np.clip(args.k * xx, -100.0, 100.0)
    m_sign = 1.0 / (1.0 + np.exp(-x_sign))
    x_src = np.clip(args.k_source * xx, -50.0, 50.0)
    m_src = 1.0 / (1.0 + np.exp(-x_src))
    b = 4.0 * m_src * (1.0 - m_src)

    approx = soft_signed_distance_2d(
        xx,
        k=args.k,
        k_source=args.k_source,
        tau=args.tau,
        kappa=args.kappa,
        iterations=args.iterations,
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    im0 = axes[0, 0].imshow(
        m_sign, origin="lower", extent=[-args.radius, args.radius, -args.radius, args.radius]
    )
    axes[0, 0].set_title("sigmoid (sign)")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im1 = axes[0, 1].imshow(
        b, origin="lower", extent=[-args.radius, args.radius, -args.radius, args.radius]
    )
    axes[0, 1].set_title("boundary likelihood b")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im2 = axes[1, 0].imshow(
        approx, origin="lower", extent=[-args.radius, args.radius, -args.radius, args.radius]
    )
    axes[1, 0].set_title("soft SDF (approx)")
    fig.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

    mid = args.radius
    axes[1, 1].plot(axis, real[mid, :], label="real", linewidth=2)
    axes[1, 1].plot(axis, approx[mid, :], label="approx", linewidth=2)
    axes[1, 1].axhline(0.0, color="black", linewidth=0.5)
    axes[1, 1].axvline(0.0, color="black", linewidth=0.5)
    axes[1, 1].set_title("cross-section at y=0")
    axes[1, 1].set_xlabel("x")
    axes[1, 1].set_ylabel("distance")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    out_path = os.path.abspath(args.output)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"saved plot: {out_path}")


if __name__ == "__main__":
    main()
