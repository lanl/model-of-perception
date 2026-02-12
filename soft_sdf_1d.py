import os

import numpy as np
import matplotlib.pyplot as plt


def soft_signed_distance_1d(
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
        eps = float(np.exp(-float(x.size)))

    # Steep sigmoids to mimic a near-binary interface band.
    x_sign = np.clip(k * x, -100.0, 100.0)
    m_sign = 1.0 / (1.0 + np.exp(-x_sign))

    x_src = np.clip(k_source * x, -50.0, 50.0)
    m_src = 1.0 / (1.0 + np.exp(-x_src))

    # Boundary likelihood and source cost (≈0 on interface, large elsewhere).
    b = 4.0 * m_src * (1.0 - m_src)
    c_scale = max(1.0, float(x.size) / 30.0)
    c = -np.log(b + eps) * c_scale
    d = c.copy()

    # Soft fast marching: log-sum-exp softmin over neighbors and source cost.
    pad_val = float(d.max()) + 1.0
    for _ in range(iterations):
        d_pad = np.pad(d, 1, mode="constant", constant_values=pad_val)
        d_l = d_pad[:-2]
        d_r = d_pad[2:]

        a0 = c
        a1 = d_l + 1.0
        a2 = d_r + 1.0

        min_a = np.minimum.reduce((a0, a1, a2))
        sum_exp = (
            np.exp(-(a0 - min_a) / tau)
            + np.exp(-(a1 - min_a) / tau)
            + np.exp(-(a2 - min_a) / tau)
        )
        d = min_a - tau * np.log(sum_exp + 1e-30)

    # Smooth sign to turn unsigned distance into signed distance.
    sign = np.tanh(kappa * (m_sign - 0.5))
    return sign * d


def global_soft_signed_distance_1d(m_boundary, m_sign, x, tau=0.1, kappa=20.0, eps=1e-6):
    """Sum-over-all-nodes soft distance for a fixed 1D grid x."""
    b = 4.0 * m_boundary * (1.0 - m_boundary)
    c = -np.log(b + eps)
    diff = np.abs(x[:, None] - x[None, :])
    rc = diff + c[None, :]
    m = np.min(rc, axis=1)
    exp_sum = np.exp(-(rc - m[:, None]) / tau)
    D = m - tau * np.log(np.sum(exp_sum, axis=1) + 1e-30)
    sign = np.tanh(kappa * (m_sign - 0.5))
    return sign * D


def build_grid(radius=50):
    """Create the 1D grid. We center it at zero to mimic the interface at x=0."""
    return np.arange(-radius, radius + 1, dtype=float)


def derive_params_from_grid(x):
    """Derive sigmoid sharpness, temperature, and iterations from grid resolution."""
    span = x[-1] - x[0]
    step = float(x[1] - x[0])
    count = int(x.size)
    k = max(150.0, count * 2.0)
    k_source = max(80.0, count)
    tau = max(0.1, step)
    kappa = min(30.0, count / 4.0)
    iterations = max(120, int(span * 2.0))
    return {
        "k": k,
        "k_source": k_source,
        "tau": tau,
        "kappa": kappa,
        "iterations": iterations,
    }


def main():
    x = build_grid()
    params = derive_params_from_grid(x)
    reference = {
        "k": 200.0,
        "k_source": 200.0,
        "tau": 0.25,
        "kappa": 20.0,
        "iterations": 120,
    }
    print("auto-derived params:", params)
    print("reference params:   ", reference)
    real = x.copy()

    def build_sigmas(p):
        x_sign = np.clip(p["k"] * x, -100.0, 100.0)
        m_sign = 1.0 / (1.0 + np.exp(-x_sign))
        x_src = np.clip(p["k_source"] * x, -50.0, 50.0)
        m_src = 1.0 / (1.0 + np.exp(-x_src))
        return m_sign, m_src

    m_sign, m_src = build_sigmas(params)
    approx = soft_signed_distance_1d(x, **params)
    m_sign_ref, m_src_ref = build_sigmas(reference)
    approx_ref = soft_signed_distance_1d(x, **reference)
    approx_global = global_soft_signed_distance_1d(
        m_boundary=m_src, m_sign=m_sign, x=x, tau=params["tau"], kappa=params["kappa"]
    )

    b = 4.0 * m_src * (1.0 - m_src)
    b_ref = 4.0 * m_src_ref * (1.0 - m_src_ref)

    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    axes[0].plot(x, real, label="real signed distance", linewidth=2)
    axes[0].plot(x, approx, label="soft SDF (auto)", linewidth=2)
    axes[0].plot(x, approx_ref, label="soft SDF (reference)", linewidth=2, linestyle="--")
    axes[0].plot(
        x,
        approx_global,
        label="global Dτ (sum over nodes)",
        linewidth=2,
        linestyle=":",
    )
    axes[0].axhline(0.0, color="black", linewidth=0.5)
    axes[0].axvline(0.0, color="black", linewidth=0.5)
    axes[0].set_ylabel("distance")
    axes[0].set_title("1D soft SDF (auto-derived params)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x, m_sign, label="sigmoid (auto)", linewidth=2)
    axes[1].plot(x, m_sign_ref, label="sigmoid (ref)", linewidth=2, linestyle="--")
    axes[1].plot(x, b, label="boundary likelihood b (auto)", linewidth=2)
    axes[1].plot(x, b_ref, label="boundary likelihood b (ref)", linewidth=2, linestyle="--")
    axes[1].axhline(0.0, color="black", linewidth=0.5)
    axes[1].axvline(0.0, color="black", linewidth=0.5)
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("value")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    out_path = os.path.abspath("soft_sdf_1d_auto.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"saved plot: {out_path}")


if __name__ == "__main__":
    main()
