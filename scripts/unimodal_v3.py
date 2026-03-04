import numpy as np

# NOTE: do NOT import matplotlib.pyplot at module level.
# Call apply_thesis_style() before any plot function.


# ----------------------------
# Geometry + utilities on S²
# ----------------------------

def unit(v, eps=1e-15):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)

def tangent_project(x, v):
    xv = np.sum(x * v, axis=-1, keepdims=True)
    return v - xv * x

def fibonacci_sphere(n_points, rng=None):
    # Approximately equal-area points on S²
    if rng is None:
        i = np.arange(n_points)
        u = (i + 0.5) / n_points
    else:
        u = (rng.random(n_points) + np.arange(n_points)) / n_points

    z = 1.0 - 2.0 * u
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    phi = (np.pi * (3.0 - np.sqrt(5.0))) * np.arange(n_points)
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    pts = np.stack([x, y, z], axis=1)
    return unit(pts)

def vmf_norm_const(kappa):
    # vMF normalization constant on S²: c(k)=k/(4π sinh k), c(0)=1/(4π)
    kappa = float(kappa)
    if kappa < 1e-10:
        return 1.0 / (4.0 * np.pi)
    return kappa / (4.0 * np.pi * np.sinh(kappa))


# ----------------------------
# Unimodal target: vMF via U(x) = -alpha (mu·x)
# ----------------------------

def grad_sphere_U(x, mu, alpha):
    # ∇_{S²}U = -alpha * P(x) mu
    mu = mu.reshape(1, 3)
    return -alpha * tangent_project(x, mu)

def drift(x, mu, alpha):
    return -grad_sphere_U(x, mu, alpha)

def target_density_vmf(grid_pts, mu, kappa_target):
    mu = mu.reshape(1, 3)
    dots = (grid_pts @ mu.T).reshape(-1)
    return vmf_norm_const(kappa_target) * np.exp(kappa_target * dots)


# ----------------------------
# Memory-safe vMF KDE (batched) + KL
# ----------------------------

def kde_vmf_density_batched(eval_pts, samples, kappa_kde, batch=1024):
    """
    rho_hat(x) = (1/N) Σ c(k) exp(k x·X_n)
    Computed in batches over samples to avoid building (M,N) matrices.
    """
    c = vmf_norm_const(kappa_kde)
    M = eval_pts.shape[0]
    N = samples.shape[0]
    acc = np.zeros(M, dtype=np.float64)
    for j in range(0, N, batch):
        Xb = samples[j:j + batch]
        dots = eval_pts @ Xb.T
        acc += np.exp(kappa_kde * dots).sum(axis=1)
    return c * (acc / N)

def kl_on_grid(rho, pi, area_weights):
    eps = 1e-15
    rho = np.maximum(rho, eps)
    pi  = np.maximum(pi,  eps)
    return np.sum(rho * (np.log(rho) - np.log(pi)) * area_weights)


# ----------------------------
# Simulation: snapshots + late-time time-averaged density (no sample storage)
# ----------------------------

def simulate_sphere_with_empirical_averaging(
    n_particles=6000,
    T=6.0,
    dt=2e-3,
    beta=4.0,
    mu=np.array([0.0, 0.0, 1.0]),
    alpha=2.0,
    seed=0,
    snapshot_times=(0.0, 3.0, 6.0),
    curve_times=None,
    avg_window=(5.0, 6.0),
    avg_stride=10,
    grid_pts=None,
    kappa_kde=35.0,
    kde_batch=1024,
):
    """
    Runs the intrinsic (Stratonovich-style tangent noise + retraction) Langevin on S²,
    starting from (approximately) uniform points, returns:
      - snapshots at snapshot_times
      - KL values at curve_times (if provided)
      - late-time averaged density rho_avg on grid_pts from time-window averaging
        without storing any samples
    """
    rng = np.random.default_rng(seed)
    mu = unit(mu.reshape(1, 3)).reshape(3)

    n_steps = int(np.round(T / dt))
    X = fibonacci_sphere(n_particles, rng=rng)

    snapshot_times = np.array(sorted(set(float(t) for t in snapshot_times)))
    snap_steps = {int(np.round(t / dt)): t for t in snapshot_times}
    snapshots = {}

    if curve_times is not None:
        curve_times = np.array(sorted(set(float(t) for t in curve_times)))
        curve_steps = {int(np.round(t / dt)): t for t in curve_times}
        kl_curve = {t: None for t in curve_times}
    else:
        curve_steps = {}
        kl_curve = None

    w0, w1 = float(avg_window[0]), float(avg_window[1])
    w0_step = int(np.round(w0 / dt))
    w1_step = int(np.round(w1 / dt))

    if grid_pts is None:
        raise ValueError("grid_pts must be provided to compute rho_avg and KL.")
    M = grid_pts.shape[0]
    c_kde = vmf_norm_const(kappa_kde)
    rho_avg_acc = np.zeros(M, dtype=np.float64)
    rho_avg_count = 0

    kappa_target = beta * alpha
    pi_grid = target_density_vmf(grid_pts, mu, kappa_target)
    area_weights = np.full(M, 4.0 * np.pi / M)

    def compute_kl_from_samples(samples):
        rho = kde_vmf_density_batched(grid_pts, samples, kappa_kde, batch=kde_batch)
        return kl_on_grid(rho, pi_grid, area_weights)

    for k in range(n_steps + 1):
        if k in snap_steps:
            snapshots[snap_steps[k]] = X.copy()

        if k in curve_steps:
            t = curve_steps[k]
            kl_curve[t] = compute_kl_from_samples(X)

        if w0_step <= k <= w1_step and ((k - w0_step) % avg_stride == 0):
            N = X.shape[0]
            for j in range(0, N, kde_batch):
                Xb = X[j:j + kde_batch]
                dots = grid_pts @ Xb.T
                rho_avg_acc += np.exp(kappa_kde * dots).sum(axis=1)
            rho_avg_count += 1

        if k == n_steps:
            break

        b = drift(X, mu, alpha)
        dW = rng.normal(size=X.shape) * np.sqrt(dt)
        dW_tan = tangent_project(X, dW)
        X_star = X + b * dt + np.sqrt(2.0 / beta) * dW_tan
        X = unit(X_star)

    if rho_avg_count > 0:
        rho_avg = c_kde * (rho_avg_acc / (rho_avg_count * n_particles))
    else:
        rho_avg = kde_vmf_density_batched(grid_pts, X, kappa_kde, batch=kde_batch)

    return snapshots, (curve_times, kl_curve, pi_grid, rho_avg)


# ----------------------------
# Public simulation wrapper (for figures.py)
# ----------------------------

def simulate_unimodal_v3(
    beta=4.0,
    alpha=2.0,
    mu=None,
    T=6.0,
    dt=2e-3,
    n_particles=6000,
    grid_M=2500,
    kappa_kde=35.0,
    kde_batch=768,
    t0=0.0,
    tm=3.0,
    tT=6.0,
    avg_window=(5.0, 6.0),
    avg_stride=10,
    seed=0,
):
    """
    Convenience wrapper: builds grid, runs simulation, returns all plot-ready data.

    Returns dict: grid_pts, curve_times, kls, log_rho0, log_rhom, log_rho_avg,
                  log_ratio_avg, t0, tm, tT, kl0, klm, kl_avg, avg_window
    """
    if mu is None:
        mu = np.array([0.0, 0.0, 1.0])

    curve_times = np.linspace(0.0, T, 41)
    grid_pts = fibonacci_sphere(grid_M)

    snapshots, (curve_times_out, kl_curve_map, pi_grid, rho_avg) = \
        simulate_sphere_with_empirical_averaging(
            n_particles=n_particles, T=T, dt=dt,
            beta=beta, mu=mu, alpha=alpha, seed=seed,
            snapshot_times=(t0, tm, tT),
            curve_times=curve_times,
            avg_window=avg_window, avg_stride=avg_stride,
            grid_pts=grid_pts, kappa_kde=kappa_kde, kde_batch=kde_batch,
        )

    rho0 = kde_vmf_density_batched(grid_pts, snapshots[t0], kappa_kde, batch=kde_batch)
    rhom = kde_vmf_density_batched(grid_pts, snapshots[tm], kappa_kde, batch=kde_batch)

    eps = 1e-15
    log_rho0     = np.log(np.maximum(rho0,     eps))
    log_rhom     = np.log(np.maximum(rhom,     eps))
    log_rho_avg  = np.log(np.maximum(rho_avg,  eps))
    log_ratio_avg = log_rho_avg - np.log(np.maximum(pi_grid, eps))

    kls = np.array([kl_curve_map[float(t)] for t in curve_times_out])
    area_weights = np.full(grid_M, 4.0 * np.pi / grid_M)
    kl0    = kl_on_grid(rho0,    pi_grid, area_weights)
    klm    = kl_on_grid(rhom,    pi_grid, area_weights)
    kl_avg = kl_on_grid(rho_avg, pi_grid, area_weights)

    return {
        "grid_pts": grid_pts,
        "curve_times": curve_times_out,
        "kls": kls,
        "log_rho0": log_rho0,
        "log_rhom": log_rhom,
        "log_rho_avg": log_rho_avg,
        "log_ratio_avg": log_ratio_avg,
        "t0": t0, "tm": tm, "tT": tT,
        "kl0": kl0, "klm": klm, "kl_avg": kl_avg,
        "avg_window": avg_window,
    }


# ----------------------------
# Plotting helpers
# ----------------------------

def sphere_scatter(ax, pts, values, title="", s=6):
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=values, s=s)
    ax.set_title(title)
    ax.set_box_aspect([1, 1, 1])
    ax.set_axis_off()


def plot_unimodal_v3(result, outpath, colors):
    """4-panel: sphere densities at t0, tm, late-time avg + KL decay curve."""
    import matplotlib.pyplot as plt

    r = result
    fig = plt.figure(figsize=(14, 4))

    ax1 = fig.add_subplot(141, projection="3d")
    sphere_scatter(ax1, r["grid_pts"], r["log_rho0"],
                   title=f"Start (t={r['t0']:g}) log ρ̂", s=6)

    ax2 = fig.add_subplot(142, projection="3d")
    sphere_scatter(ax2, r["grid_pts"], r["log_rhom"],
                   title=f"Middle (t={r['tm']:g}) log ρ̂", s=6)

    ax3 = fig.add_subplot(143, projection="3d")
    sphere_scatter(ax3, r["grid_pts"], r["log_rho_avg"],
                   title=(f"Late-time avg log ρ̂ "
                          f"(t∈[{r['avg_window'][0]}, {r['avg_window'][1]}])"), s=6)

    ax4 = fig.add_subplot(144)
    ax4.plot(r["curve_times"], r["kls"], marker="o", markersize=3)
    ax4.set_xlabel("time")
    ax4.set_ylabel(r"KL($\rho_t \| \pi$)")
    ax4.set_title("Free energy decay (relative entropy)")
    ax4.scatter([r["t0"], r["tm"], r["tT"]], [r["kl0"], r["klm"], r["kl_avg"]], zorder=3)
    ax4.annotate("start",    (r["t0"], r["kl0"]),    textcoords="offset points", xytext=(6, 6))
    ax4.annotate("mid",      (r["tm"], r["klm"]),    textcoords="offset points", xytext=(6, 6))
    ax4.annotate("late avg", (r["tT"], r["kl_avg"]), textcoords="offset points", xytext=(6, 6))

    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def plot_unimodal_v3_ratio(result, outpath, colors):
    """Convergence check: log(ρ̂/π) on sphere (flat when converged)."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111, projection="3d")
    sphere_scatter(ax, result["grid_pts"], result["log_ratio_avg"],
                   title=r"Late-time avg $\log(\hat\rho/\pi)$ (flat $\Rightarrow$ converged)", s=6)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


# ----------------------------
# Standalone
# ----------------------------

def main():
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from viz_style import apply_thesis_style
    import matplotlib.pyplot as plt

    colors = apply_thesis_style()
    result = simulate_unimodal_v3()
    plot_unimodal_v3(result, "unimodal_v3.pdf", colors)
    plot_unimodal_v3_ratio(result, "unimodal_v3_ratio.pdf", colors)
    print("Saved unimodal_v3.pdf and unimodal_v3_ratio.pdf")
    plt.show()


if __name__ == "__main__":
    main()
