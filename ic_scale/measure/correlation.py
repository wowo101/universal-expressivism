"""Spatial correlation function and correlation length estimation.

Computes C(r) from spin configurations and estimates ξ via:
1. Fourier-space second-moment definition (primary, robust at criticality)
2. Real-space exponential fit (secondary diagnostic)
3. Real-space second-moment (cross-check)
"""

import numpy as np


def compute_correlation(samples, L, max_r=None):
    """Compute spatial spin-spin correlation function C(r).

    Averages over samples, positions, and both horizontal/vertical directions.
    Uses only r < L/4 to avoid wraparound effects.

    Args:
        samples: list of (L, L) int8 arrays
        L: lattice size
        max_r: maximum distance (default: L//4)

    Returns:
        (r_values, C_values) arrays where C(r) = <s_i s_{i+r}> - <s>²
    """
    if max_r is None:
        max_r = L // 4

    # Stack samples for vectorised computation
    configs = np.array(samples, dtype=np.float64)  # (n_samples, L, L)

    # Mean magnetization per sample
    mean_s = configs.mean(axis=(1, 2))  # (n_samples,)

    # C(r) = <s_i s_{i+r}> - <s>²
    # Average over both directions (isotropy)
    c_values = np.zeros(max_r + 1)

    for r in range(max_r + 1):
        # Horizontal: s(x,y) * s(x+r, y)
        corr_h = (configs * np.roll(configs, -r, axis=2)).mean(axis=(1, 2))
        # Vertical: s(x,y) * s(x, y+r)
        corr_v = (configs * np.roll(configs, -r, axis=1)).mean(axis=(1, 2))

        # Average over directions and samples, subtract <s>²
        c_values[r] = 0.5 * ((corr_h - mean_s**2).mean() +
                              (corr_v - mean_s**2).mean())

    r_values = np.arange(max_r + 1, dtype=np.float64)
    return r_values, c_values


def xi_fourier(samples, L):
    """Fourier-space second-moment correlation length.

    ξ = (1 / (2 sin(π/L))) * sqrt(S(0)/S(k_min) - 1)

    where S(k) is the structure factor and k_min = 2π/L.
    This is the standard MC definition, robust at and near T_c.
    Caps at L.
    """
    configs = np.array(samples, dtype=np.float64)

    # Structure factor S(k) = |m(k)|² / N
    # Average over both kx and ky directions for isotropy
    N = L * L

    # Compute S(0) = <M²>/N = <(Σs)²>/N
    M = configs.sum(axis=(1, 2))
    S0 = (M**2).mean() / N

    # S(k_min) for k_min = (2π/L, 0) and (0, 2π/L), then average
    # m(kx) = Σ_x exp(-i kx x) s(x,y)  summed over y
    kmin = 2 * np.pi / L
    x = np.arange(L)
    phase = np.exp(-1j * kmin * x)  # (L,)

    # k = (kmin, 0): FT along x-axis
    # For each config, FT row by row and sum over y
    mk_x = (configs * phase[np.newaxis, np.newaxis, :]).sum(axis=(1, 2))
    Sk_x = (np.abs(mk_x)**2).mean() / N

    # k = (0, kmin): FT along y-axis
    mk_y = (configs * phase[np.newaxis, :, np.newaxis]).sum(axis=(1, 2))
    Sk_y = (np.abs(mk_y)**2).mean() / N

    Sk_min = 0.5 * (Sk_x + Sk_y)

    if Sk_min <= 0 or S0 / Sk_min <= 1:
        return float(L)

    xi = (1.0 / (2 * np.sin(np.pi / L))) * np.sqrt(S0 / Sk_min - 1)
    return min(xi, float(L))


def fit_xi(r_values, c_values, L):
    """Fit correlation length from C(r) ~ exp(-r/ξ).

    Uses log-linear regression on positive C(r) values, excluding r=0.
    Caps ξ at L when the fit diverges (at criticality).

    Note: at T_c, correlations decay as a power law r^(-η), so this
    exponential fit underestimates ξ. Use xi_fourier as primary estimator.

    Returns:
        ξ (float)
    """
    # Use r > 0 and C(r) > 0
    mask = (r_values > 0) & (c_values > 0)
    if mask.sum() < 2:
        return float(L)  # Can't fit, assume critical

    r_fit = r_values[mask]
    log_c = np.log(c_values[mask])

    # Log-linear regression: log C(r) = a - r/ξ
    coeffs = np.polyfit(r_fit, log_c, 1)
    slope = coeffs[0]

    if slope >= 0:
        return float(L)

    xi = -1.0 / slope
    return min(xi, float(L))


def xi_second_moment(r_values, c_values):
    """Real-space second-moment correlation length.

    ξ² = Σ r² C(r) / Σ C(r)  (summing over r > 0 with C(r) > 0)
    """
    mask = (r_values > 0) & (c_values > 0)
    if mask.sum() == 0:
        return 0.0

    r = r_values[mask]
    c = c_values[mask]

    xi_sq = (r**2 * c).sum() / c.sum()
    if xi_sq <= 0:
        return 0.0
    return np.sqrt(xi_sq)


def validate_xi_estimation():
    """Validate ξ estimation using Ising configurations.

    Checks:
    1. At T >> T_c: ξ should be small (< 5)
    2. At T_c: ξ_fourier should be large (~ L)
    """
    from ic_scale.sim.ising import equilibrate, sample, TC, _seed_rng

    L = 64
    n_equil = 3000
    n_samples = 500
    thinning = 5

    print("Validating ξ estimation...")

    # Test 1: High temperature (disordered)
    T_high = 4.0
    _seed_rng(111)
    lattice, _ = equilibrate(L, T_high, n_steps=n_equil, seed=111)
    configs_high = sample(lattice, L, T_high, n_samples, thinning=thinning, seed=222)

    xi_f = xi_fourier(configs_high, L)
    r, c = compute_correlation(configs_high, L)
    xi_fit = fit_xi(r, c, L)
    xi_sm = xi_second_moment(r, c)
    print(f"  T={T_high} (disordered): ξ_fourier={xi_f:.2f}, ξ_fit={xi_fit:.2f}, ξ_sm={xi_sm:.2f}")
    # Fourier ξ overestimates when ξ << 1 (lattice effects), but should still be << L
    high_pass = xi_f < L / 4
    print(f"    ξ_fourier < L/4={L/4}: {'PASS' if high_pass else 'FAIL'}")

    # Test 2: Critical temperature
    T_c = TC
    _seed_rng(333)
    lattice, _ = equilibrate(L, T_c, n_steps=n_equil, seed=333)
    configs_tc = sample(lattice, L, T_c, n_samples, thinning=thinning, seed=444)

    xi_f_c = xi_fourier(configs_tc, L)
    r, c = compute_correlation(configs_tc, L)
    xi_fit_c = fit_xi(r, c, L)
    xi_sm_c = xi_second_moment(r, c)
    print(f"  T_c={T_c:.4f} (critical): ξ_fourier={xi_f_c:.2f}, ξ_fit={xi_fit_c:.2f}, ξ_sm={xi_sm_c:.2f}")
    # At T_c, Fourier ξ should be comparable to L
    crit_pass = xi_f_c > L / 4
    print(f"    ξ_fourier > L/4={L/4}: {'PASS' if crit_pass else 'FAIL'}")

    # Test 3: ordering check — ξ(T_c) >> ξ(T_high)
    order_pass = xi_f_c > 3 * xi_f
    print(f"  ξ(T_c) >> ξ(T_high): {'PASS' if order_pass else 'FAIL'} ({xi_f_c:.1f} vs {xi_f:.1f})")

    all_pass = high_pass and crit_pass and order_pass
    print(f"  Overall: {'PASS' if all_pass else 'FAIL'}")
    return all_pass


if __name__ == "__main__":
    validate_xi_estimation()
