"""Validation for mutual information estimators."""

import numpy as np
from ic_scale.info.mi_plugin import mi_plugin, spins_to_states_batch
from ic_scale.info.mi_magnetization import mi_magnetization, spins_to_magnetizations_batch


def validate_mi_known():
    """Validate MI estimators on synthetic data with known MI.

    Creates a bivariate distribution where X has 3 equiprobable states
    and Y = X (perfect correlation), giving MI = log2(3) ≈ 1.585 bits.
    """
    print("Validating MI estimators on known distribution...")

    rng = np.random.default_rng(42)
    N = 50000
    expected_mi = np.log2(3)

    # Create synthetic data: 2 spins for interior, 2 spins for blanket
    # Interior takes 3 of 4 possible states equiprobably
    # Blanket = Interior (perfect correlation)
    n_int_spins = 2
    n_bla_spins = 2

    # States 0, 1, 2 (skip state 3) — gives H = log2(3)
    states = rng.integers(0, 3, size=N)

    # Plugin estimator test
    mi, h_int, h_bla = mi_plugin(states, states, n_int_spins, n_bla_spins, alpha=0.5)
    ratio = mi / expected_mi
    print(f"  Plugin: MI={mi:.4f}, expected={expected_mi:.4f}, ratio={ratio:.4f}")
    plugin_pass = abs(ratio - 1.0) < 0.03
    print(f"  Plugin check: {'PASS' if plugin_pass else 'FAIL'} (within 3%: {abs(ratio-1.0)*100:.1f}%)")

    # Magnetization estimator test with a different setup:
    # Use 4 interior spins and 4 blanket spins with correlated magnetizations
    n_int_spins_m = 4
    n_bla_spins_m = 4
    n_int_bins = n_int_spins_m + 1  # 5 bins
    n_bla_bins = n_bla_spins_m + 1  # 5 bins

    # Create correlated magnetizations: blanket_mag = interior_mag + noise
    int_mags = rng.choice(np.array([-4, -2, 0, 2, 4]), size=N)
    # Add some noise but keep correlation
    noise = rng.choice(np.array([-2, 0, 2]), size=N, p=[0.15, 0.7, 0.15])
    bla_mags = np.clip(int_mags + noise, -4, 4)
    # Round to valid magnetization values
    bla_mags = (np.round(bla_mags / 2) * 2).astype(np.int64)

    mi_m, h_int_m, h_bla_m = mi_magnetization(int_mags, bla_mags, n_int_spins_m, n_bla_spins_m)
    print(f"  Magnetization: MI={mi_m:.4f}, H_int={h_int_m:.4f}, H_bla={h_bla_m:.4f}")
    # MI should be positive and less than min(H_int, H_bla)
    mag_pass = mi_m > 0 and mi_m < min(h_int_m, h_bla_m)
    print(f"  Magnetization bounds check: {'PASS' if mag_pass else 'FAIL'} (0 < MI < min(H_int, H_bla))")

    # Test with independent variables — MI should be near 0
    ind_int = rng.integers(0, 4, size=N)
    ind_bla = rng.integers(0, 4, size=N)
    mi_ind, _, _ = mi_plugin(ind_int, ind_bla, n_int_spins, n_bla_spins)
    ind_pass = abs(mi_ind) < 0.05
    print(f"  Independent variables: MI={mi_ind:.4f} (should be ≈0)")
    print(f"  Independence check: {'PASS' if ind_pass else 'FAIL'}")

    all_pass = plugin_pass and mag_pass and ind_pass
    print(f"  Overall: {'PASS' if all_pass else 'FAIL'}")
    return all_pass


def validate_mi_cross():
    """Cross-validate plugin and magnetization estimators on Ising data.

    For m=2 blocks, both methods should agree on the ranking
    (higher MI at T_c than at T >> T_c).
    """
    from ic_scale.sim.ising import equilibrate, sample, TC, _seed_rng
    from ic_scale.sim.coarse_grain import extract_regions

    print("Cross-validating MI estimators on Ising data (m=2)...")

    L = 32
    m = 2
    n_equil = 2000
    n_samples = 1000
    n_blocks = 100
    thinning = 5

    results = {}
    for T, label in [(TC, "T_c"), (3.5, "T_high")]:
        _seed_rng(int(T * 10000))
        lattice, _ = equilibrate(L, T, n_steps=n_equil, seed=int(T * 10000))
        configs = sample(lattice, L, T, n_samples, thinning=thinning, seed=int(T * 10000 + 1))

        rng = np.random.default_rng(42)

        # Collect interior/blanket data
        int_spins_list = []
        bla_spins_list = []
        for config in configs:
            for _ in range(n_blocks):
                cx = rng.integers(L)
                cy = rng.integers(L)
                interior, blanket = extract_regions(config, m, cx, cy)
                int_spins_list.append(interior)
                bla_spins_list.append(blanket)

        int_spins = np.array(int_spins_list)
        bla_spins = np.array(bla_spins_list)

        # Plugin MI (using state indices)
        int_states = spins_to_states_batch(int_spins)
        bla_states = spins_to_states_batch(bla_spins)
        mi_p, h_int_p, h_bla_p = mi_plugin(
            int_states, bla_states, m * m, (m + 2)**2 - m**2
        )

        # Magnetization MI
        int_mags = spins_to_magnetizations_batch(int_spins)
        bla_mags = spins_to_magnetizations_batch(bla_spins)
        mi_m, h_int_m, h_bla_m = mi_magnetization(
            int_mags, bla_mags, m * m, (m + 2)**2 - m**2
        )

        results[label] = (mi_p, mi_m)
        print(f"  {label} (T={T:.4f}): plugin MI={mi_p:.4f}, mag MI={mi_m:.4f}")

    # Both methods should rank T_c > T_high
    plugin_ranking = results["T_c"][0] > results["T_high"][0]
    mag_ranking = results["T_c"][1] > results["T_high"][1]

    print(f"  Plugin ranks T_c > T_high: {'PASS' if plugin_ranking else 'FAIL'}")
    print(f"  Magnetization ranks T_c > T_high: {'PASS' if mag_ranking else 'FAIL'}")

    all_pass = plugin_ranking and mag_ranking
    print(f"  Overall: {'PASS' if all_pass else 'FAIL'}")
    return all_pass


if __name__ == "__main__":
    validate_mi_known()
    print()
    validate_mi_cross()
