"""Validation: L=64 pipeline reproducing browser explorer T₀(m=4) ≈ 2.272.

At L=64, we have 3 RG levels (ℓ=0,1,2 → lattices 64,32,16).
T₀ is the temperature where dκ/dℓ for m=4 crosses zero.
The browser explorer found T₀(m=4) ≈ 2.272, within 0.003 of T_c ≈ 2.269.
"""

import json
import os
import time
import numpy as np

from ic_scale import config
from ic_scale.sim.ising import TC, _seed_rng
from ic_scale.run_experiment import process_temperature, RESULTS_DIR


# Override config for L=64 run
L64_RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results_L64")


def run_L64_validation():
    """Run L=64 across critical-regime temperatures and find T₀(m=4)."""

    # Dense temperature grid around T_c for accurate T₀ determination
    t_grid = np.sort(np.concatenate([
        np.linspace(1.5, 1.9, 3),       # ordered (sparse)
        np.linspace(2.0, 2.6, 25),       # critical regime (dense)
        np.linspace(2.7, 3.5, 3),        # disordered (sparse)
    ]))

    n_samples = 2000
    n_equil = 3000
    n_blocks = 150
    n_bootstrap = 30

    # Temporarily override config for L=64
    orig_L0 = config.L0
    orig_N_RG = config.N_RG_LEVELS
    config.L0 = 64
    config.N_RG_LEVELS = 2  # levels 0,1,2 → 64,32,16

    os.makedirs(L64_RESULTS_DIR, exist_ok=True)

    # Monkey-patch result path to use L64 directory
    import ic_scale.run_experiment as run_mod
    orig_results_dir = run_mod.RESULTS_DIR
    run_mod.RESULTS_DIR = L64_RESULTS_DIR

    print(f"L=64 Validation Run")
    print(f"  T₀(m=4) target: ≈ 2.272 (browser explorer)")
    print(f"  T_c = {TC:.4f}")
    print(f"  N_SAMPLES={n_samples}, {len(t_grid)} temperatures")
    print()

    t_total_start = time.time()

    for i, T in enumerate(t_grid):
        rpath = os.path.join(L64_RESULTS_DIR, f"T_{T:.4f}.json")
        if os.path.exists(rpath):
            print(f"  [{i+1}/{len(t_grid)}] T={T:.4f} — skipped (exists)")
            continue

        t0 = time.time()
        result = process_temperature(
            T, n_samples=n_samples, n_equil=n_equil,
            n_blocks=n_blocks, n_bootstrap=n_bootstrap
        )
        dt = time.time() - t0

        # Quick summary
        m4_kappas = []
        for lv in result["levels"]:
            m4 = lv["series"].get("m4_fixed", {})
            if m4:
                m4_kappas.append(m4["kappa"])
        kstr = ", ".join(f"{k:.4f}" for k in m4_kappas)
        print(f"  [{i+1}/{len(t_grid)}] T={T:.4f}  {dt:.1f}s  κ4=[{kstr}]")

    t_total = time.time() - t_total_start
    print(f"\nTotal runtime: {t_total:.0f}s")

    # Restore config
    config.L0 = orig_L0
    config.N_RG_LEVELS = orig_N_RG
    run_mod.RESULTS_DIR = orig_results_dir

    # Analyze results
    print("\n" + "="*60)
    print("ANALYSIS: Finding T₀(m=4)")
    print("="*60)
    analyze_T0(t_grid)


def analyze_T0(t_grid):
    """Compute dκ/dℓ at each temperature and find zero crossing."""

    temperatures = []
    slopes = []
    kappa_data = []  # (T, [κ_ℓ0, κ_ℓ1, κ_ℓ2])

    for T in t_grid:
        rpath = os.path.join(L64_RESULTS_DIR, f"T_{T:.4f}.json")
        if not os.path.exists(rpath):
            continue

        with open(rpath) as f:
            result = json.load(f)

        # Extract m=4 κ values across levels
        kappas = []
        for lv in result["levels"]:
            m4 = lv["series"].get("m4_fixed", {})
            if m4 and not np.isnan(m4["kappa"]):
                kappas.append(m4["kappa"])

        if len(kappas) < 2:
            continue

        # Linear regression: κ vs ℓ
        ells = np.arange(len(kappas), dtype=np.float64)
        slope, intercept = np.polyfit(ells, kappas, 1)

        temperatures.append(T)
        slopes.append(slope)
        kappa_data.append(kappas)

    temperatures = np.array(temperatures)
    slopes = np.array(slopes)

    # Print slope table
    print(f"\n{'T':>8s}  {'slope':>8s}  {'κ(ℓ=0)':>8s}  {'κ(ℓ=1)':>8s}  {'κ(ℓ=2)':>8s}")
    print("-" * 50)
    for i, T in enumerate(temperatures):
        k = kappa_data[i]
        kstr = "  ".join(f"{kk:8.4f}" for kk in k)
        marker = " <--" if abs(slopes[i]) < 0.01 else ""
        print(f"{T:8.4f}  {slopes[i]:8.4f}  {kstr}{marker}")

    # Find T₀: linear interpolation of slope zero crossing
    # slope goes from positive (ordered) to negative (disordered)
    T0 = None
    for i in range(len(slopes) - 1):
        if slopes[i] >= 0 and slopes[i+1] < 0:
            # Linear interpolation
            frac = slopes[i] / (slopes[i] - slopes[i+1])
            T0 = temperatures[i] + frac * (temperatures[i+1] - temperatures[i])
            break

    if T0 is None:
        # Try the other direction (negative to positive shouldn't happen, but be safe)
        # Also check for closest to zero
        idx_min = np.argmin(np.abs(slopes))
        T0 = temperatures[idx_min]
        print(f"\nNo clean zero crossing found. Closest to zero: T={T0:.4f}, slope={slopes[idx_min]:.4f}")
    else:
        print(f"\nT₀(m=4) = {T0:.4f}")

    print(f"T_c      = {TC:.4f}")
    print(f"Browser  = 2.272")
    if T0 is not None:
        offset = T0 - TC
        browser_diff = abs(T0 - 2.272)
        print(f"T₀ - T_c = {offset:+.4f}")
        print(f"|T₀ - 2.272| = {browser_diff:.4f}")

        # Pass/fail: within 0.03 of browser value (allowing for statistical noise)
        passed = browser_diff < 0.03
        print(f"\nValidation: {'PASS' if passed else 'FAIL'} "
              f"(within 0.03 of browser T₀=2.272: {browser_diff:.4f})")
        return passed

    return False


if __name__ == "__main__":
    run_L64_validation()
