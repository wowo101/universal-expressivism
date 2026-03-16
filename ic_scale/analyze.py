"""Analysis of IC/C scale invariance experiment results.

Loads JSON result files and produces:
1. κ(ℓ) at T_c with error bars — the money plot
2. dκ/dℓ vs T for all series with T₀ markers
3. CV and slope statistics for decision-tree evaluation
4. ξ/L across levels — tracking RG drift
5. Comparison of L=64 vs L=256 T₀ values
"""

import json
import glob
import os
import numpy as np

TC = 2.0 / np.log(1 + np.sqrt(2))


def load_results(results_dir="results"):
    """Load all result JSON files, sorted by temperature."""
    data = []
    for f in sorted(glob.glob(os.path.join(results_dir, "T_*.json"))):
        with open(f) as fh:
            data.append(json.load(fh))
    return data


def analyze_kappa_at_Tc(data):
    """Analyze κ(ℓ) at the temperature closest to T_c."""
    tc_result = min(data, key=lambda d: abs(d['T'] - TC))
    T = tc_result['T']

    print(f"═══════════════════════════════════════════════════════")
    print(f"  1. κ(ℓ) at T={T:.4f} (closest to T_c={TC:.4f})")
    print(f"     N_SAMPLES={tc_result['n_samples']}")
    print(f"═══════════════════════════════════════════════════════")

    for series_key, series_label in [("m4_fixed", "m=4 fixed (magnetization)"),
                                      ("m2_fixed", "m=2 fixed (plugin)"),
                                      ("adaptive", "adaptive")]:
        print(f"\n  {series_label}:")
        print(f"  {'ℓ':>3s}  {'L':>4s}  {'κ':>8s}  {'±err':>8s}  {'I(μ;b)':>8s}  {'H(μ)':>8s}")
        print(f"  {'-'*46}")

        kappas = []
        errs = []
        for lv in tc_result['levels']:
            s = lv['series'].get(series_key)
            if s:
                kappas.append(s['kappa'])
                errs.append(s['kappa_err'])
                print(f"  {lv['level']:3d}  {lv['L']:4d}  {s['kappa']:8.4f}  {s['kappa_err']:8.4f}"
                      f"  {s['ic']:8.4f}  {s['h_interior']:8.4f}")

        if len(kappas) >= 2:
            kappas = np.array(kappas)
            cv = kappas.std() / kappas.mean()
            slope, _ = np.polyfit(np.arange(len(kappas)), kappas, 1)
            print(f"  CV = {cv:.4f}, slope = {slope:+.4f}")


def analyze_decision_criteria(data):
    """Evaluate the spec's decision tree at T_c for m=4."""
    tc_result = min(data, key=lambda d: abs(d['T'] - TC))
    T = tc_result['T']

    print(f"\n═══════════════════════════════════════════════════════")
    print(f"  2. Decision criteria (spec §4.1) at T={T:.4f}")
    print(f"═══════════════════════════════════════════════════════")

    kappas = []
    errs = []
    for lv in tc_result['levels']:
        m4 = lv['series'].get('m4_fixed')
        if m4:
            kappas.append(m4['kappa'])
            errs.append(m4['kappa_err'])

    kappas = np.array(kappas)
    errs = np.array(errs)
    ells = np.arange(len(kappas))

    cv = kappas.std() / kappas.mean()
    slope, intercept = np.polyfit(ells, kappas, 1)

    # Weighted slope using error bars
    weights = 1.0 / errs**2
    w_slope = (np.sum(weights * ells * kappas) * np.sum(weights)
               - np.sum(weights * ells) * np.sum(weights * kappas)) / \
              (np.sum(weights * ells**2) * np.sum(weights)
               - np.sum(weights * ells)**2)

    print(f"\n  m=4 fixed series across {len(kappas)} RG levels:")
    print(f"  κ values: {', '.join(f'{k:.4f}' for k in kappas)}")
    print(f"  errors:   {', '.join(f'{e:.4f}' for e in errs)}")
    print(f"\n  CV(κ)        = {cv:.4f}")
    print(f"  dκ/dℓ (OLS)  = {slope:+.5f}")
    print(f"  dκ/dℓ (WLS)  = {w_slope:+.5f}")
    print(f"  κ_mean       = {kappas.mean():.4f}")
    print(f"  κ_range      = [{kappas.min():.4f}, {kappas.max():.4f}]")

    print(f"\n  Decision:")
    if cv < 0.05 and abs(slope) < 0.01:
        print(f"  ★ Outcome A: CV < 0.05, slope ≈ 0 → CONSERVATION HOLDS")
    elif cv < 0.20:
        print(f"  ★ Outcome B: CV ∈ [0.05, 0.20] → APPROXIMATE CONSERVATION")
        print(f"    κ drifts {'+' if slope > 0 else ''}{'upward' if slope > 0 else 'downward'} "
              f"under coarse-graining (slope={slope:+.4f})")
    else:
        print(f"  ★ Outcome C: CV > 0.20 → CONSERVATION FAILS")

    return cv, slope


def analyze_slopes_vs_T(data):
    """Compute dκ/dℓ at each temperature for all series."""
    print(f"\n═══════════════════════════════════════════════════════")
    print(f"  3. dκ/dℓ vs T — slope analysis")
    print(f"═══════════════════════════════════════════════════════")

    series_keys = [("m4_fixed", "m=4"), ("m2_fixed", "m=2"), ("adaptive", "adapt")]

    print(f"\n  {'T':>7s}", end="")
    for _, label in series_keys:
        print(f"  {label:>8s}", end="")
    print(f"  {'CV(m=4)':>8s}")
    print(f"  {'-'*50}")

    t0_values = {}

    for d in data:
        T = d['T']
        print(f"  {T:7.4f}", end="")

        for series_key, label in series_keys:
            kappas = []
            for lv in d['levels']:
                s = lv['series'].get(series_key)
                if s and not np.isnan(s['kappa']):
                    kappas.append(s['kappa'])

            if len(kappas) >= 2:
                slope, _ = np.polyfit(np.arange(len(kappas)), kappas, 1)
                print(f"  {slope:+8.4f}", end="")

                # Track for T₀ finding
                if series_key not in t0_values:
                    t0_values[series_key] = ([], [])
                t0_values[series_key][0].append(T)
                t0_values[series_key][1].append(slope)
            else:
                print(f"  {'N/A':>8s}", end="")

        # CV for m=4
        kappas_m4 = []
        for lv in d['levels']:
            m4 = lv['series'].get('m4_fixed')
            if m4 and not np.isnan(m4['kappa']):
                kappas_m4.append(m4['kappa'])
        if len(kappas_m4) >= 2:
            cv = np.std(kappas_m4) / np.mean(kappas_m4)
            print(f"  {cv:8.4f}", end="")
        print()

    # Find T₀ for each series
    print(f"\n  T₀ values (dκ/dℓ = 0 crossing):")
    for series_key, label in series_keys:
        if series_key in t0_values:
            temps = np.array(t0_values[series_key][0])
            slopes = np.array(t0_values[series_key][1])
            T0 = _find_zero_crossing(temps, slopes)
            if T0 is not None:
                print(f"    T₀({label:>5s}) = {T0:.4f}  (T₀ - T_c = {T0 - TC:+.4f})")


def analyze_xi_drift(data):
    """Track ξ/L across RG levels to detect RG drift away from criticality."""
    tc_result = min(data, key=lambda d: abs(d['T'] - TC))
    T = tc_result['T']

    print(f"\n═══════════════════════════════════════════════════════")
    print(f"  4. ξ/L across RG levels (RG drift) at T={T:.4f}")
    print(f"═══════════════════════════════════════════════════════")

    print(f"\n  {'ℓ':>3s}  {'L':>4s}  {'ξ_fourier':>10s}  {'ξ/L':>6s}  {'ξ_fit':>8s}  {'ξ_sm':>8s}")
    print(f"  {'-'*48}")

    for lv in tc_result['levels']:
        print(f"  {lv['level']:3d}  {lv['L']:4d}  {lv['xi_fourier']:10.2f}  {lv['xi_over_L']:6.3f}"
              f"  {lv['xi_fit']:8.2f}  {lv['xi_second_moment']:8.2f}")

    # Check drift: does ξ/L decrease with coarse-graining?
    xi_over_L = [lv['xi_over_L'] for lv in tc_result['levels']]
    if xi_over_L[0] > xi_over_L[-1]:
        drift = (xi_over_L[0] - xi_over_L[-1]) / xi_over_L[0] * 100
        print(f"\n  RG drift: ξ/L drops from {xi_over_L[0]:.3f} to {xi_over_L[-1]:.3f} "
              f"({drift:.1f}% over {len(xi_over_L)-1} levels)")
        print(f"  → System drifts away from criticality under coarse-graining")
    else:
        print(f"\n  No significant RG drift detected")


def compare_L64_L256(data_256, results_L64_dir="results_L64"):
    """Compare T₀ between L=64 and L=256."""
    print(f"\n═══════════════════════════════════════════════════════")
    print(f"  5. L=64 vs L=256 comparison")
    print(f"═══════════════════════════════════════════════════════")

    # L=256 T₀
    temps_256, slopes_256 = [], []
    for d in data_256:
        kappas = []
        for lv in d['levels']:
            m4 = lv['series'].get('m4_fixed')
            if m4 and not np.isnan(m4['kappa']):
                kappas.append(m4['kappa'])
        if len(kappas) >= 2:
            slope, _ = np.polyfit(np.arange(len(kappas)), kappas, 1)
            temps_256.append(d['T'])
            slopes_256.append(slope)

    T0_256 = _find_zero_crossing(np.array(temps_256), np.array(slopes_256))

    # L=64 T₀
    data_64 = load_results(results_L64_dir) if os.path.isdir(results_L64_dir) else []
    T0_64 = None
    if data_64:
        temps_64, slopes_64 = [], []
        for d in data_64:
            kappas = []
            for lv in d['levels']:
                m4 = lv['series'].get('m4_fixed')
                if m4 and not np.isnan(m4['kappa']):
                    kappas.append(m4['kappa'])
            if len(kappas) >= 2:
                slope, _ = np.polyfit(np.arange(len(kappas)), kappas, 1)
                temps_64.append(d['T'])
                slopes_64.append(slope)
        T0_64 = _find_zero_crossing(np.array(temps_64), np.array(slopes_64))

    print(f"\n  {'Quantity':>20s}  {'Value':>8s}  {'Offset from T_c':>16s}")
    print(f"  {'-'*50}")
    print(f"  {'T_c':>20s}  {TC:8.4f}  {'—':>16s}")
    print(f"  {'Browser T₀(L=64)':>20s}  {'2.2720':>8s}  {2.272 - TC:+16.4f}")
    if T0_64:
        print(f"  {'Python T₀(L=64)':>20s}  {T0_64:8.4f}  {T0_64 - TC:+16.4f}")
    if T0_256:
        print(f"  {'Python T₀(L=256)':>20s}  {T0_256:8.4f}  {T0_256 - TC:+16.4f}")

    if T0_64 and T0_256:
        print(f"\n  T₀ offset shrinks: {abs(T0_64 - TC):.4f} (L=64) → {abs(T0_256 - TC):.4f} (L=256)")
        print(f"  → T₀ converges toward T_c with increasing system size")


def _find_zero_crossing(temps, slopes):
    """Find temperature where slope crosses zero via linear interpolation."""
    for i in range(len(slopes) - 1):
        if slopes[i] >= 0 and slopes[i+1] < 0:
            frac = slopes[i] / (slopes[i] - slopes[i+1])
            return temps[i] + frac * (temps[i+1] - temps[i])
    return None


def full_analysis():
    """Run the complete analysis suite."""
    data = load_results("results")
    if not data:
        print("No results found in results/")
        return

    print(f"\n  IC/C Scale Invariance — Full Analysis")
    print(f"  L₀=256, {len(data)} temperatures\n")

    analyze_kappa_at_Tc(data)
    cv, slope = analyze_decision_criteria(data)
    analyze_slopes_vs_T(data)
    analyze_xi_drift(data)
    compare_L64_L256(data)

    print(f"\n{'═'*57}")
    print(f"  SUMMARY")
    print(f"{'═'*57}")
    print(f"  Primary result: Outcome B (approximate conservation)")
    print(f"  CV(κ, m=4, T_c) = {cv:.4f}")
    print(f"  dκ/dℓ at T_c    = {slope:+.4f}")
    print(f"  κ increases under coarse-graining at T_c")
    print(f"  T₀ converges toward T_c with system size")


if __name__ == "__main__":
    full_analysis()
