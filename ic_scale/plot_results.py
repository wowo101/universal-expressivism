"""Generate publication-quality figures for the IC/C scale invariance results."""

import json
import glob
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

TC = 2.0 / np.log(1 + np.sqrt(2))
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figures")


def load_results(results_dir="results"):
    data = []
    for f in sorted(glob.glob(os.path.join(results_dir, "T_*.json"))):
        with open(f) as fh:
            data.append(json.load(fh))
    return data


def _find_zero_crossing(temps, slopes):
    for i in range(len(slopes) - 1):
        if slopes[i] >= 0 and slopes[i+1] < 0:
            frac = slopes[i] / (slopes[i] - slopes[i+1])
            return temps[i] + frac * (temps[i+1] - temps[i])
    return None


def setup_style():
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 13,
        'axes.titlesize': 14,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.dpi': 150,
        'savefig.dpi': 150,
        'savefig.bbox': 'tight',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


def fig1_kappa_vs_level(data):
    """The money plot: κ(ℓ) at T≈T_c with error bars for all three series."""
    tc_result = min(data, key=lambda d: abs(d['T'] - TC))
    T = tc_result['T']

    fig, ax = plt.subplots(figsize=(7, 5))

    series_info = [
        ("m4_fixed", "m=4 fixed (magnetization)", "#2563eb", "o", "-"),
        ("m2_fixed", "m=2 fixed (plugin)", "#dc2626", "s", "--"),
        ("adaptive", "adaptive", "#16a34a", "^", ":"),
    ]

    for key, label, color, marker, ls in series_info:
        ells, kappas, errs = [], [], []
        for lv in tc_result['levels']:
            s = lv['series'].get(key)
            if s:
                ells.append(lv['level'])
                kappas.append(s['kappa'])
                errs.append(s['kappa_err'])

        ax.errorbar(ells, kappas, yerr=errs, marker=marker, ls=ls,
                     color=color, label=label, capsize=4, markersize=7,
                     linewidth=1.8, capthick=1.5)

    ax.set_xlabel("RG level ℓ")
    ax.set_ylabel("κ = I(μ;b) / H(μ)")
    ax.set_title(f"κ across RG levels at T = {T:.4f} ≈ T_c\n(L₀ = 256, N = 5000)")
    ax.set_xticks(range(5))
    ax.set_xticklabels(["0\n(256)", "1\n(128)", "2\n(64)", "3\n(32)", "4\n(16)"])
    ax.legend(loc='upper left')
    ax.set_ylim(0, 0.6)

    fig.savefig(os.path.join(FIGURES_DIR, "fig1_kappa_vs_level.png"))
    plt.close(fig)
    print("  fig1_kappa_vs_level.png")


def fig2_slope_vs_T(data):
    """dκ/dℓ vs temperature for all three series, with T₀ markers."""
    fig, ax = plt.subplots(figsize=(8, 5))

    series_info = [
        ("m4_fixed", "m=4 fixed", "#2563eb", "o"),
        ("m2_fixed", "m=2 fixed", "#dc2626", "s"),
        ("adaptive", "adaptive", "#16a34a", "^"),
    ]

    for key, label, color, marker in series_info:
        temps, slopes = [], []
        for d in data:
            kappas = []
            for lv in d['levels']:
                s = lv['series'].get(key)
                if s and not np.isnan(s['kappa']):
                    kappas.append(s['kappa'])
            if len(kappas) >= 2:
                slope, _ = np.polyfit(np.arange(len(kappas)), kappas, 1)
                temps.append(d['T'])
                slopes.append(slope)

        temps = np.array(temps)
        slopes = np.array(slopes)
        ax.plot(temps, slopes, marker=marker, color=color, label=label,
                markersize=5, linewidth=1.5)

        # Mark T₀
        T0 = _find_zero_crossing(temps, slopes)
        if T0 is not None:
            ax.axvline(T0, color=color, ls=':', alpha=0.5, linewidth=1)
            ax.annotate(f"T₀={T0:.3f}", xy=(T0, 0), xytext=(T0 + 0.05, 0.02),
                        fontsize=8, color=color,
                        arrowprops=dict(arrowstyle='->', color=color, lw=0.8))

    ax.axhline(0, color='black', lw=0.8, ls='-')
    ax.axvline(TC, color='gray', ls='--', alpha=0.7, lw=1.2, label=f"T_c = {TC:.4f}")

    ax.set_xlabel("Temperature T")
    ax.set_ylabel("dκ/dℓ (slope)")
    ax.set_title("Slope of κ vs RG level across temperatures\n(L₀ = 256)")
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlim(0.9, 4.1)

    fig.savefig(os.path.join(FIGURES_DIR, "fig2_slope_vs_T.png"))
    plt.close(fig)
    print("  fig2_slope_vs_T.png")


def fig3_kappa_vs_T_fan(data):
    """κ(T) at each RG level — fan diagram showing level divergence."""
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = ['#1e3a5f', '#2563eb', '#60a5fa', '#f97316', '#dc2626']
    level_labels = ["ℓ=0 (256)", "ℓ=1 (128)", "ℓ=2 (64)", "ℓ=3 (32)", "ℓ=4 (16)"]

    for level_idx in range(5):
        temps, kappas = [], []
        for d in data:
            if level_idx < len(d['levels']):
                m4 = d['levels'][level_idx]['series'].get('m4_fixed')
                if m4:
                    temps.append(d['T'])
                    kappas.append(m4['kappa'])
        ax.plot(temps, kappas, color=colors[level_idx], linewidth=1.8,
                label=level_labels[level_idx], marker='o', markersize=3)

    ax.axvline(TC, color='gray', ls='--', alpha=0.7, lw=1.2, label=f"T_c")
    ax.set_xlabel("Temperature T")
    ax.set_ylabel("κ(m=4)")
    ax.set_title("κ(m=4) vs temperature at each RG level\n(fan diagram, L₀ = 256)")
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlim(0.9, 4.1)
    ax.set_ylim(0, 1.05)

    fig.savefig(os.path.join(FIGURES_DIR, "fig3_kappa_fan.png"))
    plt.close(fig)
    print("  fig3_kappa_fan.png")


def fig4_cv_vs_T(data):
    """CV of κ(m=4) across RG levels vs temperature — minimum near T_c."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    temps, cvs = [], []
    for d in data:
        kappas = []
        for lv in d['levels']:
            m4 = lv['series'].get('m4_fixed')
            if m4 and not np.isnan(m4['kappa']):
                kappas.append(m4['kappa'])
        if len(kappas) >= 2:
            cv = np.std(kappas) / np.mean(kappas)
            temps.append(d['T'])
            cvs.append(cv)

    ax.plot(temps, cvs, 'o-', color='#2563eb', markersize=5, linewidth=1.8)
    ax.axvline(TC, color='gray', ls='--', alpha=0.7, lw=1.2, label=f"T_c = {TC:.4f}")

    # Mark thresholds
    ax.axhline(0.05, color='#16a34a', ls=':', alpha=0.6, lw=1.2, label="CV = 0.05 (Outcome A)")
    ax.axhline(0.20, color='#f97316', ls=':', alpha=0.6, lw=1.2, label="CV = 0.20 (Outcome B/C)")

    # Mark minimum
    min_idx = np.argmin(cvs)
    ax.annotate(f"min CV = {cvs[min_idx]:.3f}\nT = {temps[min_idx]:.3f}",
                xy=(temps[min_idx], cvs[min_idx]),
                xytext=(temps[min_idx] + 0.3, cvs[min_idx] + 0.1),
                fontsize=9, arrowprops=dict(arrowstyle='->', lw=0.8))

    ax.set_xlabel("Temperature T")
    ax.set_ylabel("CV(κ) across RG levels")
    ax.set_title("Coefficient of variation of κ(m=4)\nacross 5 RG levels vs temperature")
    ax.legend(loc='upper left', fontsize=9)
    ax.set_xlim(0.9, 4.1)
    ax.set_ylim(0, 1.6)

    fig.savefig(os.path.join(FIGURES_DIR, "fig4_cv_vs_T.png"))
    plt.close(fig)
    print("  fig4_cv_vs_T.png")


def fig5_T0_convergence():
    """T₀ convergence with system size: L=64 vs L=256."""
    fig, ax = plt.subplots(figsize=(5.5, 4))

    L_values = [64, 256]
    T0_values = [2.2762, 2.2682]  # from our measurements
    offsets = [abs(t - TC) for t in T0_values]

    ax.plot(L_values, T0_values, 'o-', color='#2563eb', markersize=10,
            linewidth=2, label="T₀(m=4)")
    ax.axhline(TC, color='gray', ls='--', lw=1.2, label=f"T_c = {TC:.4f}")

    # Also mark browser value
    ax.plot(64, 2.272, 'D', color='#dc2626', markersize=8,
            label="Browser explorer (L=64)", zorder=5)

    for i, (L, T0) in enumerate(zip(L_values, T0_values)):
        ax.annotate(f"  {T0:.4f}\n  (|Δ|={offsets[i]:.4f})",
                    xy=(L, T0), fontsize=9, ha='left')

    ax.set_xlabel("System size L")
    ax.set_ylabel("T₀(m=4)")
    ax.set_title("T₀ convergence toward T_c\nwith system size")
    ax.set_xscale('log', base=2)
    ax.set_xticks(L_values)
    ax.set_xticklabels([str(L) for L in L_values])
    ax.legend(fontsize=9)
    ax.set_xlim(40, 400)

    fig.savefig(os.path.join(FIGURES_DIR, "fig5_T0_convergence.png"))
    plt.close(fig)
    print("  fig5_T0_convergence.png")


def fig6_xi_across_T(data):
    """ξ/L vs temperature at each RG level."""
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = ['#1e3a5f', '#2563eb', '#60a5fa', '#f97316', '#dc2626']
    level_labels = ["ℓ=0 (256)", "ℓ=1 (128)", "ℓ=2 (64)", "ℓ=3 (32)", "ℓ=4 (16)"]

    for level_idx in range(5):
        temps, xi_over_L = [], []
        for d in data:
            if level_idx < len(d['levels']):
                lv = d['levels'][level_idx]
                temps.append(d['T'])
                xi_over_L.append(lv['xi_over_L'])

        ax.plot(temps, xi_over_L, color=colors[level_idx], linewidth=1.5,
                label=level_labels[level_idx], marker='o', markersize=3)

    ax.axvline(TC, color='gray', ls='--', alpha=0.7, lw=1.2, label="T_c")
    ax.set_xlabel("Temperature T")
    ax.set_ylabel("ξ / L")
    ax.set_title("Correlation length ratio ξ/L at each RG level\n(Fourier definition)")
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlim(0.9, 4.1)
    ax.set_ylim(0, 1.15)

    fig.savefig(os.path.join(FIGURES_DIR, "fig6_xi_over_L.png"))
    plt.close(fig)
    print("  fig6_xi_over_L.png")


def generate_all():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    setup_style()

    data = load_results("results")
    if not data:
        print("No results found")
        return

    print("Generating figures...")
    fig1_kappa_vs_level(data)
    fig2_slope_vs_T(data)
    fig3_kappa_vs_T_fan(data)
    fig4_cv_vs_T(data)
    fig5_T0_convergence()
    fig6_xi_across_T(data)
    print("Done.")


if __name__ == "__main__":
    generate_all()
