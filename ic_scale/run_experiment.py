"""Main orchestration for the IC/C scale invariance experiment.

Generates Ising configurations, builds RG towers, and computes κ at
each level for multiple block sizes and temperatures. Results are saved
as JSON files, one per temperature, with checkpointing for restartability.
"""

import json
import os
import time
import sys
from datetime import datetime, timezone

import numpy as np

from ic_scale import config
from ic_scale.sim.ising import equilibrate, sample, _seed_rng, HAS_NUMBA
from ic_scale.sim.coarse_grain import build_rg_tower
from ic_scale.measure.correlation import xi_fourier, compute_correlation, fit_xi, xi_second_moment
from ic_scale.measure.kappa import compute_kappa, adaptive_m


RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")


def result_path(T):
    return os.path.join(RESULTS_DIR, f"T_{T:.4f}.json")


def process_temperature(T, n_samples=None, n_equil=None, n_blocks=None,
                        n_bootstrap=None, thinning=None):
    """Run the full pipeline for a single temperature point.

    Returns the result dict (also saved to disk).
    """
    n_samples = n_samples or config.N_SAMPLES
    n_equil = n_equil or config.N_EQUIL
    n_blocks = n_blocks or config.N_BLOCKS
    n_bootstrap = n_bootstrap or config.N_BOOTSTRAP
    thinning = thinning or config.THINNING

    L0 = config.L0
    seed = int(T * 10000)
    t_start = time.time()

    # 1. Generate configurations
    _seed_rng(seed)
    lattice, mean_cluster = equilibrate(L0, T, n_steps=n_equil, seed=seed)
    configs = sample(lattice, L0, T, n_samples, thinning=thinning, seed=seed + 1)

    # 2. Build RG tower for each configuration
    rng_rg = np.random.default_rng(seed + 2)
    rng_kappa = np.random.default_rng(seed + 3)

    levels_data = []

    for level_idx in range(config.N_RG_LEVELS + 1):
        L = L0 // (2 ** level_idx)

        # Get configs at this level
        if level_idx == 0:
            level_configs = configs
        else:
            level_configs = [
                _coarse_to_level(c, level_idx, rng_rg)
                for c in configs
            ]

        # 3a. Estimate ξ
        xi_f = xi_fourier(level_configs, L)
        r_vals, c_vals = compute_correlation(level_configs, L)
        xi_fit = fit_xi(r_vals, c_vals, L)
        xi_sm = xi_second_moment(r_vals, c_vals)

        # 3b-d. Compute κ for each series
        series = {}

        # m=4 fixed (magnetization MI) — primary series
        if 4 + 2 <= L:
            kr = compute_kappa(
                level_configs, L, m=4, n_blocks=n_blocks,
                mi_method='magnetization', alpha=config.ALPHA,
                n_bootstrap=n_bootstrap, rng=rng_kappa
            )
            series["m4_fixed"] = kr.to_dict()

        # m=2 fixed (plugin MI)
        if 2 + 2 <= L:
            kr = compute_kappa(
                level_configs, L, m=2, n_blocks=n_blocks,
                mi_method='plugin', alpha=config.ALPHA,
                n_bootstrap=n_bootstrap, rng=rng_kappa
            )
            series["m2_fixed"] = kr.to_dict()

        # Adaptive m
        m_adapt = adaptive_m(L, config.C_ADAPTIVE)
        if m_adapt + 2 <= L:
            kr = compute_kappa(
                level_configs, L, m=m_adapt, n_blocks=n_blocks,
                mi_method='auto', alpha=config.ALPHA,
                n_bootstrap=n_bootstrap, rng=rng_kappa
            )
            series["adaptive"] = kr.to_dict()

        levels_data.append({
            "level": level_idx,
            "L": L,
            "xi_fourier": round(xi_f, 3),
            "xi_fit": round(xi_fit, 3),
            "xi_over_L": round(xi_f / L, 4),
            "xi_second_moment": round(xi_sm, 3),
            "series": series,
        })

        # Free level configs to manage memory (except level 0 which shares with configs)
        if level_idx > 0:
            del level_configs

    runtime = time.time() - t_start

    result = {
        "T": round(T, 6),
        "L0": L0,
        "n_samples": n_samples,
        "seed": seed,
        "levels": levels_data,
        "metadata": {
            "n_equil": n_equil,
            "thinning": thinning,
            "alpha": config.ALPHA,
            "n_blocks": n_blocks,
            "n_bootstrap": n_bootstrap,
            "c_adaptive": config.C_ADAPTIVE,
            "numba": HAS_NUMBA,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "runtime_seconds": round(runtime, 1),
        }
    }

    # Save to disk
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(result_path(T), 'w') as f:
        json.dump(result, f, indent=2)

    return result


def _coarse_to_level(config_0, level, rng):
    """Coarse-grain a single configuration to the given RG level."""
    from ic_scale.sim.coarse_grain import coarse_grain
    c = config_0
    for _ in range(level):
        c = coarse_grain(c, rng=rng)
    return c


def run(t_grid=None, n_samples=None, n_equil=None, n_blocks=None,
        n_bootstrap=None, thinning=None):
    """Run the experiment across the temperature grid.

    Skips temperatures where results already exist (checkpointing).
    """
    if t_grid is None:
        t_grid = config.T_GRID

    n_total = len(t_grid)
    print(f"IC/C Scale Invariance Experiment")
    print(f"  L0={config.L0}, RG levels={config.N_RG_LEVELS}")
    print(f"  N_SAMPLES={n_samples or config.N_SAMPLES}, "
          f"N_EQUIL={n_equil or config.N_EQUIL}")
    print(f"  Temperatures: {n_total} points")
    print(f"  numba: {HAS_NUMBA}")
    print()

    for i, T in enumerate(t_grid):
        # Checkpoint: skip if already done
        if os.path.exists(result_path(T)):
            print(f"  [{i+1}/{n_total}] T={T:.4f} — skipped (exists)")
            continue

        t0 = time.time()
        result = process_temperature(
            T, n_samples=n_samples, n_equil=n_equil,
            n_blocks=n_blocks, n_bootstrap=n_bootstrap,
            thinning=thinning
        )
        dt = time.time() - t0

        # Summary line
        # Find κ(m=4) at the middle RG level as a summary stat
        mid_level = min(2, len(result["levels"]) - 1)
        series = result["levels"][mid_level].get("series", {})
        m4 = series.get("m4_fixed", {})
        kappa_str = f"κ(m=4)={m4.get('kappa', 'N/A'):.4f}" if m4 else "κ(m=4)=N/A"

        print(f"  [{i+1}/{n_total}] T={T:.4f}  {dt:.1f}s  {kappa_str}")

    print("\nDone.")


if __name__ == "__main__":
    # Parse optional --n-samples argument for pilot runs
    n_samples = None
    for arg in sys.argv[1:]:
        if arg.startswith("--n-samples="):
            n_samples = int(arg.split("=")[1])

    run(n_samples=n_samples)
