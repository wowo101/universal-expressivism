# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

IC/C Scale Invariance Python computation engine for 2D Ising model simulations. Investigates whether the informational closure ratio κ = I(μ;b)/H(μ) is conserved across renormalization group (RG) coarse-graining levels, using Wolff cluster Monte Carlo at lattice sizes up to L=256.

The implementation plan lives in `python_engine_plan.md`. Follow it closely — task ordering, module boundaries, and parameter choices are deliberate.

## Architecture

```
ic_scale/
├── sim/
│   ├── ising.py              # Wolff cluster MC (numba-jitted inner loop)
│   └── coarse_grain.py       # Block-spin RG, region extraction
├── info/
│   ├── mi_plugin.py          # Plugin MI estimator (Dirichlet, for m≤3)
│   └── mi_magnetization.py   # Magnetization-based MI (for m≥3)
├── measure/
│   ├── correlation.py        # Spatial correlation C(r), ξ estimation
│   └── kappa.py              # κ = I(μ;b)/H(μ) with bootstrap errors
├── run_experiment.py          # Main orchestration loop
├── config.py                  # All parameters (T grid, MC settings, etc.)
├── validate.py                # Validation suite (run before experiments)
└── results/                   # JSON output, one file per temperature
```

Each module does one thing. No unnecessary abstraction.

## Key Commands

```bash
# Run validation suite (must pass before any experiment)
python -m ic_scale.validate

# Run full experiment (2-4 hours, saves incrementally)
python -m ic_scale.run_experiment

# Run pilot with reduced samples (edit config.py: N_SAMPLES=1000)
python -m ic_scale.run_experiment
```

## Dependencies

- **numpy** — lattice arrays (int8), all array operations
- **numba** — JIT compilation for Wolff cluster growth loop (falls back to pure Python if unavailable)
- **scipy** — correlation fitting (log-linear regression for ξ)

## Critical Implementation Constraints

- **m=4 fixed is the primary test series.** Use magnetization MI at all 5 RG levels. Never switch MI methods across levels for a given series.
- **Numba warmup:** Run a tiny L=8 simulation at startup to trigger JIT compilation (~5-10s) so it doesn't pollute per-temperature timing.
- **Memory:** ~450 MB per temperature point at L=256 with 5000 samples. Process one temperature at a time — save results, then release memory.
- **Reproducibility:** Use `np.random.default_rng(seed)` with `seed = int(T * 10000)` per temperature point. Record seed in output JSON.
- **Checkpointing:** Skip temperatures where `results/T_{T:.4f}.json` already exists on restart.
- **ξ estimation:** Cap ξ at L when fit diverges near T_c. Always report second-moment estimate alongside exponential fit. Flag when they disagree >50%.

## Physical Constants

- T_c = 2 / ln(1 + √2) ≈ 2.2692 (exact 2D Ising critical temperature)
- At T_c with L=256: Wolff cluster should flip O(L^1.875) spins per step
- Expected: κ approximately flat across RG levels at T_c if conservation holds (Outcome A)

## Decision Criteria (from spec §4.1)

The experiment evaluates dκ/dℓ at T_c for m=4 fixed across 5 levels:
- **Outcome A:** CV < 0.05, slope ≈ 0 → conservation holds
- **Outcome B:** CV ∈ [0.05, 0.20], slope small → approximate conservation
- **Outcome C:** CV > 0.20 or significant slope → conservation fails
- **Outcome D:** regime-change phenomenology absent at L=256
