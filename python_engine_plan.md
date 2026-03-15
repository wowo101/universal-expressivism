# IC/C Scale Invariance: Python Computation Engine

## Plan for Claude Code Implementation

status: ready for implementation
date: 2026-03-15
prerequisite: browser-based adaptive-m explorer (L=64) findings

---

## 0. Context and Motivation

The browser explorer at L=64 established three findings that shape the Python engine:

1. **m=4 fixed is the primary baseline.** At L=64 (ℓ 0–2, 3 regression points), T₀(m=4) ≈ 2.272 – within 0.003 of T_c. At L=256 we get 5 regression points (ℓ 0–4, lattices 256→128→64→32→16) with m=4 fitting at every level. This is the decisive test.

2. **m=2 fixed satisfies flatness trivially** – too little interior structure to produce meaningful signal. It remains a useful lower bound and estimator validation tool, but not a primary test of conservation.

3. **The adaptive scheme introduces a geometry artefact** (perimeter-to-area ratio varies ~6× as m changes from 4 to 1). We track adaptive as a secondary series but the clean test is m=4 fixed across 5 levels.

The Python engine's job: produce κ(ℓ, T) data at L₀ = 256 with enough statistical power to evaluate the spec's decision tree (Outcomes A–D in §7).

---

## 1. Architecture

```
ic_scale/
├── sim/
│   ├── ising.py           # Wolff cluster MC
│   └── coarse_grain.py    # Block-spin RG + region extraction
├── info/
│   ├── mi_plugin.py       # Plugin MI (Dirichlet regularised)
│   └── mi_magnetization.py # Magnetization-based MI
├── measure/
│   ├── correlation.py     # ξ estimation from C(r)
│   └── kappa.py           # κ computation for given block config
├── run_experiment.py      # Main orchestration
├── config.py              # All parameters
├── validate.py            # Validation suite
├── results/               # JSON output directory
│   └── .gitkeep
└── README.md
```

Flat structure, no unnecessary abstraction. Each module does one thing.

---

## 2. Implementation Tasks

Execute these in order. Each task is a self-contained Claude Code session.

### Task 1: Core simulation (`sim/ising.py`)

**Deliverable:** Wolff cluster MC that can generate equilibrated configurations at any T for L=256.

**Requirements:**
- `wolff_step(lattice, L, beta)` – single cluster flip, modifies lattice in-place
- `equilibrate(L, T, n_steps=5000)` → lattice – creates random lattice, runs n_steps Wolff flips
- `sample(lattice, L, T, n_samples, thinning=5)` → list of np.ndarray(int8) – generates independent samples
- Lattice is `np.ndarray(dtype=np.int8, shape=(L, L))`, periodic boundary conditions
- Use numpy for the lattice, but the inner Wolff cluster growth loop must be **numba-jitted** for speed (the cluster growth is inherently sequential and numpy can't vectorise it)
- Verify: at T_c with L=256, a single Wolff step should flip O(L^(2-β/ν)) ≈ O(L^1.875) spins. Track and print mean cluster size during equilibration as a sanity check.

**Validation (built into module):**
- `validate_wolff()`: run at T_c, L=64, measure <|M|>/N vs known value (≈ 0.9117 × L^(-β/ν) for 2D Ising at T_c). Check within 5%.
- Measure autocorrelation time of energy; should be O(1) in Wolff cluster steps.

**Performance target:** generating 1000 samples at L=256 should take < 60 seconds.

**Key decision – numba vs cython:** Use numba. It's pip-installable, no compilation step, and the Wolff inner loop (BFS with random bond acceptance) jits well. If numba is unavailable, fall back to pure numpy with the cluster loop in Python (slower but correct).

```python
# Skeleton for the Wolff step
@numba.njit
def _wolff_cluster(lattice, L, p_add, seed_x, seed_y):
    """Grow and flip a Wolff cluster. Returns cluster size."""
    cluster_spin = lattice[seed_y, seed_x]
    stack_x = np.empty(L * L, dtype=np.int32)
    stack_y = np.empty(L * L, dtype=np.int32)
    visited = np.zeros((L, L), dtype=np.bool_)
    # ... BFS with p_add acceptance
```

### Task 2: Coarse-graining (`sim/coarse_grain.py`)

**Deliverable:** block-spin RG and region extraction.

**Requirements:**
- `coarse_grain(lattice)` → lattice at L/2 – majority rule on 2×2 blocks, random tie-breaking
- `build_rg_tower(lattice, n_levels)` → list of lattices at each level
- `extract_regions(lattice, m, cx, cy)` → (interior, blanket) – extract m×m interior block and width-1 blanket ring
- All functions operate on numpy arrays

**Note on block-spin RG correctness:** After coarse-graining at T_c, the effective Hamiltonian acquires next-nearest-neighbour couplings. We are *not* tuning couplings (no MCRG). This means the coarse-grained system drifts slightly away from criticality. At L=256 with 4 RG steps this drift is small but measurable – track it via ξ estimation at each level.

### Task 3: MI estimation (`info/mi_plugin.py`, `info/mi_magnetization.py`)

**Deliverable:** two MI estimators, consistent with the browser explorer but more efficient.

**Plugin estimator** (for m ≤ 3, i.e. interior ≤ 9 spins):
- `mi_plugin(interior_states, blanket_states, n_interior_spins, n_blanket_spins, alpha=0.5)` → (MI, H_interior, H_blanket)
- Interior/blanket states are integer indices (binary config → int via bitpacking)
- Dirichlet regularisation with α=0.5 (Jeffreys prior)

**Magnetization estimator** (for m ≥ 3):
- `mi_magnetization(interior_mags, blanket_mags, n_interior_spins, n_blanket_spins, alpha=0.5)` → (MI, H_interior, H_blanket)
- Magnetization bins: n+1 values for n spins (from -n to +n in steps of 2)

**Important:** for the primary m=4 series, we use magnetization MI throughout all 5 levels. This ensures method consistency. The plugin estimator is there for the m=2 validation series.

**Validation:**
- `validate_mi()`: generate samples from a known distribution (e.g. bivariate with MI = log2(3) ≈ 1.585 bits) and check both estimators recover it within 3%.
- Cross-validate: for m=2 blocks at T=2.5, plugin and magnetization should agree on the ranking (not necessarily the absolute values, since they measure different things).

### Task 4: Correlation length (`measure/correlation.py`)

**Deliverable:** ξ estimation from spin-spin correlation function.

**Requirements:**
- `compute_correlation(samples, L, max_r=None)` → array of (r, C(r)) – spatial correlation function averaged over samples and positions
- `fit_xi(correlation_data, L)` → float – fit C(r) ~ exp(-r/ξ) via log-linear regression on positive C(r) values
- `xi_second_moment(correlation_data)` → float – second-moment definition ξ² = Σr²C(r) / ΣC(r) as cross-check

**Details:**
- Average C(r) over both horizontal and vertical directions (isotropy check)
- Use only r < L/4 to avoid wraparound effects from periodic boundaries
- At T_c, the fit will return ξ ≈ L (capped) – that's correct, it means correlations span the system

### Task 5: κ computation (`measure/kappa.py`)

**Deliverable:** the main measurement function that ties everything together.

**Requirements:**
```python
def compute_kappa(
    samples: list[np.ndarray],  # configurations at this RG level
    L: int,                      # lattice size at this level
    m: int,                      # interior block size
    n_blocks: int = 200,         # blocks per sample (random positions)
    mi_method: str = 'auto'      # 'plugin' | 'magnetization' | 'auto'
) -> KappaResult:
    """Compute κ = I(μ;b) / H(μ) with error estimates."""
```

**KappaResult dataclass:**
```python
@dataclass
class KappaResult:
    kappa: float
    kappa_err: float     # bootstrap standard error
    ic: float            # I(μ;b)
    h_interior: float    # H(μ)
    h_blanket: float     # H(b)
    mi_method: str
    m: int
    n_samples: int
    n_blocks: int
```

**Bootstrap error estimation:** resample the (interior_state, blanket_state) pairs with replacement, recompute MI 50 times, report standard deviation. This gives error bars on κ.

**Block placement:** random positions, uniformly distributed, wrapped with periodic boundaries. Average over n_blocks positions per sample.

### Task 6: Configuration and orchestration (`config.py`, `run_experiment.py`)

**config.py:**
```python
# Experiment parameters
L0 = 256
N_RG_LEVELS = 4        # levels 0–4: 256, 128, 64, 32, 16
M_VALUES = [2, 4]      # block sizes to measure
C_ADAPTIVE = 2.0       # adaptive-m parameter

# MC parameters
N_EQUIL = 5000         # Wolff steps for equilibration
N_SAMPLES = 5000       # independent configurations per temperature
THINNING = 5           # Wolff steps between samples

# Temperature grid
TC = 2.0 / np.log(1 + np.sqrt(2))
T_GRID = np.sort(np.concatenate([
    np.linspace(1.0, 1.8, 5),      # ordered phase (sparse)
    np.linspace(1.9, 2.7, 17),     # critical regime (dense)
    np.linspace(2.8, 4.0, 5),      # disordered phase (sparse)
]))

# MI parameters
ALPHA = 0.5            # Dirichlet regularisation
N_BLOCKS = 200         # block positions per sample
N_BOOTSTRAP = 50       # bootstrap resamples for error bars
```

**run_experiment.py:**

The main loop structure:
```
for T in T_GRID:
    1. Generate N_SAMPLES configurations at L=256
    2. Build RG tower (5 levels)
    3. At each level:
       a. Estimate ξ from C(r)
       b. Compute κ for m=4 fixed (magnetization MI)
       c. Compute κ for m=2 fixed (plugin MI)
       d. Compute κ for adaptive m (auto MI)
    4. Save results as JSON to results/T_{T:.4f}.json
    5. Print progress and summary statistics
```

**Output format** – one JSON file per temperature:
```json
{
    "T": 2.2692,
    "L0": 256,
    "n_samples": 5000,
    "levels": [
        {
            "level": 0,
            "L": 256,
            "xi": 245.3,
            "xi_over_L": 0.959,
            "xi_second_moment": 198.7,
            "series": {
                "m4_fixed": {
                    "m": 4,
                    "kappa": 0.1823,
                    "kappa_err": 0.0041,
                    "ic": 0.412,
                    "h_interior": 2.261,
                    "h_blanket": 3.891,
                    "mi_method": "magnetization"
                },
                "m2_fixed": { ... },
                "adaptive": {
                    "m": 3,
                    "kappa": ...,
                    ...
                }
            }
        },
        ...
    ],
    "metadata": {
        "n_equil": 5000,
        "thinning": 5,
        "alpha": 0.5,
        "n_blocks": 200,
        "n_bootstrap": 50,
        "c_adaptive": 2.0,
        "timestamp": "2026-03-16T14:30:00Z",
        "runtime_seconds": 342
    }
}
```

### Task 7: Validation suite (`validate.py`)

Run this before the main experiment. All checks must pass.

```python
def validate_all():
    validate_wolff()           # cluster size, magnetisation at T_c
    validate_coarse_grain()    # output shape, spin conservation
    validate_mi_known()        # MI recovery on synthetic data
    validate_mi_cross()        # plugin vs magnetization agreement for m=2
    validate_xi_estimation()   # ξ → ∞ at T_c, ξ < 5 at T >> T_c
    validate_kappa_limits()    # κ → 0 at T >> T_c, κ finite at T_c
```

Each validation prints PASS/FAIL with the measured vs expected values.

---

## 3. Execution Plan for Claude Code

### Session 1: Foundation (Tasks 1–2)
Build `sim/ising.py` and `sim/coarse_grain.py`. Run the Wolff validation. Generate a few test configurations at L=256 to verify performance.

**Exit criterion:** `validate_wolff()` passes, 1000 samples at L=256 generated in < 60s.

### Session 2: Measurement (Tasks 3–4)
Build MI estimators and ξ estimation. Run validation on synthetic data.

**Exit criterion:** `validate_mi_known()` and `validate_xi_estimation()` pass.

### Session 3: Integration (Tasks 5–6)
Build κ computation and the orchestration script. Run a quick test at 3 temperatures (T=1.5, T_c, T=3.0) with N_SAMPLES=500 to verify the full pipeline end-to-end.

**Exit criterion:** three JSON files in `results/`, κ values are reasonable (near 0 at T=3.0, non-trivial at T_c, degenerate at T=1.5).

### Session 4: Validation and pilot run (Task 7)
Run the full validation suite. Then do a pilot run with reduced samples (N_SAMPLES=1000) across the full temperature grid. This gives a preview of the results and catches any issues before the expensive full run.

**Exit criterion:** all validations pass; pilot results show the regime-change phenomenology (slope zero crossing near T_c for m=4).

### Session 5: Full experiment
Run with N_SAMPLES=5000 across the full temperature grid. This is the long computation (~2–4 hours depending on hardware). Can be left to run overnight.

**Exit criterion:** complete JSON results for all temperatures.

---

## 4. Critical Implementation Notes

### 4.1 The m=4 consistency constraint

The whole point of the Python engine is that m=4 fits at all 5 RG levels (L=256 down to L=16, and 16 ≥ 3×4 = 12 ✓). The magnetization-based MI estimator is used throughout. **Do not switch MI methods across levels.** If a future block size doesn't fit, skip that series at that level rather than substituting a different m.

### 4.2 Numba compilation

The first call to a numba-jitted function triggers compilation (~5–10s). Structure the code so this happens once at startup (e.g. run a tiny L=8 warmup). Don't let compilation time pollute the per-temperature timing.

If numba is not available (some environments lack LLVM), the fallback is pure Python with numpy for array operations. The Wolff cluster growth will be ~50× slower but still feasible for the pilot run. Flag this clearly in output.

### 4.3 Memory management

At L=256, each configuration is 256×256 = 65536 bytes (int8). 5000 samples = 327 MB. The RG tower adds levels of 128², 64², 32², 16² = another ~110 MB. Total ~450 MB per temperature point. This fits in memory but don't hold multiple temperatures simultaneously – process one T, save results, release memory.

### 4.4 ξ estimation robustness

Near T_c, ξ → ∞ and the exponential fit may not converge. Two safeguards:
- Cap ξ at L (if the fit gives ξ > L, report ξ = L)
- Always report the second-moment estimate alongside the fit – if they disagree by more than 50%, flag the level

### 4.5 Random number generation

Use `np.random.default_rng(seed)` with a fixed seed per temperature point for reproducibility. The seed can be derived from T: `seed = int(T * 10000)`. Record the seed in the output JSON.

### 4.6 Progress and checkpointing

The orchestration script should:
- Print a one-line summary after each temperature (T, runtime, mean κ at T_c level)
- Save results after each temperature (not at the end)
- If restarted, skip temperatures where `results/T_{T:.4f}.json` already exists

---

## 5. What We're Looking For

The primary observable is **dκ/dℓ at T_c for the m=4 fixed series across 5 levels.**

The spec's decision criteria (§4.1):
- CV < 0.05 and slope ≈ 0 → **Outcome A** (conservation holds)
- CV ∈ [0.05, 0.20] and slope small → **Outcome B** (approximate conservation)
- CV > 0.20 or significant slope → **Outcome C** (conservation fails, look for universal rate)

Secondary observables:
- Does the regime-change phenomenology (slope zero crossing, CV minimum, κ-convergence near T_c) survive at L=256? If not → **Outcome D**.
- T₀(m=4) at L=256 vs T₀(m=4) at L=64: is the offset stable or shrinking with system size?
- ξ at each RG level: does the coarse-grained system stay near criticality, or does RG drift accumulate?
- Adaptive κ with bootstrap errors: is the geometry artefact (the hump-then-drop) still present? If so, how does it interact with the larger lattice?

---

## 6. After the Python Run: Analysis Dashboard

Once results are in `results/`, the next step is a React artifact (or lightweight Python+matplotlib script) that loads the JSON files and produces:

1. κ(ℓ) at T_c with error bars – the money plot
2. dκ/dℓ vs T with three series (m=4, m=2, adaptive) and T₀ markers
3. κ(ℓ, T) surface plot or fan diagram
4. ξ(ℓ) at each temperature – tracking RG drift
5. CV and slope statistics for decision-tree evaluation

This can be built in a separate Claude Code session once the data exists.
