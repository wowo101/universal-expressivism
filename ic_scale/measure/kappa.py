"""κ = I(μ;b) / H(μ) computation with bootstrap error estimation.

The main measurement function that ties together region extraction
and MI estimation to compute the informational closure ratio.
"""

import numpy as np
from dataclasses import dataclass, asdict

from ic_scale.sim.coarse_grain import extract_regions
from ic_scale.info.mi_plugin import mi_plugin, spins_to_states_batch
from ic_scale.info.mi_magnetization import mi_magnetization, spins_to_magnetizations_batch


@dataclass
class KappaResult:
    kappa: float
    kappa_err: float      # bootstrap standard error
    ic: float             # I(μ;b)
    h_interior: float     # H(μ)
    h_blanket: float      # H(b)
    mi_method: str
    m: int
    n_samples: int
    n_blocks: int

    def to_dict(self):
        return asdict(self)


def compute_kappa(samples, L, m, n_blocks=200, mi_method='auto',
                  alpha=0.5, n_bootstrap=50, rng=None):
    """Compute κ = I(μ;b) / H(μ) with error estimates.

    Args:
        samples: list of (L, L) int8 arrays at this RG level
        L: lattice size at this level
        m: interior block size
        n_blocks: block positions per sample (random positions)
        mi_method: 'plugin' | 'magnetization' | 'auto'
            auto: plugin for m <= 2, magnetization for m >= 3
        alpha: Dirichlet pseudocount
        n_bootstrap: number of bootstrap resamples for error bars
        rng: numpy random generator

    Returns:
        KappaResult
    """
    if rng is None:
        rng = np.random.default_rng()

    # Check that m fits: need m+2 <= L for blanket
    if m + 2 > L:
        return KappaResult(
            kappa=np.nan, kappa_err=np.nan, ic=np.nan,
            h_interior=np.nan, h_blanket=np.nan,
            mi_method=mi_method, m=m,
            n_samples=len(samples), n_blocks=n_blocks
        )

    # Resolve MI method
    if mi_method == 'auto':
        mi_method = 'plugin' if m <= 2 else 'magnetization'

    n_interior_spins = m * m
    n_blanket_spins = (m + 2) ** 2 - m * m

    # Collect interior/blanket data from all samples
    int_spins_list = []
    bla_spins_list = []

    for config in samples:
        for _ in range(n_blocks):
            cx = rng.integers(L)
            cy = rng.integers(L)
            interior, blanket = extract_regions(config, m, cx, cy)
            int_spins_list.append(interior)
            bla_spins_list.append(blanket)

    int_spins = np.array(int_spins_list, dtype=np.int8)
    bla_spins = np.array(bla_spins_list, dtype=np.int8)

    # Convert to appropriate representation
    if mi_method == 'plugin':
        int_data = spins_to_states_batch(int_spins)
        bla_data = spins_to_states_batch(bla_spins)
        mi_func = lambda i, b: mi_plugin(i, b, n_interior_spins, n_blanket_spins, alpha)
    else:
        int_data = spins_to_magnetizations_batch(int_spins)
        bla_data = spins_to_magnetizations_batch(bla_spins)
        mi_func = lambda i, b: mi_magnetization(i, b, n_interior_spins, n_blanket_spins, alpha)

    # Compute MI on full dataset
    mi, h_int, h_bla = mi_func(int_data, bla_data)
    kappa = mi / h_int if h_int > 0 else 0.0

    # Bootstrap error estimation
    n_total = len(int_data)
    kappa_boots = np.empty(n_bootstrap)

    for b in range(n_bootstrap):
        idx = rng.integers(0, n_total, size=n_total)
        mi_b, h_int_b, _ = mi_func(int_data[idx], bla_data[idx])
        kappa_boots[b] = mi_b / h_int_b if h_int_b > 0 else 0.0

    kappa_err = kappa_boots.std()

    return KappaResult(
        kappa=kappa, kappa_err=kappa_err,
        ic=mi, h_interior=h_int, h_blanket=h_bla,
        mi_method=mi_method, m=m,
        n_samples=len(samples), n_blocks=n_blocks
    )


def adaptive_m(L, c_adaptive=2.0):
    """Compute adaptive block size m for a given lattice size L.

    m = max(1, floor(L / (c_adaptive * sqrt(L))))
    Capped so that m+2 <= L.
    """
    m = max(1, int(L / (c_adaptive * np.sqrt(L))))
    m = min(m, L - 2)  # ensure blanket fits
    return m
