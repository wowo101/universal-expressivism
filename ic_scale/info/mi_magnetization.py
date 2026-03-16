"""Magnetization-based mutual information estimator.

For larger block sizes (m ≥ 3) where full state enumeration is impractical.
Reduces spin configurations to magnetization (sum of spins), giving n+1
bins for n spins. Uses Dirichlet regularisation with Jeffreys prior.
"""

import numpy as np


def mi_magnetization(interior_mags, blanket_mags, n_interior_spins, n_blanket_spins, alpha=0.5):
    """Compute MI between interior and blanket magnetizations.

    Args:
        interior_mags: 1D array of magnetization values (sum of spins)
        blanket_mags: 1D array of magnetization values (sum of spins)
        n_interior_spins: number of spins in interior
        n_blanket_spins: number of spins in blanket
        alpha: Dirichlet pseudocount (0.5 = Jeffreys prior)

    Returns:
        (MI, H_interior, H_blanket) in bits
    """
    N = len(interior_mags)
    assert len(blanket_mags) == N

    # Magnetization bins: n+1 values for n spins (-n, -n+2, ..., n-2, n)
    n_int_bins = n_interior_spins + 1
    n_bla_bins = n_blanket_spins + 1

    # Convert magnetization to bin index: m -> (m + n) / 2
    int_bins = ((interior_mags + n_interior_spins) // 2).astype(np.int64)
    bla_bins = ((blanket_mags + n_blanket_spins) // 2).astype(np.int64)

    # Clip to valid range (shouldn't be needed but safety)
    int_bins = np.clip(int_bins, 0, n_int_bins - 1)
    bla_bins = np.clip(bla_bins, 0, n_bla_bins - 1)

    # Joint histogram
    joint_idx = int_bins * n_bla_bins + bla_bins
    joint_counts = np.bincount(joint_idx, minlength=n_int_bins * n_bla_bins).astype(np.float64)
    joint_counts = joint_counts.reshape(n_int_bins, n_bla_bins)

    int_counts = joint_counts.sum(axis=1)
    bla_counts = joint_counts.sum(axis=0)

    # Dirichlet-regularised entropies
    H_joint = _entropy(joint_counts.ravel(), alpha, N)
    H_interior = _entropy(int_counts, alpha * n_bla_bins, N)
    H_blanket = _entropy(bla_counts, alpha * n_int_bins, N)

    MI = H_interior + H_blanket - H_joint

    # Convert to bits
    MI /= np.log(2)
    H_interior /= np.log(2)
    H_blanket /= np.log(2)

    return MI, H_interior, H_blanket


def _entropy(counts, alpha, N):
    """Entropy from Dirichlet-regularised plugin estimator."""
    K = len(counts)
    total = N + K * alpha
    probs = (counts + alpha) / total
    return -np.sum(probs * np.log(probs))


def spins_to_magnetization(spins):
    """Convert spin configuration to magnetization (sum)."""
    return spins.sum()


def spins_to_magnetizations_batch(spins_array):
    """Convert batch of spin configs to magnetizations.

    Args:
        spins_array: 2D array (n_samples, n_spins), values ±1

    Returns:
        1D array of magnetization values
    """
    return spins_array.sum(axis=1)
