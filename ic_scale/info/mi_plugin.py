"""Plugin mutual information estimator with Dirichlet regularisation.

Designed for small state spaces (m ≤ 3, i.e. ≤ 9 spins) where full
joint distribution estimation is tractable. Uses Jeffreys prior (α=0.5).
"""

import numpy as np
from scipy.special import digamma, gammaln


def mi_plugin(interior_states, blanket_states, n_interior_spins, n_blanket_spins, alpha=0.5):
    """Compute MI between interior and blanket using plugin estimator.

    Args:
        interior_states: 1D array of integer state indices for interior
        blanket_states: 1D array of integer state indices for blanket
        n_interior_spins: number of spins in interior (for state space size)
        n_blanket_spins: number of spins in blanket (for state space size)
        alpha: Dirichlet pseudocount (0.5 = Jeffreys prior)

    Returns:
        (MI, H_interior, H_blanket) in bits
    """
    N = len(interior_states)
    assert len(blanket_states) == N

    n_int_states = 2 ** n_interior_spins
    n_bla_states = 2 ** n_blanket_spins

    # Count joint, marginal distributions
    # Use flat indices for the joint
    joint_idx = interior_states * n_bla_states + blanket_states
    joint_counts = np.bincount(joint_idx, minlength=n_int_states * n_bla_states).astype(np.float64)
    joint_counts = joint_counts.reshape(n_int_states, n_bla_states)

    int_counts = joint_counts.sum(axis=1)  # marginal over blanket
    bla_counts = joint_counts.sum(axis=0)  # marginal over interior

    # Dirichlet-regularised probabilities
    joint_reg = joint_counts + alpha
    int_reg = int_counts + alpha * n_bla_states  # sum of joint row pseudocounts
    bla_reg = bla_counts + alpha * n_int_states

    # Entropies via digamma (Dirichlet-regularised plugin)
    total_reg = N + alpha * n_int_states * n_bla_states

    H_joint = _dirichlet_entropy(joint_counts.ravel(), alpha, N)
    H_interior = _dirichlet_entropy(int_counts, alpha * n_bla_states, N)
    H_blanket = _dirichlet_entropy(bla_counts, alpha * n_int_states, N)

    MI = H_interior + H_blanket - H_joint

    # Convert to bits
    MI /= np.log(2)
    H_interior /= np.log(2)
    H_blanket /= np.log(2)

    return MI, H_interior, H_blanket


def _dirichlet_entropy(counts, alpha, N):
    """Compute entropy using Dirichlet-regularised plugin estimator.

    H = -Σ p_i log(p_i) where p_i = (n_i + α) / (N + K*α)
    K is the number of categories.
    """
    K = len(counts)
    total = N + K * alpha
    probs = (counts + alpha) / total

    # Avoid log(0) — all probs > 0 due to regularisation
    return -np.sum(probs * np.log(probs))


def spins_to_state(spins):
    """Convert a spin configuration (-1/+1) to an integer state index.

    Maps -1→0, +1→1, then interprets as binary number.
    """
    bits = ((spins + 1) // 2).astype(np.int64)
    state = 0
    for b in bits:
        state = (state << 1) | int(b)
    return state


def spins_to_states_batch(spins_array):
    """Convert batch of spin configs to state indices.

    Args:
        spins_array: 2D array (n_samples, n_spins), values ±1

    Returns:
        1D array of integer state indices
    """
    bits = ((spins_array + 1) // 2).astype(np.int64)
    n_spins = bits.shape[1]
    powers = 2 ** np.arange(n_spins - 1, -1, -1, dtype=np.int64)
    return (bits * powers).sum(axis=1)
