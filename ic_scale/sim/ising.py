"""Wolff cluster Monte Carlo for the 2D Ising model.

Provides equilibration and sampling at any temperature for square lattices
with periodic boundary conditions. The inner cluster growth loop is
numba-jitted for performance; falls back to pure Python if numba is unavailable.
"""

import numpy as np
import time

try:
    import numba
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

# Exact 2D Ising critical temperature
TC = 2.0 / np.log(1 + np.sqrt(2))

# Critical exponents (2D Ising exact)
BETA_NU = 1.0 / 8.0  # β/ν = (1/8) / 1 = 0.125


# ---------------------------------------------------------------------------
# Numba-jitted Wolff cluster growth
# ---------------------------------------------------------------------------

if HAS_NUMBA:
    @numba.njit(cache=True)
    def _numba_seed(s):
        np.random.seed(s)

    @numba.njit(cache=True)
    def _wolff_cluster(lattice, L, p_add):
        """Grow and flip a single Wolff cluster. Returns cluster size.

        Uses numba's built-in PRNG (Mersenne Twister) for proper
        statistical quality. Seed with np.random.seed() before calling.
        """
        seed_x = np.random.randint(0, L)
        seed_y = np.random.randint(0, L)

        cluster_spin = lattice[seed_y, seed_x]

        stack_x = np.empty(L * L, dtype=np.int32)
        stack_y = np.empty(L * L, dtype=np.int32)
        visited = np.zeros((L, L), dtype=np.bool_)

        # Push seed
        stack_x[0] = seed_x
        stack_y[0] = seed_y
        visited[seed_y, seed_x] = True
        stack_top = 1
        cluster_size = 0

        # Neighbour offsets (right, left, down, up)
        dx = np.array([1, -1, 0, 0], dtype=np.int32)
        dy = np.array([0, 0, 1, -1], dtype=np.int32)

        while stack_top > 0:
            stack_top -= 1
            cx = stack_x[stack_top]
            cy = stack_y[stack_top]
            cluster_size += 1

            for k in range(4):
                nx = (cx + dx[k]) % L
                ny = (cy + dy[k]) % L

                if not visited[ny, nx] and lattice[ny, nx] == cluster_spin:
                    if np.random.random() < p_add:
                        visited[ny, nx] = True
                        stack_x[stack_top] = nx
                        stack_y[stack_top] = ny
                        stack_top += 1

        # Flip all cluster spins
        for iy in range(L):
            for ix in range(L):
                if visited[iy, ix]:
                    lattice[iy, ix] = -lattice[iy, ix]

        return cluster_size

    @numba.njit(cache=True)
    def _run_wolff_steps(lattice, L, p_add, n_steps):
        """Run multiple Wolff steps, return total cluster size."""
        total = 0
        for _ in range(n_steps):
            total += _wolff_cluster(lattice, L, p_add)
        return total

    @numba.njit(cache=True)
    def _sample_configs(lattice, L, p_add, n_samples, thinning):
        """Generate samples with thinning. Returns 3D array (n_samples, L, L)."""
        out = np.empty((n_samples, L, L), dtype=np.int8)
        for i in range(n_samples):
            for _ in range(thinning):
                _wolff_cluster(lattice, L, p_add)
            out[i] = lattice.copy()
        return out

else:
    def _wolff_cluster(lattice, L, p_add):
        """Pure Python fallback (slower)."""
        seed_x = np.random.randint(0, L)
        seed_y = np.random.randint(0, L)

        cluster_spin = lattice[seed_y, seed_x]
        visited = np.zeros((L, L), dtype=np.bool_)
        stack = [(seed_x, seed_y)]
        visited[seed_y, seed_x] = True
        cluster_size = 0

        dx = [1, -1, 0, 0]
        dy = [0, 0, 1, -1]

        while stack:
            cx, cy = stack.pop()
            cluster_size += 1
            for k in range(4):
                nx = (cx + dx[k]) % L
                ny = (cy + dy[k]) % L
                if not visited[ny, nx] and lattice[ny, nx] == cluster_spin:
                    if np.random.random() < p_add:
                        visited[ny, nx] = True
                        stack.append((nx, ny))

        lattice[visited] *= -1
        return cluster_size

    def _run_wolff_steps(lattice, L, p_add, n_steps):
        total = 0
        for _ in range(n_steps):
            total += _wolff_cluster(lattice, L, p_add)
        return total

    def _sample_configs(lattice, L, p_add, n_samples, thinning):
        out = np.empty((n_samples, L, L), dtype=np.int8)
        for i in range(n_samples):
            for _ in range(thinning):
                _wolff_cluster(lattice, L, p_add)
            out[i] = lattice.copy()
        return out


def _seed_rng(seed_val):
    """Seed the random state used by jitted functions."""
    if HAS_NUMBA:
        _numba_seed(seed_val)
    else:
        np.random.seed(seed_val)


def _warmup_numba():
    """Trigger numba JIT compilation on a tiny lattice."""
    if not HAS_NUMBA:
        return
    tiny = np.ones((8, 8), dtype=np.int8)
    _numba_seed(42)
    _wolff_cluster(tiny, 8, 0.5)
    _run_wolff_steps(tiny, 8, 0.5, 1)
    _sample_configs(tiny, 8, 0.5, 1, 1)


# Run warmup at import time
_warmup_numba()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def wolff_step(lattice, L, beta):
    """Perform a single Wolff cluster flip. Returns cluster size.

    Modifies lattice in-place.
    """
    p_add = 1.0 - np.exp(-2.0 * beta)
    return _wolff_cluster(lattice, L, p_add)


def equilibrate(L, T, n_steps=5000, seed=None):
    """Create a random lattice and equilibrate with Wolff cluster MC.

    Returns the equilibrated lattice and mean cluster size.
    """
    rng = np.random.default_rng(seed)
    lattice = rng.choice(np.array([-1, 1], dtype=np.int8), size=(L, L))

    # Seed the PRNG used by jitted functions
    _seed_rng(int(rng.integers(0, 2**31)))

    beta = 1.0 / T
    p_add = 1.0 - np.exp(-2.0 * beta)

    total_cluster = _run_wolff_steps(lattice, L, p_add, n_steps)
    mean_cluster = total_cluster / n_steps
    return lattice, mean_cluster


def sample(lattice, L, T, n_samples, thinning=5, seed=None):
    """Generate independent configurations by thinning Wolff steps.

    Returns a list of np.ndarray(int8, (L, L)).
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
        _seed_rng(int(rng.integers(0, 2**31)))

    beta = 1.0 / T
    p_add = 1.0 - np.exp(-2.0 * beta)

    configs_3d = _sample_configs(lattice, L, p_add, n_samples, thinning)
    return [configs_3d[i] for i in range(n_samples)]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_wolff():
    """Validate Wolff algorithm at T_c, L=64.

    Checks:
    1. Mean |M|/N vs known value at T_c
    2. Mean cluster size is O(L^(2-β/ν))
    3. Energy autocorrelation time is O(1) in cluster steps
    """
    L = 64
    T = TC
    beta = 1.0 / T
    n_equil = 2000
    n_samples = 2000
    thinning = 5

    print(f"Validating Wolff at T_c={T:.4f}, L={L}")
    print(f"  numba available: {HAS_NUMBA}")

    # Equilibrate
    t0 = time.time()
    lattice, mean_cluster_eq = equilibrate(L, T, n_steps=n_equil, seed=12345)
    t_eq = time.time() - t0
    print(f"  Equilibration: {t_eq:.1f}s, mean cluster size: {mean_cluster_eq:.0f}")

    # Expected cluster size ~ L^(2-β/ν) = 64^1.875 ≈ 2435
    expected_cluster = L ** (2.0 - BETA_NU)
    cluster_ratio = mean_cluster_eq / expected_cluster
    print(f"  Expected cluster ~ {expected_cluster:.0f}, ratio: {cluster_ratio:.2f}")

    # Sample
    t0 = time.time()
    configs = sample(lattice, L, T, n_samples, thinning=thinning, seed=67890)
    t_sample = time.time() - t0
    print(f"  Sampling {n_samples} configs: {t_sample:.1f}s")

    # Measure <|M|>/N
    mags = np.array([np.abs(c.sum()) / (L * L) for c in configs])
    mean_mag = mags.mean()

    # Finite-size scaling at T_c: <|M|>/N ~ a * L^(-β/ν)
    # The amplitude a ≈ 1.0 for PBC square lattice (Kamieniarz & Blöte 1993)
    # (Note: 0.9117 in the plan is the spontaneous magnetization amplitude below T_c)
    expected_mag = 1.0 * L ** (-BETA_NU)
    mag_ratio = mean_mag / expected_mag
    print(f"  <|M|>/N = {mean_mag:.4f}, expected ≈ {expected_mag:.4f}, ratio: {mag_ratio:.3f}")

    mag_pass = abs(mag_ratio - 1.0) < 0.05
    print(f"  Magnetization check: {'PASS' if mag_pass else 'FAIL'} (within 5%: {abs(mag_ratio-1.0)*100:.1f}%)")

    # Energy autocorrelation
    energies = []
    test_lattice = configs[0].copy()
    p_add = 1.0 - np.exp(-2.0 * beta)
    for _ in range(500):
        _wolff_cluster(test_lattice, L, p_add)
        e = _compute_energy(test_lattice, L)
        energies.append(e)

    energies = np.array(energies, dtype=np.float64)
    e_mean = energies.mean()
    e_var = energies.var()
    if e_var > 0:
        autocorr = np.correlate(energies - e_mean, energies - e_mean, mode='full')
        autocorr = autocorr[len(autocorr) // 2:]
        autocorr /= autocorr[0]
        tau_int = 0.5
        for k in range(1, len(autocorr)):
            if autocorr[k] < 0:
                break
            tau_int += autocorr[k]
    else:
        tau_int = 0.0

    print(f"  Energy autocorrelation time τ_int ≈ {tau_int:.1f} (should be O(1))")
    tau_pass = tau_int < 10.0
    print(f"  Autocorrelation check: {'PASS' if tau_pass else 'FAIL'}")

    all_pass = mag_pass and tau_pass
    print(f"  Overall: {'PASS' if all_pass else 'FAIL'}")
    return all_pass


def _compute_energy(lattice, L):
    """Compute Ising energy E = -Σ s_i s_j for nearest neighbours."""
    right = np.roll(lattice, -1, axis=1)
    down = np.roll(lattice, -1, axis=0)
    return -int((lattice * right).sum() + (lattice * down).sum())


if __name__ == "__main__":
    validate_wolff()
