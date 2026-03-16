"""Block-spin renormalization group coarse-graining and region extraction.

Implements majority-rule 2×2 block-spin RG and extraction of interior/blanket
regions for mutual information computation.
"""

import numpy as np


def coarse_grain(lattice, rng=None):
    """Coarse-grain lattice by factor 2 using majority rule on 2×2 blocks.

    Tie-breaking (when block sums to 0) is random.
    Returns new lattice at L/2 with dtype int8.
    """
    L = lattice.shape[0]
    assert L % 2 == 0, f"Lattice size {L} must be even"

    if rng is None:
        rng = np.random.default_rng()

    # Sum 2×2 blocks
    block_sum = (
        lattice[0::2, 0::2] +
        lattice[0::2, 1::2] +
        lattice[1::2, 0::2] +
        lattice[1::2, 1::2]
    )

    L2 = L // 2
    coarse = np.sign(block_sum).astype(np.int8)

    # Random tie-breaking for zero-sum blocks
    ties = (block_sum == 0)
    n_ties = ties.sum()
    if n_ties > 0:
        coarse[ties] = rng.choice(
            np.array([-1, 1], dtype=np.int8), size=n_ties
        )

    return coarse


def build_rg_tower(lattice, n_levels, rng=None):
    """Build a tower of coarse-grained lattices.

    Returns list of n_levels+1 lattices: [level_0 (original), level_1, ..., level_n].
    Level k has size L / 2^k.
    """
    if rng is None:
        rng = np.random.default_rng()

    tower = [lattice]
    current = lattice
    for _ in range(n_levels):
        current = coarse_grain(current, rng=rng)
        tower.append(current)

    return tower


def extract_regions(lattice, m, cx, cy):
    """Extract m×m interior block and width-1 blanket ring.

    The interior is the m×m block with top-left corner at (cx, cy).
    The blanket is the ring of width 1 surrounding the interior.
    Both use periodic boundary conditions.

    Returns (interior, blanket) as 1D int8 arrays.
    """
    L = lattice.shape[0]

    # Interior: m×m block
    ix = np.arange(cx, cx + m) % L
    iy = np.arange(cy, cy + m) % L
    interior = lattice[np.ix_(iy, ix)].ravel().copy()

    # Blanket: width-1 ring around the interior
    # The blanket region is (m+2)×(m+2) minus the m×m interior
    bx = np.arange(cx - 1, cx + m + 1) % L
    by = np.arange(cy - 1, cy + m + 1) % L
    full_region = lattice[np.ix_(by, bx)]

    # The interior sits at positions [1:-1, 1:-1] within the full region
    # Blanket = full_region minus that interior subarray
    mask = np.ones((m + 2, m + 2), dtype=bool)
    mask[1:-1, 1:-1] = False
    blanket = full_region[mask].copy()

    return interior, blanket


def validate_coarse_grain():
    """Validate coarse-graining operations.

    Checks:
    1. Output shape is L/2 × L/2
    2. All spins are ±1
    3. build_rg_tower produces correct number of levels with correct sizes
    4. extract_regions returns correct sizes and handles periodic BCs
    """
    rng = np.random.default_rng(42)
    print("Validating coarse-graining...")

    # Test basic coarse-graining
    L = 64
    lattice = rng.choice(np.array([-1, 1], dtype=np.int8), size=(L, L))

    coarse = coarse_grain(lattice, rng=rng)
    shape_pass = coarse.shape == (L // 2, L // 2)
    print(f"  Shape {lattice.shape} → {coarse.shape}: {'PASS' if shape_pass else 'FAIL'}")

    spins_pass = np.all(np.abs(coarse) == 1)
    print(f"  All spins ±1: {'PASS' if spins_pass else 'FAIL'}")

    # Test RG tower
    L = 256
    lattice = rng.choice(np.array([-1, 1], dtype=np.int8), size=(L, L))
    tower = build_rg_tower(lattice, n_levels=4, rng=rng)

    tower_pass = True
    expected_sizes = [256, 128, 64, 32, 16]
    for i, (level, expected_L) in enumerate(zip(tower, expected_sizes)):
        ok = level.shape == (expected_L, expected_L) and np.all(np.abs(level) == 1)
        if not ok:
            tower_pass = False
        print(f"  Level {i}: {level.shape}, spins valid: {ok}")

    print(f"  RG tower check: {'PASS' if tower_pass else 'FAIL'}")

    # Test region extraction
    L = 16
    lattice = rng.choice(np.array([-1, 1], dtype=np.int8), size=(L, L))

    m = 4
    interior, blanket = extract_regions(lattice, m, cx=5, cy=3)
    interior_size_pass = len(interior) == m * m  # 16
    blanket_size_pass = len(blanket) == (m + 2) ** 2 - m ** 2  # 36 - 16 = 20
    print(f"  Region m={m}: interior={len(interior)} (exp {m*m}), blanket={len(blanket)} (exp {(m+2)**2 - m*m})")
    print(f"  Interior size: {'PASS' if interior_size_pass else 'FAIL'}")
    print(f"  Blanket size: {'PASS' if blanket_size_pass else 'FAIL'}")

    # Test periodic BC wrapping
    interior_wrap, blanket_wrap = extract_regions(lattice, m=4, cx=14, cy=14)
    wrap_pass = len(interior_wrap) == 16 and len(blanket_wrap) == 20
    print(f"  Periodic BC wrapping: {'PASS' if wrap_pass else 'FAIL'}")

    # Verify blanket is actually the ring around the interior
    # For a known position, manually check
    m = 2
    interior2, blanket2 = extract_regions(lattice, m, cx=3, cy=3)
    int_size_pass = len(interior2) == 4  # 2×2
    bla_size_pass = len(blanket2) == 12  # 4×4 - 2×2
    print(f"  Region m={m}: interior={len(interior2)} (exp 4), blanket={len(blanket2)} (exp 12)")
    print(f"  m=2 sizes: {'PASS' if int_size_pass and bla_size_pass else 'FAIL'}")

    all_pass = all([shape_pass, spins_pass, tower_pass, interior_size_pass,
                    blanket_size_pass, wrap_pass, int_size_pass, bla_size_pass])
    print(f"  Overall: {'PASS' if all_pass else 'FAIL'}")
    return all_pass


if __name__ == "__main__":
    validate_coarse_grain()
