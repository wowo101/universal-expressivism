"""Unified validation suite. Run before the main experiment.

Usage: python -m ic_scale.validate
"""

import numpy as np

from ic_scale.sim.ising import validate_wolff, TC, _seed_rng
from ic_scale.sim.coarse_grain import validate_coarse_grain
from ic_scale.info.validate_mi import validate_mi_known, validate_mi_cross
from ic_scale.measure.correlation import validate_xi_estimation
from ic_scale.measure.kappa import compute_kappa
from ic_scale.sim.ising import equilibrate, sample
from ic_scale.sim.coarse_grain import build_rg_tower


def validate_kappa_limits():
    """Validate κ behavior at temperature extremes.

    Checks:
    1. κ → near 0 at T >> T_c (disordered, weak correlations)
    2. κ is non-trivial at T_c
    3. κ is large at T << T_c (ordered, strong correlations)
    """
    L = 64
    n_equil = 2000
    n_samples = 500
    n_blocks = 100
    n_bootstrap = 20
    m = 4

    print("Validating κ limits...")

    results = {}
    for T, label in [(1.5, "ordered"), (TC, "critical"), (3.5, "disordered")]:
        seed = int(T * 10000)
        _seed_rng(seed)
        lattice, _ = equilibrate(L, T, n_steps=n_equil, seed=seed)
        configs = sample(lattice, L, T, n_samples, thinning=5, seed=seed + 1)

        rng = np.random.default_rng(seed + 2)
        kr = compute_kappa(configs, L, m=m, n_blocks=n_blocks,
                           mi_method='magnetization', n_bootstrap=n_bootstrap,
                           rng=rng)
        results[label] = kr.kappa
        print(f"  T={T:.1f} ({label}): κ(m={m}) = {kr.kappa:.4f} ± {kr.kappa_err:.4f}")

    # Checks
    dis_pass = results["disordered"] < 0.05
    print(f"  κ(disordered) < 0.05: {'PASS' if dis_pass else 'FAIL'}")

    crit_pass = 0.05 < results["critical"] < 0.5
    print(f"  0.05 < κ(critical) < 0.5: {'PASS' if crit_pass else 'FAIL'}")

    ord_pass = results["ordered"] > 0.5
    print(f"  κ(ordered) > 0.5: {'PASS' if ord_pass else 'FAIL'}")

    order_pass = results["ordered"] > results["critical"] > results["disordered"]
    print(f"  Monotonic ordering: {'PASS' if order_pass else 'FAIL'}")

    all_pass = dis_pass and crit_pass and ord_pass and order_pass
    print(f"  Overall: {'PASS' if all_pass else 'FAIL'}")
    return all_pass


def validate_all():
    """Run all validation checks. Returns True if all pass."""
    checks = [
        ("Wolff cluster MC", validate_wolff),
        ("Coarse-graining", validate_coarse_grain),
        ("MI (known distribution)", validate_mi_known),
        ("MI (cross-validation)", validate_mi_cross),
        ("ξ estimation", validate_xi_estimation),
        ("κ limits", validate_kappa_limits),
    ]

    results = []
    for name, func in checks:
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")
        passed = func()
        results.append((name, passed))

    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {status}  {name}")
        if not passed:
            all_pass = False

    print(f"\n  {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")
    return all_pass


if __name__ == "__main__":
    validate_all()
