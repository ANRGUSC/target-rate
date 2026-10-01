"""Checks: closed form vs CVXPY, KKT conditions, structural claims."""
import numpy as np
from tralloc import *

rng = np.random.default_rng(1)
worst_gap, worst_P = 0.0, 0.0
for trial in range(300):
    N = rng.integers(2, 40)
    a = rng.exponential(1.0, N) * 10 ** rng.uniform(-1, 2)
    T = rng.uniform(0.2, 6, N) if trial % 2 else np.full(N, rng.uniform(0.5, 5))
    Pt = rng.uniform(0.05, 1.5) * caps(a, T).sum()
    P, lam, it = target_rate_ls(a, T, Pt, return_info=True)
    Pc = cvx(a, T, Pt)
    assert P.sum() <= Pt * (1 + 1e-9) and np.all(P >= 0)
    # no overshoot
    assert np.all(rates(a, P) <= T + 1e-9)
    J = objective(a, P, T)
    Jc = float(np.sum(np.maximum(T - rates(a, Pc), 0) ** 2))  # epigraph value
    worst_gap = max(worst_gap, (J - Jc) / max(Jc, 1e-9))
    if lam > 0:  # Case B: optimum unique, compare powers
        worst_P = max(worst_P, np.max(np.abs(P - Pc)) / max(P.max(), 1e-9))
    if lam > 0:
        d = T - rates(a, P)
        act = P > 1e-9
        # shortfall identity: delta_i = (lam ln2 / 2)(P_i + 1/a_i) on active channels
        assert np.allclose(d[act], lam * LN2 / 2 * (P[act] + 1 / a[act]), rtol=1e-6)
        # inactive iff a_i T_i <= lam ln2 / 2
        assert np.all((a * T <= lam * LN2 / 2 + 1e-9) == ~act)
        assert abs(P.sum() - Pt) < 1e-7 * Pt
print(f"worst relative J gap vs CVXPY: {worst_gap:.2e}; worst rel. power diff {worst_P:.2e}")

# Examples used in the paper
a8 = np.array([20, 15, 10, 7, 5, 3, 2, 1.0])
print("sum Pbar (T=3):", caps(a8, np.full(8, 3.0)).sum())
Th = np.array([5, 4, 3, 3, 2, 2, 1, 1.0])
print("sum Pbar (hetero):", caps(a8, Th).sum())
P, lam, it = target_rate_ls(a8, 3.0, 10.0, return_info=True)
print("P*", np.round(P, 3), "lam", lam, "iters", it, "J", objective(a8, P, 3.0))
print("r*", np.round(rates(a8, P), 3))
Ps = slsqp(a8, 3.0, 10.0)
print("SLSQP J", objective(a8, Ps, 3.0), "max|dP|", np.abs(Ps - P).max())
