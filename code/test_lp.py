import numpy as np, warnings
warnings.filterwarnings("ignore")
from tralloc import *
rng = np.random.default_rng(3)
Jp = lambda a, P, T, p: np.sum(np.maximum(T - rates(a, P), 0) ** p)
worst = {}; fails = 0
for p in [1.25, 1.3, 1.5, 2.0, 3.0, 5.0, 10.0]:
    for trial in range(60):
        N = rng.integers(2, 30)
        a = rng.exponential(1.0, N) * 10 ** rng.uniform(-0.5, 1.5)
        T = rng.uniform(0.3, 5, N)
        Pt = rng.uniform(0.1, 0.95) * caps(a, T).sum()
        P = target_rate_lp(a, T, Pt, p)
        try:
            Pc = cvx_p(a, T, Pt, p)
        except Exception:
            fails += 1; continue
        g = (Jp(a, P, T, p) - Jp(a, Pc, T, p)) / Jp(a, Pc, T, p)
        worst[p] = max(worst.get(p, -1), g)
        if p == 2.0:
            assert np.allclose(P, target_rate_ls(a, T, Pt), rtol=1e-6, atol=1e-9)
print({p: f"{v:.1e}" for p, v in worst.items()}, "(positive = ours worse than CVXPY); CVXPY failures:", fails)
