"""Runtime benchmark (Table III).

Algorithm 1 is timed in two forms:
  * NumPy (vectorized Python), bisection on log(lambda) with the explicit bracket;
  * compiled (Numba), with bisection and with a safeguarded Newton step on log(lambda).
References:
  * CVXPY + Clarabel on the epigraph (DPP-parameterized: compiled once, then re-solved);
    we report the solver's own solve time (no Python/canonicalization overhead);
  * SciPy SLSQP with analytic gradients.
Instances: a_i = 10 * Exp(1) (10 dB), T_i = 2, P_tot = min(N, sum(Pbar)/2) (regime B).
"""
import json, os, time, warnings
import numpy as np
import numba as nb
import cvxpy as cp
warnings.filterwarnings("ignore")
from tralloc import target_rate_ls, slsqp, caps, objective, LN2

HERE = os.path.dirname(os.path.abspath(__file__))
C2H = LN2 * LN2 / 2.0


@nb.njit(cache=True, fastmath=False)
def lambertw0(z):
    """Principal branch W0(z), z >= 0, Halley iteration."""
    if z == 0.0:
        return 0.0
    w = np.log1p(z) if z < 3.0 else np.log(z) - np.log(np.log(z))
    for _ in range(40):
        ew = np.exp(w)
        f = w * ew - z
        wp1 = w + 1.0
        dw = f / (ew * wp1 - (w + 2.0) * f / (2.0 * wp1))
        w -= dw
        if abs(dw) <= 1e-15 * (1.0 + abs(w)):
            break
    return w


@nb.njit(cache=True)
def _S(lam, a, T, q, P, om):
    s = 0.0
    for i in range(a.size):
        w = lambertw0(lam * C2H * q[i])
        d = w / LN2
        if d >= T[i]:
            P[i] = 0.0
            om[i] = -1.0          # inactive marker
        else:
            P[i] = (np.exp((T[i] - d) * LN2) - 1.0) / a[i]
            om[i] = w
        s += P[i]
    return s


@nb.njit(cache=True)
def alg1_nb(a, T, Ptot, eps, newton):
    n = a.size
    q = np.empty(n); Pbar = np.empty(n)
    sPbar = 0.0; sq2 = 0.0; lam_hi = 0.0
    for i in range(n):
        q[i] = np.exp(T[i] * LN2) / a[i]
        Pbar[i] = (np.exp(T[i] * LN2) - 1.0) / a[i]
        sPbar += Pbar[i]; sq2 += q[i] * q[i]
        lam_hi = max(lam_hi, 2.0 * a[i] * T[i] / LN2)
    if sPbar <= Ptot:
        return Pbar, 0
    lo = np.log((sPbar - Ptot) / (LN2 * LN2 * sq2)); hi = np.log(lam_hi)
    P = np.empty(n); om = np.empty(n)
    u = 0.5 * (lo + hi)
    it = 0
    while it < 200:
        it += 1
        lam = np.exp(u)
        g = _S(lam, a, T, q, P, om) - Ptot
        if g > 0.0:
            lo = u
        else:
            hi = u
        if newton:
            if abs(g) <= eps * Ptot:
                break
            # dS/du = lam * S'(lam) = -sum_active (1 + a P) / a * w / (1 + w)
            dS = 0.0
            for i in range(n):
                if om[i] >= 0.0:
                    dS -= (1.0 + a[i] * P[i]) / a[i] * om[i] / (1.0 + om[i])
            un = u - g / dS if dS < 0.0 else 0.5 * (lo + hi)
            u = un if (un > lo and un < hi) else 0.5 * (lo + hi)   # safeguard
        else:
            if hi - lo <= eps:
                break
            u = 0.5 * (lo + hi)
    lam = np.exp(hi if not newton else u)
    _S(lam, a, T, q, P, om)
    s = P.sum()
    if s > Ptot:
        P *= Ptot / s
    return P, it


def build_cvx(N):
    a = cp.Parameter(N, nonneg=True); T = cp.Parameter(N); Pt = cp.Parameter(nonneg=True)
    P = cp.Variable(N, nonneg=True); t = cp.Variable(N, nonneg=True)
    prob = cp.Problem(cp.Minimize(cp.sum_squares(t)),
                      [cp.sum(P) <= Pt, t >= T - cp.log(1 + cp.multiply(a, P)) / LN2])
    return prob, a, T, Pt, P


def best(f, reps):
    t = np.inf
    for _ in range(reps):
        s = time.perf_counter(); f(); t = min(t, time.perf_counter() - s)
    return t


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    # warm up JIT
    a0 = 10 * rng.exponential(1, 8); alg1_nb(a0, np.full(8, 2.0), 4.0, 1e-12, False); alg1_nb(a0, np.full(8, 2.0), 4.0, 1e-12, True)
    rows = []
    for N in [8, 64, 256, 1024, 4096]:
        n_inst = 10 if N <= 1024 else 3
        rec = {k: [] for k in ["np", "bis", "bis6", "newt", "clar", "cvx_total", "slsqp",
                               "it_bis", "it_bis6", "it_newt", "gap_newt", "gap_bis6", "gap_clar", "gap_slsqp", "clar_fail"]}
        prob, ap, Tp, Ptp, Pv = build_cvx(N)
        for k in range(n_inst):
            a = 10 * rng.exponential(1.0, N); T = np.full(N, 2.0)
            Pt = min(float(N), 0.5 * caps(a, T).sum())
            ref = target_rate_ls(a, T, Pt)                    # NumPy Algorithm 1, eps = 1e-12
            Jref = objective(a, ref, T)
            rec["np"].append(best(lambda: target_rate_ls(a, T, Pt), 5))
            P1, it1 = alg1_nb(a, T, Pt, 1e-12, False)
            P6, it6 = alg1_nb(a, T, Pt, 1e-6, False)
            P2, it2 = alg1_nb(a, T, Pt, 1e-12, True)
            rec["bis"].append(best(lambda: alg1_nb(a, T, Pt, 1e-12, False), 20)); rec["it_bis"].append(it1)
            rec["bis6"].append(best(lambda: alg1_nb(a, T, Pt, 1e-6, False), 20)); rec["it_bis6"].append(it6)
            rec["newt"].append(best(lambda: alg1_nb(a, T, Pt, 1e-12, True), 20)); rec["it_newt"].append(it2)
            rec["gap_newt"].append(abs(objective(a, P2, T) - Jref) / Jref)
            rec["gap_bis6"].append(abs(objective(a, P6, T) - Jref) / Jref)
            ap.value = a; Tp.value = T; Ptp.value = Pt
            try:
                prob.solve(solver=cp.CLARABEL)                # first solve compiles (excluded)
                s = time.perf_counter(); prob.solve(solver=cp.CLARABEL); rec["cvx_total"].append(time.perf_counter() - s)
                rec["clar"].append(prob.solver_stats.solve_time)
                rec["gap_clar"].append((objective(a, np.maximum(Pv.value, 0), T) - Jref) / Jref)
            except cp.error.SolverError:
                rec["clar_fail"].append(1)
            if N <= 1024 and k < (3 if N >= 256 else n_inst):
                s = time.perf_counter(); Ps = slsqp(a, T, Pt); rec["slsqp"].append(time.perf_counter() - s)
                rec["gap_slsqp"].append((objective(a, Ps, T) - Jref) / Jref)
        row = {"N": N}
        for k, v in rec.items():
            if v and k != "clar_fail":
                row[k] = float(np.median(v)) if not k.startswith("gap") else float(np.max(np.abs(v)))
        row["clar_fail"] = len(rec["clar_fail"]); row["n_inst"] = n_inst
        row["it_bis_range"] = [int(min(rec["it_bis"])), int(max(rec["it_bis"]))]
        row["it_newt_range"] = [int(min(rec["it_newt"])), int(max(rec["it_newt"]))]
        row["it_bis6_range"] = [int(min(rec["it_bis6"])), int(max(rec["it_bis6"]))]
        rows.append(row)
        print(json.dumps(row), flush=True)
    json.dump(rows, open(os.path.join(HERE, "results_runtime.json"), "w"), indent=1)
