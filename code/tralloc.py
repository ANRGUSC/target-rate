"""Target-rate least-squares power allocation over parallel Gaussian channels.

    minimize   J(P) = sum_i (log2(1 + a_i P_i) - T_i)^2
    subject to sum_i P_i <= P_tot,  P_i >= 0.

Closed form (Theorem 2 of the paper): for a dual price lam > 0,
    delta_i(lam) = min{ T_i, W0(lam * ln2^2 * q_i / 2) / ln2 },   q_i = 2^{T_i} / a_i,
    P_i(lam)     = (2^{T_i - delta_i} - 1) / a_i,
and lam* solves sum_i P_i(lam) = P_tot on the explicit bracket (0, max_i 2 a_i T_i / ln2].

Baselines: classical waterfilling, target-capped waterfilling, uniform,
proportional fairness (max sum log r_i) and max-min target satisfaction.
"""
import numpy as np
from scipy.special import lambertw

LN2 = np.log(2.0)


def rates(a, P):
    return np.log2(1.0 + a * P)


def objective(a, P, T):
    return float(np.sum((rates(a, P) - T) ** 2))


def caps(a, T):
    """Power that meets each target exactly, Pbar_i = (2^T_i - 1)/a_i."""
    return (np.exp2(T) - 1.0) / a


# ----------------------------------------------------------------- proposed
def shortfall(lam, a, T):
    """Optimal rate shortfall delta_i(lam) (bits/s/Hz), clipped to [0, T_i]."""
    q = np.exp2(T) / a
    z = lam * LN2 ** 2 * q / 2.0
    d = np.real(lambertw(z, 0)) / LN2
    return np.minimum(T, d)


def power_at(lam, a, T):
    d = shortfall(lam, a, T)
    return (np.exp2(T - d) - 1.0) / a


def target_rate_ls(a, T, P_tot, tol=1e-12, max_iter=200, return_info=False):
    """Algorithm 1: closed-form per-channel powers + bisection on log(lam)."""
    a = np.asarray(a, float)
    T = np.broadcast_to(np.asarray(T, float), a.shape).copy()
    Pbar = caps(a, T)
    if Pbar.sum() <= P_tot:                      # Case A: every target met
        out = (Pbar, 0.0, 0)
        return out if return_info else Pbar
    q = np.exp2(T) / a
    lam_hi = np.max(2.0 * a * T / LN2)                        # S(lam_hi) = 0 < P_tot
    lam_lo = (Pbar.sum() - P_tot) / (LN2 ** 2 * np.sum(q ** 2))  # S(lam_lo) > P_tot (Prop. 3)
    it = 0
    while it < max_iter:
        it += 1
        lam = np.sqrt(lam_lo * lam_hi)           # bisection in the log domain
        S = power_at(lam, a, T).sum()
        if S > P_tot:
            lam_lo = lam
        else:
            lam_hi = lam
        if lam_hi / lam_lo - 1.0 < tol:
            break
    lam = lam_hi                                  # feasible side
    P = power_at(lam, a, T)
    P *= min(1.0, P_tot / max(P.sum(), 1e-300))
    return (P, lam, it) if return_info else P


# ----------------------------------------------------------------- baselines
def _bisect(fun, lo, hi, target, iters=200, increasing=True):
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        v = fun(mid)
        if (v < target) == increasing:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def waterfilling(a, P_tot):
    a = np.asarray(a, float)
    s = lambda mu: np.maximum(0.0, mu - 1.0 / a).sum()
    mu = _bisect(s, 0.0, P_tot + np.max(1.0 / a), P_tot)
    P = np.maximum(0.0, mu - 1.0 / a)
    return P * (P_tot / P.sum())


def capped_waterfilling(a, T, P_tot):
    """Maximize sum_i min(r_i, T_i): waterfilling with per-channel caps Pbar_i."""
    a = np.asarray(a, float)
    Pbar = caps(a, np.broadcast_to(T, a.shape))
    if Pbar.sum() <= P_tot:
        return Pbar.copy()
    f = lambda mu: np.clip(mu - 1.0 / a, 0.0, Pbar).sum()
    mu = _bisect(f, 0.0, np.max(1.0 / a + Pbar), P_tot)
    return np.clip(mu - 1.0 / a, 0.0, Pbar)


def uniform(a, P_tot):
    return np.full(len(a), P_tot / len(a))


def proportional_fair(a, P_tot):
    """Maximize sum_i log r_i. KKT: a_i / ((1+a_i P_i) ln(1+a_i P_i)) = lam."""
    a = np.asarray(a, float)

    def P_of(lam):
        # g(P) = a / ((1+aP) ln(1+aP)) is decreasing from +inf; solve g(P) = lam.
        lo = np.zeros_like(a)
        hi = np.full_like(a, 10.0 * P_tot + 1.0)
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            g = a / ((1.0 + a * mid) * np.log1p(a * mid))
            big = g > lam
            lo = np.where(big, mid, lo)
            hi = np.where(big, hi, mid)
        return 0.5 * (lo + hi)

    lo, hi = 1e-12, 1e12
    for _ in range(200):
        lam = np.sqrt(lo * hi)
        if P_of(lam).sum() > P_tot:
            lo = lam
        else:
            hi = lam
    P = P_of(hi)
    return P * (P_tot / P.sum())


def maxmin_satisfaction(a, T, P_tot):
    """Maximize min_i r_i / T_i (capped at 1): P_i = (2^{s T_i} - 1)/a_i."""
    a = np.asarray(a, float)
    T = np.broadcast_to(np.asarray(T, float), a.shape)
    if caps(a, T).sum() <= P_tot:
        return caps(a, T)
    f = lambda s: ((np.exp2(s * T) - 1.0) / a).sum()
    s = _bisect(f, 0.0, 1.0, P_tot)
    return (np.exp2(s * T) - 1.0) / a


# ----------------------------------------------------------------- references
def slsqp(a, T, P_tot):
    """General-purpose NLP baseline (SciPy SLSQP with analytic gradient)."""
    from scipy.optimize import minimize
    a = np.asarray(a, float)
    T = np.broadcast_to(np.asarray(T, float), a.shape)
    N = len(a)

    def f(P):
        r = np.log2(1.0 + a * P)
        g = 2.0 * (r - T) * a / ((1.0 + a * P) * LN2)
        return np.sum((r - T) ** 2), g

    res = minimize(f, np.full(N, P_tot / N), jac=True, method="SLSQP",
                   bounds=[(0.0, None)] * N,
                   constraints=[{"type": "ineq", "fun": lambda P: P_tot - P.sum(),
                                 "jac": lambda P: -np.ones(N)}],
                   options={"maxiter": 2000, "ftol": 1e-14})
    return res.x


def cvx(a, T, P_tot):
    """Independent global optimum via CVXPY/Clarabel (epigraph of shortfall)."""
    import cvxpy as cp
    a = np.asarray(a, float)
    T = np.broadcast_to(np.asarray(T, float), a.shape)
    P = cp.Variable(len(a), nonneg=True)
    t = cp.Variable(len(a), nonneg=True)
    cons = [cp.sum(P) <= P_tot, t >= T - cp.log(1 + cp.multiply(a, P)) / LN2]
    cp.Problem(cp.Minimize(cp.sum_squares(t)), cons).solve(solver=cp.CLARABEL)
    return np.maximum(P.value, 0.0)


# ----------------------------------------------------------------- l_p family
def shortfall_p(lam, a, T, p):
    """delta_i(lam) for J_p = sum ((T_i - r_i)^+)^p, p > 1:
    delta = (m/ln2) W0( (ln2/m) (lam ln2 q / p)^{1/m} ),  m = p - 1,  clipped to T_i."""
    m = p - 1.0
    q = np.exp2(T) / a
    # log-domain argument for numerical range: (lam ln2 q / p)^{1/m}
    arg = (LN2 / m) * np.exp(np.log(lam * LN2 * q / p) / m)
    d = (m / LN2) * np.real(lambertw(arg, 0))
    return np.minimum(T, d)


def target_rate_lp(a, T, P_tot, p, tol=1e-12, max_iter=400):
    a = np.asarray(a, float)
    T = np.broadcast_to(np.asarray(T, float), a.shape).copy()
    if p == 1:
        return capped_waterfilling(a, T, P_tot)
    if np.isinf(p):
        return minimax_shortfall(a, T, P_tot)
    Pbar = caps(a, T)
    if Pbar.sum() <= P_tot:
        return Pbar
    PofL = lambda lam: (np.exp2(T - shortfall_p(lam, a, T, p)) - 1.0) / a
    # inactive threshold: delta = T  <=>  lam = p T^{p-1} a / ln2
    lam_hi = np.max(p * T ** (p - 1) * a / LN2)
    lam_lo = lam_hi * 1e-30
    while PofL(lam_lo).sum() <= P_tot:
        lam_lo *= 1e-10
    for _ in range(max_iter):
        lam = np.sqrt(lam_lo * lam_hi)
        if PofL(lam).sum() > P_tot:
            lam_lo = lam
        else:
            lam_hi = lam
        if lam_hi / lam_lo - 1.0 < tol:
            break
    P = PofL(lam_hi)
    return P * min(1.0, P_tot / P.sum())


def minimax_shortfall(a, T, P_tot):
    """p -> infinity: minimize max_i (T_i - r_i)^+ : common shortfall d, P_i = (2^{(T_i-d)^+}-1)/a_i."""
    a = np.asarray(a, float)
    T = np.broadcast_to(np.asarray(T, float), a.shape)
    if caps(a, T).sum() <= P_tot:
        return caps(a, T)
    f = lambda d: ((np.exp2(np.maximum(T - d, 0)) - 1.0) / a).sum()
    d = _bisect(f, 0.0, T.max(), P_tot, increasing=False)
    return (np.exp2(np.maximum(T - d, 0)) - 1.0) / a


def cvx_p(a, T, P_tot, p):
    import cvxpy as cp
    a = np.asarray(a, float)
    T = np.broadcast_to(np.asarray(T, float), a.shape)
    P = cp.Variable(len(a), nonneg=True)
    t = cp.Variable(len(a), nonneg=True)
    cons = [cp.sum(P) <= P_tot, t >= T - cp.log(1 + cp.multiply(a, P)) / LN2]
    cp.Problem(cp.Minimize(cp.sum(cp.power(t, p))), cons).solve(solver=cp.CLARABEL)
    return np.maximum(P.value, 0.0)


# ----------------------------------------------------------------- weighted
def target_rate_wls(a, T, P_tot, w, tol=1e-12, max_iter=200):
    """Weighted variant: minimize sum_i w_i (r_i - T_i)^2. Same closed form with q_i -> q_i / w_i.
    w_i = T_i^{-2} gives the normalized (relative) deviation sum_i ((r_i - T_i)/T_i)^2."""
    a = np.asarray(a, float)
    T = np.broadcast_to(np.asarray(T, float), a.shape).copy()
    w = np.broadcast_to(np.asarray(w, float), a.shape)
    Pbar = caps(a, T)
    if Pbar.sum() <= P_tot:
        return Pbar
    qw = np.exp2(T) / a / w
    PofL = lambda lam: (np.exp2(T - np.minimum(T, np.real(lambertw(lam * LN2 ** 2 * qw / 2)) / LN2)) - 1) / a
    lam_hi = np.max(2.0 * w * a * T / LN2)
    lam_lo = (Pbar.sum() - P_tot) / (LN2 ** 2 * np.sum(np.exp2(T) / a * qw))
    for _ in range(max_iter):
        lam = np.sqrt(lam_lo * lam_hi)
        if PofL(lam).sum() > P_tot:
            lam_lo = lam
        else:
            lam_hi = lam
        if lam_hi / lam_lo - 1.0 < tol:
            break
    P = PofL(lam_hi)
    return P * min(1.0, P_tot / P.sum())


# ----------------------------------------------------------------- overflow-safe versions (large targets, e.g. T_i = backlog)
def _lambertw_log(lnz):
    """W0(exp(lnz)) for any real lnz (no overflow)."""
    lnz = np.asarray(lnz, float)
    out = np.empty_like(lnz)
    small = lnz < 600
    out[small] = np.real(lambertw(np.exp(lnz[small]), 0))
    if np.any(~small):
        L = lnz[~small]
        w = L - np.log(L)
        for _ in range(8):                       # Newton on w + ln w = L
            w = w - (w + np.log(w) - L) / (1 + 1 / w)
        out[~small] = w
    return out


def _logsumexp(x):
    m = np.max(x)
    return m + np.log(np.sum(np.exp(x - m)))


def target_rate_ls_safe(a, T, P_tot, w=None, tol=1e-12, max_iter=200):
    """Same optimum as target_rate_ls (weighted if w given), computed in the log domain."""
    a = np.asarray(a, float)
    T = np.broadcast_to(np.asarray(T, float), a.shape).copy()
    w = np.ones_like(a) if w is None else np.broadcast_to(np.asarray(w, float), a.shape)
    lnq = T * LN2 - np.log(a) - np.log(w)              # ln(q_i / w_i)
    lnPbar = T * LN2 + np.log1p(-np.exp2(-T)) - np.log(a)  # ln Pbar_i
    lnSumPbar = _logsumexp(lnPbar)
    if lnSumPbar <= np.log(P_tot):
        return np.exp(lnPbar)

    def P_of(lnlam):
        d = np.minimum(T, _lambertw_log(lnlam + np.log(LN2 ** 2 / 2) + lnq) / LN2)
        return np.expm1((T - d) * LN2) / a

    lnhi = np.log(np.max(2 * w * a * T / LN2))
    # ln lam_lo = ln(sum Pbar - P_tot) - ln(c^2 sum q_i^2 / w_i)
    lnlo = lnSumPbar + np.log(-np.expm1(np.log(P_tot) - lnSumPbar)) \
        - np.log(LN2 ** 2) - _logsumexp(2 * (T * LN2 - np.log(a)) - np.log(w))
    for _ in range(max_iter):
        mid = 0.5 * (lnlo + lnhi)
        if P_of(mid).sum() > P_tot:
            lnlo = mid
        else:
            lnhi = mid
        if lnhi - lnlo < tol:
            break
    P = P_of(lnhi)
    return P * min(1.0, P_tot / P.sum())


def capped_waterfilling_safe(a, T, P_tot):
    a = np.asarray(a, float)
    T = np.broadcast_to(np.asarray(T, float), a.shape)
    Pbar = np.where(T * LN2 < 700, (np.exp2(np.minimum(T, 1000)) - 1) / a, np.inf)
    if np.sum(Pbar) <= P_tot:
        return Pbar.copy()
    lo, hi = 0.0, P_tot + np.max(1 / a)
    for _ in range(200):
        mu = 0.5 * (lo + hi)
        if np.clip(mu - 1 / a, 0, Pbar).sum() < P_tot:
            lo = mu
        else:
            hi = mu
    return np.clip(0.5 * (lo + hi) - 1 / a, 0, Pbar)
