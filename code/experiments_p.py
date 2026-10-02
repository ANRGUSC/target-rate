"""Efficiency-equity trade-off across the l_p shortfall family (Fig. 4)."""
import json, os, sys, warnings
import numpy as np
from multiprocessing import Pool
warnings.filterwarnings("ignore")
from tralloc import *
import experiments_style as S  # shared matplotlib style
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
QUICK = "--quick" in sys.argv
PS = [1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 10.0, np.inf]
SNRS = [5, 10]
N, TT = 64, 2.0
REAL = 60 if QUICK else 500


def one(args):
    snr, seed = args
    rng = np.random.default_rng(seed)
    a = 10 ** (snr / 10) * rng.exponential(1.0, N)
    T = np.full(N, TT)
    out = []
    for p in PS:
        P = target_rate_lp(a, T, float(N), p)
        sat = np.minimum(rates(a, P), T) / T
        out.append((sat.mean(), np.percentile(sat, 5), np.mean(P < 1e-12)))
    return snr, np.array(out)


if __name__ == "__main__":
    jobs = [(s, 1000 * i + s) for s in SNRS for i in range(REAL)]  # same seeds as experiments.py
    with Pool(2) as pool:
        res = pool.map(one, jobs, chunksize=20)
    agg = {s: np.mean([o for ss, o in res if ss == s], axis=0) for s in SNRS}
    out = {str(s): [dict(p=("inf" if np.isinf(p) else p), served=float(v[0]), p5=float(v[1]),
                         starved=float(v[2])) for p, v in zip(PS, agg[s])] for s in SNRS}
    json.dump(out, open(os.path.join(HERE, "results_p.json"), "w"), indent=1)
    for s in SNRS:
        for r in out[str(s)]:
            print(s, r)

    fig, axs = plt.subplots(1, 2, figsize=(S.COLW, 2.1), gridspec_kw=dict(wspace=0.38))
    for ax, s, col, mk in [(axs[0], 5, "#2a78d6", "o"), (axs[1], 10, "#eb6834", "s")]:
        v = agg[s]
        ax.plot(100 * v[:, 2], 100 * v[:, 0], color=col, marker=mk, ms=3.5, lw=1.2)
        for j, p in enumerate(PS):
            lab = {1.0: "$p{=}1$", 2.0: "$p{=}2$", 5.0: "$p{=}5$", np.inf: "$p{\\to}\\infty$"}.get(p)
            if lab:
                off = {1.0: (-3, -9), 2.0: (7, -9), 5.0: (4, 1), np.inf: (4, -2)}[p]
                ax.annotate(lab, (100 * v[j, 2], 100 * v[j, 0]), textcoords="offset points",
                            xytext=off, fontsize=7, ha="right" if p == 1.0 else "left")
        j2 = PS.index(2.0)
        ax.plot(100 * v[j2, 2], 100 * v[j2, 0], marker=mk, ms=7, mfc="none", mec="#0b0b0b",
                mew=0.8, ls="none")
        ax.set_title(f"mean SNR {s} dB", fontsize=8)
        ax.set_xlabel("starved channels (%)")
    axs[0].set_ylabel("mean satisfaction (%)")
    fig.savefig(os.path.join(HERE, "..", "figs", "fig_pfamily.pdf"))
