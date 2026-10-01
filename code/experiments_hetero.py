"""Mixed traffic classes: per-subcarrier targets drawn uniformly from {1, 2, 4} b/s/Hz."""
import json, os, warnings
import numpy as np
from multiprocessing import Pool
warnings.filterwarnings("ignore")
from tralloc import *

HERE = os.path.dirname(os.path.abspath(__file__))
N, REAL, SNRS = 64, 500, [5, 10]
CLASSES = np.array([1.0, 2.0, 4.0])
METHODS = {
    "TR-LS": lambda a, T, P: target_rate_ls(a, T, P),
    "TR-NLS": lambda a, T, P: target_rate_wls(a, T, P, 1.0 / T ** 2),
    "Capped WF": lambda a, T, P: capped_waterfilling(a, T, P),
    "Max-min": lambda a, T, P: maxmin_satisfaction(a, T, P),
    "PF": lambda a, T, P: proportional_fair(a, P),
    "WF": lambda a, T, P: waterfilling(a, P),
    "Uniform": lambda a, T, P: uniform(a, P),
}


def one(args):
    snr, seed = args
    rng = np.random.default_rng(seed)
    a = 10 ** (snr / 10) * rng.exponential(1.0, N)
    T = rng.choice(CLASSES, N)
    out = {}
    for k, f in METHODS.items():
        P = f(a, T, float(N))
        sat = np.minimum(rates(a, P), T) / T
        starved = P < 1e-12
        out[k] = dict(served=sat.mean(), bits=np.minimum(rates(a, P), T).sum() / T.sum(),
                      p5=np.percentile(sat, 5), starved=starved.mean(),
                      starved_hi=starved[T == 4].mean() if np.any(T == 4) else 0.0,
                      starved_lo=starved[T == 1].mean() if np.any(T == 1) else 0.0,
                      jain=sat.sum() ** 2 / (N * np.sum(sat ** 2)), used=P.sum() / N)
    return snr, out


if __name__ == "__main__":
    jobs = [(s, 7000 + 1000 * i + s) for s in SNRS for i in range(REAL)]
    with Pool(2) as pool:
        res = pool.map(one, jobs, chunksize=20)
    agg = {}
    for s in SNRS:
        agg[s] = {k: {m: float(np.mean([o[k][m] for ss, o in res if ss == s]))
                      for m in res[0][1][k]} for k in METHODS}
    json.dump({str(s): v for s, v in agg.items()}, open(os.path.join(HERE, "results_hetero.json"), "w"), indent=1)
    for s in SNRS:
        print("SNR", s)
        for k, v in agg[s].items():
            print("  %-10s " % k + " ".join(f"{m} {x:.3f}" for m, x in v.items()))
