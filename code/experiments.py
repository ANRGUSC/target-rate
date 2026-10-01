"""All numerical results for the ICC version. Writes figures to ../figs and numbers to results.json.

Run:  python3 experiments.py            (full run, ~several minutes on 2 cores)
"""
import json, time, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from multiprocessing import Pool
from tralloc import *

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "figs")
os.makedirs(FIG, exist_ok=True)
QUICK = "--quick" in sys.argv

# ---- style: IEEE column width, serif to sit next to Times text --------------
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": 0.6, "axes.edgecolor": "#52514e",
    "xtick.color": "#52514e", "ytick.color": "#52514e",
    "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.4, "lines.markersize": 4,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
COLW = 3.45
STYLE = {  # color = identity, marker/linestyle = secondary encoding for print/CVD
    "TR-LS":     dict(color="#2a78d6", marker="o", ls="-",  label="Target-rate LS (proposed)"),
    "WF":        dict(color="#eb6834", marker="s", ls="--", label="Waterfilling"),
    "Capped WF": dict(color="#1baf7a", marker="^", ls="-.", label="Capped waterfilling"),
    "Max-min":   dict(color="#4a3aa7", marker="D", ls=":",  label="Max-min satisfaction"),
    "PF":        dict(color="#e87ba4", marker="v", ls="--", label="Proportional fair"),
    "Uniform":   dict(color="#8a8985", marker="x", ls=":",  label="Uniform"),
}
ORDER = ["TR-LS", "Capped WF", "Max-min", "WF", "PF", "Uniform"]


def allocate(name, a, T, Pt):
    if name == "TR-LS":     return target_rate_ls(a, T, Pt)
    if name == "WF":        return waterfilling(a, Pt)
    if name == "Capped WF": return capped_waterfilling(a, T, Pt)
    if name == "Max-min":   return maxmin_satisfaction(a, T, Pt)
    if name == "PF":        return proportional_fair(a, Pt)
    if name == "Uniform":   return uniform(a, Pt)


R = {}

# ============================================================ E1: worked example
a8 = np.array([20, 15, 10, 7, 5, 3, 2, 1.0]); T8 = 3.0; P8 = 10.0
P, lam, it = target_rate_ls(a8, T8, P8, return_info=True)
Pw, Pc = waterfilling(a8, P8), capped_waterfilling(a8, T8, P8)
Pm = maxmin_satisfaction(a8, T8, P8)
R["example"] = dict(
    a=a8.tolist(), T=T8, Ptot=P8, sum_caps=float(caps(a8, np.full(8, T8)).sum()),
    lam=lam, iters=it,
    P={k: np.round(v, 4).tolist() for k, v in
       [("TR-LS", P), ("WF", Pw), ("Capped WF", Pc), ("Max-min", Pm)]},
    r={k: np.round(rates(a8, v), 4).tolist() for k, v in
       [("TR-LS", P), ("WF", Pw), ("Capped WF", Pc), ("Max-min", Pm)]},
    J={k: objective(a8, v, T8) for k, v in
       [("TR-LS", P), ("WF", Pw), ("Capped WF", Pc), ("Max-min", Pm)]},
    served={k: float(np.minimum(rates(a8, v), T8).sum() / (8 * T8)) for k, v in
            [("TR-LS", P), ("WF", Pw), ("Capped WF", Pc), ("Max-min", Pm)]},
)
Th = np.array([5, 4, 3, 3, 2, 2, 1, 1.0])
R["hetero"] = dict(T=Th.tolist(), sum_caps=float(caps(a8, Th).sum()),
                   unused_at_15=float(15 - target_rate_ls(a8, Th, 15.0).sum()),
                   J_at_5=objective(a8, target_rate_ls(a8, Th, 5.0), Th))

# Fig. 1: (a) per-channel rates, (b) water levels P_i + 1/a_i
fig, ax = plt.subplots(1, 2, figsize=(COLW * 2 * 0.98 if False else COLW, 1.75),
                       gridspec_kw=dict(width_ratios=[1, 1], wspace=0.42))
idx = np.arange(8)
w = 0.2
for j, k in enumerate(["WF", "Capped WF", "Max-min", "TR-LS"]):
    Pk = dict(WF=Pw, **{"Capped WF": Pc, "Max-min": Pm, "TR-LS": P})[k]
    ax[0].bar(idx + (j - 1.5) * w, rates(a8, Pk), w * 0.9, color=STYLE[k]["color"],
              label=STYLE[k]["label"].replace(" (proposed)", ""), edgecolor="white",
              linewidth=0.3, hatch={"WF": "////", "Capped WF": "", "Max-min": "....",
                                    "TR-LS": ""}[k])
ax[0].axhline(T8, color="#0b0b0b", lw=0.8, ls="--")
ax[0].text(7.6, T8 + 0.12, "target $T$", ha="right", va="bottom", fontsize=6.5)
ax[0].set_xticks(idx, [f"{int(v)}" for v in a8])
ax[0].set_xlabel("channel gain $a_i$")
ax[0].set_ylabel("rate $r_i$ (b/s/Hz)")
ax[0].set_ylim(0, 6.2)
ax[0].grid(axis="x", visible=False)
ax[0].set_title("(a) per-channel rate", fontsize=8)

inv = 1 / a8
ax[1].bar(idx, inv, 0.62, color="#d9d8d3", edgecolor="white", linewidth=0.3,
          label="floor $1/a_i$")
ax[1].bar(idx, P, 0.62, bottom=inv, color=STYLE["TR-LS"]["color"], edgecolor="white",
          linewidth=0.3, label="power $P_i^\\star$")
h = P + inv
ax[1].plot(idx, h, color="#0b0b0b", lw=0.9, marker="o", ms=2.5,
           label="level $\\propto$ shortfall $\\delta_i$")
mu = (Pw + inv)[Pw > 0][0]
ax[1].axhline(mu, color=STYLE["WF"]["color"], ls="--", lw=1.0,
              label="waterfilling level $\\mu$")
ax[1].set_xticks(idx, [f"{int(v)}" for v in a8])
ax[1].set_xlabel("channel gain $a_i$")
ax[1].set_ylabel("$P_i + 1/a_i$")
ax[1].grid(axis="x", visible=False)
ax[1].set_title("(b) level $P_i+1/a_i$", fontsize=8)
ax[1].legend(loc="upper left", frameon=False, fontsize=6, handlelength=1.4)
h0, l0 = ax[0].get_legend_handles_labels()
fig.legend(h0, l0, loc="lower center", ncol=4, frameon=False, fontsize=6,
           bbox_to_anchor=(0.5, 1.0), handlelength=1.2, columnspacing=0.8)
fig.savefig(os.path.join(FIG, "fig_example.pdf"))
plt.close(fig)

# ============================================================ E2: budget sweep
Pts = np.linspace(0.5, 25, 50)
sweep = {k: dict(J=[], used=[], served=[]) for k in ORDER}
for Pt in Pts:
    for k in ORDER:
        Pk = allocate(k, a8, np.full(8, T8), Pt)
        sweep[k]["J"].append(objective(a8, Pk, T8))
        sweep[k]["used"].append(float(Pk.sum() / Pt))
        sweep[k]["served"].append(float(np.minimum(rates(a8, Pk), T8).sum() / 24))
R["sweep"] = dict(Ptot=Pts.tolist(), **sweep)

fig, ax = plt.subplots(1, 2, figsize=(COLW, 1.6), gridspec_kw=dict(wspace=0.45))
for k in ["TR-LS", "Capped WF", "Max-min", "WF", "Uniform"]:
    s = STYLE[k]
    ax[0].plot(Pts, np.maximum(sweep[k]["J"], 1e-3), color=s["color"], ls=s["ls"],
               marker=s["marker"], markevery=6, label=s["label"])
    ax[1].plot(Pts, 100 * np.array(sweep[k]["used"]), color=s["color"], ls=s["ls"],
               marker=s["marker"], markevery=6)
for x in ax:
    x.axvline(R["example"]["sum_caps"], color="#52514e", lw=0.6, ls=":")
    x.set_xlabel("power budget $P_{\\mathrm{tot}}$")
ax[0].set_yscale("log"); ax[0].set_ylim(1e-3, 300)
ax[0].set_ylabel("$J=\\sum_i (r_i-T)^2$")
ax[0].text(R["example"]["sum_caps"] + 0.4, 60, "$\\sum_i\\bar P_i$", fontsize=6.5)
ax[1].set_ylabel("power used (%)"); ax[1].set_ylim(0, 108)
ax[0].set_title("(a) squared deviation", fontsize=8)
ax[1].set_title("(b) budget spent", fontsize=8)
h0, l0 = ax[0].get_legend_handles_labels()
fig.legend(h0, l0, loc="lower center", ncol=3, frameon=False, fontsize=6,
           bbox_to_anchor=(0.5, 1.0), handlelength=1.8, columnspacing=0.8)
fig.savefig(os.path.join(FIG, "fig_sweep.pdf"))
plt.close(fig)

# ============================================================ E3: OFDM fading
N_SC, T_SC = 64, 2.0
SNRS_DB = [0, 5, 10, 15, 20]
REAL = 60 if QUICK else 500
METHODS_F = ["TR-LS", "Capped WF", "Max-min", "WF", "PF", "Uniform"]


def one_real(args):
    snr_db, seed = args
    rng = np.random.default_rng(seed)
    a = 10 ** (snr_db / 10) * rng.exponential(1.0, N_SC)   # P_tot = N, so abar = mean SNR
    T = np.full(N_SC, T_SC)
    out = {}
    for k in METHODS_F:
        Pk = allocate(k, a, T, float(N_SC))
        r = rates(a, Pk)
        sat = np.minimum(r, T) / T
        out[k] = dict(msd=float(np.mean((r - T) ** 2)), short=float(np.mean(np.maximum(T - r, 0) ** 2)),
                      served=float(sat.mean()), p10=float(np.percentile(sat, 10)),
                      p5=float(np.percentile(sat, 5)),
                      jain=float(sat.sum() ** 2 / (len(sat) * np.sum(sat ** 2) + 1e-300)),
                      starved=float(np.mean(Pk < 1e-12)), used=float(Pk.sum() / N_SC),
                      sat=sat)
    caseA = bool(caps(a, T).sum() <= N_SC)
    return snr_db, out, caseA


jobs = [(s, 1000 * i + s) for s in SNRS_DB for i in range(REAL)]
t0 = time.time()
with Pool(2) as pool:
    res = pool.map(one_real, jobs, chunksize=20)
print(f"fading done in {time.time() - t0:.1f}s")
fad = {s: {k: {} for k in METHODS_F} for s in SNRS_DB}
sat_pool = {s: {k: [] for k in METHODS_F} for s in SNRS_DB}
caseA = {s: 0 for s in SNRS_DB}
for s, out, ca in res:
    caseA[s] += ca
    for k in METHODS_F:
        for m, v in out[k].items():
            if m == "sat":
                sat_pool[s][k].append(v)
            else:
                fad[s][k].setdefault(m, []).append(v)
fad_mean = {s: {k: {m: float(np.mean(v)) for m, v in fad[s][k].items()} for k in METHODS_F}
            for s in SNRS_DB}
R["fading"] = dict(N=N_SC, T=T_SC, realizations=REAL, snr_db=SNRS_DB, mean=fad_mean,
                   caseA_frac={s: caseA[s] / REAL for s in SNRS_DB})

# Fig. 3: (a) CDF of per-channel satisfaction at 10 dB, (b) served vs SNR, (c) p10 vs SNR
fig = plt.figure(figsize=(COLW, 2.9))
gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1], hspace=0.62, wspace=0.45)
axc = fig.add_subplot(gs[0, :])
s0 = 10
for k in ["TR-LS", "Capped WF", "Max-min", "WF", "PF", "Uniform"]:
    v = np.sort(np.concatenate(sat_pool[s0][k]))
    y = np.arange(1, len(v) + 1) / len(v)
    st = STYLE[k]
    axc.plot(v, y, color=st["color"], ls=st["ls"], label=st["label"], lw=1.2)
axc.set_xlabel("per-channel satisfaction $\\min(r_i,T)/T$")
axc.set_ylabel("CDF")
axc.set_xlim(-0.02, 1.02); axc.set_ylim(0, 1.0)
axc.set_title(f"(a) distribution at mean SNR {s0} dB", fontsize=8)
axc.legend(loc="upper left", frameon=False, fontsize=6, ncol=2, handlelength=1.8)
ax1 = fig.add_subplot(gs[1, 0]); ax2 = fig.add_subplot(gs[1, 1])
for k in ["TR-LS", "Capped WF", "Max-min", "WF", "Uniform"]:
    st = STYLE[k]
    ax1.plot(SNRS_DB, [100 * fad_mean[s][k]["served"] for s in SNRS_DB], color=st["color"],
             ls=st["ls"], marker=st["marker"])
for k in ["TR-LS", "Capped WF", "WF"]:   # max-min, PF and uniform never starve
    st = STYLE[k]
    ax2.plot(SNRS_DB, [100 * fad_mean[s][k]["starved"] for s in SNRS_DB],
             color=st["color"], ls=st["ls"], marker=st["marker"])
ax1.set_ylabel("mean satisfaction (%)"); ax2.set_ylabel("zero-power channels (%)")
ax2.set_yscale("symlog", linthresh=0.1, linscale=0.5); ax2.set_ylim(0, 60)
ax2.set_yticks([0, 0.1, 1, 10]); ax2.set_yticklabels(["0", "0.1", "1", "10"])
for x in (ax1, ax2):
    x.set_xlabel("mean SNR (dB)"); x.set_xticks(SNRS_DB)
ax1.set_title("(b) demand served", fontsize=8)
ax2.set_title("(c) starved channels", fontsize=8)
fig.savefig(os.path.join(FIG, "fig_fading.pdf"))
plt.close(fig)

# Runtime results: see runtime_bench.py (results_runtime.json).

with open(os.path.join(HERE, "results.json"), "w") as f:
    json.dump(R, f, indent=1, default=float)
print(json.dumps({k: R[k] for k in ["example", "hetero"]}, indent=1, default=float))
for s in SNRS_DB:
    print(s, "caseA", R["fading"]["caseA_frac"][s],
          {k: {m: round(v, 3) for m, v in fad_mean[s][k].items()} for k in METHODS_F})
