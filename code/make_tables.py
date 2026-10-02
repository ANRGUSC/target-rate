"""Turn results.json / results_p.json into numbers.tex, tab_fading.tex, tab_runtime.tex."""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "paper")
R = json.load(open(os.path.join(HERE, "results.json")))
Rp = json.load(open(os.path.join(HERE, "results_p.json")))
F = R["fading"]["mean"]
LO, MID = "5", "10"

ex = R["example"]
r_ls = np.array(ex["r"]["TR-LS"]); r_wf = np.array(ex["r"]["WF"])
sweep = R["sweep"]
i25 = int(np.argmin(np.abs(np.array(sweep["Ptot"]) - 25)))
rt = [r for r in json.load(open(os.path.join(HERE, "results_runtime.json"))) if r["N"] <= 1024]
row1024 = [r for r in rt if r["N"] == 1024][0]


def pct(x, d=1):
    return f"{100 * x:.{d}f}"


macros = {
    "SNRlo": LO, "SNRmid": MID,
    "ServedLoLS": pct(F[LO]["TR-LS"]["served"]), "ServedLoCWF": pct(F[LO]["Capped WF"]["served"]),
    "StarvedLoLS": pct(F[LO]["TR-LS"]["starved"]), "StarvedLoCWF": pct(F[LO]["Capped WF"]["starved"]),
    "SpeedupClar": f"{row1024['clar'] / row1024['newt']:.0f}",
    "SpeedupClarMin": f"{min(r['clar'] / r['newt'] for r in rt):.0f}",
    "NewtItMin": str(min(r["it_newt_range"][0] for r in rt)), "NewtItMax": str(max(r["it_newt_range"][1] for r in rt)),
    "BisItMin": str(min(r["it_bis_range"][0] for r in rt)), "BisItMax": str(max(r["it_bis_range"][1] for r in rt)),
    "BisSixItMin": str(min(r["it_bis6_range"][0] for r in rt)), "BisSixItMax": str(max(r["it_bis6_range"][1] for r in rt)),
    "SumCaps": f"{ex['sum_caps']:.2f}",
    "WFOvershoot": f"{(r_wf - 3).max():.2f}", "WFWeakest": f"{r_wf.min():.2f}",
    "LSShortMin": f"{(3 - r_ls).min():.2f}", "LSShortMax": f"{(3 - r_ls).max():.2f}",
    "UsedAtTwentyFive": f"{100 * sweep['TR-LS']['used'][i25]:.0f}",
    "Realizations": str(R["fading"]["realizations"]),
}
with open(os.path.join(OUT, "numbers.tex"), "w") as f:
    for k, v in macros.items():
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")

# ---- fading table (target-aware methods first; best in bold, second best underlined)
aware = [("TR-LS", "\\textit{Target-rate}"), ("Capped WF", "Capped WF"), ("Max-min", "Max-min")]
agnostic = [("PF", "Prop.\\ fair"), ("WF", "Waterfilling"), ("Uniform", "Uniform")]
names = aware + agnostic
cols = [("served", "Mean"), ("p5", "P5"), ("starved", "Starved"), ("msd", "Dev.")]
LOWER = {"starved", "msd"}
def shown(c, v):
    if c == "msd":
        return round(v, 2) if v >= 0.1 else round(v, 3)
    return round(100 * v, 2) if c == "starved" else round(100 * v, 1)
lines = [r"\footnotesize\begin{tabular}{@{}l" + "rrrr" * 2 + "@{}}", r"\toprule",
         r" & \multicolumn{4}{c}{$\bar\gamma=%s$ dB} & \multicolumn{4}{c}{$\bar\gamma=%s$ dB}\\" % (LO, MID),
         r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
         "Method & " + " & ".join([c[1] for c in cols] * 2) + r"\\", r"\midrule"]
mark = {}
for s_ in (LO, MID):
    for c, _ in cols:
        vals = sorted({shown(c, F[s_][k][c]) for k, _ in names}, reverse=c not in LOWER)
        best = vals[0]
        n_best = sum(shown(c, F[s_][k][c]) == best for k, _ in names)
        second = vals[1] if (n_best == 1 and len(vals) > 1) else None   # no runner-up when the best is tied
        mark[(s_, c)] = (best, second)
def row(k, lab):
    cells = []
    for s_ in (LO, MID):
        for c, _ in cols:
            v = shown(c, F[s_][k][c])
            txt = (f"{v:.3f}" if (c == "msd" and v < 0.1) else f"{v:.2f}") if c in ("msd", "starved") else f"{v:.1f}"
            best, second = mark[(s_, c)]
            if v == best:
                txt = r"\textbf{" + txt + "}"
            elif second is not None and v == second:
                txt = r"\underline{" + txt + "}"
            cells.append(txt)
    return lab + " & " + " & ".join(cells) + r"\\"
lines += [row(k, lab) for k, lab in aware] + [r"\midrule"] + [row(k, lab) for k, lab in agnostic]
lines += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(OUT, "tab_fading.tex"), "w").write("\n".join(lines) + "\n")

# ---- runtime table
def ms(x):
    v = 1e3 * x
    return f"{v:,.0f}" if v >= 100 else (f"{v:.1f}" if v >= 10 else (f"{v:.2f}" if v >= 0.1 else f"{v:.3f}"))
lines = [r"\footnotesize\begin{tabular}{@{}rrrrrrr@{}}", r"\toprule",
         r" & \multicolumn{3}{c}{Algorithm~\ref{alg:tr}} & \multicolumn{2}{c}{Reference solvers} & \\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}",
         r"$N$ & Newton & Bisect. & NumPy & Clarabel & SLSQP & Speedup\\", r"\midrule"]
for r in rt:
    sl = ms(r["slsqp"]).replace(",", "{,}") if "slsqp" in r else "--"
    lines.append(f"{r['N']:,}".replace(",", "{,}") + f" & {ms(r['newt'])} & {ms(r['bis'])} & {ms(r['np'])} & {ms(r['clar'])} & {sl} & {r['clar'] / r['newt']:.0f}$\\times$\\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(OUT, "tab_runtime.tex"), "w").write("\n".join(lines) + "\n")
print(macros)

