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

# ---- fading table
names = [("TR-LS", "\\textit{Target-rate}"), ("Capped WF", "Capped WF"), ("Max-min", "Max-min"),
         ("PF", "Prop.\\ fair"), ("WF", "Waterfilling"), ("Uniform", "Uniform")]
cols = [("served", "Mean"), ("p5", "P5"), ("starved", "Starved"), ("jain", "Jain")]
lines = [r"\footnotesize\begin{tabular}{@{}l" + "rrrr" * 2 + "@{}}", r"\toprule",
         r" & \multicolumn{4}{c}{$\bar\gamma=%s$ dB} & \multicolumn{4}{c}{$\bar\gamma=%s$ dB}\\" % (LO, MID),
         r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
         "Method & " + " & ".join([c[1] for c in cols] * 2) + r"\\",
         r" & \% & \% & \% & & \% & \% & \% & \\", r"\midrule"]
best = {}
for s in (LO, MID):
    for c, _ in cols:
        vals = {k: F[s][k][c] for k, _ in names}
        best[(s, c)] = min(vals.values()) if c == "starved" else max(vals.values())
for k, lab in names:
    cells = []
    for s in (LO, MID):
        for c, _ in cols:
            v = F[s][k][c]
            txt = f"{v:.2f}" if c == "jain" else (pct(v, 2) if c == "starved" else pct(v))
            if abs(v - best[(s, c)]) < 5e-4:
                txt = r"\textbf{" + txt + "}"
            cells.append(txt)
    lines.append(lab + " & " + " & ".join(cells) + r"\\")
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

# ---- mixed-traffic table
H = json.load(open(os.path.join(HERE, "results_hetero.json")))
hn = [("TR-LS", "\\textit{Target-rate}"), ("TR-NLS", "\\textit{Target-rate}, $w_i{=}T_i^{-2}$"),
      ("Capped WF", "Capped WF"), ("Max-min", "Max-min"), ("PF", "Prop.\\ fair"),
      ("WF", "Waterfilling"), ("Uniform", "Uniform")]
hc = [("served", "Sat."), ("bits", "Bits"), ("p5", "P5"), ("starved", "Starved")]
lines = [r"\footnotesize\begin{tabular}{@{}l" + "rrrr" * 2 + "@{}}", r"\toprule",
         r" & \multicolumn{4}{c}{$\bar\gamma=5$ dB} & \multicolumn{4}{c}{$\bar\gamma=10$ dB}\\",
         r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
         "Method & " + " & ".join([c[1] for c in hc] * 2) + r"\\", r"\midrule"]
hb = {}
for s in ("5", "10"):
    for c, _ in hc:
        vals = [H[s][k][c] for k, _ in hn]
        hb[(s, c)] = min(vals) if c == "starved" else max(vals)
for k, lab in hn:
    cells = []
    for s in ("5", "10"):
        for c, _ in hc:
            v = H[s][k][c]
            txt = pct(v, 2) if c == "starved" else pct(v)
            if abs(v - hb[(s, c)]) < 5e-4:
                txt = r"\textbf{" + txt + "}"
            cells.append(txt)
    lines.append(lab + " & " + " & ".join(cells) + r"\\")
lines += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(OUT, "tab_hetero.tex"), "w").write("\n".join(lines) + "\n")
