# Target-Rate Least-Squares Power Allocation — code for the ICC submission

- `tralloc.py` — closed-form allocation (Algorithm 1), l_p family, weighted variant, baselines, CVXPY/SLSQP references, log-domain (overflow-safe) solver.
- `test_tralloc.py`, `test_lp.py` — correctness checks vs CVXPY/Clarabel.
- `experiments.py` — Figs. 1–3 and Table II (Rayleigh-faded channels). `experiments_p.py` — Fig. 4.
- `experiments_hetero.py` — unequal-target study quoted in Remark 2 (`results_hetero.json`).
- `runtime_bench.py` — Table III (Numba-compiled bisection and safeguarded Newton; Clarabel solve time with the CVXPY problem compiled once; SLSQP).
- `make_tables.py` — writes `numbers.tex` and `tab_*.tex` into ../paper from the `results*.json` files.

Requires numpy, scipy, matplotlib, cvxpy (+clarabel), numba.
