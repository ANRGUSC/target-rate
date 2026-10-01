# Target-Rate Least-Squares Power Allocation — ICC version code

- `tralloc.py` — closed-form allocation (Algorithm 1), l_p family, weighted variant, baselines, CVXPY/SLSQP references, log-domain (overflow-safe) solver.
- `test_tralloc.py`, `test_lp.py` — correctness checks vs CVXPY/Clarabel.
- `experiments.py` — Figs. 1–3 and Table I. `experiments_p.py` — Fig. 4. `experiments_hetero.py` — Table II.
- `runtime_bench.py` — Table III (Numba-compiled bisection and safeguarded Newton; Clarabel solve time with the CVXPY problem compiled once; SLSQP).
- `make_tables.py` — writes `numbers.tex` and `tab_*.tex` into ../paper from the `results*.json` files.

Requires numpy, scipy, matplotlib, cvxpy (+clarabel), numba.
