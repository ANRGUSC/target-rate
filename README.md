# Target-Rate Least-Squares Power Allocation over Parallel Channels

Reference implementation and experiment code for the paper:

> **Target-Rate Least-Squares Power Allocation over Parallel Channels**
> Bhaskar Krishnamachari, Ming Hsieh Department of Electrical and Computer Engineering,
> University of Southern California, Los Angeles, CA, USA.

**Status:** This paper is currently **in submission**. A related, longer paper is available
on arXiv: [arXiv:2603.06893](https://arxiv.org/abs/2603.06893).

The paper PDF (submission version) is included in this repository:
[`Krishnamachari_target_rate_LS_submission.pdf`](Krishnamachari_target_rate_LS_submission.pdf).

## Overview

We study power allocation over `N` parallel Gaussian channels (e.g., OFDM subcarriers or the
zero-forcing streams of a multiuser MIMO downlink) where each channel has a target spectral
efficiency `T_i` set by the demand of its receiver. The goal is to minimize the total squared
deviation between achieved and target rates subject to a sum-power constraint:

```
minimize   J(P) = sum_i ( log2(1 + a_i P_i) - T_i )^2
subject to sum_i P_i <= P_tot,   P_i >= 0,
```

where `a_i` is the gain-to-noise ratio of channel `i`. The optimum never overshoots a target,
the problem is equivalent to a strictly convex program, and it exhibits a sharp two-regime
structure: if the budget suffices, every target is met exactly and surplus power is left
unused; otherwise the solution has a waterfilling-like closed form in which each channel has
its own water level, governed by the Lambert `W` function, with a shortfall proportional to
the channel's rate shortfall. The optimum is found by a one-dimensional dual search with an
explicit bracket.

## Repository layout

```
.
├── README.md
├── LICENSE
├── Krishnamachari_target_rate_LS_submission.pdf   # submission version of the paper
└── code/                                           # Python implementation and experiments
```

## Code

See [`code/README.md`](code/README.md) for details. In brief:

- `tralloc.py` — closed-form allocation (Algorithm 1), the `l_p` family, the weighted variant,
  baselines (waterfilling, target-capped waterfilling, uniform, proportional fairness,
  max-min target satisfaction), CVXPY/SLSQP reference solvers, and a log-domain
  (overflow-safe) solver.
- `test_tralloc.py`, `test_lp.py` — correctness checks against CVXPY/Clarabel.
- `experiments.py` — Figs. 1–3 and Table I. `experiments_p.py` — Fig. 4.
  `experiments_hetero.py` — Table II.
- `runtime_bench.py` — Table III (Numba-compiled bisection and safeguarded Newton; Clarabel
  solve time; SLSQP).
- `make_tables.py` — writes the LaTeX tables from the `results*.json` files.
- `results*.json` — cached experiment outputs.

### Requirements

Python 3 with `numpy`, `scipy`, `matplotlib`, `cvxpy` (with `clarabel`), and `numba`.

```
pip install numpy scipy matplotlib cvxpy clarabel numba
```

### Running

```
cd code
python test_tralloc.py      # verify the closed-form solver against CVXPY
python experiments.py       # reproduce Figs. 1-3 and Table I
```

## Citation

If you use this code, please cite the paper. A BibTeX entry will be added once the paper is
published; in the meantime, please refer to the submission PDF and the related arXiv paper
[arXiv:2603.06893](https://arxiv.org/abs/2603.06893).

## License

This project is released under the BSD 3-Clause License. See [`LICENSE`](LICENSE).
