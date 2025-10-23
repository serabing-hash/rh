
# NB/BD Constructive and Numerical (Exploratory)

This repo provides exploratory code and data to visualize the theoretical upper bound
for the least-squares approximation error in the Nyman-Baez-Duarte (NB/BD) program:

E_N^2 <= C / (log N)^2.

Scope: This shows the trend of the theoretical upper bound. It is not a full constructive proof of RH.
Implementing exact Gram-matrix entries and solving the least-squares system for large N remains future work.

## What is included
- src/plot_error_bound.py  - regenerates CSV + plot for the theoretical bound
- data/error_bound_logspace.csv  - table of (N, bound) up to N=10^4
- data/error_bound_table.md      - a markdown table suitable for README/Notion
- figs/error_decay_bound.png     - figure used in the paper

## Quick start
```bash
python3 src/plot_error_bound.py
```

## Limitations
- Exploratory: Does not compute the actual least-squares error from NB/BD bases; it plots the theoretical upper bound.
- Scaling: Current data goes to N=10^4. Extending to N=10^6+ requires optimized quadrature for Gram entries, sparse solvers, and possibly GPUs.
- References: Classical NB/BD and zeta literature is cited in the paper. A separate RELATED_WORK.md (planned) will track 2023-2025 updates (random-matrix statistics, de Bruijn-Newman, numerical zero statistics, etc.).

## Roadmap
1) Implement closed-form / efficient quadrature for Gram matrix A and vector b under logarithmic lattices.
2) Solve normal equations using stable methods (QR/LSQR) and compare actual E_N^2 vs the bound.
3) Push to N = 10^6+ with batched quadrature + mixed precision.
4) Add RELATED_WORK.md with recent references (2023-2025).
