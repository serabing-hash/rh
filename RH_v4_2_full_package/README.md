RH v4.2 — Full Executable Package (NB/BD Program)
=================================================

This repository contains an executable reference implementation to reproduce
the NB/BD experiments reported in the paper, including Gram matrix builds,
projection errors E_N^2, and log–log convergence plots.

Quick Start
-----------
1) (Optional) Create a virtualenv with Python 3.11+
   $ python -m venv .venv && source .venv/bin/activate

2) Install dependencies
   $ pip install -r requirements.txt

3) Run a sweep up to N=2000 (fast demo)
   $ python nbdb_main.py --mode sweep --N_list 50,100,200,500,1000,2000 --grid medium

4) Generate a single configuration (N=1000, ridge=1e-10)
   $ python nbdb_main.py --mode single --N 1000 --grid medium

Expected Results
----------------
- The regression slope on the log–log plot E_N^2 vs log N is near -2.
- The product E_N^2 * (log N)^2 is roughly constant (approx 1/pi^2) up to small residuals.

Notes
-----
- This is a clean, readable template emphasizing reproducibility over speed.
- Integrals are computed via composite trapezoidal rules on mixed grids.
- Use --grid large for higher accuracy (slower).

Files
-----
- nbdb_main.py          : CLI entry, orchestration
- nbdb_sweep.py         : sweep driver, plotting, CSV output
- nbdb_gram.py          : Gram/b vector construction; phi_theta(x) implementation
- utils/kernel_tools.py : helpers for grids and regression
- data/                 : output directory (created automatically)

License: MIT
