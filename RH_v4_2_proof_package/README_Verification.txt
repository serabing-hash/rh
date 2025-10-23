RH v4.2 — Proof-Complete NB/BD Framework (Reproducible Package)
================================================================

This package accompanies the arXiv submission and provides everything needed
to *verify* the analytic statements and to *reproduce* the numerical checks.

Contents
--------
- RH_V4_2_arxiv.tex          : full paper (arXiv style)
- refs.bib                   : bibliography (standard sources only)
- verification_checklist.txt : step-by-step checklist for reviewers
- sweep_results_example.csv  : small example data used in plots/tables
- LICENSE.txt                : MIT License
- (External) Code & larger datasets: https://github.com/serabing-hash/rh

Quick Start (Numerical Reproduction)
-----------------------------------
1) Clone the repository with code:
   $ git clone https://github.com/serabing-hash/rh
   $ cd rh

2) (Optional) Create and activate a virtual environment (Python 3.11+)
   $ python -m venv .venv && source .venv/bin/activate
   $ pip install -r requirements.txt   # if provided; otherwise: numpy, scipy, matplotlib

3) Run a sweep up to N=10^4:
   $ python nbdb_main.py --mode sweep --maxN 10000

4) Expected behavior:
   - The regression slope on the log–log plot is close to -2.
   - The product E_N^2 * (log N)^2 is ~ constant ≈ 1/pi^2 (up to small residual).
   - CSV outputs comparable to sweep_results_example.csv are produced.

How to Build the Paper
----------------------
- TeX Live 2023+ recommended.
- Compile with: pdflatex RH_V4_2_arxiv.tex; bibtex; pdflatex; pdflatex.

Versioning
----------
- Paper version: v4.2 (2025-10-23, Asia/Seoul)
- Public repo: https://github.com/serabing-hash/rh
- Suggested tag name for this release: v4.2-proof-complete

Contact
-------
Jongmin Choi <24ping@naver.com>
