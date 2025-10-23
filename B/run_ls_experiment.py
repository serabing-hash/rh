
# run_ls_experiment.py
# Build theta lattice, compute A and b, solve least squares, and measure E_N^2.

import numpy as np
import argparse
from pathlib import Path

try:
    import scipy.linalg as la
except Exception:
    la = None

from nbbd_quadrature import build_A_thetas, build_b_thetas

def theta_log_lattice(N, theta_min=1e-3, theta_max=1.0):
    t0 = np.log(theta_min)
    L  = np.log(theta_max) - t0
    if N <= 1:
        return np.array([theta_max])
    t  = t0 + (np.arange(N) * (L / (N-1)))
    return np.exp(t)

def run(N=50, theta_min=1e-3, theta_max=1.0, quad=\"gauss\", n_gauss=4, outdir=\"experiments\"):
    thetas = theta_log_lattice(N, theta_min, theta_max)
    A = build_A_thetas(thetas, quad=quad, n_gauss=n_gauss)
    b = build_b_thetas(thetas, quad=quad, n_gauss=n_gauss)
    # Solve A c = b
    if la is not None:
        c = la.solve(A, b, assume_a=\"sym\")
    else:
        c = np.linalg.solve(A, b)
    # E_N^2 = ||chi||^2 - b^T c   (since c = A^{-1} b)
    chi_norm2 = 1.0  # ||chi||^2 over (0,1]
    E2 = chi_norm2 - float(b.T @ c)
    # Save
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f\"A_N{N}.npy\", A)
    np.save(out / f\"b_N{N}.npy\", b)
    np.save(out / f\"c_N{N}.npy\", c)
    with open(out / f\"summary_N{N}.txt\", \"w\") as f:
        f.write(f\"N={N}\\n\")
        f.write(f\"theta_min={theta_min}, theta_max={theta_max}\\n\")
        f.write(f\"quad={quad}, n_gauss={n_gauss}\\n\")
        f.write(f\"E2={E2:.6e}\\n\")
    return E2

if __name__ == \"__main__\":
    ap = argparse.ArgumentParser()
    ap.add_argument(\"--N\", type=int, default=50)
    ap.add_argument(\"--theta_min\", type=float, default=1e-3)
    ap.add_argument(\"--theta_max\", type=float, default=1.0)
    ap.add_argument(\"--quad\", type=str, default=\"gauss\", choices=[\"gauss\",\"mid\"])
    ap.add_argument(\"--n_gauss\", type=int, default=4)
    ap.add_argument(\"--outdir\", type=str, default=\"experiments\")
    args = ap.parse_args()
    E2 = run(**vars(args))
    print(f\"E_N^2 = {E2:.6e}\")
