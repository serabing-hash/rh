python - <<'PY'
import numpy as np
import pandas as pd
from pathlib import Path
from src.nbbd_quadrature import build_A_thetas, build_b_thetas
import numpy.linalg as la
import matplotlib.pyplot as plt

def theta_log_lattice(N, tmin=1e-3, tmax=1.0):
    t0 = np.log(tmin); L = np.log(tmax) - t0
    if N<=1: return np.array([tmax])
    t = t0 + (np.arange(N)*(L/(N-1)))
    return np.exp(t)

def run_once(N, lam=1e-6, quad="mid", n_gauss=4):
    th = theta_log_lattice(N)
    A = build_A_thetas(th, quad=quad, n_gauss=n_gauss)
    b = build_b_thetas(th, quad=quad, n_gauss=n_gauss)
    c = la.solve(A + lam*np.eye(N), b)
    return 1.0 - float(b@c)

Ns = [10,15,20,25,30]
rows = []
for N in Ns:
    E2 = run_once(N, lam=1e-6, quad="mid", n_gauss=4)
    rows.append({"N":N, "E2_actual_ridge":E2, "inv_log2":1/(np.log(N)**2)})
df = pd.DataFrame(rows)
out = Path("experiments"); out.mkdir(exist_ok=True)
df.to_csv(out/"sweep_results_small_ridge.csv", index=False)

plt.figure(figsize=(6,4))
plt.loglog(df["N"], df["E2_actual_ridge"], "o-", linewidth=2, label="Actual $E_N^2$ (ridge)")
plt.loglog(df["N"], df["inv_log2"], "s--", linewidth=2, label="$1/\\log^2 N$")
plt.xlabel("N"); plt.ylabel("$E_N^2$")
plt.title("Small Sweep: $E_N^2$ vs $1/\\log^2 N$")
plt.grid(True, which='both', ls='--', alpha=0.7)
plt.legend(); plt.tight_layout()
plt.savefig(out/"sweep_E2_vs_logN_small_ridge.png", dpi=300)
PY