
"""
v19B_numerics.py — Numerical verification for Functional Equation Transfer (v19A → v19B)
- Builds NB/BD kernel K_{mn} = exp(-0.5 * |log(m/n)|) for m,n <= M.
- Applies functional-equation transfer via diagonal weight W(s)=m^{s-1/2} (tested at s = 1/2 + it grid).
- Checks symmetry residuals against s -> 1-s mapping on the critical line.
- Exports: figures, CSV summary, and a JSON log.
Note: Self-contained (no internet). Keep M modest for speed.
"""

import numpy as np
import matplotlib.pyplot as plt
import json, csv, math, os

def K_matrix(M):
    m = np.arange(1, M+1, dtype=float)
    n = np.arange(1, M+1, dtype=float)
    mm = m[:,None]
    nn = n[None,:]
    return np.exp(-0.5 * np.abs(np.log(mm/nn)))

def W_diag(M, s):
    m = np.arange(1, M+1, dtype=float)
    return np.power(m, s - 0.5)

def symmetry_residual(K, t_grid):
    """
    For s = 1/2 + it on the critical line, compare W(s) K W(s)^{-1} vs W(1-s) K W(1-s)^{-1}.
    On the critical line, 1 - s = 1/2 - it, so expect conjugation symmetry.
    Return mean Frobenius residuals over t_grid.
    """
    M = K.shape[0]
    res = []
    for t in t_grid:
        s = 0.5 + 1j*t
        w1 = W_diag(M, s)
        w2 = W_diag(M, 1-s)
        # Similarity transforms
        A = (w1[:,None] * K) / w1[None,:]
        B = (w2[:,None] * K) / w2[None,:]
        # Frobenius norm of difference / baseline
        diff = A - B
        num = np.linalg.norm(diff, 'fro')
        den = np.linalg.norm(K, 'fro')
        res.append(num/den)
    return np.array(res)

def spectrum_stats(K):
    # symmetric proxy (since K is symmetric positive kernel)
    eigvals = np.linalg.eigvalsh(K)
    return dict(min=float(eigvals.min()), max=float(eigvals.max()), 
                mean=float(eigvals.mean()), std=float(eigvals.std()),
                eigvals=eigvals)

def main(out_dir=".", M=220, T=40, t_max=3.0, seed=123):
    np.random.seed(seed)
    K = K_matrix(M)
    # Symmetry residuals across t grid
    t_grid = np.linspace(-t_max, t_max, T)
    res = symmetry_residual(K, t_grid)
    stats = spectrum_stats(K)
    os.makedirs(out_dir, exist_ok=True)

    # Figure 1: residual vs t
    plt.figure()
    plt.plot(t_grid, res, marker='o')
    plt.xlabel('t (s = 1/2 + it)')
    plt.ylabel('relative Frobenius residual')
    plt.title('Functional-Equation Transfer Residuals (v19B)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    f1 = os.path.join(out_dir, "figure_residuals.png")
    plt.savefig(f1, dpi=200)
    plt.close()

    # Figure 2: eigenvalue histogram
    plt.figure()
    plt.hist(stats["eigvals"], bins=30)
    plt.xlabel('eigenvalue')
    plt.ylabel('count')
    plt.title('Spectrum of K (NB/BD kernel)')
    plt.tight_layout()
    f2 = os.path.join(out_dir, "figure_spectrum.png")
    plt.savefig(f2, dpi=200)
    plt.close()

    # Figure 3: heatmap of K (coarse)
    plt.figure()
    plt.imshow(K, origin='lower', aspect='auto')
    plt.colorbar(label='K_{mn}')
    plt.title('Kernel Heatmap K_{mn}')
    plt.xlabel('n')
    plt.ylabel('m')
    plt.tight_layout()
    f3 = os.path.join(out_dir, "figure_heatmap.png")
    plt.savefig(f3, dpi=200)
    plt.close()

    # CSV summary
    import csv
    csv_path = os.path.join(out_dir, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["M","T","t_min","t_max","res_mean","res_std","eig_min","eig_max","eig_mean","eig_std"])
        w.writerow([M, T, -t_max, t_max, float(res.mean()), float(res.std()),
                    stats["min"], stats["max"], stats["mean"], stats["std"]])

    # JSON log
    stats_safe = dict(min=stats['min'], max=stats['max'], mean=stats['mean'], std=stats['std'], eigvals=stats['eigvals'].tolist())
    log = dict(M=M, T=T, t_max=t_max, residuals=res.tolist(), spectrum=stats_safe)
    with open(os.path.join(out_dir, "log.json"), "w") as f:
        json.dump(log, f, indent=2)

if __name__ == "__main__":
    main(out_dir="v19B_out", M=220, T=41, t_max=3.0)
