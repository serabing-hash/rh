
# v20_runner.py — execute E1–E3 sweeps and save results/figures
import json, os
import numpy as np
import matplotlib.pyplot as plt
from v20_kernel import kernel_K1, kernel_K2_realblock, kernel_K3
from v20_metrics import basic_metrics

def panel_plot(w, xs, dens, title, out_png):
    w_sorted = np.sort(w)
    gaps = np.diff(w_sorted)
    gaps = gaps[gaps > 1e-15]

    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1); plt.plot(w_sorted); plt.title("w_i sorted"); plt.xlabel("rank"); plt.ylabel("weight")
    plt.subplot(1,3,2); plt.plot(xs, dens); plt.title("Eigenvalue KDE"); plt.xlabel("eigenvalue"); plt.ylabel("density")
    plt.subplot(1,3,3); 
    if len(gaps)>0: plt.plot(gaps)
    plt.title("Gap distribution"); plt.xlabel("i"); plt.ylabel("Δw_i")
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()

def sweep_K1(out_dir, N=600, taus=None, gamma=1.0):
    if taus is None:
        taus = np.linspace(0, 1, 6)
    results = []
    for tau in taus:
        G = kernel_K1(N, float(tau), gamma=gamma)
        m, (w, xs, dens) = basic_metrics(G)
        panel_plot(w, xs, dens, f"K1 tau={tau:.2f}, N={N}", os.path.join(out_dir, f"K1_tau{tau:.2f}_N{N}.png"))
        m.update(dict(kernel="K1", tau=float(tau), N=int(N), gamma=float(gamma)))
        results.append(m)
    return results

def sweep_K2(out_dir, N=300, taus=None):  # doubled dimension → default N smaller
    if taus is None:
        taus = np.linspace(0, 2*np.pi, 7)
    results = []
    for tau in taus:
        B = kernel_K2_realblock(N, float(tau))
        m, (w, xs, dens) = basic_metrics(B)
        panel_plot(w, xs, dens, f"K2 (real block) tau={tau:.2f}, N={N}", os.path.join(out_dir, f"K2_tau{tau:.2f}_N{N}.png"))
        m.update(dict(kernel="K2", tau=float(tau), N=int(N)))
        results.append(m)
    return results

def sweep_K3(out_dir, N=600, taus=None, alpha=1.5):
    if taus is None:
        taus = [0.0, 0.5, 1.0]
    results = []
    for tau in taus:
        G = kernel_K3(N, float(tau), alpha=float(alpha))
        m, (w, xs, dens) = basic_metrics(G)
        panel_plot(w, xs, dens, f"K3 tau={tau:.2f}, alpha={alpha}, N={N}", os.path.join(out_dir, f"K3_tau{tau:.2f}_a{alpha}_N{N}.png"))
        m.update(dict(kernel="K3", tau=float(tau), N=int(N), alpha=float(alpha)))
        results.append(m)
    return results

def run_all(base_out=".", N=600):
    os.makedirs(base_out, exist_ok=True)
    all_results = []
    all_results += sweep_K1(base_out, N=N, taus=np.linspace(0,1,6), gamma=1.0)
    all_results += sweep_K2(base_out, N=max(200, N//2), taus=np.linspace(0, 2*np.pi, 7))
    all_results += sweep_K3(base_out, N=N, taus=[0.0,0.5,1.0], alpha=1.5)
    with open(os.path.join(base_out, "results.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    return all_results

if __name__ == "__main__":
    out = "results"
    os.makedirs(out, exist_ok=True)
    run_all(out, N=600)
    print("[v20] Sweep completed. See ./results for figures and results.json")
