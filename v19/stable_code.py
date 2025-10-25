
# stable_code.py — v19D.2 (Frobenius-normalized) — Appendix artifact
# NOTE: Replace/merge this stub with your finalized implementation if you already have one.
import numpy as np
import matplotlib.pyplot as plt

def nb_bd_kernel(N):
    m = np.arange(1, N+1).reshape(-1, 1)
    n = np.arange(1, N+1).reshape(1, -1)
    K = np.exp(-0.5*np.abs(np.log(m/n)))
    return K

def frobenius_normalize(K):
    fro = np.linalg.norm(K, 'fro')
    return K / fro

def three_panel_plot(N=600, out_png='figure.png'):
    K = nb_bd_kernel(N)
    K = frobenius_normalize(K)
    # Symmetrized Gram (for numerical stability)
    G = (K + K.T) / 2.0
    w, _ = np.linalg.eigh(G)
    # Sorted weights
    w_sorted = np.sort(w)

    # Naive KDE for eigenvalues (for appendix visualization only)
    xs = np.linspace(w_sorted.min(), w_sorted.max(), 400)
    bw = (w_sorted.max() - w_sorted.min()) / 50.0 if w_sorted.max() > w_sorted.min() else 1e-6
    kde = np.zeros_like(xs)
    for wi in w_sorted:
        kde += np.exp(-0.5*((xs-wi)/bw)**2)
    kde /= (np.sqrt(2*np.pi)*bw*len(w_sorted))

    # Gap distribution
    gaps = np.diff(w_sorted)
    # Avoid zeros for log plots in Overleaf
    gaps = gaps[gaps > 1e-15]

    # Plot
    plt.figure(figsize=(12, 4))
    # (1) w_i sorted
    plt.subplot(1, 3, 1)
    plt.plot(w_sorted)
    plt.title("Sorted spectral weights $w_i$")
    plt.xlabel("i")
    plt.ylabel("$w_i$")

    # (2) eigenvalue KDE
    plt.subplot(1, 3, 2)
    plt.plot(xs, kde)
    plt.title("Eigenvalue KDE")
    plt.xlabel("$\lambda$")
    plt.ylabel("density")

    # (3) gap distribution
    plt.subplot(1, 3, 3)
    plt.plot(gaps)
    plt.title("Gap distribution")
    plt.xlabel("i")
    plt.ylabel("$\Delta w_i$")

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    return out_png

if __name__ == "__main__":
    print("[v19D] Generating three-panel plot (N=600)...")
    png = three_panel_plot(N=600, out_png="figure.png")
    print(f"[v19D] Saved: {png}")
