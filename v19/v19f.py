import numpy as np
import scipy.linalg as la
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

def build_K_norm(N: int):
    i = np.arange(1, N+1)
    M, Nn = i[:, None], i[None, :]
    K = np.sqrt(np.minimum(M, Nn) / np.maximum(M, Nn))
    # Frobenius norm normalization
    K /= np.linalg.norm(K, 'fro')
    return K

def spectral_measure_e1(K):
    evals, evecs = la.eigh(K)
    w = evecs[0, :]**2
    # normalize weights to sum = 1
    w /= w.sum()
    return evals, w

def atomness_metrics(evals, w):
    kde = gaussian_kde(evals)
    grid = np.linspace(evals.min(), evals.max(), 512)
    dens = kde(grid)
    med_w, med_d = np.median(w), np.median(dens)
    top_weight_ratio = w.max() / med_w
    peak_ratio = dens.max() / med_d
    gaps = np.diff(np.sort(evals)); gaps = gaps[gaps > 1e-14]
    gap_ratio = gaps.max() / np.median(gaps)
    return top_weight_ratio, peak_ratio, gap_ratio, grid, dens

def run_stable(N=600):
    K = build_K_norm(N)
    evals, w = spectral_measure_e1(K)
    T, P, G, grid, dens = atomness_metrics(evals, w)

    fig, ax = plt.subplots(1,3,figsize=(15,4))
    ax[0].plot(np.sort(w)[::-1],'.-'); ax[0].set_title(f'Top/Median={T:.2f}')
    ax[1].hist(evals,bins=40,density=True,alpha=0.4)
    ax[1].plot(grid,dens,lw=2); ax[1].set_title(f'KDE peak/median={P:.2f}')
    ax[2].hist(np.diff(np.sort(evals)),bins=40,density=True,alpha=0.5)
    ax[2].set_title(f'Gap ratio={G:.2f}')
    plt.tight_layout(); plt.show()
    return T,P,G

if __name__ == "__main__":
    T,P,G = run_stable(600)
    print(f"Stabilized metrics: Top/Median={T:.3f}, KDE peak/median={P:.3f}, Gap ratio={G:.3f}")