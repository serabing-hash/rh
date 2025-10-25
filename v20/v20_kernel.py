
# v20_kernel.py — Continuous Deformation (Hybrid Kernel) — skeleton
import numpy as np

def nb_bd_kernel(N: int) -> np.ndarray:
    m = np.arange(1, N+1).reshape(-1, 1)
    n = np.arange(1, N+1).reshape(1, -1)
    K = np.exp(-0.5*np.abs(np.log(m/n)))
    return K

def log_gaussian_kernel(N: int, gamma: float=1.0) -> np.ndarray:
    m = np.arange(1, N+1).reshape(-1, 1)
    n = np.arange(1, N+1).reshape(1, -1)
    D = (np.log(m) - np.log(n))**2
    return np.exp(-gamma*D)

def frobenius_normalize(K: np.ndarray) -> np.ndarray:
    fro = np.linalg.norm(K, 'fro')
    return K / fro if fro != 0 else K

# (K1) Additive hybrid: convex mix of NB/BD and log-Gaussian
def kernel_K1(N: int, tau: float, gamma: float=1.0) -> np.ndarray:
    K = nb_bd_kernel(N)
    G = log_gaussian_kernel(N, gamma=gamma)
    return frobenius_normalize((1.0 - tau)*K + tau*G)

# (K2) Multiplicative phase twist on NB/BD
# Returns a real symmetric 2N×2N block matrix representing Re/Im parts.
def kernel_K2_realblock(N: int, tau: float) -> np.ndarray:
    m = np.arange(1, N+1).reshape(-1, 1)
    n = np.arange(1, N+1).reshape(1, -1)
    K = np.exp(-0.5*np.abs(np.log(m/n)))
    phase = tau*(np.log(m) - np.log(n))
    C = K * np.cos(phase)
    S = K * np.sin(phase)
    top = np.concatenate([C, -S], axis=1)
    bot = np.concatenate([S,  C], axis=1)
    B = np.concatenate([top, bot], axis=0)
    return frobenius_normalize(B)

# (K3) Local thinning (diagonal emphasis)
def kernel_K3(N: int, tau: float, alpha: float=1.5) -> np.ndarray:
    m = np.arange(1, N+1).reshape(-1, 1)
    n = np.arange(1, N+1).reshape(1, -1)
    K = np.exp(-0.5*np.abs(np.log(m/n)))
    W = np.exp(-tau*np.abs(np.log(m/n))**alpha)
    return frobenius_normalize(K * W)
