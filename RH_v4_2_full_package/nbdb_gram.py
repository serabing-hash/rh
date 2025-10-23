import numpy as np
from utils.kernel_tools import make_mixed_grid, trapz_integral

def frac(x):
    """Fractional part robust for large inputs."""
    return x - np.floor(x)

def phi_theta(theta, x):
    """phi_theta(x) = {theta/x} - theta*{1/x} for vector x."""
    x = np.asarray(x, dtype=float)
    y1 = theta / x
    y2 = 1.0 / x
    return frac(y1) - theta * frac(y2)

def build_gram_and_b(N, L=None, theta_min=None, x=None, grid="medium"):
    """Construct theta-grid (logarithmic) and compute Gram matrix A and vector b."""
    if theta_min is None and L is None:
        theta_min = 1.0 / max(N, 2)
    theta_max = 1.0
    if L is None:
        L = np.log(theta_max / theta_min)
    if theta_min is None:
        theta_min = np.exp(-L)

    theta = np.exp(np.linspace(np.log(theta_min), np.log(theta_max), N))

    if x is None:
        if grid == "small":
            x = make_mixed_grid(n_small=1000, n_large=2000)
        elif grid == "large":
            x = make_mixed_grid(n_small=8000, n_large=12000)
        else:
            x = make_mixed_grid(n_small=3000, n_large=5000)

    Phi = np.zeros((N, x.size), dtype=float)
    for k, th in enumerate(theta):
        Phi[k, :] = phi_theta(th, x)

    A = np.zeros((N, N), dtype=float)
    for i in range(N):
        for j in range(i, N):
            Aij = trapz_integral(Phi[i, :] * Phi[j, :], x)
            A[i, j] = Aij
            A[j, i] = Aij

    b = np.zeros(N, dtype=float)
    for i in range(N):
        b[i] = trapz_integral(Phi[i, :], x)

    return theta, A, b, x

def projection_error(A, b):
    """E_N^2 = 1 - b^T A^{-1} b (since ||chi||^2=1)."""
    try:
        coef = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        lam = 1e-12
        coef = np.linalg.solve(A + lam*np.eye(A.shape[0]), b)
    val = float(b.T @ coef)
    return max(0.0, 1.0 - val)
