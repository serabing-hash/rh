import numpy as np

def frac_theta_over_x(theta, x):
    y = theta / x
    return y - np.floor(y)

def frac_one_over_x(x):
    y = 1.0 / x
    return y - np.floor(y)

def logspace_bins(num_bins=256, x_min=1e-6, x_max=1.0):
    lx0 = np.log(x_min)
    lx1 = np.log(x_max)
    edges = np.exp(np.linspace(lx0, lx1, num_bins + 1))
    edges[0]  = max(edges[0], 1e-12)
    edges[-1] = 1.0
    mids  = 0.5 * (edges[:-1] + edges[1:])
    widths = (edges[1:] - edges[:-1])
    return edges, mids, widths

def theta_log_lattice(N, theta_min=1e-3, theta_max=1.0):
    t0 = np.log(theta_min)
    L  = np.log(theta_max) - t0
    if N <= 1:
        return np.array([theta_max], dtype=float)
    t  = t0 + (np.arange(N, dtype=float) * (L / (N - 1)))
    return np.exp(t)

def build_A_b_coarse(thetas, num_bins=256, x_min=1e-6):
    _, mids, widths = logspace_bins(num_bins=num_bins, x_min=x_min, x_max=1.0)
    f1 = frac_one_over_x(mids)
    m = len(thetas)
    A = np.zeros((m, m), dtype=float)
    b = np.zeros(m, dtype=float)
    V = np.empty((len(mids), m), dtype=float)
    for i, th in enumerate(thetas):
        V[:, i] = frac_theta_over_x(th, mids) - th * f1
        b[i] = np.sum(V[:, i] * widths)
    for i in range(m):
        vi = V[:, i]
        Ai = A[i]
        for j in range(i, m):
            val = np.sum(vi * V[:, j] * widths)
            Ai[j] = val
            if j != i:
                A[j, i] = val
    return A, b

def solve_system(A, b, lam=1e-6, use_lsqr=False, lsqr_tol=1e-8, lsqr_iter=2000):
    m = A.shape[0]
    if lam and lam > 0:
        A = A + lam * np.eye(m, dtype=float)
    if use_lsqr:
        try:
            import scipy.sparse as sp
            import scipy.sparse.linalg as spla
            Asp = sp.csr_matrix(A)
            sol = spla.lsqr(Asp, b, atol=lsqr_tol, btol=lsqr_tol, iter_lim=lsqr_iter)
            return sol[0]
        except Exception:
            pass
    return np.linalg.solve(A, b)

def compute_E2(N, theta_min=1e-3, theta_max=1.0, num_bins=256, x_min=1e-6, lam=1e-6, use_lsqr=False):
    thetas = theta_log_lattice(N, theta_min, theta_max)
    A, b = build_A_b_coarse(thetas, num_bins=num_bins, x_min=x_min)
    c = solve_system(A, b, lam=lam, use_lsqr=use_lsqr)
    return 1.0 - float(b @ c)
