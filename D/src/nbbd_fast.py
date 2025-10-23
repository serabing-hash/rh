# src/nbbd_fast.py
import numpy as np

# ---- Optional: uncomment to use numba JIT for speed ----
# from numba import njit

def _leggauss(n):
    xi, wi = np.polynomial.legendre.leggauss(n)
    return xi.astype(np.float64), wi.astype(np.float64)

def _breakpoints_theta(theta, x_min=1e-8, x_max=1.0, cap=100000):
    m_max = min(int(np.ceil(theta / x_min)), cap)
    pts = theta / np.arange(1, m_max + 1, dtype=np.float64)
    pts = pts[(pts > x_min) & (pts <= x_max)]
    return np.sort(pts)

def _breakpoints_one_over_x(x_min=1e-8, x_max=1.0, cap=100000):
    n_max = min(int(np.ceil(1.0 / x_min)), cap)
    pts = 1.0 / np.arange(1, n_max + 1, dtype=np.float64)
    pts = pts[(pts > x_min) & (pts <= x_max)]
    return np.sort(pts)

def _frac_theta_over_x(theta, x):
    y = theta / x
    return y - np.floor(y)

def _frac_one_over_x(x):
    y = 1.0 / x
    return y - np.floor(y)

def _integrate_piecewise(f, edges, xi, wi):
    # Gauss-Legendre on each [a,b], reuse nodes
    total = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a: 
            continue
        xm = 0.5 * ((b - a) * xi + (b + a))
        w  = 0.5 * (b - a) * wi
        total += np.sum(f(xm) * w)
    return total

def build_b_thetas(thetas, x_min=1e-8, quad_n=6):
    pts_common = _breakpoints_one_over_x(x_min=x_min)
    xi, wi = _leggauss(quad_n)
    b = np.zeros(len(thetas), dtype=np.float64)
    edges_common = np.concatenate(([0.0], pts_common, [1.0]))
    # precompute {1/x} contribution once at Gauss nodes per interval? we embed into f for correctness.
    for i, th in enumerate(thetas):
        pts_th = _breakpoints_theta(th, x_min=x_min)
        edges = np.unique(np.concatenate(([0.0], pts_th, pts_common, [1.0])))
        def f(x, th=th):
            return _frac_theta_over_x(th, x) - th * _frac_one_over_x(x)
        b[i] = _integrate_piecewise(f, edges, xi, wi)
    return b

def build_A_thetas(thetas, x_min=1e-8, quad_n=6, symmetric=True):
    m = len(thetas)
    A = np.zeros((m, m), dtype=np.float64)
    pts_common = _breakpoints_one_over_x(x_min=x_min)
    xi, wi = _leggauss(quad_n)
    cache_bp = [ _breakpoints_theta(th, x_min=x_min) for th in thetas ]
    for i in range(m):
        for j in range(i if symmetric else 0, m):
            pts = np.unique(np.concatenate(([0.0], cache_bp[i], cache_bp[j], pts_common, [1.0])))
            def f(x, a=thetas[i], b=thetas[j]):
                vi = _frac_theta_over_x(a, x) - a * _frac_one_over_x(x)
                vj = _frac_theta_over_x(b, x) - b * _frac_one_over_x(x)
                return vi * vj
            Aij = _integrate_piecewise(f, pts, xi, wi)
            A[i, j] = Aij
            if symmetric and j != i:
                A[j, i] = Aij
    return A

def theta_log_lattice(N, theta_min=1e-3, theta_max=1.0):
    t0 = np.log(theta_min)
    L  = np.log(theta_max) - t0
    if N <= 1:
        return np.array([theta_max], dtype=np.float64)
    t  = t0 + (np.arange(N, dtype=np.float64) * (L / (N - 1)))
    return np.exp(t)

def solve_least_squares(A, b, lam=1e-8, use_lsqr=False):
    m = A.shape[0]
    if lam > 0:
        A = A + lam * np.eye(m, dtype=np.float64)
    if use_lsqr:
        # (Optional) LSQR via scipy.sparse.linalg.lsqr
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla
        Asp = sp.csr_matrix(A)
        c, *_ = spla.lsqr(Asp, b, atol=1e-12, btol=1e-12, iter_lim=2000)
        return c
    else:
        import numpy.linalg as npla
        return npla.solve(A, b)

def compute_E2(N, theta_min=1e-3, theta_max=1.0, x_min=1e-8, quad_n=6, lam=1e-6, use_lsqr=False):
    thetas = theta_log_lattice(N, theta_min, theta_max)
    A = build_A_thetas(thetas, x_min=x_min, quad_n=quad_n)
    b = build_b_thetas(thetas, x_min=x_min, quad_n=quad_n)
    c = solve_least_squares(A, b, lam=lam, use_lsqr=use_lsqr)
    return 1.0 - float(b @ c)
