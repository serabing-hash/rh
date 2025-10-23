
# nbbd_quadrature.py
# Quadrature utilities to build Gram matrix A and vector b for the NB/BD basis.
# Note: For large N this will be computationally heavy; start with N<=200 and profile.
# Dependencies: numpy (required), scipy (optional for LSQR), numba (optional for speed).

import numpy as np

def breakpoints_theta(theta, x_min=0.0, x_max=1.0, m_max=None):
    r\"\"\"
    Return sorted breakpoints for {theta/x}: points are theta/m.
    Only keep those within (x_min, x_max].
    r\"\"\"
    if theta <= 0:
        raise ValueError(\"theta must be > 0\")
    if x_min <= 0:
        x_min = 1e-12
    if m_max is None:
        m_max = int(np.ceil(theta / x_min))
        m_max = min(m_max, 100000)
    pts = theta / np.arange(1, m_max + 1, dtype=float)
    pts = pts[(pts > x_min) & (pts <= x_max)]
    return np.sort(pts)

def breakpoints_one_over_x(x_min=0.0, x_max=1.0, n_max=None):
    r\"\"\"
    Return sorted breakpoints for {1/x}: points are 1/n.
    Keep within (x_min, x_max].
    r\"\"\"
    if x_min <= 0:
        x_min = 1e-12
    if n_max is None:
        n_max = int(np.ceil(1.0 / x_min))
        n_max = min(n_max, 100000)
    pts = 1.0 / np.arange(1, n_max + 1, dtype=float)
    pts = pts[(pts > x_min) & (pts <= x_max)]
    return np.sort(pts)

def frac_theta_over_x(theta, x):
    y = theta / x
    return y - np.floor(y)

def frac_one_over_x(x):
    y = 1.0 / x
    return y - np.floor(y)

def integrate_piecewise(f, pts, quad=\"gauss\", n_gauss=4):
    r\"\"\"
    Integrate f on (0,1] piecewise over intervals defined by 'pts' (sorted breakpoints subset of (0,1]).
    Intervals: (0, pts[0]], (pts[0], pts[1}], ..., (pts[-1], 1].
    quad: 'mid' (midpoint), or 'gauss' (Gauss-Legendre with n_gauss points).
    r\"\"\"
    edges = np.concatenate(([0.0], pts, [1.0]))
    total = 0.0
    if quad == \"mid\":
        for a, b in zip(edges[:-1], edges[1:]):
            if b <= a:
                continue
            xm = 0.5*(a+b)
            total += f(xm)*(b-a)
        return total
    # Gauss-Legendre nodes/weights on [-1,1]
    xi, wi = np.polynomial.legendre.leggauss(n_gauss)
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a:
            continue
        xm = 0.5*((b-a)*xi + (b+a))
        w  = 0.5*(b-a)*wi
        total += np.sum(f(xm)*w)
    return total

def build_b_thetas(thetas, quad=\"gauss\", n_gauss=4):
    r\"\"\"
    Compute b_i = <chi, varphi_theta_i> = int_0^1 ({theta/x} - theta {1/x}) dx
    r\"\"\"
    pts_common = breakpoints_one_over_x()
    b = np.zeros(len(thetas), dtype=float)
    for i, th in enumerate(thetas):
        pts_th = breakpoints_theta(th)
        pts = np.unique(np.concatenate((pts_th, pts_common)))
        f = lambda x, th=th: frac_theta_over_x(th, x) - th*frac_one_over_x(x)
        b[i] = integrate_piecewise(f, pts, quad=quad, n_gauss=n_gauss)
    return b

def build_A_thetas(thetas, quad=\"gauss\", n_gauss=4, symmetric=True):
    r\"\"\"
    Compute Gram matrix A_{ij} = <varphi_theta_i, varphi_theta_j>.
    r\"\"\"
    m = len(thetas)
    A = np.zeros((m, m), dtype=float)
    pts_common = breakpoints_one_over_x()
    cache_bp = [breakpoints_theta(th) for th in thetas]
    for i in range(m):
        for j in range(i if symmetric else 0, m):
            th_i, th_j = thetas[i], thetas[j]
            pts = np.unique(np.concatenate((cache_bp[i], cache_bp[j], pts_common)))
            def f(x, a=th_i, b=th_j):
                vi = frac_theta_over_x(a, x) - a*frac_one_over_x(x)
                vj = frac_theta_over_x(b, x) - b*frac_one_over_x(x)
                return vi*vj
            Aij = integrate_piecewise(f, pts, quad=quad, n_gauss=n_gauss)
            A[i, j] = Aij
            if symmetric and j != i:
                A[j, i] = Aij
    return A
