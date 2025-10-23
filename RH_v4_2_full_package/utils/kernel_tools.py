import numpy as np

def make_mixed_grid(n_small=4000, n_large=6000, x_small_min=1e-6, x_small_max=1e-2):
    """Create a composite grid: dense near 0 (logspace), uniform near 1."""
    small = np.geomspace(x_small_min, x_small_max, num=n_small, endpoint=True, dtype=float)
    large = np.linspace(x_small_max, 1.0, num=n_large, endpoint=True, dtype=float)
    x = np.unique(np.concatenate([small, large]))
    return x

def trapz_integral(y, x):
    """Composite trapezoidal rule integral of y with respect to x."""
    return np.trapz(y, x)

def regress_loglog(logN, E2):
    """Linear regression on log–log scale: log(E2) vs log(log N)."""
    x = np.log(np.array(logN, dtype=float))
    y = np.log(np.array(E2, dtype=float))
    A = np.vstack([x, np.ones_like(x)]).T
    coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    slope, intercept = coef[0], coef[1]
    return slope, intercept
