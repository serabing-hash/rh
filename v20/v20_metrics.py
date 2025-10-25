
# v20_metrics.py — metrics for atom-free robustness
import numpy as np

def eigen_spectrum(G: np.ndarray):
    Gs = (G + G.T) / 2.0
    w = np.linalg.eigvalsh(Gs)
    return np.sort(w)

def entropy_from_weights(w: np.ndarray) -> float:
    wpos = np.clip(w, 0.0, None)
    s = wpos.sum()
    if s <= 0:
        return 0.0
    p = wpos / s
    p = p[p > 0]
    return float(-np.sum(p*np.log(p)))

def kde_scott(x: np.ndarray, grid: int=400):
    x = np.sort(x)
    n = len(x)
    if n == 0:
        xs = np.linspace(0, 1, grid)
        return xs, np.zeros_like(xs), 1.0
    sigma = np.std(x)
    bw = 1.06*sigma*np.power(n, -1/5)  # Scott/Silverman rule of thumb
    if bw <= 1e-12:
        bw = max(1e-6, 0.01*np.max(x) if np.max(x) > 0 else 1e-6)
    xs = np.linspace(x[0], x[-1], grid)
    dens = np.zeros_like(xs)
    inv = 1.0/(np.sqrt(2*np.pi)*bw*n)
    for xi in x:
        z = (xs - xi)/bw
        dens += np.exp(-0.5*z*z)
    dens *= inv
    return xs, dens, bw

def gap_stats(w: np.ndarray):
    gaps = np.diff(w)
    gaps = gaps[gaps > 1e-15]
    if len(gaps) == 0:
        return dict(median=0.0, max=0.0, ratio=float('inf'))
    med = float(np.median(gaps))
    mx = float(np.max(gaps))
    ratio = (mx/med) if med > 0 else float('inf')
    return dict(median=med, max=mx, ratio=ratio)

def basic_metrics(G: np.ndarray):
    w = eigen_spectrum(G)
    xs, dens, bw = kde_scott(w)
    ent = entropy_from_weights(w)
    g = gap_stats(w)
    eps = 1e-15
    wpos = np.abs(w)
    wpos = wpos[wpos > eps]
    cn = float((wpos.max()/wpos.min()) if len(wpos) else float('inf'))
    return {
        "lambda_min": float(w.min() if len(w) else 0.0),
        "lambda_max": float(w.max() if len(w) else 0.0),
        "entropy": ent,
        "gap_median": g["median"],
        "gap_max": g["max"],
        "gap_ratio": g["ratio"],
        "kde_peak": float(dens.max() if len(dens) else 0.0),
        "kde_median": float(np.median(dens) if len(dens) else 0.0),
        "kde_peak_over_median": float((dens.max()/np.median(dens)) if np.median(dens)>0 else float('inf')),
        "bw": float(bw),
        "cond_est": cn
    }, (w, xs, dens)
