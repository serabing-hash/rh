# clark_measure_simulation.py
# Author: Jongmin Choi
# Purpose: Numerical visualization of the Clark measure μΘ for the zeta-encoded inner function Θ(s)

import mpmath as mp
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
mp.dps = 30         # precision (digits)
RANGE = 40          # range for t in [-RANGE, RANGE]
POINTS = 2000       # fewer points for weak CPUs
LOG_SCALE = True    # log scale for better visibility

# --- CORE FUNCTIONS ---
def theta(s):
    """Zeta-encoded inner function."""
    return mp.zeta(1 - s) / mp.zeta(s)

def mu_density(t):
    """Clark measure density μΘ(t) ∝ |ζ(1/2+it)|⁻²."""
    s = 0.5 + 1j * t
    z = mp.zeta(s)
    return 1.0 / abs(z)**2 if z != 0 else mp.nan

# --- NUMERICAL SAMPLING ---
ts = np.linspace(-RANGE, RANGE, POINTS)
mu_vals = []

for i, t in enumerate(ts):
    try:
        mu_vals.append(mu_density(t))
    except Exception:
        mu_vals.append(mp.nan)

# --- PLOTTING ---
plt.figure(figsize=(10, 4))
plt.plot(ts, mu_vals, lw=1, color="navy")

plt.title(r"Clark Measure Density $\mu_{\Theta}(t) \propto |ζ(1/2+it)|^{-2}$", fontsize=11)
plt.xlabel("t (imaginary axis of s = 1/2 + it)")
plt.ylabel(r"$\mu_{\Theta}(t)$")

if LOG_SCALE:
    plt.yscale("log")

plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()