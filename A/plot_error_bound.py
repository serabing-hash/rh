#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

def generate_error_bound_data(N_list=None, out_csv="data/error_bound_logspace.csv", out_fig="figs/error_decay_bound.png"):
    if N_list is None:
        N_list = [10, 20, 30, 50, 100, 200, 300, 500, 1000, 2000, 3000, 5000, 10000]
    N = np.array(N_list, dtype=int)
    E2_bound = 1.0 / (np.log(N) ** 2)
    df = pd.DataFrame({"N": N, "E2_bound_theoretical": E2_bound})
    df.to_csv(out_csv, index=False)
    plt.figure(figsize=(6,4))
    plt.loglog(N, E2_bound, 'o-', linewidth=2)
    plt.xlabel("N (logarithmic lattice size)")
    plt.ylabel(r"$E_N^2$ (theoretical upper bound)")
    plt.title("Upper Bound Trend: $E_N^2 \\lesssim 1/\\log^2 N$")
    plt.grid(True, which='both', ls='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(out_fig, dpi=300)
    print(f"Saved: {out_csv} and {out_fig}")
    return df

if __name__ == "__main__":
    generate_error_bound_data()