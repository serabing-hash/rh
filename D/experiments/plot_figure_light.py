import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    df = pd.read_csv("experiments/sweep_light_results.csv")
    out = Path("experiments"); out.mkdir(exist_ok=True)

    plt.figure(figsize=(6,4))
    plt.loglog(df["N"], df["E2_actual"], 'o-', linewidth=2, label="Actual $E_N^2$ (light)")
    plt.loglog(df["N"], df["inv_log2"], 's--', linewidth=2, label="$1/\\log^2 N$")
    plt.xlabel("N")
    plt.ylabel(r"$E_N^2$")
    plt.title("Lightweight: $E_N^2$ vs $1/\\log^2 N$")
    plt.grid(True, which='both', ls='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out/"figure_light_E2.png", dpi=300)
    plt.close()
    print("Saved:", out/"figure_light_E2.png")

if __name__ == "__main__":
    main()
