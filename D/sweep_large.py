# experiments/sweep_large.py
import numpy as np
import pandas as pd
from pathlib import Path
from src.nbbd_fast import compute_E2

def main():
    Ns = [100, 200, 300, 500, 800, 1000, 1500, 2000, 3000, 5000, 8000, 10000]
    rows = []
    for N in Ns:
        E2 = compute_E2(
            N,
            theta_min=1e-3, theta_max=1.0,
            x_min=1e-8, quad_n=6,
            lam=1e-6,
            use_lsqr=(N>=2000)  # 큰 N은 LSQR 권장
        )
        rows.append({"N": N, "E2_actual": E2, "inv_log2": 1.0/(np.log(N)**2)})
        print(f"N={N:>5d}  E2={E2:.6e}")
    out = Path("experiments"); out.mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out/"sweep_large_results.csv", index=False)
    print("Saved:", out/"sweep_large_results.csv")

if __name__ == "__main__":
    main()
