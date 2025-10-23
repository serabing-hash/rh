import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from nbdb_gram import build_gram_and_b, projection_error

def sweep(N_list, grid="medium", out_csv="data/sweep_results.csv", out_png="data/sweep_loglog.png"):
    rows = []
    for N in N_list:
        theta, A, b, x = build_gram_and_b(N=N, grid=grid)
        E2 = projection_error(A, b)
        logN = np.log(N)
        rows.append({"N": N, "logN": logN, "E2": E2, "E2_log2": E2*(logN**2)})
        print(f"[sweep] N={N:6d}  E_N^2={E2:.6e}  E_N^2*log^2N={E2*(logN**2):.6f}")

    df = pd.DataFrame(rows).sort_values("N")
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    # Log–log plot of E2 vs log N
    plt.figure()
    x_vals = np.log(df["logN"].values)  # log(log N)
    y_vals = np.log(df["E2"].values)    # log(E2)
    A = np.vstack([x_vals, np.ones_like(x_vals)]).T
    slope, intercept = np.linalg.lstsq(A, y_vals, rcond=None)[0]
    plt.scatter(x_vals, y_vals, label="data")
    plt.plot(x_vals, slope*x_vals + intercept, label=f"fit slope ≈ {slope:.2f}")
    plt.xlabel("log(log N)")
    plt.ylabel("log(E_N^2)")
    plt.legend()
    plt.title("Convergence: E_N^2 vs log N (log–log scale)")
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()

    return df, slope, intercept
