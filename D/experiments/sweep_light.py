import argparse, numpy as np, pandas as pd
from pathlib import Path
from src.nbbd_light import compute_E2

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--Ns", type=str, default="100,200,300,500,800,1000")
    p.add_argument("--theta_min", type=float, default=1e-3)
    p.add_argument("--theta_max", type=float, default=1.0)
    p.add_argument("--num_bins", type=int, default=256)
    p.add_argument("--x_min", type=float, default=1e-6)
    p.add_argument("--lam", type=float, default=1e-6)
    p.add_argument("--lsqr", action="store_true")
    args = p.parse_args()

    Ns = [int(s) for s in args.Ns.split(",")]
    rows = []
    for N in Ns:
        E2 = compute_E2(
            N, theta_min=args.theta_min, theta_max=args.theta_max,
            num_bins=args.num_bins, x_min=args.x_min,
            lam=args.lam, use_lsqr=args.lsqr
        )
        rows.append({"N": N, "E2_actual": E2, "inv_log2": 1.0/(np.log(N)**2)})
        print(f"N={N:>5d}  E2={E2:.6e}  (1/log^2 N={1.0/(np.log(N)**2):.3e})")

    out = Path("experiments"); out.mkdir(exist_ok=True)
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(out/"sweep_light_results.csv", index=False)
    print("Saved:", out/"sweep_light_results.csv")

if __name__ == "__main__":
    main()
