import argparse
import numpy as np
from pathlib import Path
from nbdb_gram import build_gram_and_b, projection_error
from nbdb_sweep import sweep

def parse_args():
    p = argparse.ArgumentParser(description="NB/BD constructive experiments")
    p.add_argument("--mode", choices=["single","sweep"], required=True)
    p.add_argument("--N", type=int, default=1000, help="Basis size for single mode")
    p.add_argument("--N_list", type=str, default="50,100,200,500,1000,2000", help="Comma-separated list for sweep")
    p.add_argument("--grid", choices=["small","medium","large"], default="medium", help="Quadrature grid size")
    p.add_argument("--outdir", type=str, default="data", help="Output directory")
    return p.parse_args()

def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.mode == "single":
        theta, A, b, x = build_gram_and_b(N=args.N, grid=args.grid)
        E2 = projection_error(A, b)
        print(f"[single] N={args.N}  E_N^2={E2:.6e}  E_N^2*log^2N={E2*(np.log(args.N)**2):.6f}")
    else:
        N_list = [int(s) for s in args.N_list.split(",") if s.strip()]
        csv_path = outdir / "sweep_results.csv"
        png_path = outdir / "sweep_loglog.png"
        df, slope, intercept = sweep(N_list, grid=args.grid, out_csv=str(csv_path), out_png=str(png_path))
        print(f"[sweep] Regression slope (log–log): {slope:.3f}")
        print(f"[sweep] CSV saved to: {csv_path}")
        print(f"[sweep] Plot saved to: {png_path}")

if __name__ == "__main__":
    main()
