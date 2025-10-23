# RH Lightweight Sweep (NB/BD Approximation)

Fast, low-memory approximation for NB/BD least-squares using coarse log-space quadrature.
Intended for modest hardware while tracking the theoretical $E_N^2 \sim 1/\log^2 N$.

## Install & Run
```
pip install numpy scipy matplotlib pandas
python experiments/sweep_light.py --Ns 100,200,500,1000 --num_bins 256 --x_min 1e-6 --lam 1e-6 --lsqr
python experiments/plot_figure_light.py
```

## Tips
- Speed up: reduce `--num_bins` (e.g., 128 or 64), increase `--x_min` (e.g., 1e-5), enable `--lsqr`.
- Accuracy up: increase `--num_bins` (512+), decrease `--x_min` (1e-7), keep `--lam` small.
