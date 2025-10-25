
# v20 — Continuous Deformation (Hybrid Kernel)

## Files
- `v20_kernel.py` : K0/K1/K2/K3 kernels + Frobenius normalization
- `v20_metrics.py`: M1–M5 metrics (KDE: Scott rule, entropy, gaps, cond estimate)
- `v20_runner.py` : E1–E3 sweeps, saves PNG panels and `results.json`
- `v20_notes.tex` : Overleaf notes
- `figs/`, `results/` : output dirs

## Quick Start
```bash
python v20_runner.py
```
- outputs go to `results/`.
