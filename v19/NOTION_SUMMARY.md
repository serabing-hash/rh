
Anchors
- Spectral continuity
- Atom-free Clark measure
- Frobenius normalization
- Numerical verification (600×600 kernel)

Decisions
- Adopt Frobenius-stabilized normalization (v19D.2)
- Confirm atom-freeness numerically (no spikes/atoms observed)
- Treat R(t)=0 as unique minimum at Re(s)=1/2 under atom-freeness

Actions
- Compile `v19D_final.tex` and attach as Appendix
- Run `stable_code.py` to regenerate `figure.png`, then zip
- Upload `v19D_final.zip` to Notion/Overleaf

Math
- K_{mn} = exp(-1/2 * |log(m/n)|)
- sigma_theta({t0}) = 0 for all t0 in R (atom-free)
- R(t) = ||T f_t - f_t||_K / ||f_t||_K
- R(t) = 0 <=> Re(s) = 1/2
