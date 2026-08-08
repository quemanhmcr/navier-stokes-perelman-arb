# Navier–Stokes / Perelman-style verification lab

Public, reproducible numerical/symbolic checks for geometric identities arising from a Lagrangian/Hodge/heat-flow view of 3D incompressible Navier–Stokes.

**Policy:** numerical experiments run only on GitHub Actions. Rigorous interval checks use `python-flint` (Arb backend) at **160-bit precision**.

This repository does **not** claim a proof of Navier–Stokes regularity. It separates exact identities, rigorously enclosed numerical experiments, and conjectural coercive inequalities.

## Current verification ladder

The `Core geometry verification — Arb 160-bit` workflow separates claims into:
- **EXACT** — symbolic identities (SymPy exact rationals).
- **ARB-RIGOROUS** — 160-bit Arb interval checks; CI fails unless the sampled sign/equality is certified.
- **NUMERICAL-EVIDENCE** — rigorously enclosed point samples for patterns, never labeled as proof.

Targets: tensor square-completion, Fourier Hodge projection and strain norm, isotropic-shell cancellation, moving-Hodge geometry, covariance no-collapse, anisotropy defect, Cauchy–Fisher precision identity, and the small-time precision expansion.
