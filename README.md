# Navier–Stokes / Perelman-style verification lab

Public, reproducible numerical/symbolic checks for geometric identities arising from a Lagrangian/Hodge/heat-flow view of 3D incompressible Navier–Stokes.

**Policy:** numerical experiments run only on GitHub Actions. Rigorous interval checks use `python-flint` (Arb backend) at **160-bit precision**.

This repository does **not** claim a proof of Navier–Stokes regularity. It separates exact identities, rigorously enclosed numerical experiments, and conjectural coercive inequalities.
