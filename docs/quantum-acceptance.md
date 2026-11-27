# Acceptance record for modules 13–20

Scope: numerical simulation of finite closed and open systems, with an exact finite-horizon line simulation. Evidence is from software runs. Hardware and release acceptance remain outside this milestone.

| Module | Test or independent comparison | Evidence |
|---|---|---|
|13|Full four-cycle operator from elementary Qiskit gates, local-port shifts, line and cycle marginals|tests/test_quantum_references.py and tests/test_quantum.py; benchmark A|
|14|Two-site analytic amplitudes and inverse propagation; dense, spectral and sparse propagation on six graph families|tests/test_quantum.py; benchmark F, 36 graph/size/time combinations|
|15|Small-chain isometries, reflections and principal-angle phases; directed chain with explicit reverse|tests/test_quantum.py; benchmark G|
|16|Geometric first-detection law, a coherent target, survival accounting and a dark sector|tests/test_quantum.py; benchmark H, 10,000 independent measurement trajectories|
|17|Return amplitudes and independently monitored first returns; finite line light cone|tests/test_quantum.py; benchmark I|
|18|Degenerate and nearly degenerate phases, basis grouping, finite-time averages, limiting projectors and position dephasing|tests/test_quantum.py; benchmark I|
|19|Analytic four-item search; marked coined and Szegedy trajectories checked step by step, classical hitting and uniform sampling baselines|tests/test_quantum.py; benchmark K|
|20|Closed-form damping law, two-state classical jump limit, trace and positivity diagnostics, independent driven dissipative QuTiP trajectory|tests/test_quantum.py and tests/test_quantum_references.py; benchmark J|

With the initial state split equally between a bright and a dark component, 49.68% of trajectories were detected within twelve attempts (standard error 0.50 percentage points). The finite-horizon reference was 49.9878% and the eventual probability was 50%. For the continuous walks, the largest L2 difference between sparse action and the dense reference over the grid was below 3e-14. For the amplitude-damping example the exponential and adaptive ODE solutions differed by at most 5.1e-10 in any density-matrix entry.

The stored benchmark record comes from the Linux Python 3.12 reference run. The local macOS NumPy 2.0.2 / SciPy 1.13.1 stack emitted floating-point warnings inside the dense exponential benchmark. The independent Linux run completed with numerical warnings promoted to errors; the two systems agreed on the sampled detection fraction and had comparable propagation residuals. Wall-clock measurements are environment-specific.

[API, assumptions and references](quantum.md) · [Numerical record](quantum-benchmarks.json) · [CI history](https://github.com/Phasetensor/MarkovQ/actions)
