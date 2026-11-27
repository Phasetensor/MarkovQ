# MarkovQ — Master Scientific & Engineering Roadmap

## Classical & Quantum Stochastic Dynamics

Organization: PhaseTensor

Research division: Quantum Computing Suite

Repository:

https://github.com/Phasetensor/MarkovQ

Proposed Python package: markovq

License: Apache 2.0

Development language: Python

Project classification: Scientific Computing, Stochastic Processes, Quantum Computing, Spectral Graph Theory and Applied Network Science.

---

# 1. Project mission

Develop MarkovQ as a comprehensive scientific Python framework for the mathematical analysis, numerical simulation and experimental validation of classical and quantum stochastic dynamics.

The project must address:

- Discrete-time Markov chains.
- Continuous-time Markov chains.
- Classical random walks.
- Quantum random walks.
- Recurrence and transience.
- First-passage processes.
- Absorbing processes.
- Stationary distributions.
- Spectral graph dynamics.
- Quantum search.
- Open quantum systems.
- Biological network analysis.
- Large-scale computational experiments.

MarkovQ must bridge classical probability theory with quantum dynamics while preserving their mathematical distinctions.

It must not assume that quantum dynamics inherits classical Markov properties.

The library must provide reproducible, independently verifiable scientific results.

Do not claim quantum advantage without a properly defined computational task, relevant classical baselines and complete resource accounting.

# 2. PhaseTensor ecosystem

MarkovQ belongs to PhaseTensor and must complement its existing libraries.

Quantum Computing Suite:

- ShotWeave — Quantum Measurement Resource Optimization.
- BornLens — Quantum Statistical Inference.
- CalibMap — Quantum Calibration Analytics.

Quantitative Finance Suite:

- VintageCheck — Financial Data Integrity.
- CurveMetric — Fixed-Income Curve Diagnostics.
- SurfaceCheck — Volatility Surface Validation.

Verify the current public releases of the relevant libraries before integration.

MarkovQ's principal interoperability workflow is:

Graph and stochastic model → MarkovQ → Quantum circuit → CalibMap → ShotWeave → Quantum execution → BornLens → Scientific validation.

Do not modify the existing libraries without authorization.

# 3. Repository initialization

Use the official GitHub MCP server available in Work mode.

Verify:

- Repository availability.
- Organization permissions.
- PyPI distribution-name availability.
- Existing projects using the MarkovQ name.
- Potential naming conflicts.

If the repository does not exist, create it only when authorized.

Initialize with:

- README.md.
- Apache 2.0 license.
- Python .gitignore.
- CONTRIBUTING.md.
- CITATION.cff.
- CHANGELOG.md.
- ROADMAP.md.
- pyproject.toml.
- Scientific documentation.
- GitHub Actions.

All public documentation must be in English.

Use a dedicated development branch.

Do not publish incomplete implementations.

# 4. Scientific literature and competitive analysis

Investigate primary literature and existing implementations covering:

- Classical Markov chains.
- Random walks on graphs.
- Absorbing chains.
- First-passage processes.
- Mixing times.
- Continuous-time Markov processes.
- Lévy processes.
- Spectral graph theory.
- Discrete-time quantum walks.
- Continuous-time quantum walks.
- Szegedy quantum walks.
- Quantum first detection.
- Quantum recurrence.
- Quantum search.
- Quantum stochastic walks.
- Lindblad master equations.
- Biological network propagation.
- Quantum hardware implementations.

Review relevant research by:

- Andrey Markov.
- Paul Lévy.
- William Feller.
- Joseph Doob.
- Elliott Montroll.
- Michael Aizenman.
- Edward Farhi.
- Sam Gutmann.
- Dorit Aharonov.
- Miklos Santha.
- Mario Szegedy.
- Norio Konno.
- Viv Kendon.
- Oliver Mülken.
- Relevant contemporary researchers.

Verify the specific relevance and bibliographic details of every cited work.

Review available implementations in:

- SciPy.
- NetworkX.
- QuTiP.
- Qiskit.
- Qiskit Aer.
- PennyLane.
- Hiperwalk.
- Existing quantum-walk libraries.
- Existing Markov-chain analysis libraries.
- Relevant Algorithmiq research.

Produce a competitive analysis identifying concrete unmet research workflows.

Do not invent scientific novelty.

---

# SCIENTIFIC DIVISION I — CLASSICAL STOCHASTIC PROCESSES

# 5. Module 01 — Markov Chain Engine

Implement discrete-time Markov chains on finite state spaces, with explicit extensions to countable-state models where mathematically justified.

Define the Markov property:

P(X[n+1] = j | X[0], ..., X[n] = i) = P(X[n+1] = j | X[n] = i).

Support:

- Transition matrix construction.
- Row-stochastic matrix validation.
- Homogeneous Markov chains.
- Time-inhomogeneous Markov chains.
- Initial probability distributions.
- Multi-step transitions.
- State trajectories.
- Dense matrices.
- Sparse matrices.
- Reproducible simulation.

Validate the Chapman-Kolmogorov identity for homogeneous chains:

P^(n+m) = P^n P^m.

For inhomogeneous chains preserve chronological matrix multiplication.

Proposed APIs:

- MarkovChain
- validate_transition_matrix()
- transition_probability()
- evolve()
- sample_paths()
- n_step_matrix()

Independent validation:

- Analytical transition matrices.
- Exhaustive finite-state calculations.
- Matrix multiplication identities.
- Monte Carlo comparison.

Do not silently normalize invalid transition matrices.

# 6. Module 02 — Classical Random Walk

Implement random walks on:

- Integer lattices.
- Finite line graphs.
- Cycles.
- Two-dimensional grids.
- Weighted graphs.
- Directed graphs.
- Arbitrary finite networks.

For the one-dimensional walk:

X[n] = X[0] + sum(Z[k]).

Support:

- Symmetric walks.
- Biased walks.
- Lazy random walks.
- Reflecting boundaries.
- Absorbing boundaries.
- Weighted transitions.
- Random increments.
- Path statistics.

Measure:

- Expected position.
- Variance.
- Mean squared displacement.
- Return probabilities.
- First-passage probabilities.
- Occupation times.

Validate against exact binomial distributions and known analytical results.

For the standard symmetric walk on the integers, verify variance growth proportional to n.

# 7. Module 03 — State Classification

Implement classification of finite-state Markov chains.

Support:

- Accessibility.
- Communication.
- Communicating classes.
- Closed classes.
- Reducibility.
- Irreducibility.
- Periodicity.
- Aperiodicity.
- Absorbing states.
- Transient and recurrent classes.

Use graph-theoretic strongly connected components.

Proposed APIs:

- communicates()
- communicating_classes()
- closed_classes()
- state_period()
- classify_states()
- is_irreducible()

Validate against hand-constructed transition matrices and independent graph algorithms.

Do not transfer finite-state classification conclusions directly to infinite-state chains.

# 8. Module 04 — Recurrence & Transience

Implement classical recurrence and transience diagnostics.

Analyze:

- First-return probabilities.
- Eventual return probabilities.
- Expected return times.
- Recurrent states.
- Transient states.
- Positive recurrence.
- Null recurrence.
- Green functions.
- Return generating functions.

Investigate the identity:

P_ii(z) = 1 + F_ii(z) P_ii(z).

Where mathematically valid:

F_ii(z) = 1 - 1/P_ii(z).

Support analytical benchmark problems on infinite graphs without claiming that finite truncation proves recurrence or transience of an infinite chain.

Distinguish finite-horizon return probabilities from eventual return probabilities.

Validate:

- Symmetric random walk on Z.
- Biased random walk on Z.
- Symmetric walks on multidimensional lattices.
- Finite irreducible chains.

# 9. Module 05 — Absorbing Markov Chains

Implement absorbing-chain analysis.

Support canonical block representations:

P = [[Q, R], [0, I]].

When absorption from the transient states is guaranteed, use the fundamental matrix:

N = (I - Q)^(-1).

Prefer solving linear systems to explicitly forming inverses where appropriate.

Implement:

- Absorption probabilities.
- Expected absorption times.
- Absorption-time distributions.
- Expected state visits.
- Fundamental matrix diagnostics.
- Multiple absorbing classes.
- Gambler's ruin.

Validate analytical gambler's ruin solutions.

Detect singular or nonabsorbing subproblems instead of reporting invalid finite expectations.

# 10. Module 06 — First Passage & Hitting Time

Implement first-passage analysis.

For a target set A:

T_A = inf{n >= 0 : X[n] belongs to A}.

Distinguish first hitting time from first return time.

Support:

- First-passage distributions.
- Survival probabilities.
- Expected hitting times.
- Conditional hitting times.
- Multiple target sets.
- Censored simulations.
- Finite-horizon reachability.
- Eventual hitting probabilities.

Solve appropriate boundary-value equations.

For states outside a target, investigate:

h(i) = 1 + sum(P_ij h(j)).

Detect infinite expected hitting times and nonunit eventual hitting probabilities.

Validate against analytical examples and independent simulation.

# 11. Module 07 — Stationary Distribution Engine

Implement stationary distribution analysis.

For row-stochastic P:

pi P = pi.

Subject to:

pi_i >= 0 and sum(pi_i) = 1.

Support:

- Linear-system methods.
- Eigenvector methods.
- Power iteration.
- Cesàro averages.
- Detailed balance.
- Reversibility.
- Multiple stationary distributions.
- Closed-class stationary distributions.

Validate against:

- Uniform distributions on regular undirected graphs.
- Degree-proportional distributions on undirected connected graphs.
- Reducible chains.
- Periodic chains.
- Analytically known examples.

Distinguish stationarity from convergence to stationarity.

# 12. Module 08 — Classical Mixing & Convergence

Implement convergence diagnostics.

Support:

- Total variation distance.
- Relative entropy, where defined.
- L1 and L2 distances.
- Spectral gaps.
- Mixing-time estimates.
- Convergence bounds.
- Periodicity diagnostics.
- Reversibility diagnostics.
- Initial-state sensitivity.

Define:

d_TV(p,q) = 0.5 sum_i |p_i-q_i|.

Distinguish empirical mixing from mathematically established mixing-time bounds.

Do not apply reversible-chain spectral bounds indiscriminately to nonreversible chains.

Detect chains that do not converge under the specified assumptions.

# 13. Module 09 — Continuous-Time Markov Chains

Implement continuous-time Markov dynamics.

Support generator matrices Q satisfying:

Q_ij >= 0 for i != j.

Q_ii = -sum_{j != i} Q_ij.

Evolution:

P(t) = exp(tQ).

Implement:

- Generator validation.
- Transition semigroups.
- Exponential holding times.
- Jump-chain construction.
- Kolmogorov equations.
- Uniformization.
- Event-driven simulation.
- Birth-death processes.
- Queueing models.
- Continuous-time absorption.

Validate against:

- Poisson processes.
- Two-state continuous-time chains.
- Birth-death models.
- Independent matrix exponential calculations.

Maintain explicit time units.

# 14. Module 10 — Lévy & Jump Processes

Develop a controlled set of classical stochastic-process extensions.

Implement initially:

- Poisson processes.
- Compound Poisson processes.
- Brownian motion.
- Selected jump-diffusion processes.
- Characteristic functions.
- Increment statistics.

Evaluate the Lévy-Khintchine representation within the supported mathematical scope.

Support:

- Independent increments.
- Stationary increments.
- Jump-size distributions.
- Reproducible path simulation.
- Jump statistics.
- Moment diagnostics where moments exist.

Investigate stable processes as an extension, documenting parameterization and numerical limitations.

Do not claim general Lévy-process simulation from a limited collection of examples.

Validate analytical characteristic functions and distributional properties.

---

# SCIENTIFIC DIVISION II — GRAPH THEORY & SPECTRAL MATHEMATICS

# 15. Module 11 — Graph Hamiltonian Engine

Implement mathematical representations of graph structures.

Support:

- Adjacency matrices.
- Weighted adjacency matrices.
- Degree matrices.
- Graph Laplacians.
- Normalized Laplacians.
- Hermitian graph Hamiltonians.
- Sparse representations.
- Connected components.
- Directed and undirected graphs.

For an undirected graph:

L = D - A.

Evaluate Hamiltonian conventions:

H = gamma A.

H = gamma L.

These are different models and must not be treated as universally equivalent.

Directed graphs require a justified Hermitian construction for unitary evolution.

Reject arbitrary non-Hermitian generators in APIs intended for closed-system quantum evolution.

# 16. Module 12 — Spectral Analysis Engine

Implement:

- Eigenvalues.
- Eigenvectors.
- Spectral gaps.
- Spectral decomposition.
- Degenerate eigenspaces.
- Spectral projectors.
- Matrix conditioning.
- Sparse eigensolvers.
- Orthogonality checks.
- Graph spectral diagnostics.

Compare dense and sparse reference methods.

Validate:

- Paths.
- Cycles.
- Complete graphs.
- Star graphs.
- Regular lattices.
- Disconnected graphs.

Handle eigenvalue degeneracy explicitly.

Do not compare individual eigenvectors as unique when eigenspaces are degenerate.

Use subspace or projector comparisons where appropriate.

---

# SCIENTIFIC DIVISION III — QUANTUM WALK DYNAMICS

# 17. Module 13 — Discrete-Time Quantum Walk

Implement discrete-time coined quantum walks.

Use a Hilbert space with position and coin degrees of freedom.

Evolution:

|psi[n+1]> = U |psi[n]>.

For a coined model:

U = S(C tensor I),

with the tensor-order convention explicitly documented.

Support:

- Hadamard coin.
- Grover coin.
- Custom unitary coins.
- Conditional shift operators.
- Flip-flop shifts.
- Position probability distributions.
- Coin marginal distributions.
- Arbitrary initial states.
- Finite graphs.
- Integer-line approximations.
- Variable-degree graph constructions.

Verify unitary evolution.

Verify state normalization and probability conservation.

For finite integer-line approximations, avoid unintended wraparound and boundary effects.

Compare analytical small-step examples and independent Qiskit simulations.

# 18. Module 14 — Continuous-Time Quantum Walk

Implement continuous-time quantum walks.

Evolution:

|psi(t)> = exp(-iHt/hbar) |psi(0)>.

Allow units in which hbar = 1 when documented.

Support:

- Adjacency Hamiltonians.
- Laplacian Hamiltonians.
- Weighted Hamiltonians.
- Initial localized states.
- Initial superposition states.
- Dense matrix exponential.
- Sparse exponential action.
- Spectral evolution.
- Position measurement probabilities.

Compare:

- Direct diagonalization.
- Matrix exponential.
- scipy.sparse.linalg.expm_multiply.
- Independent reference implementations.

Validate probability conservation, Hermitian generators and evolution composition.

Benchmark numerical accuracy versus graph size and propagation time.

# 19. Module 15 — Szegedy Quantum Walk

Implement Szegedy-style quantization of Markov chains.

For row-stochastic P, define transition-amplitude states:

|psi_i> = |i> tensor sum_j sqrt(P_ij)|j>.

Construct the relevant isometries and reflection operators.

For the standard two-reflection representation:

U = R_B R_A.

Document precisely how the second subspace is constructed, including whether a reversed chain is used.

Support:

- Reversible chains.
- Explicitly defined nonreversible constructions where justified.
- Reflection operators.
- Discriminant matrices.
- Quantum walk operators.
- Spectral correspondence.
- Initial coherent distributions.
- Marked states.

Validate:

- Isometry.
- Reflection identities.
- Unitarity.
- Small-chain exact solutions.
- Spectral relations under documented assumptions.

Do not claim quadratic mixing or hitting-time speedup for arbitrary chains.

# 20. Module 16 — Quantum First Detection

Implement quantum first-detection distributions under explicit measurement protocols.

Let Pi represent the target projector and U the unitary step.

The first-detection probability at step n can be expressed as:

F_n = ||Pi U [(I-Pi)U]^(n-1) |psi_0>||².

Support:

- Repeated projective detection.
- Target subspaces.
- Survival probabilities.
- Conditional postmeasurement evolution.
- First-detection distributions.
- Total detection probability.
- Conditional mean detection time.
- Dark subspaces.

Distinguish:

- Unmonitored occupancy probability.
- First-detection probability.
- Classical first hitting time.
- Quantum algorithmic hitting-time definitions.

Validate probability accounting and exact small-system examples.

Do not sum unmonitored target occupancy probabilities as if they were first-detection events.

# 21. Module 17 — Quantum Recurrence

Implement quantum recurrence diagnostics.

Support:

- Return amplitudes.
- Return probabilities.
- First-return detection distributions.
- Monitored recurrence.
- Unmonitored recurrence.
- Finite-horizon diagnostics.
- Spectral recurrence.
- Quantum renewal relations where justified.

Investigate return amplitude:

a_n = <psi_0|U^n|psi_0>.

Return probability:

p_n = |a_n|².

Do not automatically identify sums of unmonitored return probabilities with monitored recurrence.

Document the initial state, observation protocol and relevant recurrence definition.

Validate analytical finite-dimensional examples and documented infinite-graph approximations.

# 22. Module 18 — Quantum Mixing

Implement quantum distribution convergence diagnostics.

Support:

- Instantaneous position probabilities.
- Time-averaged distributions.
- Cesàro averages.
- Total variation comparisons.
- Spectral degeneracy diagnostics.
- Time-averaged mixing criteria.
- Initial-state sensitivity.
- Dephasing effects.

For discrete time:

p_bar(T) = (1/T) sum_{t=0}^{T-1} p(t).

Distinguish convergence of time-averaged distributions from convergence of instantaneous distributions.

Closed finite-dimensional unitary systems generally do not exhibit classical dissipative mixing.

Validate limiting time-average formulas through spectral projectors.

Do not claim that quantum walks necessarily converge to the classical stationary distribution.

# 23. Module 19 — Quantum Search

Implement quantum-walk-based search on graphs.

Evaluate:

- Marked-vertex search.
- Coin-based quantum search.
- Continuous-time search.
- Oracle Hamiltonians.
- Amplitude amplification.
- Szegedy-based search constructions.

For a supported continuous-time model, evaluate Hamiltonians such as:

H = -gamma A - |w><w|.

Support:

- Success probability.
- Optimal evolution-time search within declared candidate ranges.
- Oracle accounting.
- Measurement resource accounting.
- Graph-dependent performance.
- Initial-state preparation cost.

Compare with appropriate classical search and random-walk baselines.

Distinguish evolution time, oracle complexity, circuit complexity and total execution cost.

Do not claim quantum advantage from a higher success probability at a single selected time.

---

# SCIENTIFIC DIVISION IV — OPEN QUANTUM SYSTEMS

# 24. Module 20 — Open Quantum Dynamics

Implement controlled open quantum-system evolution.

Use density matrices:

rho >= 0 and Tr(rho) = 1.

For Markovian open-system models, implement the Lindblad equation:

d rho/dt = -i[H,rho]
+ sum_k (L_k rho L_k† - 0.5 {L_k†L_k,rho}).

Support:

- Density matrix validation.
- Unitary evolution.
- Lindblad generators.
- Dissipative evolution.
- Quantum stochastic walks.
- Population dynamics.
- Coherence diagnostics.
- Trace preservation.
- Positivity diagnostics.

Use QuTiP as an independent reference.

Implement only mathematically justified numerical schemes.

Do not silently repair invalid density matrices and then report the original evolution as valid.

Report numerical positivity violations and integration errors.

# 25. Module 21 — Quantum Noise & Decoherence

Implement controlled quantum noise models.

Support:

- Bit-flip noise.
- Phase-flip noise.
- Depolarizing noise.
- Dephasing.
- Amplitude damping.
- Readout error.
- Explicit Kraus operators.

Validate Kraus completeness:

sum_k K_k† K_k = I.

Analyze:

- Probability conservation.
- Purity.
- Coherence decay.
- Distribution discrepancy.
- Quantum-to-classical crossover under specific protocols.
- Noise sensitivity.
- Quantum walk spreading.

Compare with Qiskit Aer and QuTiP where appropriate.

Integrate CalibMap calibration information without treating provider estimates as complete noise-channel identification.

---

# SCIENTIFIC DIVISION V — APPLIED NETWORK SCIENCE & COMPUTATION

# 26. Module 22 — Biological Network Analysis

Develop graph-based scientific workflows for biological networks.

Investigate:

- Protein-protein interaction networks.
- Gene association networks.
- Disease-associated subnetworks.
- Molecular interaction graphs.
- Network propagation.
- Seed-based ranking.
- Community structure.
- Graph centrality.

Compare:

- Classical random walks.
- Random walk with restart.
- Personalized PageRank.
- Diffusion methods.
- Classical spectral baselines.
- Quantum walk-based node ranking.

Use legally accessible biological datasets with documented provenance.

Investigate relevant Algorithmiq research on quantum walks in biological networks.

Define evaluation protocols using known held-out labels, appropriate negative examples and reproducible dataset splits.

Measure:

- Ranking precision.
- Recall.
- AUROC.
- AUPRC.
- Ranking stability.
- Sensitivity to seed selection.
- Computational cost.

Prevent information leakage across network evaluation splits.

Do not equate network-ranking improvements with validated disease mechanisms, medical efficacy or quantum advantage.

# 27. Module 23 — Sparse & Distributed Simulation

Develop scalable numerical methods.

Support:

- Sparse transition matrices.
- Sparse graph Hamiltonians.
- Sparse eigensolvers.
- Krylov-style exponential action.
- Matrix-free operators.
- Vectorized Monte Carlo.
- Parallel independent trajectories.
- Reproducible random number streams.
- Batch processing.
- Memory accounting.

Evaluate optional acceleration with:

- NumPy.
- SciPy.
- Numba.
- CuPy.
- Joblib.
- Azure Batch or equivalent distributed infrastructure.

Cloud execution must remain optional.

Do not require cloud credentials for core library operation.

Separate classical sparse graph simulation from circuit statevector simulation, whose memory requirements may grow exponentially with qubit count.

Benchmark increasing graph sizes, state dimensions and trajectory counts within available resources.

Report:

- Runtime.
- Peak RSS.
- Sparse matrix storage.
- Numerical error.
- Throughput.
- Scaling efficiency.
- Cloud resource costs, where actually incurred.

Do not invent large-scale benchmark results.

# 28. Module 24 — Quantum Hardware & PhaseTensor Integration

Implement optional quantum execution interfaces.

Support Qiskit-based:

- Coined walk circuits.
- Small graph-walk circuits.
- State preparation.
- Measurement.
- Circuit validation.
- Simulation.
- Backend resource accounting.

Verify circuit output against mathematical reference operators up to global phase.

Measure:

- Logical qubit count.
- Ancillary qubit count.
- Circuit depth.
- Two-qubit gate count.
- Shot budget.
- Job count.
- Runtime.

Integrate the published PhaseTensor libraries.

CalibMap:

- Calibration snapshots.
- Qubit and gate stability.
- Readout uncertainty.
- Experiment readiness.

ShotWeave:

- Measurement-resource optimization.
- Supported observable allocation.
- Independent final-stage estimation.
- Shot accounting.

BornLens:

- Statistical comparison of supported observables.
- Multiple-testing corrections.
- Distribution diagnostics.
- Sequential inference under appropriate assumptions.

Do not assume that BornLens's Pauli-expectation comparisons establish equality of arbitrary position distributions.

Use distribution-level methods when necessary.

Distinguish:

- Mathematical simulation.
- Ideal circuit simulation.
- Noisy simulation.
- Offline hardware snapshots.
- Newly executed hardware experiments.

No paid quantum hardware jobs or cloud workloads without explicit authorization.

# 29. Module 25 — Scientific Validation & Benchmarking

Implement a complete framework for reproducible scientific validation.

Cover:

- Analytical references.
- Independent numerical implementations.
- Exact small-system enumeration.
- Monte Carlo experiments.
- Statistical uncertainty.
- Numerical convergence.
- Performance measurements.
- Hardware resource accounting.
- Scientific reporting.

Support:

- Structured Python results.
- JSON.
- CSV.
- HTML.
- Reproducible figures.
- Dataset manifests.
- Software version metadata.
- Input fingerprints.

For every scientific experiment record:

- Mathematical model.
- Initial state.
- Graph definition.
- Boundary conditions.
- Measurement protocol.
- Random seeds.
- Numerical solver.
- Tolerances.
- Independent baseline.
- Runtime.
- Memory.
- Scientific limitations.

Do not use coverage percentages as substitutes for scientific validation.

Do not represent SHA-256 fingerprints as proof of the authenticity of external data.

---

# 30. Scientific benchmark suite

Develop the following mandatory benchmark families.

## Benchmark A — Classical versus quantum spreading

Compare a symmetric classical walk with a standard Hadamard quantum walk on the integer line.

Measure:

- Position distributions.
- Mean squared displacement.
- Variance scaling.
- Return probability.
- Distribution symmetry under the selected initial state.

Investigate classical diffusive scaling and quantum ballistic scaling within the respective model assumptions.

## Benchmark B — Gambler's ruin

Validate classical absorption probabilities and expected stopping times.

Compare analytical solutions, linear-system solutions and Monte Carlo.

## Benchmark C — Markov classification

Validate state classes, periodicity, recurrence and absorption on controlled transition matrices.

## Benchmark D — Classical mixing

Compare power iteration, total variation diagnostics and spectral estimates.

Include periodic and reducible counterexamples.

## Benchmark E — CTMC validation

Compare event-driven simulation, uniformization and matrix exponential methods.

## Benchmark F — Quantum graph evolution

Compare dense, sparse and spectral evolution on:

- Paths.
- Cycles.
- Complete graphs.
- Stars.
- Grids.
- Random graphs.

## Benchmark G — Szegedy quantization

Verify reflections, unitarity and relevant discriminant-matrix spectral identities.

## Benchmark H — Quantum first detection

Compare exact measurement-protocol calculations with independent simulated trajectories.

Include incomplete total detection and dark-state examples.

## Benchmark I — Quantum recurrence and mixing

Compare instantaneous and monitored returns, time averages and spectral formulas.

## Benchmark J — Open quantum dynamics

Compare supported Lindblad models with QuTiP.

Measure trace errors, positivity errors and numerical convergence.

## Benchmark K — Quantum search

Compare classical and quantum search under explicitly defined resource models.

## Benchmark L — Biological networks

Compare classical network propagation and quantum ranking on public or synthetic networks.

## Benchmark M — Noise sensitivity

Evaluate controlled readout, depolarization and dephasing scenarios.

## Benchmark N — Computational scalability

Compare dense and sparse simulation methods and optional distributed execution.

## Benchmark O — PhaseTensor integration

Execute an end-to-end synthetic experiment with the public distributions of CalibMap, ShotWeave and BornLens.

Use at least 10,000 independent trajectories for principal statistical benchmarks where computationally appropriate.

For more expensive experiments, justify the number of replications.

Report Monte Carlo uncertainty.

Do not selectively publish only favourable results.

# 31. Scientific case studies

Develop at least six reproducible application studies.

Case Study 1:

Classical versus quantum random walk on a line.

Case Study 2:

Gambler's ruin, first passage and absorbing states.

Case Study 3:

Classical Markov chain and its Szegedy quantization.

Case Study 4:

Monitored quantum recurrence and first detection.

Case Study 5:

Biological network ranking, comparing quantum walks with relevant classical baselines.

Case Study 6:

Noise-aware quantum walk experiments using the PhaseTensor Quantum Computing Suite.

Clearly label all synthetic datasets and simulation results.

New hardware experiments and access to proprietary biological datasets are not mandatory for an honestly scoped initial release.

# 32. Software architecture

Proposed package architecture:

markovq/
    classical/
    stochastic/
    graphs/
    spectral/
    quantum/
    detection/
    open_systems/
    noise/
    networks/
    simulation/
    adapters/
    reporting/

Supporting directories:

tests/
benchmarks/
examples/
docs/
scripts/
.github/workflows/

The architecture is provisional.

Organize modules by actual responsibilities.

Do not create empty files or unnecessary abstractions solely to reproduce the proposed directory structure.

Use:

- Public API type annotations.
- Structured result types.
- Explicit conventions.
- Numerical stability.
- Robust input validation.
- Deterministic random seeds.
- Optional integrations.
- Clear exception semantics.
- py.typed.

Use NumPy and SciPy as primary numerical dependencies.

Other scientific and quantum dependencies should be optional when practical.

# 33. Testing requirements

Implement:

- Unit tests.
- Integration tests.
- Property-based tests.
- Numerical reference tests.
- Analytical reference tests.
- Statistical simulation tests.
- Sparse-versus-dense comparisons.
- Boundary-condition tests.
- Invalid-input tests.
- Packaging tests.

Verify fundamental properties:

1. Classical transition matrices remain stochastic.
2. Classical probability distributions remain normalized.
3. Quantum unitary operators satisfy U†U = I.
4. Closed-system quantum states preserve norm.
5. Density matrices preserve trace within numerical tolerance.
6. Valid physical evolution respects positivity within tolerance.
7. Quantum measurement probabilities sum to one.
8. First-detection and survival probabilities obey probability accounting.
9. Sparse and dense implementations agree within specified tolerances.
10. Historical hardware information respects declared availability.
11. Independent statistical experiments preserve their sampling assumptions.
12. Numerical failures remain visible.

Mandatory quality gates:

- Statement coverage >= 95%.
- Branch coverage >= 90%.
- Ruff linting passed.
- Ruff formatting passed.
- Mypy passed.
- All required CI jobs passed.

Enforce statement and branch coverage separately.

Do not artificially exclude difficult code paths.

# 34. Documentation

All public documentation must be in English.

Create:

- README.md.
- CHANGELOG.md.
- CITATION.cff.
- ROADMAP.md.
- LICENSE.
- CONTRIBUTING.md.
- Scientific methodology.
- Mathematical conventions.
- API reference.
- Benchmark protocol.
- Benchmark results.
- Traceability matrix.
- Acceptance report.
- Known limitations.
- Release checklist.

Every module must document:

- Mathematical objective.
- Formula.
- Assumptions.
- Public API.
- Input domain.
- Output interpretation.
- Computational complexity.
- Validation evidence.
- Numerical limitations.

Verify all scientific references.

# 35. Scientific visualization

Generate reproducible figures for:

- Classical random-walk distributions.
- Quantum walk distributions.
- Diffusive versus ballistic spreading.
- Markov transition graphs.
- Communicating classes.
- Absorption probabilities.
- First-passage distributions.
- Return probabilities.
- Stationary distributions.
- Spectral gaps.
- Quantum probability heatmaps.
- Continuous-time graph evolution.
- Szegedy walk spectra.
- Quantum first-detection distributions.
- Quantum mixing diagnostics.
- Quantum search success curves.
- Decoherence.
- Density matrix evolution.
- Biological network ranking.
- Computational scalability.

Use actual calculated results.

Do not present illustrative mockups as experimental evidence.

# 36. Continuous integration

Configure GitHub Actions with:

- Supported Python version matrix.
- Core numerical dependencies.
- Optional integration testing.
- Ruff linting.
- Ruff formatting.
- Mypy.
- Separate coverage gates.
- Mathematical reference tests.
- Property-based tests.
- Scientific smoke benchmarks.
- Package builds.
- Strict Twine checks.
- Clean installation tests.

Run computationally intensive benchmarks in separate reproducible workflows when necessary.

Do not declare compatibility with untested Python environments.

# 37. Development milestones

## Milestone A — Classical Probability Foundations

Modules 1–10.

Deliver:

- Markov chains.
- Random walks.
- Classification.
- Recurrence.
- Absorption.
- First passage.
- Stationarity.
- Mixing.
- Continuous-time processes.
- Selected Lévy processes.

## Milestone B — Spectral Graph Mathematics

Modules 11–12.

Deliver:

- Graph operators.
- Hamiltonians.
- Laplacians.
- Spectral analysis.
- Sparse numerical foundations.

## Milestone C — Quantum Walk Core

Modules 13–15.

Deliver:

- Discrete-time quantum walks.
- Continuous-time quantum walks.
- Szegedy quantum walks.

## Milestone D — Advanced Quantum Probability

Modules 16–19.

Deliver:

- Quantum first detection.
- Quantum recurrence.
- Quantum mixing.
- Quantum search.

## Milestone E — Open Quantum Systems

Modules 20–21.

Deliver:

- Density matrix evolution.
- Lindblad dynamics.
- Noise and decoherence.

## Milestone F — Applied Networks & Computing

Modules 22–24.

Deliver:

- Biological network analysis.
- Sparse and distributed simulation.
- Quantum execution interfaces.
- PhaseTensor interoperability.

## Milestone G — Scientific Consolidation

Module 25.

Deliver:

- Independent validation.
- Scientific benchmarks.
- Six case studies.
- Complete documentation.
- Release-quality package.
- Final acceptance report.

Milestones may remain internal.

Do not inflate version numbers to reflect the number of implemented modules.

# 38. Scientific traceability matrix

Maintain a matrix of all 25 functional areas.

For each area record:

- Module identifier.
- Mathematical specification.
- Source files.
- Public API.
- Literature references.
- Associated tests.
- Independent mathematical validation.
- Benchmark evidence.
- Scientific limitations.
- Release status.

Distinguish:

- Planned.
- Implemented.
- Tested.
- Validated.
- Partially validated.
- Released.
- Blocked.

Do not equate implementation with scientific acceptance.

# 39. Release acceptance criteria

MarkovQ is ready for initial release when:

1. The declared 25 functional areas have implementation and evidence appropriate to their documented scope.
2. Classical and quantum dynamics are mathematically distinguished.
3. Numerical implementations agree with independent references.
4. Quantum measurement protocols are explicit.
5. Classical and quantum hitting-time definitions are not conflated.
6. Infinite-state conclusions are not inferred from arbitrary finite truncations.
7. Szegedy operator construction is verified.
8. Open-system physicality is checked.
9. Scientific benchmark results are reproducible.
10. Biological network results are evaluated against relevant classical baselines.
11. PhaseTensor interoperability is verified.
12. Statement coverage reaches at least 95%.
13. Branch coverage reaches at least 90%.
14. Ruff and Mypy pass.
15. Mandatory CI jobs pass.
16. Documentation and traceability are complete.
17. Wheel and source distributions pass strict validation.
18. Clean installation works outside the repository.
19. All remaining scientific limitations are documented.

Declare any unmet criterion explicitly.

Do not conceal incomplete modules or unsupported assumptions.

# 40. Publication process

Proposed initial release:

v0.1.0

Choose the actual release version according to API maturity.

Obtain explicit project-owner approval before publication.

Configure PyPI Trusted Publishing.

Verify:

- Approved source commit.
- Release tag.
- Version metadata.
- GitHub Release.
- Wheel.
- Source distribution.
- SHA-256 integrity.
- PyPI distribution.
- Public installation.
- Public API smoke tests.

The GitHub and PyPI distributions must match the approved artifact build.

Do not overwrite published versions.

Do not claim publication when a publishing job was skipped.

# 41. Final execution report

Report:

- Repository URL.
- Branch.
- Commit SHA.
- Version.
- Implemented modules.
- Independently validated modules.
- Actual test count.
- Statement coverage.
- Branch coverage.
- Ruff status.
- Mypy status.
- Benchmark results.
- Numerical accuracy.
- Statistical uncertainty.
- Hardware evidence status.
- Biological network validation.
- PhaseTensor interoperability.
- Computational scaling.
- Release artifacts.
- SHA-256 digests.
- Scientific limitations.
- Remaining blockers.

Do not use arbitrary completion percentages.

# 42. Operating methodology

Proceed autonomously with research, mathematical specification, implementation, testing and documentation.

Do not stop at the roadmap.

Before implementing each module:

1. Establish its mathematical definition.
2. Identify valid input domains.
3. Identify necessary assumptions.
4. Choose independent validation methods.
5. Define measurable acceptance criteria.
6. Implement the functionality.
7. Execute tests and benchmarks.
8. Document adverse results and limitations.

Do not introduce unrelated features.

Keep the project focused on classical stochastic processes, quantum walks, spectral graph dynamics and their scientifically justified applications.

The final objective is to establish MarkovQ as a rigorous, reproducible open-source scientific framework connecting classical probability mathematics with quantum graph dynamics.
## Development checkpoint — numerical consolidation of modules 01–20

The [consolidation review](docs/consolidation-review.md) records the evidence and limits for R01–R08. The next five modules remain outside this checkpoint.

## Development checkpoint — modules 21–25

The [applied module index](docs/applied.md) records implemented capabilities, evidence and limits. Publication remains subject to owner approval.
