<h1 align="center"><img src="docs/assets/markovq-logo.png" width="520" alt="MarkovQ — Classical &amp; Quantum Markov Dynamics"></h1>

PhaseTensor’s scientific Python framework for classical and quantum stochastic dynamics. MarkovQ implements modules 01–25, from classical Markov processes and spectral graph operators to quantum walks, monitored detection, graph search, Lindblad evolution, noise channels, network ranking, partitioned simulation, circuit integration and scientific reporting.

## Install MarkovQ 0.1.0

```sh
python -m pip install markovq==0.1.0
```

## Try the 25 modules

After installation, run this setup once in a Python session or notebook. Then copy any example from the tables below. Module 24 uses the optional Qiskit dependency: install it with `python -m pip install "markovq[references]==0.1.0"`.

```python
import numpy as np
import markovq as mq
from markovq import quantum as q
from markovq.batch import simulate_endpoints
from markovq.networks import rank_nodes
from markovq.noise import apply_channel, kraus_model
from markovq.reporting import render
from markovq.circuits import cycle_circuit

P = [[0.75, 0.25], [0.5, 0.5]]  # row-stochastic transition matrix
A = np.array([[0, 1], [1, 0]])  # two connected sites
v = np.array([1., 0.])          # start at site 0
target = np.diag([1., 0.])      # monitor site 0
```

### Classical processes and graph operators

| ID | Module | What you get | Python example |
|---|---|---|---|
| 01 | Markov chains | Law after five transitions | `mq.MarkovChain(P).evolve(v, 5)` |
| 02 | Classical random walks | One seeded path on five sites | `mq.line_walk(5).sample_paths([0, 0, 1, 0, 0], 8, 1, 42)` |
| 03 | State classification | Communicating classes and periods | `mq.classify_states(A)` |
| 04 | Classical recurrence | Return probability and mean return time | `mq.recurrence(A, 0)` |
| 05 | Absorption | Mean steps to either endpoint from each interior site | `mq.absorption(mq.line_walk(9, boundary='absorb').P).mean` |
| 06 | First passage | Mean steps to site 1 from each site | `mq.hitting(P, [1]).mean` |
| 07 | Stationarity | Stationary probability distribution | `mq.stationary_distributions(P)` |
| 08 | Classical mixing | Mixing time at the default TV threshold | `mq.mixing_time(P)` |
| 09 | Continuous-time Markov chains | One-hour transition matrix | `mq.birth_death([2], [1], 'hours').transition(1)` |
| 10 | Lévy processes | Mean and variance at time 1 | `mq.LevyProcess(volatility=1, intensity=2).moments(1)` |
| 11 | Graph Hamiltonians | Laplacian Hamiltonian | `mq.hamiltonian(A, kind='laplacian')` |
| 12 | Spectral analysis | Adjacency eigenvalues | `mq.spectrum(A).values` |

### Quantum dynamics

| ID | Module | What you get | Python example |
|---|---|---|---|
| 13 | Discrete-time quantum walks | Positions and probabilities after four steps | `q.line_distribution(4)` |
| 14 | Continuous-time quantum walks | Site probabilities at time 0.5 | `abs(q.continuous_walk(A, v, 0.5)) ** 2` |
| 15 | Szegedy walks | Eigenphases of a two-reflection walk | `np.angle(np.linalg.eigvals(q.szegedy([[0.75, 0.25], [0.25, 0.75]]).operator))` |
| 16 | Quantum first detection | First-detection probabilities by step | `q.first_detection(q.hadamard(), v, target, 5).first` |
| 17 | Quantum recurrence | Return occupancy by step | `q.recurrence(q.hadamard(), v, 5).occupancy` |
| 18 | Quantum mixing | Infinite-time averaged probabilities | `q.quantum_mixing(A, v, 8).limiting_average` |
| 19 | Quantum search | Success after zero, one and two oracle queries | `q.amplitude_amplification(np.ones(4) / 2, [0], [0, 1, 2]).success` |
| 20 | Open quantum dynamics | Populations during amplitude damping | `q.evolve_density(0 * A, np.diag([0, 1]), [0, 1], [np.array([[0, 1], [0, 0]])]).populations` |

### Applied modules

| ID | Module | What you get | Python example |
|---|---|---|---|
| 21 | Noise and decoherence | State after a 10% bit-flip channel | `apply_channel(target, kraus_model('bit-flip', 0.1))` |
| 22 | Biological network ranking | Quantum propagation scores on a toy contact graph | `rank_nodes([[0, 1, 0], [1, 0, 1], [0, 1, 0]], [0])['quantum']` |
| 23 | Sparse and distributed simulation | Endpoint counts from 1,000 seeded trajectories | `simulate_endpoints(P, v, 8, 1000, seed=42).counts` |
| 24 | Circuit and PhaseTensor adapters | Build a local Qiskit circuit and inspect its depth | `cycle_circuit(4, steps=2).depth()` |
| 25 | Scientific reporting | Export a computed stationary law as JSON | `render({'stationary': mq.stationary_distributions(P)})` |

For the full test suite and the six reproducible case studies, use a source checkout and follow the [applied module guide](docs/applied.md).

[Quantum API and scientific conventions](https://github.com/Phasetensor/MarkovQ/blob/v0.1.0/docs/quantum.md) · [Quantum benchmark results](https://github.com/Phasetensor/MarkovQ/blob/v0.1.0/docs/quantum-benchmarks.json)

[Scientific specifications and traceability](https://github.com/Phasetensor/MarkovQ/blob/v0.1.0/docs/foundations.md) · [Measured benchmark results](https://github.com/Phasetensor/MarkovQ/blob/v0.1.0/docs/benchmark-results.json) · [License](LICENSE)

[Numerical consolidation review](docs/consolidation-review.md) · [Precision and solver guidance](docs/numerical-stability.md) · [Scaling measurements](docs/scalability-benchmarks.md)

[Applied modules, case studies and benchmark index](docs/applied.md)

Hardware-adapter evidence uses simulators. [Final remediation](docs/final-remediation.md) · [Release notes](docs/release-notes.md).
