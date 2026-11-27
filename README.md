# MarkovQ

PhaseTensor’s scientific Python framework for classical and quantum stochastic dynamics. The [master roadmap](ROADMAP.md) defines 25 modules. The [development branch](https://github.com/Phasetensor/MarkovQ/tree/classical-spectral-foundations) implements modules 01–25, from classical Markov processes and spectral graph operators to quantum walks, monitored detection, graph search, Lindblad evolution, noise channels, network ranking, partitioned simulation, circuit integration and scientific reporting.

## Try the development version

```sh
git clone --branch classical-spectral-foundations https://github.com/Phasetensor/MarkovQ.git
cd MarkovQ
python -m pip install .
```

```python
from markovq import absorption, line_walk

walk = line_walk(9, boundary="absorb")
result = absorption(walk.P)
print(dict(zip(result.transient, result.mean)))  # state 4: 16 steps
```

Install the independent reference solvers with `python -m pip install '.[references]' pytest` and run `python -m pytest`. Reproduce the benchmarks with `python -m benchmarks.foundations` and `python -m benchmarks.quantum`.

[Quantum API and scientific conventions](https://github.com/Phasetensor/MarkovQ/blob/classical-spectral-foundations/docs/quantum.md) · [Quantum benchmark results](https://github.com/Phasetensor/MarkovQ/blob/classical-spectral-foundations/docs/quantum-benchmarks.json)

[Scientific specifications and traceability](https://github.com/Phasetensor/MarkovQ/blob/classical-spectral-foundations/docs/foundations.md) · [Measured benchmark results](https://github.com/Phasetensor/MarkovQ/blob/classical-spectral-foundations/docs/benchmark-results.json) · [License](LICENSE)

[Numerical consolidation review](docs/consolidation-review.md) · [Precision and solver guidance](docs/numerical-stability.md) · [Scaling measurements](docs/scalability-benchmarks.md)

[Applied modules, case studies and benchmark index](docs/applied.md)

The 25 areas cover chains, classical walks, classification, recurrence, absorption, first passage, stationarity, mixing, continuous-time chains, Lévy models, graph Hamiltonians, spectral analysis, discrete and continuous quantum walks, Szegedy walks, monitored detection, quantum recurrence, quantum mixing, search, open dynamics, noise channels, biological networks, partitioned simulation, circuit adapters and scientific reporting. Hardware-adapter evidence uses simulators. [Final remediation](docs/final-remediation.md) · [Proposed release notes](docs/release-notes.md).
