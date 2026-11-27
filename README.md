# MarkovQ

PhaseTensor’s scientific Python framework for classical and quantum stochastic dynamics. The [master roadmap](ROADMAP.md) defines 25 modules. The [development branch](https://github.com/Phasetensor/MarkovQ/tree/classical-spectral-foundations) implements modules 01–20, from classical Markov processes and spectral graph operators to quantum walks, monitored detection, graph search and Lindblad evolution.

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
