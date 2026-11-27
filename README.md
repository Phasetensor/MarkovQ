# MarkovQ

PhaseTensor’s scientific Python framework for classical and quantum stochastic dynamics. The [master roadmap](ROADMAP.md) defines 25 modules. The [development branch](https://github.com/Phasetensor/MarkovQ/tree/classical-spectral-foundations) implements the first twelve, spanning finite-state Markov models, first passage, selected Lévy processes and spectral graph operators.

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

Install pytest to run `python -m pytest`. Reproduce the statistical benchmarks with `python -m benchmarks.foundations`.

[Scientific specifications and traceability](https://github.com/Phasetensor/MarkovQ/blob/classical-spectral-foundations/docs/foundations.md) · [Measured benchmark results](https://github.com/Phasetensor/MarkovQ/blob/classical-spectral-foundations/docs/benchmark-results.json) · [License](LICENSE)
