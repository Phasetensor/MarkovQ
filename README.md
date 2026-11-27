# MarkovQ

PhaseTensor's scientific Python library for classical and quantum stochastic dynamics. The classical-spectral-foundations branch implements roadmap modules 01–12. The quantum modules are planned.

Install this checkout with `python -m pip install .`. Run the scientific tests with `python -m pytest` and reproduce the statistical benchmark with `python -m benchmarks.foundations`.

```python
from markovq import absorption, line_walk

walk = line_walk(9, boundary="absorb")
result = absorption(walk.P)
print(dict(zip(result.transient, result.mean)))  # state 4: 16 steps
```

[Scientific specifications and traceability](docs/foundations.md) · [Measured benchmark results](docs/benchmark-results.json) · [Master roadmap](ROADMAP.md)

Development version 0.1.0.dev0. Apache License 2.0.
