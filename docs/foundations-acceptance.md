# Foundations acceptance record

Scope: development modules 01–12 only. Refer to the [module matrix](foundations.md) for the public API, domain, independent references and numerical limitations.

- The test suite includes exhaustive finite-path enumeration, Boolean reachability, closed-form absorption, stationarity, finite and countable-state return benchmarks, a two-state CTMC, Poisson counts, increment moments and characteristic functions, known graph spectra and spectral projector checks.
- On Python 3.12 the 89 tests passed with 99.84% statement coverage and 99.40% branch coverage. Ruff lint, Ruff formatting, Mypy, wheel and sdist builds, strict Twine checks and an out-of-checkout wheel installation passed. [CI evidence](https://github.com/Phasetensor/MarkovQ/actions/runs/46065668905).
- The four principal Monte Carlo examples use 10,000 trajectories each. Sampling estimates, standard errors, reference values and software versions appear in [the benchmark record](benchmark-results.json). The two-state CTMC terminal occupancy estimate was 0.6229 versus 0.6334753 (estimated standard error 0.00485); this was the largest standardized discrepancy of the four examples.
- Finite-chain recurrence follows communicating-class structure. Infinite-lattice conclusions are restricted to the documented analytical models. Graph dilation doubles the state space. Sparse input support does not imply every downstream solver preserves sparsity.
- The current development distribution is 0.1.0.dev0. This record covers classical and spectral foundations; overall acceptance of the 25-module project is pending.
