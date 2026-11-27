"""Run from repository root with python -m benchmarks.foundations."""

import json
import platform
from time import perf_counter

import numpy as np
import scipy

from markovq import ContinuousChain, LevyProcess, lattice_paths, line_walk, spectrum
from markovq.walks import empirical_passage


def estimate(x, reference):
    return {
        "estimate": float(x.mean()),
        "reference": reference,
        "standard_error": float(x.std(ddof=1) / np.sqrt(x.size)),
    }


def run():
    start = perf_counter()
    n = 10000
    delta = lattice_paths(1, 20, n, 231)[:, -1, 0]
    ruin = line_walk(9, boundary="absorb").sample_paths(np.eye(9)[4], 320, n, 232)
    times, censor = empirical_passage(ruin, [0, 8])
    generator = ContinuousChain([[-2, 2], [1, -1]], "hour")
    rng = np.random.default_rng(233)
    endpoint = np.array([generator.events(0, 1, rng)[1][-1] for _ in range(n)])
    expected = 2 / 3 * (1 - np.exp(-3))
    jump = LevyProcess(0.1, 0.2, 2, 0.3, 0.4)
    paths, _ = jump.sample([0, 1], n, 234)
    star = np.zeros((10, 10))
    star[1:, 0] = star[0, 1:] = 1
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "trajectories": n,
        "seeds": [231, 232, 233, 234],
        "lattice_msd": estimate(delta**2, 20),
        "restricted_ruin_time": estimate(times, 16),
        "censored_ruin_paths": int(censor.sum()),
        "ctmc_endpoint": estimate(endpoint, expected),
        "ctmc_uniformization_error": float(
            abs(generator.uniformize(1) - generator.transition(1)).max()
        ),
        "levy_endpoint": estimate(paths[:, -1], jump.moments(1)[0]),
        "star_spectral_error": float(
            abs(spectrum(star).values - np.r_[-3, np.zeros(8), 3]).max()
        ),
        "elapsed_seconds": perf_counter() - start,
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
