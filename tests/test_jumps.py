import numpy as np
import pytest

from markovq.jumps import compound_poisson


def test_compound_poisson_exponential():
    def jumps(rng, n):
        return rng.exponential(0.5, n)

    paths, counts = compound_poisson([0, 0.3, 0.3, 1], 10000, 2, jumps, 102)
    np.testing.assert_array_equal(paths[:, 1], paths[:, 2])
    np.testing.assert_array_equal(counts[:, 1], 0)
    assert abs(paths[:, -1].mean() - 1) < 0.05
    assert abs(paths[:, -1].var() - 1) < 0.06


@pytest.mark.parametrize("grid, rate", [([], 1), ([0, -1], 1), ([0, 1], -1)])
def test_invalid_parameters(grid, rate):
    with pytest.raises(ValueError):
        compound_poisson(grid, 1, rate, lambda rng, n: np.ones(n))


def test_invalid_sample():
    with pytest.raises(ValueError):
        compound_poisson([0, 1], 1, 100, lambda rng, n: np.ones(n + 1), 1)
