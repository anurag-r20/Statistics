from __future__ import annotations

import numpy as np
import pytest

from Metropolis_Hastings import gaussian_random_walk
from Simulated_Annealing import (
    geometric_cooling,
    linear_cooling,
    logarithmic_cooling,
    simulated_annealing,
)

MINIMUM = np.array([1.0, -2.0])


def quadratic(x: np.ndarray) -> float:
    return float(np.sum((x - MINIMUM) ** 2))


def test__simulated_annealing__given_quadratic_energy__finds_minimum() -> None:
    # ARRANGE / ACT
    res = simulated_annealing(quadratic, x0=[5.0, 5.0], n_iters=20000, T0=5.0,
                              propose=gaussian_random_walk(0.2), seed=0)

    # ASSERT
    np.testing.assert_allclose(res.x_best, MINIMUM, atol=0.05)
    assert res.f_best == pytest.approx(0.0, abs=1e-2)


def test__simulated_annealing__given_any_run__best_value_is_lowest_energy_visited() -> None:
    # ARRANGE / ACT
    res = simulated_annealing(quadratic, x0=[5.0, 5.0], n_iters=2000, seed=1)

    # ASSERT
    assert res.f_best == pytest.approx(quadratic(res.x_best))
    assert res.f_best <= res.energies.min()


def test__simulated_annealing__given_default_schedule__cools_to_one_thousandth_of_T0() -> None:
    # ARRANGE / ACT
    res = simulated_annealing(quadratic, x0=[0.0, 0.0], n_iters=500, T0=2.0, seed=2)

    # ASSERT
    assert res.temperatures[0] == pytest.approx(2.0)
    assert res.temperatures[-1] == pytest.approx(2e-3)
    assert np.all(np.diff(res.temperatures) < 0)


def test__simulated_annealing__given_infinite_energy_region__never_enters_it() -> None:
    # ARRANGE
    energy = lambda x: quadratic(x) if np.all(x > 0) else np.inf

    # ACT
    res = simulated_annealing(energy, x0=[3.0, 3.0], n_iters=2000, T0=5.0, seed=3)

    # ASSERT
    assert np.all(np.isfinite(res.energies))
    assert np.all(res.x_best > 0)


@pytest.mark.parametrize("schedule,t,expected", [
    (geometric_cooling(0.5), 3, 10.0 * 0.5 ** 3),
    (linear_cooling(1.0), 0, 10.0),
    (linear_cooling(1.0), 99, 1.0),
    (logarithmic_cooling(), 0, 10.0),
])
def test__cooling_schedules__given_iteration__return_expected_temperature(
    schedule, t: int, expected: float
) -> None:
    # ACT
    T = schedule(t, 10.0, 100)

    # ASSERT
    assert T == pytest.approx(expected)


@pytest.mark.parametrize("n_iters,T0", [(0, 1.0), (10, 0.0), (10, -1.0)])
def test__simulated_annealing__given_invalid_settings__raises_value_error(n_iters: int, T0: float) -> None:
    # ACT / ASSERT
    with pytest.raises(ValueError):
        simulated_annealing(quadratic, x0=[0.0, 0.0], n_iters=n_iters, T0=T0)


def test__simulated_annealing__given_start_with_infinite_energy__raises_value_error() -> None:
    # ACT / ASSERT
    with pytest.raises(ValueError):
        simulated_annealing(lambda x: np.inf, x0=[0.0], n_iters=10)
