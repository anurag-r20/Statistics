from __future__ import annotations

import numpy as np
import pytest

from Metropolis_Hastings import gaussian_random_walk
from SAMC import energy_regions, samc, sqrt_cooling, stochastic_approximation_annealing


# A small discrete target with a hand-computable answer: states 0..9 with energy U(x) = 0.5 x,
# each state its own subregion, so P(E_i) = exp(-0.5 i) / sum_j exp(-0.5 j).
N_STATES = 10


def discrete_energy(x: np.ndarray) -> float:
    return 0.5 * x[0]


def state_region(x: np.ndarray, u: float) -> int:
    return int(x[0]) if 0 <= x[0] < N_STATES else -1


def step_left_or_right(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return x + (1.0 if rng.uniform() < 0.5 else -1.0)


def exact_discrete_probabilities() -> np.ndarray:
    p = np.exp(-0.5 * np.arange(N_STATES))
    return p / p.sum()


def log_bimodal_energy(x: np.ndarray) -> float:
    """Energy of 0.5 N(-5, 1) + 0.5 N(5, 1): plain MH with unit steps stays in one mode."""
    return -float(np.logaddexp(-0.5 * (x[0] + 5) ** 2, -0.5 * (x[0] - 5) ** 2))


# ---------------------------------------------------------------- energy_regions


@pytest.mark.parametrize("u,expected", [
    (0.5, 0),   # below the first level
    (1.0, 0),   # E_1 includes its upper level
    (1.5, 1),
    (3.0, 2),
    (3.5, 3),   # above the last level: the open top subregion
], ids=["below", "on-first-level", "between", "on-last-level", "above"])
def test__energy_regions__given_energy__returns_subregion_from_hand_partition(u: float, expected: int) -> None:
    # ARRANGE
    region, m = energy_regions([1.0, 2.0, 3.0])

    # ACT
    j = region(np.zeros(1), u)

    # ASSERT
    assert m == 4
    assert j == expected


@pytest.mark.parametrize("levels", [[], [1.0, 1.0], [2.0, 1.0]], ids=["empty", "repeated", "decreasing"])
def test__energy_regions__given_invalid_levels__raises_value_error(levels: list[float]) -> None:
    # ACT / ASSERT
    with pytest.raises(ValueError):
        energy_regions(levels)


# ---------------------------------------------------------------- sqrt_cooling


@pytest.mark.parametrize("t,expected", [
    (1, 11.0),     # before t0 the temperature is T_low + T_high
    (100, 11.0),
    (400, 6.0),    # 1 + 10 * sqrt(100 / 400)
    (10000, 2.0),  # 1 + 10 * sqrt(100 / 10000)
])
def test__sqrt_cooling__given_iteration__returns_hand_computed_temperature(t: int, expected: float) -> None:
    # ARRANGE
    schedule = sqrt_cooling(T_high=10.0, T_low=1.0, t0=100)

    # ACT
    tau = schedule(t)

    # ASSERT
    assert tau == pytest.approx(expected)


# ---------------------------------------------------------------- samc


def test__samc__given_discrete_target__estimates_subregion_probabilities() -> None:
    # The estimate uses only the final weights, whose Monte Carlo error is still about 10%
    # after 300,000 iterations (up to 15% over six seeds), hence rtol = 0.2.
    # ARRANGE / ACT
    res = samc(discrete_energy, x0=0.0, n_iters=300000, region=state_region, n_regions=N_STATES,
               propose=step_left_or_right, t0=200, seed=0)

    # ASSERT
    np.testing.assert_allclose(res.region_probabilities(), exact_discrete_probabilities(), rtol=0.2)


def test__samc__given_non_uniform_desired_distribution__still_estimates_subregion_probabilities() -> None:
    # pi changes how often each subregion is visited, not the target, so the estimate must
    # undo it: P(E_i) is proportional to pi_i exp(theta_i), not exp(theta_i).
    # ARRANGE
    desired = np.linspace(1.0, 3.0, N_STATES)

    # ACT
    res = samc(discrete_energy, x0=0.0, n_iters=300000, region=state_region, n_regions=N_STATES,
               desired=desired, propose=step_left_or_right, t0=200, seed=7)

    # ASSERT
    np.testing.assert_allclose(res.region_probabilities(), exact_discrete_probabilities(), rtol=0.2)


def test__samc__given_discrete_target__importance_weights_recover_the_mean() -> None:
    # ARRANGE
    exact_mean = np.sum(np.arange(N_STATES) * exact_discrete_probabilities())

    # ACT
    res = samc(discrete_energy, x0=0.0, n_iters=100000, region=state_region, n_regions=N_STATES,
               propose=step_left_or_right, t0=200, burn_in=10000, seed=1)

    # ASSERT
    assert np.sum(res.importance_weights() * res.samples[:, 0]) == pytest.approx(exact_mean, rel=0.05)


def test__samc__given_desired_distribution__visits_subregions_in_that_proportion() -> None:
    # The weight updates drive the visit frequencies towards pi, whatever the target.
    # ARRANGE
    desired = np.linspace(1.0, 3.0, N_STATES)

    # ACT
    res = samc(discrete_energy, x0=0.0, n_iters=100000, region=state_region, n_regions=N_STATES,
               desired=desired, propose=step_left_or_right, t0=200, burn_in=20000, seed=2)

    # ASSERT
    frequencies = res.region_counts / res.region_counts.sum()
    np.testing.assert_allclose(frequencies, desired / desired.sum(), atol=0.02)


def test__samc__given_bimodal_target_started_in_one_mode__weights_both_modes_equally() -> None:
    # ARRANGE
    region, m = energy_regions(np.arange(0.5, 15.0, 0.5))

    # ACT
    res = samc(log_bimodal_energy, x0=-5.0, n_iters=300000, region=region, n_regions=m,
               propose=gaussian_random_walk(1.0), burn_in=30000, seed=3)

    # ASSERT
    assert np.sum(res.importance_weights() * (res.samples[:, 0] > 0)) == pytest.approx(0.5, abs=0.05)


def test__samc__given_states_outside_sample_space__never_accepts_them() -> None:
    # ARRANGE / ACT
    res = samc(discrete_energy, x0=0.0, n_iters=5000, region=state_region, n_regions=N_STATES,
               propose=step_left_or_right, seed=4)

    # ASSERT
    assert np.all((res.samples >= 0) & (res.samples < N_STATES))
    assert np.all((res.regions >= 0) & (res.regions < N_STATES))


def test__samc__given_unreachable_subregion__gives_it_zero_probability() -> None:
    # Subregion 3 of 4 lies above every reachable energy, so it is never visited.
    # ARRANGE
    region = lambda x, u: min(int(x[0]), 2)

    # ACT
    res = samc(lambda x: 0.0, x0=0.0, n_iters=2000, region=region, n_regions=4,
               propose=lambda x, rng: rng.integers(0, 3, size=1).astype(float), seed=5)
    p = res.region_probabilities()

    # ASSERT
    assert not res.visited[3]
    assert p[3] == 0.0
    assert p.sum() == pytest.approx(1.0)


def test__samc__given_same_seed__returns_identical_runs() -> None:
    # ARRANGE / ACT
    a = samc(discrete_energy, x0=0.0, n_iters=500, region=state_region, n_regions=N_STATES,
             propose=step_left_or_right, seed=42)
    b = samc(discrete_energy, x0=0.0, n_iters=500, region=state_region, n_regions=N_STATES,
             propose=step_left_or_right, seed=42)

    # ASSERT
    np.testing.assert_array_equal(a.samples, b.samples)
    np.testing.assert_array_equal(a.theta, b.theta)


@pytest.mark.parametrize("kwargs", [
    {"n_iters": 0},
    {"thin": 0},
    {"t0": 0.0},
    {"eta": 0.5},
    {"eta": 1.5},
    {"desired": [1.0, 2.0]},             # wrong length
    {"desired": [0.0] + [1.0] * 9},      # a zero entry
], ids=["no-iterations", "thin-0", "t0-0", "eta-0.5", "eta-1.5", "desired-length", "desired-zero"])
def test__samc__given_invalid_settings__raises_value_error(kwargs: dict) -> None:
    # ARRANGE
    settings = {"n_iters": 10, **kwargs}

    # ACT / ASSERT
    with pytest.raises(ValueError):
        samc(discrete_energy, x0=0.0, region=state_region, n_regions=N_STATES,
             propose=step_left_or_right, **settings)


def test__samc__given_start_outside_sample_space__raises_value_error() -> None:
    # ACT / ASSERT
    with pytest.raises(ValueError):
        samc(discrete_energy, x0=-1.0, n_iters=10, region=state_region, n_regions=N_STATES)


# ---------------------------------------------------------------- stochastic_approximation_annealing


def test__stochastic_approximation_annealing__given_quadratic_energy__finds_minimum() -> None:
    # ARRANGE
    minimum = np.array([1.0, -2.0])
    energy = lambda x: float(np.sum((x - minimum) ** 2))

    # ACT
    res = stochastic_approximation_annealing(energy, x0=[5.0, 5.0], n_iters=20000,
                                             levels=np.arange(0.1, 10.0, 0.5), T_high=2.0, t0_cool=100,
                                             propose=gaussian_random_walk(0.2), t0=500, seed=6)

    # ASSERT
    np.testing.assert_allclose(res.x_best, minimum, atol=0.1)
    assert res.f_best == pytest.approx(energy(res.x_best))
