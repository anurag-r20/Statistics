from __future__ import annotations

import numpy as np
import pytest

from Metropolis_Hastings import gaussian_random_walk
from Simulated_Tempering import geometric_ladder, simulated_tempering


def log_bimodal(x: np.ndarray) -> float:
    """Equal mixture of N(-5, 1) and N(5, 1): plain MH with small steps stays in one mode."""
    return float(np.logaddexp(-0.5 * (x[0] + 5) ** 2, -0.5 * (x[0] - 5) ** 2))


BETAS = geometric_ladder(6, beta_min=0.02)
PROPOSALS = [gaussian_random_walk(1.0 / np.sqrt(b)) for b in BETAS]


def test__simulated_tempering__given_bimodal_target__visits_both_modes_equally() -> None:
    # ARRANGE / ACT
    res = simulated_tempering(log_bimodal, x0=-5.0, n_samples=40000, betas=BETAS,
                              propose=PROPOSALS, burn_in=10000, seed=0)

    # ASSERT
    assert (res.samples > 0).mean() == pytest.approx(0.5, abs=0.1)


def test__simulated_tempering__given_adapted_weights__spends_similar_time_at_each_level() -> None:
    # The Wang-Landau weights learned during burn-in aim for equal time at every level.
    # ARRANGE / ACT
    res = simulated_tempering(log_bimodal, x0=-5.0, n_samples=20000, betas=BETAS,
                              propose=PROPOSALS, burn_in=20000, seed=1)

    # ASSERT
    fractions = res.level_counts / res.level_counts.sum()
    np.testing.assert_allclose(fractions, 1 / BETAS.size, atol=0.05)
    assert res.log_weights[0] == 0.0


def test__simulated_tempering__given_fixed_weights__returns_them_unchanged() -> None:
    # ARRANGE
    weights = 0.5 * np.log(BETAS)

    # ACT
    res = simulated_tempering(log_bimodal, x0=0.0, n_samples=100, betas=BETAS,
                              log_weights=weights, burn_in=100, seed=2)

    # ASSERT
    np.testing.assert_allclose(res.log_weights, weights)


def test__simulated_tempering__given_any_run__samples_are_the_target_level_states() -> None:
    # ARRANGE / ACT
    res = simulated_tempering(log_bimodal, x0=0.0, n_samples=1000, betas=BETAS, propose=PROPOSALS, seed=3)

    # ASSERT
    assert np.all((0 <= res.k_trace) & (res.k_trace < BETAS.size))
    np.testing.assert_array_equal(res.samples, res.x_trace[res.k_trace == 0])


@pytest.mark.parametrize("kwargs", [
    {"betas": [0.5, 0.1]},                         # does not start at the target
    {"betas": [1.0, 0.1, 0.5]},                    # not decreasing
    {"betas": [1.0, 0.5], "log_weights": [0.0]},   # one weight missing
    {"n_samples": 0},
])
def test__simulated_tempering__given_invalid_settings__raises_value_error(kwargs: dict) -> None:
    # ARRANGE
    settings = {"n_samples": 10, **kwargs}

    # ACT / ASSERT
    with pytest.raises(ValueError):
        simulated_tempering(log_bimodal, x0=0.0, **settings)
