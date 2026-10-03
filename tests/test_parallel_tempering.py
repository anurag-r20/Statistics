from __future__ import annotations

import numpy as np
import pytest

from Metropolis_Hastings import gaussian_random_walk
from Parallel_Tempering import geometric_ladder, parallel_tempering


def log_bimodal(x: np.ndarray) -> float:
    """Equal mixture of N(-5, 1) and N(5, 1): plain MH with small steps stays in one mode."""
    return float(np.logaddexp(-0.5 * (x[0] + 5) ** 2, -0.5 * (x[0] - 5) ** 2))


def test__geometric_ladder__given_several_chains__runs_from_one_down_to_beta_min() -> None:
    # ACT
    betas = geometric_ladder(5, beta_min=0.01)

    # ASSERT
    assert betas[0] == pytest.approx(1.0)
    assert betas[-1] == pytest.approx(0.01)
    assert np.all(np.diff(betas) < 0)


def test__geometric_ladder__given_one_chain__returns_only_the_target() -> None:
    # ACT / ASSERT
    np.testing.assert_array_equal(geometric_ladder(1), [1.0])


def test__parallel_tempering__given_bimodal_target__visits_both_modes_equally() -> None:
    # ARRANGE
    betas = geometric_ladder(6, beta_min=0.02)
    proposals = [gaussian_random_walk(1.0 / np.sqrt(b)) for b in betas]

    # ACT
    res = parallel_tempering(log_bimodal, x0=-5.0, n_samples=10000, betas=betas,
                             propose=proposals, burn_in=1000, seed=0)

    # ASSERT
    assert (res.samples > 0).mean() == pytest.approx(0.5, abs=0.1)


def test__parallel_tempering__given_any_run__reports_rates_for_each_chain_and_pair() -> None:
    # ARRANGE / ACT
    res = parallel_tempering(log_bimodal, x0=0.0, n_samples=200, betas=geometric_ladder(4),
                             keep_all=True, seed=1)

    # ASSERT
    assert res.acceptance_rates.shape == (4,)
    assert res.swap_rates.shape == (3,)
    assert np.all((0 <= res.swap_rates) & (res.swap_rates <= 1))
    assert res.all_samples.shape == (200, 4, 1)
    np.testing.assert_array_equal(res.all_samples[:, 0], res.samples)


@pytest.mark.parametrize("betas", [
    [0.5, 0.1],          # does not start at the target
    [1.0, 0.1, 0.5],     # not decreasing
    [1.0, 0.0],          # zero inverse temperature
])
def test__parallel_tempering__given_invalid_ladder__raises_value_error(betas: list[float]) -> None:
    # ACT / ASSERT
    with pytest.raises(ValueError):
        parallel_tempering(log_bimodal, x0=0.0, n_samples=10, betas=betas)


def test__parallel_tempering__given_wrong_number_of_proposals__raises_value_error() -> None:
    # ARRANGE
    proposals = [gaussian_random_walk(1.0)] * 2

    # ACT / ASSERT
    with pytest.raises(ValueError):
        parallel_tempering(log_bimodal, x0=0.0, n_samples=10, betas=geometric_ladder(3), propose=proposals)
