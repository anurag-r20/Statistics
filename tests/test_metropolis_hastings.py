from __future__ import annotations

import numpy as np
import pytest

from Metropolis_Hastings import (
    double_metropolis_hastings,
    gaussian_random_walk,
    metropolis_hastings,
    mh_step,
)


def log_std_normal(x: np.ndarray) -> float:
    return -0.5 * float(x @ x)


# ---------------------------------------------------------------- gaussian_random_walk


def test__gaussian_random_walk__given_covariance_matrix__draws_steps_with_that_covariance() -> None:
    # ARRANGE
    cov = np.array([[1.0, 0.8], [0.8, 2.0]])
    propose = gaussian_random_walk(cov)
    rng = np.random.default_rng(0)
    x = np.zeros(2)

    # ACT
    steps = np.array([propose(x, rng) for _ in range(20000)])

    # ASSERT
    np.testing.assert_allclose(np.cov(steps.T), cov, atol=0.06)


# ---------------------------------------------------------------- mh_step


def test__mh_step__given_proposal_with_higher_density__always_accepts() -> None:
    # ARRANGE
    x = np.array([3.0])
    propose = lambda x, rng: np.array([0.0])    # the mode of the target

    # ACT
    accepted = [mh_step(x, log_std_normal(x), log_std_normal, propose, np.random.default_rng(s))[2]
                for s in range(50)]

    # ASSERT
    assert all(accepted)


def test__mh_step__given_proposal_outside_support__rejects_and_keeps_state() -> None:
    # ARRANGE
    log_target = lambda x: 0.0 if x[0] > 0 else -np.inf
    x = np.array([1.0])
    propose = lambda x, rng: np.array([-1.0])

    # ACT
    x_next, logp_next, accepted = mh_step(x, 0.0, log_target, propose, np.random.default_rng(0))

    # ASSERT
    assert not accepted
    assert x_next[0] == 1.0
    assert logp_next == 0.0


def test__mh_step__given_beta_zero__accepts_any_proposal_inside_support() -> None:
    # beta = 0 flattens the target to uniform, which is what the hottest level of a
    # tempering ladder relies on. A much lower density must still be accepted.
    # ARRANGE
    x = np.array([0.0])
    propose = lambda x, rng: np.array([30.0])

    # ACT
    accepted = [mh_step(x, 0.0, log_std_normal, propose, np.random.default_rng(s), beta=0.0)[2]
                for s in range(50)]

    # ASSERT
    assert all(accepted)


def test__mh_step__given_accepted_move__returns_untempered_log_density() -> None:
    # Callers that change beta between steps reuse the returned value, so it must be
    # log pi(x), not beta * log pi(x).
    # ARRANGE
    x = np.array([2.0])
    propose = lambda x, rng: np.array([1.0])

    # ACT
    _, logp_next, accepted = mh_step(x, log_std_normal(x), log_std_normal, propose,
                                     np.random.default_rng(0), beta=0.5)

    # ASSERT
    assert accepted
    assert logp_next == pytest.approx(-0.5)


# ---------------------------------------------------------------- metropolis_hastings


def test__metropolis_hastings__given_standard_normal_target__recovers_mean_and_variance() -> None:
    # ARRANGE / ACT
    res = metropolis_hastings(log_std_normal, x0=0.0, n_samples=20000,
                              propose=gaussian_random_walk(2.4), burn_in=1000, seed=0)

    # ASSERT
    assert res.samples.mean() == pytest.approx(0.0, abs=0.06)
    assert res.samples.var() == pytest.approx(1.0, rel=0.06)


def test__metropolis_hastings__given_asymmetric_proposal_with_correction__recovers_gamma_mean() -> None:
    # A multiplicative log-normal proposal is asymmetric. Without the Hastings correction
    # the chain would sample pi(x) * x, i.e. Gamma(4, 1) with mean 4 instead of 3.
    # ARRANGE
    log_gamma3 = lambda x: 2 * np.log(x[0]) - x[0] if x[0] > 0 else -np.inf
    propose = lambda x, rng: x * np.exp(0.5 * rng.standard_normal(x.size))
    log_q = lambda to, frm: -np.log(to[0]) - 0.5 * ((np.log(to[0]) - np.log(frm[0])) / 0.5) ** 2

    # ACT
    res = metropolis_hastings(log_gamma3, x0=1.0, n_samples=20000, propose=propose,
                              log_proposal=log_q, burn_in=1000, seed=1)

    # ASSERT
    assert res.samples.mean() == pytest.approx(3.0, rel=0.05)


def test__metropolis_hastings__given_two_dimensional_target__returns_one_row_per_sample() -> None:
    # ARRANGE / ACT
    res = metropolis_hastings(log_std_normal, x0=[0.0, 0.0], n_samples=100, burn_in=10, thin=3, seed=0)

    # ASSERT
    assert res.samples.shape == (100, 2)
    assert res.log_target.shape == (100,)
    assert 0.0 <= res.acceptance_rate <= 1.0


def test__metropolis_hastings__given_same_seed__returns_identical_chains() -> None:
    # ARRANGE / ACT
    a = metropolis_hastings(log_std_normal, x0=0.0, n_samples=200, seed=42)
    b = metropolis_hastings(log_std_normal, x0=0.0, n_samples=200, seed=42)

    # ASSERT
    np.testing.assert_array_equal(a.samples, b.samples)


@pytest.mark.parametrize("kwargs", [
    {"n_samples": 0},
    {"n_samples": 10, "burn_in": -1},
    {"n_samples": 10, "thin": 0},
])
def test__metropolis_hastings__given_invalid_run_lengths__raises_value_error(kwargs: dict) -> None:
    # ACT / ASSERT
    with pytest.raises(ValueError):
        metropolis_hastings(log_std_normal, x0=0.0, **kwargs)


def test__metropolis_hastings__given_start_outside_support__raises_value_error() -> None:
    # ARRANGE
    log_target = lambda x: 0.0 if x[0] > 0 else -np.inf

    # ACT / ASSERT
    with pytest.raises(ValueError):
        metropolis_hastings(log_target, x0=-1.0, n_samples=10)


# ---------------------------------------------------------------- double_metropolis_hastings
#
# Test model with a known posterior: x_i ~ N(0, 1/theta) with unnormalised likelihood
# g(x, theta) = exp(-theta * sum(x^2) / 2) and a Gamma(a, b) prior, so the posterior is
# Gamma(a + n/2, b + sum(x^2)/2). Drawing the auxiliary data exactly makes double MH
# exact, so its samples must match this posterior.

A_PRIOR, B_PRIOR = 2.0, 1.0


def gamma_log_prior(theta: np.ndarray) -> float:
    return (A_PRIOR - 1) * np.log(theta[0]) - B_PRIOR * theta[0] if theta[0] > 0 else -np.inf


def normal_log_unnorm_lik(x: np.ndarray, theta: np.ndarray) -> float:
    return -0.5 * theta[0] * float(np.sum(x ** 2))


def exact_normal_draw(x: np.ndarray, theta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return rng.normal(0.0, 1.0 / np.sqrt(theta[0]), size=x.shape)


def test__double_metropolis_hastings__given_exact_auxiliary_draws__recovers_known_posterior_mean() -> None:
    # ARRANGE
    data = np.random.default_rng(0).normal(0.0, 1.0 / np.sqrt(2.0), size=50)
    posterior_mean = (A_PRIOR + data.size / 2) / (B_PRIOR + np.sum(data ** 2) / 2)

    # ACT
    res = double_metropolis_hastings(gamma_log_prior, normal_log_unnorm_lik, data, theta0=1.0,
                                     n_samples=8000, propose=gaussian_random_walk(0.6),
                                     aux_sampler=exact_normal_draw, burn_in=500, seed=1)

    # ASSERT
    assert res.samples.mean() == pytest.approx(posterior_mean, rel=0.05)


def test__double_metropolis_hastings__given_asymmetric_proposal_with_correction__recovers_known_posterior_mean() -> None:
    # Same check with a multiplicative proposal theta' = theta * exp(sigma Z), whose density
    # is proportional to 1/theta'; the log_proposal correction must make it exact too.
    # ARRANGE
    data = np.random.default_rng(0).normal(0.0, 1.0 / np.sqrt(2.0), size=50)
    posterior_mean = (A_PRIOR + data.size / 2) / (B_PRIOR + np.sum(data ** 2) / 2)
    propose = lambda theta, rng: theta * np.exp(0.4 * rng.standard_normal(theta.size))
    log_q = lambda to, frm: -np.log(to[0])

    # ACT
    res = double_metropolis_hastings(gamma_log_prior, normal_log_unnorm_lik, data, theta0=1.0,
                                     n_samples=8000, propose=propose, log_proposal=log_q,
                                     aux_sampler=exact_normal_draw, burn_in=500, seed=4)

    # ASSERT
    assert res.samples.mean() == pytest.approx(posterior_mean, rel=0.05)


def test__double_metropolis_hastings__given_proposals_outside_prior_support__keeps_theta_in_support() -> None:
    # ARRANGE
    data = np.random.default_rng(0).normal(size=20)

    # ACT
    res = double_metropolis_hastings(gamma_log_prior, normal_log_unnorm_lik, data, theta0=0.05,
                                     n_samples=500, propose=gaussian_random_walk(1.0),
                                     aux_sampler=exact_normal_draw, seed=2)

    # ASSERT
    assert np.all(res.samples > 0)


def test__double_metropolis_hastings__given_default_inner_chain__returns_finite_samples() -> None:
    # ARRANGE
    data = np.random.default_rng(0).normal(size=10)

    # ACT
    res = double_metropolis_hastings(gamma_log_prior, normal_log_unnorm_lik, data, theta0=1.0,
                                     n_samples=200, propose=gaussian_random_walk(0.5),
                                     aux_propose=gaussian_random_walk(0.5), n_aux_steps=20, seed=3)

    # ASSERT
    assert res.samples.shape == (200, 1)
    assert np.all(np.isfinite(res.samples)) and np.all(res.samples > 0)


def test__double_metropolis_hastings__given_no_auxiliary_sampler__raises_value_error() -> None:
    # ACT / ASSERT
    with pytest.raises(ValueError):
        double_metropolis_hastings(gamma_log_prior, normal_log_unnorm_lik, np.zeros(3), theta0=1.0,
                                   n_samples=10, propose=gaussian_random_walk(0.5))
