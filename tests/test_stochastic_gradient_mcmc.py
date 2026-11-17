from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import solve_discrete_lyapunov
from scipy.special import expit

from Stochastic_Gradient_MCMC import (
    gaussian_prior_grad,
    gradient_noise_variance,
    linear_loglik_grad,
    logistic_loglik_grad,
    minibatch_gradient,
    sghmc,
    sgld,
    weighted_moments,
)


def no_data(theta, idx):
    """A log-likelihood gradient of zero, so the target is the N(0, prior_var) prior alone."""
    return np.zeros_like(theta)


def small_regression(n: int = 400, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 2))
    y = X @ np.array([1.0, -0.5]) + rng.standard_normal(n)
    return X, y


def exact_posterior(X, y, prior_var: float, noise_var: float = 1.0):
    """Conjugate Gaussian posterior of Bayesian linear regression."""
    S = np.linalg.inv(X.T @ X / noise_var + np.eye(X.shape[1]) / prior_var)
    return S @ X.T @ y / noise_var, S


# ---------------------------------------------------------------- gradients


def test__logistic_loglik_grad__given_coefficients__matches_finite_differences() -> None:
    # ARRANGE
    rng = np.random.default_rng(1)
    X = rng.standard_normal((50, 3))
    y = (rng.uniform(size=50) < 0.5).astype(float)
    theta, h, idx = np.array([0.3, -0.2, 0.5]), 1e-6, np.arange(50)

    def loglik(t):
        z = X @ t
        return np.sum(y * z - np.logaddexp(0.0, z))

    # ACT
    g = logistic_loglik_grad(X, y)(theta, idx)

    # ASSERT
    numeric = [(loglik(theta + h * e) - loglik(theta - h * e)) / (2 * h) for e in np.eye(3)]
    np.testing.assert_allclose(g, numeric, rtol=1e-6)


def test__linear_loglik_grad__given_hand_example__returns_scaled_residual_sum() -> None:
    # ARRANGE: residuals are 1 - 2 = -1 and 3 - 1 = 2 at theta = (1, 1).
    X = np.array([[1.0, 1.0], [0.0, 1.0]])
    y = np.array([1.0, 3.0])

    # ACT
    g = linear_loglik_grad(X, y, noise_var=2.0)(np.array([1.0, 1.0]), np.array([0, 1]))

    # ASSERT: (x_1 * (-1) + x_2 * 2) / 2 = ((-1, -1) + (0, 2)) / 2.
    np.testing.assert_allclose(g, [-0.5, 0.5])


def test__minibatch_gradient__given_full_batch__equals_full_gradient() -> None:
    # ARRANGE
    X, y = small_regression()
    theta = np.array([0.2, 0.4])
    full = -theta / 5.0 + X.T @ (y - X @ theta)

    # ACT
    g = minibatch_gradient(gaussian_prior_grad(5.0), linear_loglik_grad(X, y, 1.0), len(y), len(y),
                           np.random.default_rng(0))(theta)

    # ASSERT
    np.testing.assert_allclose(g, full)


def test__minibatch_gradient__given_many_draws__is_unbiased_for_full_gradient() -> None:
    # ARRANGE
    X, y = small_regression()
    theta = np.array([0.2, 0.4])
    full = -theta / 5.0 + X.T @ (y - X @ theta)
    estimate = minibatch_gradient(gaussian_prior_grad(5.0), linear_loglik_grad(X, y, 1.0), len(y), 10,
                                  np.random.default_rng(1))

    # ACT
    draws = np.array([estimate(theta) for _ in range(40000)])

    # ASSERT: compare with 4 Monte Carlo standard errors.
    se = draws.std(axis=0) / np.sqrt(len(draws))
    assert np.all(np.abs(draws.mean(axis=0) - full) < 4 * se)


def test__weighted_moments__given_hand_example__returns_weighted_mean_and_sd() -> None:
    # ARRANGE: weights 1 and 3 on the values 0 and 4.
    samples = np.array([[0.0], [4.0]])

    # ACT
    mean, sd = weighted_moments(samples, [1.0, 3.0])

    # ASSERT: mean 3; variance 0.25 * 9 + 0.75 * 1 = 3.
    assert mean[0] == pytest.approx(3.0)
    assert sd[0] == pytest.approx(np.sqrt(3.0))


# ---------------------------------------------------------------- sgld


def test__sgld__given_gaussian_target_and_full_gradient__matches_exact_discretised_variance() -> None:
    # For U = theta^2 / 2, theta' = (1 - eps/2) theta + sqrt(eps) z has stationary
    # variance 1 / (1 - eps/4), not 1: the discretisation bias of unadjusted Langevin.
    # ARRANGE
    eps = 0.5

    # ACT
    res = sgld(gaussian_prior_grad(1.0), no_data, 1, np.zeros(1), 200000, 1, eps, burn_in=1000, seed=2)

    # ASSERT
    assert res.samples.var() == pytest.approx(1 / (1 - eps / 4), rel=0.03)
    assert res.samples.mean() == pytest.approx(0.0, abs=0.03)


def test__sgld__given_linear_regression_large_batch_small_step__recovers_exact_posterior_mean() -> None:
    # ARRANGE
    X, y = small_regression()
    mean, S = exact_posterior(X, y, prior_var=5.0)

    # ACT
    res = sgld(gaussian_prior_grad(5.0), linear_loglik_grad(X, y, 1.0), len(y), np.zeros(2), 30000, 200,
               step_size=2e-4, burn_in=2000, seed=3)

    # ASSERT: within a quarter of a posterior standard deviation.
    np.testing.assert_allclose(res.samples.mean(axis=0), mean, atol=0.25 * np.sqrt(np.diag(S)).min())


def test__sgld__given_step_schedule__records_step_used_for_each_stored_state() -> None:
    # ARRANGE / ACT
    res = sgld(gaussian_prior_grad(1.0), no_data, 1, np.zeros(2), 20, 1, lambda t: 1.0 / (t + 1),
               burn_in=4, thin=5, seed=4)

    # ASSERT: stored at t = 4, 9, 14, 19.
    np.testing.assert_allclose(res.step_sizes, [1 / 5, 1 / 10, 1 / 15, 1 / 20])
    assert res.samples.shape == (4, 2)
    assert res.n_iter == 20 and res.batch_size == 1


def test__sgld__given_same_seed__returns_identical_chains() -> None:
    # ARRANGE
    X, y = small_regression()
    args = (gaussian_prior_grad(5.0), linear_loglik_grad(X, y, 1.0), len(y), np.zeros(2), 200, 20, 1e-4)

    # ACT / ASSERT
    np.testing.assert_array_equal(sgld(*args, seed=5).samples, sgld(*args, seed=5).samples)


# ---------------------------------------------------------------- sghmc


def test__sghmc__given_gaussian_target_and_full_gradient__matches_exact_stationary_variance() -> None:
    # With g(theta) = -theta, the state (theta, v) follows a linear recursion s' = A s + w, so
    # its stationary covariance solves the discrete Lyapunov equation P = A P A' + Q.
    # ARRANGE
    eta, alpha = 0.05, 0.3
    A = np.array([[1.0, 1.0], [-eta, 1 - alpha - eta]])
    P = solve_discrete_lyapunov(A, np.diag([0.0, 2 * alpha * eta]))

    # ACT
    res = sghmc(gaussian_prior_grad(1.0), no_data, 1, np.zeros(1), 300000, 1, eta, alpha, burn_in=2000, seed=6)

    # ASSERT
    assert res.samples.var() == pytest.approx(P[0, 0], rel=0.05)
    assert res.samples.mean() == pytest.approx(0.0, abs=0.05)


def test__sghmc__given_noise_estimate__injects_only_the_remaining_noise() -> None:
    # With beta_hat > 0 the injected noise variance is 2 (alpha - beta_hat) eta, so with an exact
    # gradient the stationary covariance solves P = A P A' + diag(0, 2 (alpha - beta_hat) eta).
    # ARRANGE
    eta, alpha, beta_hat = 0.05, 0.3, 0.2
    A = np.array([[1.0, 1.0], [-eta, 1 - alpha - eta]])
    P = solve_discrete_lyapunov(A, np.diag([0.0, 2 * (alpha - beta_hat) * eta]))

    # ACT
    res = sghmc(gaussian_prior_grad(1.0), no_data, 1, np.zeros(1), 300000, 1, eta, alpha,
                noise_estimate=beta_hat, burn_in=2000, seed=8)

    # ASSERT
    assert res.samples.var() == pytest.approx(P[0, 0], rel=0.05)


def test__gradient_noise_variance__given_linear_regression__matches_with_replacement_formula() -> None:
    # Drawing b indices with replacement and scaling by N / b gives variance N^2 / b times the
    # (divisor-N) variance of the per-observation gradients.
    # ARRANGE
    X, y = small_regression()
    theta, b = np.array([0.5, 0.5]), 20
    per_point = X * (y - X @ theta)[:, None]
    expected = len(y) ** 2 / b * per_point.var(axis=0)

    # ACT
    V = gradient_noise_variance(gaussian_prior_grad(5.0), linear_loglik_grad(X, y, 1.0), len(y), b, theta,
                                n_draws=20000, seed=9)

    # ASSERT
    np.testing.assert_allclose(V, expected, rtol=0.05)


def test__sghmc__given_linear_regression_full_batch__recovers_exact_posterior() -> None:
    # ARRANGE
    X, y = small_regression()
    mean, S = exact_posterior(X, y, prior_var=5.0)

    # ACT
    res = sghmc(gaussian_prior_grad(5.0), linear_loglik_grad(X, y, 1.0), len(y), np.zeros(2), 40000, len(y),
                learning_rate=1e-5, friction=0.05, burn_in=4000, seed=7)

    # ASSERT
    sd = np.sqrt(np.diag(S))
    np.testing.assert_allclose(res.samples.mean(axis=0), mean, atol=0.25 * sd.min())
    np.testing.assert_allclose(res.samples.std(axis=0), sd, rtol=0.15)


@pytest.mark.parametrize("kwargs", [
    {"friction": 0.0},
    {"friction": 1.5},
    {"friction": 0.1, "noise_estimate": 0.1},
    {"friction": 0.1, "noise_estimate": -0.01},
], ids=["no-friction", "friction-above-one", "noise-equals-friction", "negative-noise"])
def test__sghmc__given_invalid_friction_or_noise_estimate__raises_value_error(kwargs: dict) -> None:
    # ACT / ASSERT
    with pytest.raises(ValueError):
        sghmc(gaussian_prior_grad(1.0), no_data, 1, np.zeros(1), 10, 1, 0.01, **kwargs)


# ---------------------------------------------------------------- shared argument checks


@pytest.mark.parametrize("sampler", [sgld, sghmc], ids=["sgld", "sghmc"])
@pytest.mark.parametrize("kwargs", [
    {"batch_size": 0},
    {"batch_size": 11},
    {"burn_in": 20},
    {"thin": 0},
    {"step": -0.1},
], ids=["batch-zero", "batch-above-n", "burn-in-too-long", "thin-zero", "negative-step"])
def test__samplers__given_invalid_settings__raise_value_error(sampler, kwargs: dict) -> None:
    # ARRANGE
    settings = {"batch_size": 5, "burn_in": 0, "thin": 1, "step": 0.01, **kwargs}
    step = settings.pop("step")
    extra = {"step_size": step} if sampler is sgld else {"learning_rate": step, "friction": 0.1}

    # ACT / ASSERT
    with pytest.raises(ValueError):
        sampler(gaussian_prior_grad(1.0), no_data, 10, np.zeros(1), 20, **settings, **extra)
