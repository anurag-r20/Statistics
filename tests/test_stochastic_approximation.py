from __future__ import annotations

import numpy as np
import pytest
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from Stochastic_Approximation import (
    classification_error,
    constant_step,
    logistic_gradient,
    logistic_loss,
    minibatch_sgd,
    polynomial_step,
    robbins_monro,
    sgd_logistic_regression,
)


def draws_from(values):
    """H(theta, rng) = x_k - theta, where x_k runs through a fixed list (no randomness)."""
    it = iter(values)
    return lambda theta, rng: next(it) - theta


def small_logistic_data(n: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(n), rng.standard_normal((n, 2))])
    y = (rng.uniform(size=n) < expit(X @ np.array([0.3, 1.0, -1.0]))).astype(float)
    return X, y


# ---------------------------------------------------------------- step sizes


@pytest.mark.parametrize("step,k,expected", [
    (constant_step(0.3), 7, 0.3),
    (polynomial_step(2.0), 4, 0.5),
    (polynomial_step(1.0, power=0.5), 16, 0.25),
    (polynomial_step(6.0, power=1.0, offset=2.0), 1, 2.0),
], ids=["constant", "a/k", "k^-0.5", "a/(k+offset)"])
def test__step_sizes__given_iteration__return_hand_computed_value(step, k: int, expected: float) -> None:
    # ACT / ASSERT
    assert step(k) == pytest.approx(expected)


# ---------------------------------------------------------------- robbins_monro


def test__robbins_monro__given_steps_one_over_k__returns_running_mean_of_draws() -> None:
    # theta_k = theta_{k-1} + (x_k - theta_{k-1}) / k is the running mean, whatever theta_0 is.
    # ARRANGE
    x = np.array([3.0, -1.0, 4.0, 1.0, 5.0, 9.0])

    # ACT
    res = robbins_monro(draws_from(x), theta0=100.0, n_iter=x.size, step_size=polynomial_step(1.0))

    # ASSERT
    np.testing.assert_allclose(res.path[1:], np.cumsum(x) / np.arange(1, x.size + 1))


def test__robbins_monro__given_noiseless_linear_h_and_constant_step__shrinks_error_geometrically() -> None:
    # With h(theta) = 3 - theta and a_k = 0.5, theta_k - 3 = 0.5^k (theta_0 - 3).
    # ARRANGE / ACT
    res = robbins_monro(lambda th, rng: 3.0 - th, theta0=11.0, n_iter=10, step_size=constant_step(0.5))

    # ASSERT
    np.testing.assert_allclose(res.path - 3.0, 8.0 * 0.5 ** np.arange(11))


def test__robbins_monro__given_replicates__matches_exact_mean_squared_error_recursion() -> None:
    # For H = X - theta with X ~ N(mu, sigma^2): MSE_{k} = (1 - a_k)^2 MSE_{k-1} + a_k^2 sigma^2.
    # ARRANGE
    mu, sigma, a = 2.0, 1.5, polynomial_step(2.0)
    mse = 4.0                                   # (theta_0 - mu)^2 with theta_0 = 0
    for k in range(1, 501):
        mse = (1 - a(k)) ** 2 * mse + a(k) ** 2 * sigma ** 2

    # ACT
    res = robbins_monro(lambda th, rng: rng.normal(mu, sigma, th.shape) - th, np.zeros(20000), 500, a, seed=0)

    # ASSERT
    assert np.mean((res.theta - mu) ** 2) == pytest.approx(mse, rel=0.05)


def test__robbins_monro__given_record_at__stores_only_those_iterations() -> None:
    # ARRANGE / ACT
    res = robbins_monro(lambda th, rng: 1.0 - th, np.zeros(3), 50, constant_step(0.1), record_at=[50, 0, 10])

    # ASSERT
    np.testing.assert_array_equal(res.iterations, [0, 10, 50])
    assert res.path.shape == (3, 3)
    np.testing.assert_allclose(res.path[0], 0.0)
    np.testing.assert_allclose(res.path[-1], res.theta)


@pytest.mark.parametrize("kwargs", [{"n_iter": 0}, {"n_iter": 5, "record_at": [6]}, {"n_iter": 5, "record_at": [-1]}],
                         ids=["no-iterations", "record-past-end", "record-negative"])
def test__robbins_monro__given_invalid_settings__raises_value_error(kwargs: dict) -> None:
    # ACT / ASSERT
    with pytest.raises(ValueError):
        robbins_monro(lambda th, rng: -th, 0.0, step_size=constant_step(0.1), **kwargs)


# ---------------------------------------------------------------- logistic loss and gradient


def test__logistic_loss__given_zero_coefficients__equals_log_two() -> None:
    # ARRANGE
    X, y = small_logistic_data()

    # ACT / ASSERT: every predicted probability is 1/2, so each term is log 2.
    assert logistic_loss(np.zeros(3), X, y) == pytest.approx(np.log(2.0))


def test__logistic_gradient__given_penalty__matches_finite_differences_of_loss() -> None:
    # ARRANGE
    X, y = small_logistic_data()
    theta, lam, h = np.array([0.2, -0.4, 0.7]), 0.05, 1e-6

    # ACT
    g = logistic_gradient(theta, X, y, lam)

    # ASSERT
    numeric = [(logistic_loss(theta + h * e, X, y, lam) - logistic_loss(theta - h * e, X, y, lam)) / (2 * h)
               for e in np.eye(3)]
    np.testing.assert_allclose(g, numeric, rtol=1e-6, atol=1e-9)


def test__classification_error__given_hand_example__returns_fraction_misclassified() -> None:
    # ARRANGE: predictions are 1, 0, 1, 0; the second and third are wrong.
    X = np.array([[1.0], [-1.0], [2.0], [-3.0]])
    y = np.array([1.0, 1.0, 0.0, 0.0])

    # ACT / ASSERT
    assert classification_error(np.array([1.0]), X, y) == pytest.approx(0.5)


# ---------------------------------------------------------------- minibatch_sgd


def test__minibatch_sgd__given_any_run__uses_every_observation_once_per_epoch() -> None:
    # ARRANGE
    seen = []

    def grad(theta, idx):
        seen.append(idx.copy())
        return np.zeros_like(theta)

    # ACT
    minibatch_sgd(grad, np.zeros(1), n=10, batch_size=4, step_size=constant_step(0.1), n_epochs=3, seed=0)

    # ASSERT: ceil(10 / 4) = 3 minibatches per epoch, of sizes 4, 4, 2.
    assert len(seen) == 9
    for epoch in range(3):
        batches = seen[3 * epoch:3 * epoch + 3]
        assert [len(b) for b in batches] == [4, 4, 2]
        np.testing.assert_array_equal(np.sort(np.concatenate(batches)), np.arange(10))


def test__sgd_logistic_regression__given_full_batch_and_constant_step__equals_gradient_descent() -> None:
    # ARRANGE
    X, y = small_logistic_data()
    theta = np.zeros(3)
    for _ in range(25):
        theta = theta - 0.5 * logistic_gradient(theta, X, y, 0.01)

    # ACT
    res = sgd_logistic_regression(X, y, batch_size=len(y), step_size=constant_step(0.5), n_epochs=25,
                                  lam=0.01, seed=1)

    # ASSERT
    np.testing.assert_allclose(res.theta, theta, rtol=1e-10)


def test__sgd_logistic_regression__given_penalty__one_step_is_shrinkage_plus_data_gradient() -> None:
    # The derived update: theta <- (1 - 2 eta lam) theta + eta (1/|B|) sum_B (y_i - sigmoid(x_i'theta)) x_i.
    # ARRANGE
    X, y = small_logistic_data(n=8)
    theta0, eta, lam = np.array([0.5, -0.2, 0.1]), 0.3, 0.4
    expected = (1 - 2 * eta * lam) * theta0 + eta * X.T @ (y - expit(X @ theta0)) / len(y)

    # ACT
    res = sgd_logistic_regression(X, y, batch_size=8, step_size=constant_step(eta), n_epochs=1,
                                  lam=lam, theta0=theta0, seed=2)

    # ASSERT
    np.testing.assert_allclose(res.theta, expected, rtol=1e-12)


def test__sgd_logistic_regression__given_decaying_steps__converges_to_independent_ridge_solution() -> None:
    # Minimising (1/n) sum loss + lam ||theta||^2 is scikit-learn's problem with C = 1 / (2 n lam).
    # ARRANGE
    X, y = small_logistic_data(n=400)
    lam = 0.01
    reference = LogisticRegression(C=1 / (2 * len(y) * lam), fit_intercept=False, tol=1e-12,
                                   max_iter=10000).fit(X, y).coef_.ravel()

    # ACT
    res = sgd_logistic_regression(X, y, batch_size=10, step_size=lambda t: 1.0 / (1 + t / 40) ** 0.7,
                                  n_epochs=400, lam=lam, seed=3)

    # ASSERT
    np.testing.assert_allclose(res.theta, reference, atol=0.02)


def test__sgd_logistic_regression__given_test_data__records_metrics_each_epoch() -> None:
    # ARRANGE
    X, y = small_logistic_data()

    # ACT
    res = sgd_logistic_regression(X, y, batch_size=20, step_size=constant_step(0.1), n_epochs=5,
                                  X_test=X, y_test=y, seed=4)

    # ASSERT
    np.testing.assert_array_equal(res.epochs, np.arange(6))
    assert res.history["train_loss"].shape == (6,)
    assert res.history["test_error"].shape == (6,)
    assert res.history["train_loss"][0] == pytest.approx(np.log(2.0))
    assert res.history["train_loss"][-1] < res.history["train_loss"][0]


@pytest.mark.parametrize("batch_size", [0, 201], ids=["zero", "larger-than-n"])
def test__sgd_logistic_regression__given_invalid_batch_size__raises_value_error(batch_size: int) -> None:
    # ARRANGE
    X, y = small_logistic_data()

    # ACT / ASSERT
    with pytest.raises(ValueError):
        sgd_logistic_regression(X, y, batch_size=batch_size, step_size=constant_step(0.1), n_epochs=1)


def test__sgd_logistic_regression__given_labels_not_zero_one__raises_value_error() -> None:
    # ARRANGE
    X, y = small_logistic_data()

    # ACT / ASSERT
    with pytest.raises(ValueError):
        sgd_logistic_regression(X, 2 * y - 1, batch_size=10, step_size=constant_step(0.1), n_epochs=1)
