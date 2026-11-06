from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from Expectation_Maximization import normal_mixture_em, normal_mixture_loglik


def simulated_mixture(n: int = 3000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z = rng.choice(2, size=n, p=[0.4, 0.6])
    return rng.normal(np.array([-3.0, 3.0])[z], np.array([1.0, 0.5])[z])


# ---------------------------------------------------------------- normal_mixture_loglik


def test__normal_mixture_loglik__given_two_components__matches_scipy_density() -> None:
    # ARRANGE
    x = np.array([-1.0, 0.0, 2.5])
    w, m, v = np.array([0.3, 0.7]), np.array([-1.0, 2.0]), np.array([1.0, 4.0])

    # ACT
    ll = normal_mixture_loglik(x, w, m, v)

    # ASSERT
    expected = np.sum(np.log(w[0] * norm.pdf(x, m[0], 1.0) + w[1] * norm.pdf(x, m[1], 2.0)))
    assert ll == pytest.approx(expected)


# ---------------------------------------------------------------- normal_mixture_em


def test__normal_mixture_em__given_one_component__returns_sample_mean_and_variance_in_one_step() -> None:
    # With K = 1 every responsibility is 1, so the M-step gives the normal MLE (divisor n).
    # ARRANGE
    x = np.array([1.0, 2.0, 4.0, 7.0])

    # ACT
    res = normal_mixture_em(x, 1, means=[0.0], variances=[1.0])

    # ASSERT
    assert res.weights[0] == pytest.approx(1.0)
    assert res.means[0] == pytest.approx(3.5)
    assert res.variances[0] == pytest.approx(5.25)
    assert res.log_likelihood[1] == pytest.approx(np.sum(norm.logpdf(x, 3.5, np.sqrt(5.25))))


def test__normal_mixture_em__given_starting_values__first_iteration_matches_hand_computed_m_step() -> None:
    # ARRANGE
    x = np.array([-2.0, -1.0, 0.5, 3.0, 4.0])
    w0, m0, v0 = np.array([0.5, 0.5]), np.array([-1.0, 3.0]), np.array([1.0, 2.0])
    dens = w0 * norm.pdf(x[:, None], m0, np.sqrt(v0))
    r = dens / dens.sum(axis=1, keepdims=True)
    nk = r.sum(axis=0)
    mu = (r * x[:, None]).sum(axis=0) / nk
    var = (r * (x[:, None] - mu) ** 2).sum(axis=0) / nk

    # ACT
    res = normal_mixture_em(x, 2, weights=w0, means=m0, variances=v0, max_iter=1)

    # ASSERT
    np.testing.assert_allclose(res.weights, nk / x.size)
    np.testing.assert_allclose(res.means, mu)
    np.testing.assert_allclose(res.variances, var)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test__normal_mixture_em__given_random_start__never_decreases_log_likelihood(seed: int) -> None:
    # ARRANGE
    x = simulated_mixture(500, seed=10 + seed)

    # ACT
    res = normal_mixture_em(x, 3, seed=seed, max_iter=300)

    # ASSERT
    assert np.all(np.diff(res.log_likelihood) >= -1e-9 * np.abs(res.log_likelihood[1:]))


def test__normal_mixture_em__given_tolerance__stops_at_first_small_relative_increase() -> None:
    # Documented rule: stop when the increase is below tol * (1 + |log-likelihood|).
    # ARRANGE
    tol = 1e-10

    # ACT
    res = normal_mixture_em(simulated_mixture(500, seed=8), 3, tol=tol, seed=9)

    # ASSERT
    ll = res.log_likelihood
    small = np.diff(ll) < tol * (1 + np.abs(ll[1:]))
    assert res.converged
    assert small[-1]
    assert not np.any(small[:-1])


def test__normal_mixture_em__given_well_separated_mixture__recovers_parameters() -> None:
    # ARRANGE
    x = simulated_mixture()

    # ACT
    res = normal_mixture_em(x, 2, seed=0)

    # ASSERT
    assert res.converged
    np.testing.assert_allclose(res.weights, [0.4, 0.6], atol=0.03)
    np.testing.assert_allclose(res.means, [-3.0, 3.0], atol=0.1)
    np.testing.assert_allclose(np.sqrt(res.variances), [1.0, 0.5], atol=0.05)


def test__normal_mixture_em__given_any_fit__returns_valid_probabilities_sorted_by_mean() -> None:
    # ARRANGE / ACT
    res = normal_mixture_em(simulated_mixture(800, seed=5), 3, seed=6)

    # ASSERT
    assert np.all(np.diff(res.means) > 0)
    assert res.weights.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(res.responsibilities.sum(axis=1), 1.0)


def test__normal_mixture_em__given_shifted_data__shifts_means_and_keeps_weights_and_variances() -> None:
    # Metamorphic relation: x -> x + c with starting means shifted by c.
    # ARRANGE
    x, c = simulated_mixture(600, seed=7), 10.0
    start = {"weights": [0.5, 0.5], "means": [-1.0, 1.0], "variances": [2.0, 2.0]}

    # ACT
    a = normal_mixture_em(x, 2, **start)
    b = normal_mixture_em(x + c, 2, weights=start["weights"], means=np.add(start["means"], c),
                          variances=start["variances"])

    # ASSERT
    np.testing.assert_allclose(b.means, a.means + c, atol=1e-8)
    np.testing.assert_allclose(b.weights, a.weights, atol=1e-8)
    np.testing.assert_allclose(b.variances, a.variances, atol=1e-8)


def test__normal_mixture_em__given_component_on_single_point__keeps_variance_at_floor() -> None:
    # The component started at 10 owns only that point, so its variance would collapse to 0.
    # ARRANGE
    x = np.array([0.0, 0.1, -0.1, 0.05, 10.0])

    # ACT
    res = normal_mixture_em(x, 2, means=[0.0, 10.0], variances=[1.0, 1.0], min_variance=1e-6, max_iter=50)

    # ASSERT
    assert res.variances[1] == pytest.approx(1e-6 * np.var(x))
    assert np.all(np.isfinite(res.log_likelihood))


@pytest.mark.parametrize("kwargs", [
    {"n_components": 0},
    {"n_components": 6},                                    # more components than observations
    {"n_components": 2, "weights": [0.7, 0.7]},             # weights do not sum to 1
    {"n_components": 2, "variances": [1.0, -1.0]},
    {"n_components": 2, "means": [0.0]},                    # wrong length
    {"n_components": 2, "max_iter": 0},
], ids=["zero-components", "too-many-components", "weights-sum", "negative-variance", "means-length", "no-iterations"])
def test__normal_mixture_em__given_invalid_settings__raises_value_error(kwargs: dict) -> None:
    # ACT / ASSERT
    with pytest.raises(ValueError):
        normal_mixture_em(np.arange(5.0), **kwargs)
