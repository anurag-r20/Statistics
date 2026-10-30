from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from High_Dimensional_Regression import desparsified_lasso, nodewise_lasso


def orthogonal_design() -> np.ndarray:
    """Two centred, unit-variance columns with X.T X / n = I."""
    return np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=float)


# ---------------------------------------------------------------- nodewise_lasso


def test__nodewise_lasso__given_two_correlated_columns__matches_soft_threshold_solution() -> None:
    # ARRANGE
    Q = orthogonal_design()
    X = np.column_stack([Q[:, 0], 0.6 * Q[:, 0] + 0.8 * Q[:, 1]])

    # ACT
    res = nodewise_lasso(X, alpha=0.2)

    # ASSERT: each one-variable Lasso coefficient is 0.6 - 0.2 = 0.4.
    np.testing.assert_allclose(res.coefficients, [[0, 0.4], [0.4, 0]], atol=1e-12)
    np.testing.assert_allclose(res.tau2, 0.76, atol=1e-12)
    np.testing.assert_allclose(res.precision, np.array([[1, -0.4], [-0.4, 1]]) / 0.76)
    np.testing.assert_allclose(np.mean(X * res.residuals, axis=0), res.tau2, atol=1e-12)


def test__nodewise_lasso__given_orthogonal_columns__returns_empty_graph_and_identity_inverse() -> None:
    # ARRANGE / ACT
    res = nodewise_lasso(orthogonal_design(), alpha=0.1)

    # ASSERT
    np.testing.assert_allclose(res.precision, np.eye(2))
    assert not np.any(res.adjacency())


def test__adjacency__given_one_direction_selected__distinguishes_and_from_or() -> None:
    # ARRANGE
    res = nodewise_lasso(orthogonal_design(), alpha=0.1)
    res.coefficients[0, 1] = 0.3

    # ACT / ASSERT
    assert not res.adjacency("and").any()
    np.testing.assert_array_equal(res.adjacency("or"), [[False, True], [True, False]])
    assert not np.diag(res.adjacency("or")).any()


def test__nodewise_lasso__given_rescaled_shifted_columns__preserves_standardised_fit() -> None:
    # ARRANGE
    rng = np.random.default_rng(9)
    X = rng.normal(size=(100, 4))
    X[:, 1] += 0.7 * X[:, 0]

    # ACT
    a = nodewise_lasso(X, alpha=0.1)
    b = nodewise_lasso(X * [2, 3, 4, 5] + [1, -2, 3, -4], alpha=0.1)

    # ASSERT
    np.testing.assert_allclose(a.coefficients, b.coefficients, atol=1e-12)
    np.testing.assert_allclose(a.precision, b.precision, atol=1e-12)


@pytest.mark.parametrize("X", [
    np.ones((5, 2)),
    np.array([[1, 2], [3, np.nan], [4, 5]]),
    np.ones(5),
    np.ones((2, 3)),
], ids=["constant", "nonfinite", "one-dimensional", "too-few-rows"])
def test__nodewise_lasso__given_invalid_design__raises_value_error(X: np.ndarray) -> None:
    # ACT / ASSERT
    with pytest.raises(ValueError):
        nodewise_lasso(X, alpha=0.1)


@pytest.mark.parametrize("alpha", [0, -1, np.inf, np.nan])
def test__nodewise_lasso__given_invalid_penalty__raises_value_error(alpha: float) -> None:
    # ACT / ASSERT
    with pytest.raises(ValueError):
        nodewise_lasso(orthogonal_design(), alpha=alpha)


def test__nodewise_lasso__given_unconverged_solver__raises_runtime_error() -> None:
    # ARRANGE
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 20))
    X[:, 1:] += X[:, [0]]

    # ACT / ASSERT
    with pytest.raises(RuntimeError, match="did not converge"):
        nodewise_lasso(X, alpha=0.001, max_iter=1, tol=1e-14)


# ---------------------------------------------------------------- desparsified_lasso


def test__desparsified_lasso__given_orthogonal_design__removes_known_shrinkage() -> None:
    # ARRANGE
    X = orthogonal_design()
    y = 3 + X @ np.array([2.0, -1.0])

    # ACT
    res = desparsified_lasso(X, y, alpha=0.2, sigma=1)

    # ASSERT
    np.testing.assert_allclose(res.initial_coefficients, [1.8, -0.8], atol=1e-12)
    np.testing.assert_allclose(res.coefficients, [2, -1], atol=1e-12)
    np.testing.assert_allclose(res.standard_errors, 0.5)
    np.testing.assert_allclose(res.confidence_intervals,
                               np.array([[2], [-1]]) + np.array([-1, 1]) * norm.ppf(0.975) / 2)
    np.testing.assert_allclose(res.p_values, 2 * norm.sf([4, 2]))
    np.testing.assert_allclose(res.adjusted_p_values, np.minimum(1, 4 * norm.sf([4, 2])))
    assert res.intercept == pytest.approx(3)
    assert res.intercept_standard_error == pytest.approx(0.5)


def test__desparsified_lasso__given_original_units_and_shift__matches_full_ols_covariance() -> None:
    # ARRANGE
    X = orthogonal_design() * [2, 5] + [3, -4]
    y = 7 + X @ np.array([1.0, -2.0])
    design = np.column_stack([np.ones(len(X)), X])
    covariance = 4 * np.linalg.inv(design.T @ design)

    # ACT
    res = desparsified_lasso(X, y, alpha=0.2, sigma=2)

    # ASSERT
    np.testing.assert_allclose(res.coefficients, [1, -2], atol=1e-12)
    np.testing.assert_allclose(res.standard_errors, np.sqrt(np.diag(covariance)[1:]))
    assert res.intercept == pytest.approx(7)
    assert res.intercept_standard_error == pytest.approx(np.sqrt(covariance[0, 0]))


def test__desparsified_lasso__given_correlated_columns__uses_residual_score_variance() -> None:
    # ARRANGE: nodewise coefficient 0.4 gives residual variance 0.68 and tau^2 = 0.76.
    Q = orthogonal_design()
    X = np.column_stack([Q[:, 0], 0.6 * Q[:, 0] + 0.8 * Q[:, 1]])
    y = X @ [2, -1]

    # ACT
    res = desparsified_lasso(X, y, alpha=0.2, sigma=1)

    # ASSERT: using only the diagonal of the approximate inverse would give a different SE.
    np.testing.assert_allclose(res.standard_errors, np.sqrt(0.68) / (2 * 0.76), atol=1e-12)


def test__nodewise_lasso__given_more_columns_than_rows__satisfies_nodewise_kkt_conditions() -> None:
    # ARRANGE
    rng = np.random.default_rng(546)
    X = rng.normal(size=(40, 60))
    X[:, 1] += 0.8 * X[:, 0]
    alpha = 0.2

    # ACT
    res = nodewise_lasso(X, alpha=alpha)

    # ASSERT: the score for each own column is 1, and the other scores are bounded by lambda/tau^2.
    Z = (X - res.mean) / res.scale
    scores = res.precision @ (Z.T @ Z / len(X))
    np.testing.assert_allclose(np.diag(scores), 1, atol=1e-7)
    np.fill_diagonal(scores, 0)
    assert np.all(np.abs(scores) <= alpha / res.tau2[:, None] + 1e-7)


def test__desparsified_lasso__given_unknown_noise__uses_stated_residual_df_estimate() -> None:
    # ARRANGE: the interaction is orthogonal to both columns and the intercept.
    X = orthogonal_design()
    y = 2 + 3 * X[:, 0] + X[:, 0] * X[:, 1]

    # ACT
    res = desparsified_lasso(X, y, alpha=0.2)

    # ASSERT: RSS = 4 (1 + 0.2^2), with df = 4 - 1 slope - 1 intercept.
    assert res.sigma == pytest.approx(np.sqrt(4 * 1.04 / 2))


def test__desparsified_lasso__given_reused_nodewise_fit__matches_fresh_fit() -> None:
    # ARRANGE
    X = orthogonal_design()
    y = X @ [1, 2]
    nodewise = nodewise_lasso(X, alpha=0.15)

    # ACT
    a = desparsified_lasso(X, y, alpha=0.2, nodewise_alpha=0.15, sigma=1)
    b = desparsified_lasso(X, y, alpha=0.2, nodewise=nodewise, sigma=1)

    # ASSERT
    np.testing.assert_allclose(a.coefficients, b.coefficients)
    np.testing.assert_allclose(a.standard_errors, b.standard_errors)


def test__desparsified_lasso__given_nodewise_fit_from_different_rows__rejects_reuse() -> None:
    # ARRANGE
    X = orthogonal_design()
    nodewise = nodewise_lasso(X, alpha=0.15)

    # ACT / ASSERT: reordering rows preserves means and scales, but not the residual vectors.
    with pytest.raises(ValueError, match="same X"):
        desparsified_lasso(X[::-1], X[:, 0], alpha=0.2, sigma=1, nodewise=nodewise)


@pytest.mark.parametrize("kwargs", [
    {"confidence": 1},
    {"confidence": 0},
    {"sigma": 0},
    {"sigma": np.nan},
    {"nodewise_alpha": -1},
], ids=["confidence-1", "confidence-0", "sigma-zero", "sigma-nan", "nodewise-penalty"])
def test__desparsified_lasso__given_invalid_settings__raises_value_error(kwargs: dict) -> None:
    # ARRANGE
    X = orthogonal_design()

    # ACT / ASSERT
    with pytest.raises(ValueError):
        desparsified_lasso(X, X[:, 0], alpha=0.2, **kwargs)
