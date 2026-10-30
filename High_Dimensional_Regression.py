"""
Nodewise Lasso for Gaussian graphs, and desparsified Lasso for linear regression.

Each column is centred and scaled to mean square 1 before fitting. Nodewise
regression estimates its neighbours by regressing it on all remaining columns.
The same regressions give an approximate inverse of the sample correlation matrix,
which removes the leading shrinkage bias from a Lasso regression estimate.

The Lasso objective is ||y - Z b||^2 / (2 n) + alpha ||b||_1. Coordinate descent
is supplied by scikit-learn; the graph construction, correction and inference
are computed here. Returned regression coefficients are in the original units.
"""

from dataclasses import dataclass
from typing import Optional
import warnings

import numpy as np
from scipy.stats import norm
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso


@dataclass
class NodewiseResult:
    coefficients: np.ndarray    # row j: regression of standardised column j on the others
    residuals: np.ndarray       # shape (n, p), one residual vector per column
    tau2: np.ndarray            # residual mean square plus alpha times coefficient l1 norm
    precision: np.ndarray       # approximate inverse correlation matrix; not symmetrised
    mean: np.ndarray            # original column means
    scale: np.ndarray           # original column standard deviations, with divisor n
    alpha: float
    n_iter: np.ndarray          # coordinate-descent iterations for each regression
    dual_gap: np.ndarray        # solver optimality gaps

    def adjacency(self, rule: str = "and", threshold: float = 1e-8) -> np.ndarray:
        """Undirected graph: both regressions select an edge (and), or either does (or)."""
        if rule not in ("and", "or"):
            raise ValueError("rule must be 'and' or 'or'.")
        if not np.isfinite(threshold) or threshold < 0:
            raise ValueError("threshold must be finite and non-negative.")
        selected = np.abs(self.coefficients) > threshold
        graph = selected & selected.T if rule == "and" else selected | selected.T
        np.fill_diagonal(graph, False)
        return graph


@dataclass
class DesparsifiedResult:
    initial_coefficients: np.ndarray
    coefficients: np.ndarray
    standard_errors: np.ndarray
    confidence_intervals: np.ndarray   # pointwise intervals, shape (p, 2)
    z_scores: np.ndarray
    p_values: np.ndarray               # two-sided tests of each coefficient being zero
    adjusted_p_values: np.ndarray      # Bonferroni adjustment over the p slopes
    intercept: float
    intercept_standard_error: float
    intercept_confidence_interval: np.ndarray
    sigma: float
    nodewise: NodewiseResult
    alpha: float
    confidence: float


def _standardize(X):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] < 3 or X.shape[1] < 1 or not np.all(np.isfinite(X)):
        raise ValueError("X must be a finite matrix with at least 3 rows and 1 column.")
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    if np.any(scale == 0):
        raise ValueError("Every column of X must have positive variance.")
    return (X - mean) / scale, mean, scale


def _check_settings(alpha, max_iter, tol):
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive.")
    if not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer.")
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be finite and positive.")


def _fit_lasso(X, y, alpha, max_iter, tol):
    model = Lasso(alpha=alpha, fit_intercept=False, max_iter=max_iter, tol=tol)
    # An unconverged fit would contaminate both edge selection and standard errors.
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        try:
            model.fit(X, y)
        except ConvergenceWarning as exc:
            raise RuntimeError("Lasso did not converge; increase max_iter or check the penalty.") from exc
    return model


def nodewise_lasso(X, alpha: float, max_iter: int = 10000, tol: float = 1e-8) -> NodewiseResult:
    """
    Fit one Lasso regression per variable to learn a Gaussian graph.

    Parameters
    ----------
    X : array_like, shape (n, p)
        Observations in rows. Columns are centred and standardised internally.
    alpha : float
        Penalty in ||Z_j - Z_{-j} gamma_j||^2 / (2 n) + alpha ||gamma_j||_1.
    max_iter, tol : int, float
        Coordinate-descent iteration limit and convergence tolerance.

    The nodewise residual r_j gives tau_j^2 = ||r_j||^2 / n + alpha ||gamma_j||_1.
    Row j of the approximate inverse is (e_j - gamma_j) / tau_j^2. This matrix
    need not be symmetric or positive definite; it is used for debiasing, not as
    a fitted covariance model. The AND/OR graph is available through adjacency.
    """
    _check_settings(alpha, max_iter, tol)
    Z, mean, scale = _standardize(X)
    n, p = Z.shape
    coefficients = np.zeros((p, p))
    residuals = np.empty((n, p))
    n_iter = np.zeros(p, dtype=int)
    dual_gap = np.zeros(p)

    for j in range(p):
        others = np.arange(p) != j
        if p > 1:
            model = _fit_lasso(Z[:, others], Z[:, j], alpha, max_iter, tol)
            coefficients[j, others] = model.coef_
            n_iter[j], dual_gap[j] = model.n_iter_, model.dual_gap_
        residuals[:, j] = Z[:, j] - Z @ coefficients[j]

    tau2 = np.mean(residuals ** 2, axis=0) + alpha * np.abs(coefficients).sum(axis=1)
    precision = (np.eye(p) - coefficients) / tau2[:, None]
    return NodewiseResult(coefficients, residuals, tau2, precision, mean, scale,
                          float(alpha), n_iter, dual_gap)


def desparsified_lasso(
    X,
    y,
    alpha: float,
    nodewise_alpha: Optional[float] = None,
    confidence: float = 0.95,
    max_iter: int = 10000,
    tol: float = 1e-8,
    sigma: Optional[float] = None,
    nodewise: Optional[NodewiseResult] = None,
) -> DesparsifiedResult:
    """
    Correct a Lasso estimate and compute pointwise normal intervals and p-values.

    Parameters
    ----------
    X, y : array_like
        Design matrix (n, p) and response vector (n,). The intercept is unpenalised.
    alpha : float
        Lasso penalty for centred y and standardised X.
    nodewise_alpha : float, optional
        Nodewise regression penalty. Defaults to alpha when no fit is supplied.
    confidence : float
        Pointwise confidence level. The intervals are not simultaneous intervals.
    max_iter, tol : int, float
        Coordinate-descent controls; an unconverged fit raises RuntimeError.
    sigma : float, optional
        Known error standard deviation. If omitted, estimate it from the initial
        Lasso residual sum of squares divided by n - number of nonzero slopes - 1.
        This degrees-of-freedom adjustment is an estimate, not an exact pivot.
    nodewise : NodewiseResult, optional
        A nodewise fit for the same X, allowing its reuse between graph learning
        and regression. If nodewise_alpha is also supplied, it must agree.

    For standardised design Z and approximate inverse M, the corrected estimate
    is b = beta_lasso + M Z.T (y_centred - Z beta_lasso) / n. Its estimated
    covariance is sigma^2 M (Z.T Z / n) M.T / n. The normal approximation requires
    sparse coefficients and sparse inverse covariance, with sufficiently small
    regularisation bias; it is not an exact finite-sample confidence statement.
    """
    _check_settings(alpha, max_iter, tol)
    Z, mean, scale = _standardize(X)
    n, p = Z.shape
    y = np.asarray(y, dtype=float)
    if y.shape != (n,) or not np.all(np.isfinite(y)):
        raise ValueError("y must be a finite vector with one value per row of X.")
    if not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must be strictly between 0 and 1.")
    if sigma is not None and (not np.isfinite(sigma) or sigma <= 0):
        raise ValueError("sigma must be finite and positive.")

    if nodewise is None:
        penalty = alpha if nodewise_alpha is None else nodewise_alpha
        nodewise = nodewise_lasso(X, penalty, max_iter=max_iter, tol=tol)
    else:
        if nodewise_alpha is not None and not np.isclose(nodewise_alpha, nodewise.alpha):
            raise ValueError("nodewise_alpha does not match the supplied nodewise fit.")
        if (nodewise.coefficients.shape != (p, p) or nodewise.residuals.shape != (n, p)
                or not np.allclose(nodewise.mean, mean, rtol=1e-10, atol=1e-12)
                or not np.allclose(nodewise.scale, scale, rtol=1e-10, atol=1e-12)
                or not np.allclose(nodewise.residuals, Z - Z @ nodewise.coefficients.T,
                                   rtol=1e-8, atol=1e-10)):
            raise ValueError("The supplied nodewise fit must use the same X.")

    centred_y = y - y.mean()
    initial = _fit_lasso(Z, centred_y, alpha, max_iter, tol).coef_
    residual = centred_y - Z @ initial
    if sigma is None:
        df = n - np.count_nonzero(initial) - 1
        if df <= 0:
            raise ValueError("Too many selected variables to estimate sigma; supply a known sigma.")
        sigma = float(np.sqrt(residual @ residual / df))
        if sigma == 0:
            raise ValueError("The residual variance is zero; supply a positive known sigma.")

    # Each column of scores is r_j / tau_j^2; no p-by-p covariance product is needed.
    scores = Z @ nodewise.precision.T
    coefficients = (initial + scores.T @ residual / n) / scale
    standard_errors = sigma * np.sqrt(np.sum(scores ** 2, axis=0)) / (n * scale)
    critical = norm.ppf((1 + confidence) / 2)
    confidence_intervals = coefficients[:, None] + np.array([-1, 1]) * critical * standard_errors[:, None]
    z_scores = coefficients / standard_errors
    p_values = 2 * norm.sf(np.abs(z_scores))

    intercept = float(y.mean() - mean @ coefficients)
    intercept_influence = (1 - scores @ (mean / scale)) / n
    intercept_standard_error = float(sigma * np.linalg.norm(intercept_influence))
    intercept_confidence_interval = intercept + np.array([-1, 1]) * critical * intercept_standard_error

    return DesparsifiedResult(
        initial_coefficients=initial / scale,
        coefficients=coefficients,
        standard_errors=standard_errors,
        confidence_intervals=confidence_intervals,
        z_scores=z_scores,
        p_values=p_values,
        adjusted_p_values=np.minimum(1.0, p * p_values),
        intercept=intercept,
        intercept_standard_error=intercept_standard_error,
        intercept_confidence_interval=intercept_confidence_interval,
        sigma=float(sigma),
        nodewise=nodewise,
        alpha=float(alpha),
        confidence=float(confidence),
    )
