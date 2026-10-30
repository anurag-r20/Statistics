"""
STAT546 HW5, Problem 2: high-dimensional inference by desparsified Lasso.

The design is exactly the one simulated in Q1. The intercept is 1 and the
first five slopes are 2, 2.5, 3, 3.5 and 4; the other slopes are zero. Since the
assignment does not specify the errors, use independent N(0, 1) errors.
Five-fold cross-validation chooses the initial Lasso penalty and a separate
nodewise penalty for the bias correction. Inference covers all 400 slopes, with pointwise 95%
intervals and Bonferroni-adjusted two-sided p-values.
"""

import warnings

import numpy as np
from scipy.stats import norm
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, lasso_path
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import common
from common import BLUE, ORANGE, TEXT_2, RESULTS_DIR, plt, save_fig, save_results
from data import N, P, SEED, simulate_design
from High_Dimensional_Regression import desparsified_lasso

ERROR_SEED, ERROR_SD, INTERCEPT = 547, 1.0, 1.0
ALPHAS, FOLDS = np.geomspace(0.01, 0.5, 40), 5
NODEWISE_ALPHAS = np.geomspace(0.3, 0.01, 20)
CONFIDENCE, TOL, MAX_ITER = 0.95, 1e-8, 10000


def choose_alpha(X, y):
    # Fit scaling on each training fold, so validation observations do not enter it.
    model = make_pipeline(StandardScaler(), Lasso(max_iter=MAX_ITER, tol=TOL))
    cv = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    search = GridSearchCV(model, {"lasso__alpha": ALPHAS}, cv=cv,
                          scoring="neg_mean_squared_error", n_jobs=1, error_score="raise")
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        search.fit(X, y)
    return float(search.best_params_["lasso__alpha"]), -search.cv_results_["mean_test_score"]


def choose_nodewise_alpha(X):
    """Choose a shared penalty by prediction error over all nodewise regressions.

    Only X enters this choice. Each fold centres and scales using its training
    observations. A coefficient path avoids refitting each penalty from scratch.
    """
    cv = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    mse = np.zeros((FOLDS, len(NODEWISE_ALPHAS)))
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        for fold, (train, test) in enumerate(cv.split(X)):
            mean, scale = X[train].mean(axis=0), X[train].std(axis=0)
            Z_train, Z_test = (X[train] - mean) / scale, (X[test] - mean) / scale
            for j in range(X.shape[1]):
                others = np.arange(X.shape[1]) != j
                _, coefficients, _ = lasso_path(Z_train[:, others], Z_train[:, j],
                                                alphas=NODEWISE_ALPHAS, tol=TOL, max_iter=MAX_ITER)
                residuals = Z_test[:, [j]] - Z_test[:, others] @ coefficients
                mse[fold] += np.mean(residuals ** 2, axis=0) / X.shape[1]
            print(f"  nodewise cross-validation fold {fold + 1}/{FOLDS}", flush=True)
    mean_mse = mse.mean(axis=0)
    return float(NODEWISE_ALPHAS[np.argmin(mean_mse)]), mean_mse


def plot(res, beta):
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.5), layout="constrained")
    ax = axes[0]
    j = np.arange(10)
    half_width = res.confidence_intervals[:10, 1] - res.coefficients[:10]
    ax.errorbar(j + 1, res.coefficients[:10], yerr=half_width, fmt="o", color=BLUE,
                markersize=4, capsize=2, label="Estimate and 95% interval")
    ax.scatter(j + 1, beta[:10], marker="x", color=ORANGE, s=35, zorder=3, label="True coefficient")
    ax.axhline(0, color=TEXT_2, lw=0.7)
    ax.set(xlabel="Variable", ylabel="Coefficient", xticks=j + 1, title="First ten regression coefficients")
    ax.legend(fontsize=7, loc="upper left")

    ax = axes[1]
    ax.hist(res.z_scores[beta == 0], bins=np.linspace(-4, 4, 25), density=True,
            color=BLUE, alpha=0.7, label="395 null coefficients")
    z = np.linspace(-4, 4, 300)
    ax.plot(z, norm.pdf(z), color=ORANGE, label="Standard normal")
    ax.set(xlabel="z statistic", ylabel="Density", title="Null statistics in this simulation")
    ax.legend(fontsize=7)
    save_fig(fig, "q2_desparsified_lasso")


if __name__ == "__main__":
    X, _ = simulate_design()
    beta = np.zeros(P)
    beta[:5] = [2.0, 2.5, 3.0, 3.5, 4.0]
    epsilon = np.random.default_rng(ERROR_SEED).normal(0, ERROR_SD, N)
    y = INTERCEPT + X @ beta + epsilon
    with threadpool_limits(limits=1):
        alpha, cv_mse = choose_alpha(X, y)
        nodewise_alpha, nodewise_cv_mse = choose_nodewise_alpha(X)
        res = desparsified_lasso(X, y, alpha, nodewise_alpha=nodewise_alpha,
                                 confidence=CONFIDENCE, max_iter=MAX_ITER, tol=TOL)

    covered = (res.confidence_intervals[:, 0] <= beta) & (beta <= res.confidence_intervals[:, 1])
    significant = res.adjusted_p_values < 1 - CONFIDENCE
    pointwise = res.p_values < 1 - CONFIDENCE
    active = beta != 0
    print(f"Q2: desparsified Lasso, alpha = {alpha:.5f}, estimated error SD = {res.sigma:.3f}")
    print(f"  first five estimates: {res.coefficients[:5].round(3)}")
    print(f"  Bonferroni significant variables: {(np.flatnonzero(significant) + 1).tolist()}")
    print(f"  pointwise intervals covering truth: {covered.sum()}/{P}")

    save_results("q2", {
        "settings": {"n": N, "p": P, "design_seed": SEED, "error_seed": ERROR_SEED,
                     "error_distribution": "independent N(0, 1)", "error_sd": ERROR_SD,
                     "method": "desparsified Lasso", "alpha": alpha,
                     "alpha_grid": ALPHAS.tolist(), "cv_folds": FOLDS, "cv_seed": SEED,
                     "nodewise_alpha": nodewise_alpha, "nodewise_alpha_grid": NODEWISE_ALPHAS.tolist(),
                     "confidence": CONFIDENCE,
                     "p_adjustment": "Bonferroni over 400 slopes", "solver_tolerance": TOL,
                     "max_iterations": MAX_ITER},
        "cv_mean_squared_errors": cv_mse.tolist(),
        "nodewise_cv_mean_squared_errors": nodewise_cv_mse.tolist(),
        "intercept": {"truth": INTERCEPT, "estimate": float(res.intercept),
                      "standard_error": float(res.intercept_standard_error),
                      "confidence_interval": res.intercept_confidence_interval.tolist()},
        "estimated_error_sd": float(res.sigma),
        "initial_selected_variables": (np.flatnonzero(res.initial_coefficients != 0) + 1).tolist(),
        "significant_variables": (np.flatnonzero(significant) + 1).tolist(),
        "summary": {"true_positives": int(np.sum(significant & active)),
                    "false_positives": int(np.sum(significant & ~active)),
                    "false_negatives": int(np.sum(~significant & active)),
                    "pointwise_false_positives": int(np.sum(pointwise & ~active)),
                    "intervals_covering_truth": int(covered.sum()),
                    "active_intervals_covering_truth": int(covered[active].sum()),
                    "null_intervals_covering_truth": int(covered[~active].sum()),
                    "coefficient_rmse": float(np.sqrt(np.mean((res.coefficients - beta) ** 2)))},
        "coefficients": [{"variable": j + 1, "truth": float(beta[j]),
                          "initial_lasso": float(res.initial_coefficients[j]),
                          "estimate": float(res.coefficients[j]),
                          "standard_error": float(res.standard_errors[j]),
                          "confidence_interval": res.confidence_intervals[j].tolist(),
                          "z": float(res.z_scores[j]), "p_value": float(res.p_values[j]),
                          "bonferroni_p_value": float(res.adjusted_p_values[j]),
                          "covers_truth": bool(covered[j])} for j in range(P)],
    })
    np.savez_compressed(RESULTS_DIR / "q2_data.npz", y=y, beta=beta, epsilon=epsilon)
    print("  saved Results/q2_data.npz")
    plot(res, beta)
