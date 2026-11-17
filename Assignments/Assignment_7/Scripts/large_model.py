"""
The moderately large Bayesian model of Problem 3, and how each run is scored.

Bayesian linear regression with known noise variance:
    y_i = x_i'beta + e_i,  e_i ~ N(0, 1),  beta ~ N(0, PRIOR_VAR I),
with n = 100,000 training and 20,000 test observations and d = 100 predictors whose rows
are N(0, Sigma_x) with AR(1) correlation rho^{|j-k|}. The posterior is exactly
N(m, S), with S = (X'X + I / PRIOR_VAR)^{-1} and m = S X'y, so posterior estimation
error can be measured against the exact answer.
"""

import numpy as np

import common
from Stochastic_Gradient_MCMC import gaussian_prior_grad, linear_loglik_grad

N_TRAIN, N_TEST, D, RHO, PRIOR_VAR, NOISE_VAR, DATA_SEED = 100_000, 20_000, 100, 0.5, 1.0, 1.0, 546


def simulate():
    rng = np.random.default_rng(DATA_SEED)
    L = np.linalg.cholesky(RHO ** np.abs(np.subtract.outer(np.arange(D), np.arange(D))))
    beta = rng.normal(0.0, 0.5, D)

    def draw(n):
        X = rng.standard_normal((n, D)) @ L.T
        return X, X @ beta + np.sqrt(NOISE_VAR) * rng.standard_normal(n)

    (X, y), (Xt, yt) = draw(N_TRAIN), draw(N_TEST)
    return X, y, Xt, yt, beta


def exact_posterior(X, y):
    S = np.linalg.inv(X.T @ X / NOISE_VAR + np.eye(D) / PRIOR_VAR)
    return S @ X.T @ y / NOISE_VAR, S


def gradients(X, y):
    return gaussian_prior_grad(PRIOR_VAR), linear_loglik_grad(X, y, NOISE_VAR)


def score(samples, cpu_time, m, sd, Xt, yt, rmse_exact):
    """Measures for one run (samples after burn-in)."""
    import arviz as az
    est_mean, est_sd = samples.mean(axis=0), samples.std(axis=0)
    ess = np.array([float(az.ess(samples[None, :, j])) for j in range(samples.shape[1])])
    rmse = float(np.sqrt(np.mean((yt - Xt @ est_mean) ** 2)))
    return {
        "cpu_time": cpu_time,
        "ess_min": float(ess.min()), "ess_median": float(np.median(ess)),
        "ess_per_second": float(np.median(ess) / cpu_time),
        "mean_error": float(np.sqrt(np.mean(((est_mean - m) / sd) ** 2))),
        "sd_ratio": float(np.mean(est_sd / sd)),
        "test_rmse": rmse,
        "test_rmse_excess": rmse - rmse_exact,
    }
