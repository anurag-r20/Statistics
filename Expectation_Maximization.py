"""
EM algorithm for a finite mixture of univariate normal distributions.

The model is f(x) = sum_k pi_k N(x; mu_k, sigma_k^2). With latent membership
indicators z_ik, one iteration is

    E-step:  r_ik = pi_k N(x_i; mu_k, sigma_k^2) / sum_j pi_j N(x_i; mu_j, sigma_j^2)
    M-step:  n_k = sum_i r_ik,  pi_k = n_k / n,  mu_k = sum_i r_ik x_i / n_k,
             sigma_k^2 = sum_i r_ik (x_i - mu_k)^2 / n_k.

The observed-data log-likelihood never decreases from one iteration to the next.
All computations use logarithms, so tiny densities do not underflow.
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy.special import logsumexp


@dataclass
class EMResult:
    weights: np.ndarray            # pi_k, sorted by increasing mean
    means: np.ndarray              # mu_k
    variances: np.ndarray          # sigma_k^2
    log_likelihood: np.ndarray     # observed-data log-likelihood: at the start, then after each iteration
    responsibilities: np.ndarray   # r_ik at the final parameters, shape (n, K)
    n_iter: int
    converged: bool


def _log_component_densities(x, weights, means, variances):
    """log(pi_k N(x_i; mu_k, sigma_k^2)), shape (n, K)."""
    return (np.log(weights) - 0.5 * np.log(2 * np.pi * variances)
            - 0.5 * (x[:, None] - means) ** 2 / variances)


def normal_mixture_loglik(x, weights, means, variances) -> float:
    """Observed-data log-likelihood sum_i log sum_k pi_k N(x_i; mu_k, sigma_k^2)."""
    x = np.asarray(x, dtype=float)
    return float(np.sum(logsumexp(_log_component_densities(x, np.asarray(weights, float),
                                                            np.asarray(means, float),
                                                            np.asarray(variances, float)), axis=1)))


def normal_mixture_em(
    x,
    n_components: int,
    weights: Optional[Sequence[float]] = None,
    means: Optional[Sequence[float]] = None,
    variances: Optional[Sequence[float]] = None,
    max_iter: int = 1000,
    tol: float = 1e-10,
    min_variance: float = 1e-8,
    seed=None,
) -> EMResult:
    """
    Fit a K-component univariate normal mixture by EM.

    Starting values not supplied are: equal weights, means at K distinct data points
    chosen at random, and every variance equal to the sample variance. Iteration stops
    when the log-likelihood increases by less than tol * (1 + |log-likelihood|), or after
    max_iter iterations. Variances are kept at least min_variance times the sample
    variance, so a component cannot collapse onto one point. Components are returned
    sorted by mean.
    """
    x = np.asarray(x, dtype=float).ravel()
    n, K = x.size, n_components
    if K < 1 or n < K:
        raise ValueError("Require at least one component and at least as many observations.")
    if max_iter < 1 or tol <= 0:
        raise ValueError("Require max_iter >= 1 and tol > 0.")
    rng = np.random.default_rng(seed)
    floor = min_variance * np.var(x)

    pi = np.full(K, 1.0 / K) if weights is None else np.asarray(weights, dtype=float).copy()
    mu = rng.choice(x, size=K, replace=False) if means is None else np.asarray(means, dtype=float).copy()
    var = np.full(K, np.var(x)) if variances is None else np.asarray(variances, dtype=float).copy()
    if not (pi.shape == mu.shape == var.shape == (K,)):
        raise ValueError("weights, means and variances must each have n_components entries.")
    if np.any(pi <= 0) or not np.isclose(pi.sum(), 1.0) or np.any(var <= 0):
        raise ValueError("weights must be positive and sum to 1; variances must be positive.")

    log_dens = _log_component_densities(x, pi, mu, var)
    loglik = [float(np.sum(logsumexp(log_dens, axis=1)))]
    converged = False
    for it in range(1, max_iter + 1):
        # E-step: responsibilities r_ik = P(z_ik = 1 | x_i, current parameters).
        r = np.exp(log_dens - logsumexp(log_dens, axis=1, keepdims=True))
        # M-step: closed-form maximisers of the expected complete-data log-likelihood.
        nk = r.sum(axis=0)
        pi = nk / n
        mu = (r * x[:, None]).sum(axis=0) / nk
        var = np.maximum((r * (x[:, None] - mu) ** 2).sum(axis=0) / nk, floor)

        log_dens = _log_component_densities(x, pi, mu, var)
        loglik.append(float(np.sum(logsumexp(log_dens, axis=1))))
        if loglik[-1] - loglik[-2] < tol * (1 + abs(loglik[-1])):
            converged = True
            break

    r = np.exp(log_dens - logsumexp(log_dens, axis=1, keepdims=True))
    order = np.argsort(mu)
    return EMResult(weights=pi[order], means=mu[order], variances=var[order],
                    log_likelihood=np.array(loglik), responsibilities=r[:, order],
                    n_iter=it, converged=converged)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    z = rng.choice(3, size=2000, p=[0.3, 0.5, 0.2])
    x = rng.normal(np.array([-2.0, 1.0, 5.0])[z], np.array([0.7, 1.0, 1.5])[z])
    fit = normal_mixture_em(x, 3, seed=1)
    print(f"EM converged={fit.converged} after {fit.n_iter} iterations")
    print("  weights  ", fit.weights.round(3), " true [0.3 0.5 0.2]")
    print("  means    ", fit.means.round(3), " true [-2. 1. 5.]")
    print("  std devs ", np.sqrt(fit.variances).round(3), " true [0.7 1. 1.5]")
