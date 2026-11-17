"""
Stochastic gradient MCMC: SGLD and SGHMC.

Both sample a posterior pi(theta | data) proportional to p(theta) prod_i p(x_i | theta)
using a minibatch estimate of the gradient of the log posterior,

    g(theta) = grad log p(theta) + (N / b) sum_{i in B} grad log p(x_i | theta),

where B holds b indices drawn uniformly with replacement, so E[g] is the full gradient.

SGLD (stochastic gradient Langevin dynamics):

    theta <- theta + (eps_t / 2) g(theta) + N(0, eps_t I).

SGHMC (stochastic gradient Hamiltonian Monte Carlo), in the parameterisation with
learning rate eta, friction alpha and gradient-noise estimate beta_hat (unit mass;
beta_hat may be a scalar or one value per coordinate):

    theta <- theta + v,
    v     <- (1 - alpha) v + eta g(theta) + N(0, 2 (alpha - beta_hat) eta I).

Neither applies a Metropolis-Hastings correction.
"""

from dataclasses import dataclass
import time
from typing import Callable, Union

import numpy as np
from scipy.special import expit

GradFn = Callable[[np.ndarray, np.ndarray], np.ndarray]
StepFn = Union[float, Callable[[int], float]]


@dataclass
class SGMCMCResult:
    samples: np.ndarray      # stored states after burn-in, shape (n_stored, d)
    step_sizes: np.ndarray   # step size (SGLD) or learning rate (SGHMC) used for each stored state
    cpu_time: float          # process time spent in the sampling loop, in seconds
    n_iter: int
    batch_size: int


def weighted_moments(samples: np.ndarray, weights: np.ndarray):
    """Weighted mean and standard deviation of each coordinate (weights need not be normalised)."""
    w = np.asarray(weights, dtype=float) / np.sum(weights)
    mean = w @ samples
    sd = np.sqrt(w @ (samples - mean) ** 2)
    return mean, sd


# ---------------------------------------------------------------- model gradients


def gaussian_prior_grad(prior_var: float) -> Callable[[np.ndarray], np.ndarray]:
    """grad log N(theta; 0, prior_var I) = -theta / prior_var."""
    return lambda theta: -theta / prior_var


def logistic_loglik_grad(X: np.ndarray, y: np.ndarray) -> GradFn:
    """(theta, idx) -> sum_{i in idx} (y_i - sigmoid(x_i'theta)) x_i, for labels in {0, 1}."""
    return lambda theta, idx: X[idx].T @ (y[idx] - expit(X[idx] @ theta))


def linear_loglik_grad(X: np.ndarray, y: np.ndarray, noise_var: float) -> GradFn:
    """(theta, idx) -> sum_{i in idx} x_i (y_i - x_i'theta) / noise_var."""
    return lambda theta, idx: X[idx].T @ (y[idx] - X[idx] @ theta) / noise_var


# ---------------------------------------------------------------- samplers


def _check(n_data, batch_size, n_iter, burn_in, thin):
    if n_data < 1 or not 1 <= batch_size <= n_data:
        raise ValueError("Require n_data >= 1 and 1 <= batch_size <= n_data.")
    if n_iter < 1 or not 0 <= burn_in < n_iter or thin < 1:
        raise ValueError("Require n_iter >= 1, 0 <= burn_in < n_iter and thin >= 1.")


def minibatch_gradient(grad_log_prior, grad_loglik, n_data: int, batch_size: int, rng) -> Callable:
    """theta -> unbiased minibatch estimate of the log-posterior gradient, with b indices drawn
    uniformly with replacement and scaled by N / b. Uses all the data if batch_size == n_data."""
    if batch_size == n_data:
        everything = np.arange(n_data)
        return lambda theta: grad_log_prior(theta) + grad_loglik(theta, everything)
    scale = n_data / batch_size
    return lambda theta: grad_log_prior(theta) + scale * grad_loglik(theta, rng.integers(0, n_data, batch_size))


def gradient_noise_variance(grad_log_prior, grad_loglik, n_data: int, batch_size: int, theta,
                            n_draws: int = 2000, seed=None) -> np.ndarray:
    """Per-coordinate variance of the minibatch gradient estimate at theta, from n_draws draws."""
    rng = np.random.default_rng(seed)
    estimate = minibatch_gradient(grad_log_prior, grad_loglik, n_data, batch_size, rng)
    return np.var([estimate(np.asarray(theta, dtype=float)) for _ in range(n_draws)], axis=0)


def _schedule(step: StepFn) -> Callable[[int], float]:
    return step if callable(step) else (lambda t: step)


def sgld(
    grad_log_prior: Callable[[np.ndarray], np.ndarray],
    grad_loglik: GradFn,
    n_data: int,
    theta0,
    n_iter: int,
    batch_size: int,
    step_size: StepFn,
    burn_in: int = 0,
    thin: int = 1,
    seed=None,
) -> SGMCMCResult:
    """
    Stochastic gradient Langevin dynamics.

    step_size is a constant eps or a schedule t -> eps_t for t = 0, 1, ... . States after
    burn_in iterations are stored every thin iterations, with the step size used to reach
    them, so decreasing-step runs can be summarised with weighted_moments.
    """
    _check(n_data, batch_size, n_iter, burn_in, thin)
    rng = np.random.default_rng(seed)
    grad = minibatch_gradient(grad_log_prior, grad_loglik, n_data, batch_size, rng)
    eps = _schedule(step_size)
    theta = np.array(theta0, dtype=float)
    samples, steps = [], []

    start = time.process_time()
    for t in range(n_iter):
        e = eps(t)
        if e <= 0:
            raise ValueError("Step sizes must be positive.")
        theta = theta + 0.5 * e * grad(theta) + np.sqrt(e) * rng.standard_normal(theta.shape)
        if t >= burn_in and (t - burn_in) % thin == 0:
            samples.append(theta)
            steps.append(e)
    cpu = time.process_time() - start
    return SGMCMCResult(np.array(samples), np.array(steps), cpu, n_iter, batch_size)


def sghmc(
    grad_log_prior: Callable[[np.ndarray], np.ndarray],
    grad_loglik: GradFn,
    n_data: int,
    theta0,
    n_iter: int,
    batch_size: int,
    learning_rate: StepFn,
    friction: float,
    noise_estimate=0.0,
    burn_in: int = 0,
    thin: int = 1,
    seed=None,
) -> SGMCMCResult:
    """
    Stochastic gradient Hamiltonian Monte Carlo with unit mass:

        theta <- theta + v,
        v     <- (1 - alpha) v + eta g(theta) + N(0, 2 (alpha - beta_hat) eta I),

    with learning rate eta (constant or a schedule t -> eta_t), friction alpha in (0, 1] and
    gradient-noise estimate beta_hat in [0, alpha), a scalar or one value per coordinate.
    beta_hat = eta V / 2 matches the variance V of the minibatch gradient (see
    gradient_noise_variance); 0 ignores it. The momentum starts at v ~ N(0, eta_0 I).
    """
    _check(n_data, batch_size, n_iter, burn_in, thin)
    beta_hat = np.asarray(noise_estimate, dtype=float)
    if not 0 < friction <= 1 or np.any(beta_hat < 0) or np.any(beta_hat >= friction):
        raise ValueError("Require 0 < friction <= 1 and 0 <= noise_estimate < friction.")
    rng = np.random.default_rng(seed)
    grad = minibatch_gradient(grad_log_prior, grad_loglik, n_data, batch_size, rng)
    eta = _schedule(learning_rate)
    theta = np.array(theta0, dtype=float)
    if eta(0) <= 0:
        raise ValueError("Learning rates must be positive.")
    v = np.sqrt(eta(0)) * rng.standard_normal(theta.shape)
    samples, steps = [], []

    start = time.process_time()
    for t in range(n_iter):
        h = eta(t)
        if h <= 0:
            raise ValueError("Learning rates must be positive.")
        theta = theta + v
        v = ((1 - friction) * v + h * grad(theta)
             + np.sqrt(2 * (friction - beta_hat) * h) * rng.standard_normal(theta.shape))
        if t >= burn_in and (t - burn_in) % thin == 0:
            samples.append(theta)
            steps.append(h)
    cpu = time.process_time() - start
    return SGMCMCResult(np.array(samples), np.array(steps), cpu, n_iter, batch_size)


if __name__ == "__main__":
    # Bayesian linear regression with a known Gaussian posterior.
    rng = np.random.default_rng(0)
    n, d, prior_var = 2000, 3, 10.0
    X = rng.standard_normal((n, d))
    y = X @ np.array([1.0, -2.0, 0.5]) + rng.standard_normal(n)
    S = np.linalg.inv(X.T @ X + np.eye(d) / prior_var)
    m = S @ X.T @ y
    args = (gaussian_prior_grad(prior_var), linear_loglik_grad(X, y, 1.0), n, np.zeros(d), 40000, 100)
    a = sgld(*args, step_size=1e-4, burn_in=5000, seed=1)
    b = sghmc(*args, learning_rate=1e-5, friction=0.1, burn_in=5000, seed=2)
    print("exact posterior mean", m.round(4), " sd", np.sqrt(np.diag(S)).round(4))
    for name, r in [("SGLD ", a), ("SGHMC", b)]:
        print(f"{name} mean {r.samples.mean(0).round(4)}  sd {r.samples.std(0).round(4)}  cpu {r.cpu_time:.2f}s")
