"""
General Metropolis-Hastings sampler.

Draws samples from a target distribution pi(x) known only up to a
normalising constant. At each step a candidate x' ~ q(x' | x) is proposed and
accepted with probability

    alpha(x, x') = min(1, [pi(x') q(x | x')] / [pi(x) q(x' | x)])

All computations are done in log space for numerical stability.

Also provides the double Metropolis-Hastings sampler for models whose
likelihood has an intractable normalising constant.
"""

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


@dataclass
class MHResult:
    samples: np.ndarray          # shape (n_samples, dim)
    log_target: np.ndarray       # log pi at each stored sample
    acceptance_rate: float       # fraction of accepted proposals (after burn-in)


def gaussian_random_walk(scale):
    """Symmetric Gaussian random-walk proposal x' = x + N(0, scale^2 I) (or N(0, scale) if scale is a matrix)."""
    scale = np.asarray(scale, dtype=float)

    def propose(x, rng):
        if scale.ndim == 2:
            return x + rng.multivariate_normal(np.zeros(x.size), scale)
        return x + scale * rng.standard_normal(x.size)

    return propose


def mh_step(
    x: np.ndarray,
    logp_x: float,
    log_target: Callable[[np.ndarray], float],
    propose: Callable[[np.ndarray, np.random.Generator], np.ndarray],
    rng: np.random.Generator,
    log_proposal: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    beta: float = 1.0,
):
    """
    One Metropolis-Hastings transition targeting pi(x)^beta.

    `logp_x` is the *untempered* log pi(x); the returned log density is also
    untempered, so callers that change beta (annealing, tempering) can reuse it.

    Returns (x_next, logp_next, accepted).
    """
    x_new = np.atleast_1d(np.asarray(propose(x, rng), dtype=float))
    logp_new = log_target(x_new)

    log_alpha = beta * (logp_new - logp_x)
    if log_proposal is not None:
        log_alpha += log_proposal(x, x_new) - log_proposal(x_new, x)

    # log_alpha may be nan if both logp are -inf; the comparison is then False (reject).
    if np.log(rng.uniform()) < log_alpha:
        return x_new, logp_new, True
    return x, logp_x, False


def metropolis_hastings(
    log_target: Callable[[np.ndarray], float],
    x0,
    n_samples: int,
    propose: Optional[Callable[[np.ndarray, np.random.Generator], np.ndarray]] = None,
    log_proposal: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    burn_in: int = 0,
    thin: int = 1,
    seed=None,
) -> MHResult:
    """
    Run a Metropolis-Hastings chain.

    Parameters
    ----------
    log_target : callable
        x -> log pi(x), up to an additive constant. May return -inf outside the support.
    x0 : array_like
        Initial state (must have finite log_target).
    n_samples : int
        Number of samples to return (after burn-in and thinning).
    propose : callable, optional
        (x, rng) -> x', a draw from q(. | x). Defaults to a Gaussian random walk with unit scale.
    log_proposal : callable, optional
        (x_to, x_from) -> log q(x_to | x_from). Leave as None for symmetric
        proposals, in which case the Hastings correction cancels (plain Metropolis).
    burn_in : int
        Number of initial iterations to discard.
    thin : int
        Keep every `thin`-th sample after burn-in.
    seed : int or np.random.Generator, optional
        Random seed or generator for reproducibility.
    """
    if n_samples < 1 or burn_in < 0 or thin < 1:
        raise ValueError("Require n_samples >= 1, burn_in >= 0, thin >= 1.")

    rng = np.random.default_rng(seed)
    if propose is None:
        propose = gaussian_random_walk(1.0)

    x = np.atleast_1d(np.asarray(x0, dtype=float)).copy()
    logp_x = log_target(x)
    if not np.isfinite(logp_x):
        raise ValueError("log_target(x0) must be finite; choose a starting point inside the support.")

    total_iters = burn_in + n_samples * thin
    samples = np.empty((n_samples, x.size))
    log_targets = np.empty(n_samples)
    n_accepted = 0
    k = 0

    for i in range(total_iters):
        x, logp_x, accepted = mh_step(x, logp_x, log_target, propose, rng, log_proposal)
        if accepted and i >= burn_in:
            n_accepted += 1

        if i >= burn_in and (i - burn_in) % thin == thin - 1:
            samples[k] = x
            log_targets[k] = logp_x
            k += 1

    return MHResult(
        samples=samples,
        log_target=log_targets,
        acceptance_rate=n_accepted / (n_samples * thin),
    )


def double_metropolis_hastings(
    log_prior: Callable[[np.ndarray], float],
    log_unnorm_lik: Callable[[np.ndarray, np.ndarray], float],
    data,
    theta0,
    n_samples: int,
    propose: Callable[[np.ndarray, np.random.Generator], np.ndarray],
    log_proposal: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    aux_propose: Optional[Callable[[np.ndarray, np.random.Generator], np.ndarray]] = None,
    n_aux_steps: int = 1,
    aux_sampler: Optional[Callable[[np.ndarray, np.ndarray, np.random.Generator], np.ndarray]] = None,
    burn_in: int = 0,
    thin: int = 1,
    seed=None,
) -> MHResult:
    """
    Double Metropolis-Hastings sampler (Liang, 2010) for posteriors whose
    likelihood f(x | theta) = g(x, theta) / Z(theta) has an intractable Z(theta).

    Each iteration proposes theta' ~ q(. | theta), then draws an auxiliary
    dataset y approximately from f(. | theta') by running a short MH chain at
    theta' started from the observed data. Z(theta) cancels in

        r = p(theta') g(x, theta') g(y, theta) q(theta | theta')
            / [p(theta) g(x, theta) g(y, theta') q(theta' | theta)].

    Parameters
    ----------
    log_prior : callable
        theta -> log p(theta). Return -inf outside the support.
    log_unnorm_lik : callable
        (x, theta) -> log g(x, theta), the unnormalised log likelihood.
    data : array_like
        Observed data x (any shape the likelihood and aux proposal understand).
    theta0 : array_like
        Initial parameter value (must have finite log_prior).
    n_samples : int
        Number of posterior draws to return (after burn-in and thinning).
    propose, log_proposal :
        Proposal for theta and its log density (None if symmetric).
    aux_propose : callable, optional
        (y, rng) -> y', the MH proposal for the auxiliary data chain.
    n_aux_steps : int
        Number of inner MH steps used to generate y (e.g. one sweep over the data).
    aux_sampler : callable, optional
        (x_start, theta', rng) -> y. Replaces the default inner MH chain, e.g.
        with a faster model-specific sampler. Either this or aux_propose is required.
    burn_in, thin, seed :
        As in metropolis_hastings.

    The returned MHResult.log_target holds log p(theta) + log g(x, theta) at
    each draw (the log posterior up to the unknown -log Z(theta)).
    """
    if n_samples < 1 or burn_in < 0 or thin < 1 or n_aux_steps < 1:
        raise ValueError("Require n_samples >= 1, burn_in >= 0, thin >= 1, n_aux_steps >= 1.")
    if aux_sampler is None and aux_propose is None:
        raise ValueError("Provide aux_propose (for the default inner MH chain) or aux_sampler.")

    rng = np.random.default_rng(seed)
    x = np.asarray(data, dtype=float)

    def run_aux_chain(theta_new):
        if aux_sampler is not None:
            return aux_sampler(x, theta_new, rng)
        log_g = lambda y: log_unnorm_lik(y, theta_new)
        y, logp_y = x.copy(), log_g(x)
        for _ in range(n_aux_steps):
            y, logp_y, _ = mh_step(y, logp_y, log_g, aux_propose, rng)
        return y

    theta = np.atleast_1d(np.asarray(theta0, dtype=float)).copy()
    log_prior_theta = log_prior(theta)
    if not np.isfinite(log_prior_theta):
        raise ValueError("log_prior(theta0) must be finite.")
    log_g_x_theta = log_unnorm_lik(x, theta)

    total_iters = burn_in + n_samples * thin
    samples = np.empty((n_samples, theta.size))
    log_targets = np.empty(n_samples)
    n_accepted = 0
    k = 0

    for i in range(total_iters):
        theta_new = np.atleast_1d(np.asarray(propose(theta, rng), dtype=float))
        log_prior_new = log_prior(theta_new)

        if np.isfinite(log_prior_new):
            y = run_aux_chain(theta_new)
            log_g_x_new = log_unnorm_lik(x, theta_new)
            log_alpha = (
                log_prior_new - log_prior_theta
                + log_g_x_new - log_g_x_theta
                + log_unnorm_lik(y, theta) - log_unnorm_lik(y, theta_new)
            )
            if log_proposal is not None:
                log_alpha += log_proposal(theta, theta_new) - log_proposal(theta_new, theta)

            if np.log(rng.uniform()) < log_alpha:
                theta, log_prior_theta, log_g_x_theta = theta_new, log_prior_new, log_g_x_new
                if i >= burn_in:
                    n_accepted += 1

        if i >= burn_in and (i - burn_in) % thin == thin - 1:
            samples[k] = theta
            log_targets[k] = log_prior_theta + log_g_x_theta
            k += 1

    return MHResult(
        samples=samples,
        log_target=log_targets,
        acceptance_rate=n_accepted / (n_samples * thin),
    )


if __name__ == "__main__":
    # Example 1: 2D correlated Gaussian with a symmetric random-walk proposal.
    mean = np.array([1.0, -2.0])
    cov = np.array([[1.0, 0.8], [0.8, 1.0]])
    cov_inv = np.linalg.inv(cov)

    def log_gauss(x):
        d = x - mean
        return -0.5 * d @ cov_inv @ d

    res = metropolis_hastings(
        log_gauss, x0=[0.0, 0.0], n_samples=20000,
        propose=gaussian_random_walk(0.8), burn_in=2000, seed=0,
    )
    print("Gaussian target")
    print("  acceptance rate:", round(res.acceptance_rate, 3))
    print("  sample mean:    ", res.samples.mean(axis=0).round(3), " true:", mean)
    print("  sample cov:\n", np.cov(res.samples.T).round(3))

    # Example 2: Gamma(shape=3, rate=1) with an asymmetric log-normal proposal,
    # which requires the Hastings correction.
    shape = 3.0

    def log_gamma(x):
        return (shape - 1) * np.log(x[0]) - x[0] if x[0] > 0 else -np.inf

    sigma = 0.5

    def propose_lognormal(x, rng):
        return x * np.exp(sigma * rng.standard_normal(x.size))

    def log_q_lognormal(x_to, x_from):
        # density of log-normal centred at log(x_from), up to a constant
        z = np.log(x_to[0]) - np.log(x_from[0])
        return -np.log(x_to[0]) - 0.5 * (z / sigma) ** 2

    res = metropolis_hastings(
        log_gamma, x0=1.0, n_samples=20000,
        propose=propose_lognormal, log_proposal=log_q_lognormal,
        burn_in=2000, seed=1,
    )
    print("\nGamma(3, 1) target")
    print("  acceptance rate:", round(res.acceptance_rate, 3))
    print("  sample mean:    ", res.samples.mean().round(3), " true:", shape)
    print("  sample var:     ", res.samples.var().round(3), " true:", shape)
