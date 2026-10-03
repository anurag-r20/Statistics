"""
Parallel tempering (replica-exchange MCMC) built on Metropolis-Hastings.

Runs K chains in parallel, chain k targeting pi(x)^beta_k with
1 = beta_0 > beta_1 > ... > beta_{K-1} > 0. Hot chains (small beta) cross
between modes easily; periodic swaps between neighbouring chains pass those
moves down to the cold chain (beta = 1), which samples the actual target.

A swap between chains i and j is accepted with probability

    min(1, exp[(beta_i - beta_j) * (log pi(x_j) - log pi(x_i))]).
"""

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Union

import numpy as np

from Metropolis_Hastings import gaussian_random_walk, mh_step


ProposalFn = Callable[[np.ndarray, np.random.Generator], np.ndarray]


def geometric_ladder(n_chains: int, beta_min: float = 0.01) -> np.ndarray:
    """Inverse temperatures spaced geometrically from 1 down to beta_min."""
    if n_chains == 1:
        return np.ones(1)
    return np.geomspace(1.0, beta_min, n_chains)


@dataclass
class PTResult:
    samples: np.ndarray            # cold-chain (beta = 1) samples, shape (n_samples, dim)
    log_target: np.ndarray         # log pi at each cold-chain sample
    betas: np.ndarray
    acceptance_rates: np.ndarray   # MH acceptance rate per chain
    swap_rates: np.ndarray         # swap acceptance rate per adjacent pair (k, k+1)
    all_samples: Optional[np.ndarray] = None   # shape (n_samples, K, dim) if requested


def parallel_tempering(
    log_target: Callable[[np.ndarray], float],
    x0,
    n_samples: int,
    betas: Optional[Sequence[float]] = None,
    propose: Optional[Union[ProposalFn, Sequence[ProposalFn]]] = None,
    log_proposal: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    swap_every: int = 1,
    burn_in: int = 0,
    thin: int = 1,
    keep_all: bool = False,
    seed=None,
) -> PTResult:
    """
    Run parallel tempering.

    Parameters
    ----------
    log_target : callable
        x -> log pi(x), up to an additive constant.
    x0 : array_like
        Initial state: either one point (copied to all chains) or an array of
        shape (K, dim) with one starting point per chain.
    n_samples : int
        Number of cold-chain samples to return (after burn-in and thinning).
    betas : sequence of float, optional
        Inverse temperatures with betas[0] == 1. Defaults to geometric_ladder(8).
    propose : callable or list of callables, optional
        MH proposal, either shared by all chains or one per chain (hot chains
        usually want larger steps). Defaults to a unit Gaussian random walk.
    log_proposal : callable, optional
        log q(x_to | x_from) for asymmetric proposals; None if symmetric.
    swap_every : int
        Attempt swaps after every `swap_every` MH sweeps.
    burn_in, thin : int
        As in metropolis_hastings, counted in sweeps over all chains.
    keep_all : bool
        Also store samples from every chain.
    seed : int or np.random.Generator, optional
    """
    if n_samples < 1 or burn_in < 0 or thin < 1 or swap_every < 1:
        raise ValueError("Require n_samples >= 1, burn_in >= 0, thin >= 1, swap_every >= 1.")

    rng = np.random.default_rng(seed)
    betas = geometric_ladder(8) if betas is None else np.asarray(betas, dtype=float)
    if betas[0] != 1.0 or np.any(betas <= 0) or np.any(np.diff(betas) >= 0):
        raise ValueError("betas must be strictly decreasing, start at 1 and stay positive.")
    K = betas.size

    if propose is None:
        propose = gaussian_random_walk(1.0)
    proposals = list(propose) if isinstance(propose, (list, tuple)) else [propose] * K
    if len(proposals) != K:
        raise ValueError("Need one proposal per chain.")

    x0 = np.asarray(x0, dtype=float)
    xs = np.array([np.atleast_1d(x0)] * K) if x0.ndim <= 1 else x0.copy()
    if xs.shape[0] != K:
        raise ValueError("x0 must be a single point or have one row per chain.")
    logps = np.array([log_target(x) for x in xs])
    if not np.all(np.isfinite(logps)):
        raise ValueError("log_target must be finite at every starting point.")

    dim = xs.shape[1]
    samples = np.empty((n_samples, dim))
    log_targets = np.empty(n_samples)
    all_samples = np.empty((n_samples, K, dim)) if keep_all else None

    n_accepted = np.zeros(K)
    swap_attempts = np.zeros(K - 1)
    swap_accepts = np.zeros(K - 1)

    total_sweeps = burn_in + n_samples * thin
    k_out = 0
    n_swap_rounds = 0

    for i in range(total_sweeps):
        # 1. One MH step in every chain at its own temperature.
        for k in range(K):
            xs[k], logps[k], accepted = mh_step(
                xs[k], logps[k], log_target, proposals[k], rng, log_proposal, beta=betas[k]
            )
            if accepted and i >= burn_in:
                n_accepted[k] += 1

        # 2. Swap proposals between neighbours, alternating even and odd pairs.
        if (i + 1) % swap_every == 0:
            for k in range(n_swap_rounds % 2, K - 1, 2):
                log_alpha = (betas[k] - betas[k + 1]) * (logps[k + 1] - logps[k])
                swap_attempts[k] += 1
                if np.log(rng.uniform()) < log_alpha:
                    xs[[k, k + 1]] = xs[[k + 1, k]]
                    logps[[k, k + 1]] = logps[[k + 1, k]]
                    swap_accepts[k] += 1
            n_swap_rounds += 1

        if i >= burn_in and (i - burn_in) % thin == thin - 1:
            samples[k_out] = xs[0]
            log_targets[k_out] = logps[0]
            if keep_all:
                all_samples[k_out] = xs
            k_out += 1

    return PTResult(
        samples=samples,
        log_target=log_targets,
        betas=betas,
        acceptance_rates=n_accepted / (n_samples * thin),
        swap_rates=np.divide(swap_accepts, swap_attempts, out=np.zeros(K - 1), where=swap_attempts > 0),
        all_samples=all_samples,
    )


if __name__ == "__main__":
    from Metropolis_Hastings import metropolis_hastings

    # Bimodal target: equal mixture of N(-5, 1) and N(5, 1). Plain MH with a
    # small step size gets stuck in whichever mode it starts in.
    def log_bimodal(x):
        return np.logaddexp(-0.5 * (x[0] + 5) ** 2, -0.5 * (x[0] - 5) ** 2)

    mh = metropolis_hastings(log_bimodal, x0=-5.0, n_samples=20000,
                             propose=gaussian_random_walk(1.0), burn_in=1000, seed=0)

    betas = geometric_ladder(6, beta_min=0.02)
    proposals = [gaussian_random_walk(1.0 / np.sqrt(b)) for b in betas]
    pt = parallel_tempering(log_bimodal, x0=-5.0, n_samples=20000, betas=betas,
                            propose=proposals, burn_in=1000, seed=0)

    print("Bimodal target 0.5 N(-5,1) + 0.5 N(5,1), started at x = -5")
    print("  true:               mean = 0,     P(x > 0) = 0.5")
    print(f"  plain MH:           mean = {mh.samples.mean():6.3f}, P(x > 0) = {(mh.samples > 0).mean():.3f}")
    print(f"  parallel tempering: mean = {pt.samples.mean():6.3f}, P(x > 0) = {(pt.samples > 0).mean():.3f}")
    print("  betas:              ", pt.betas.round(3))
    print("  MH acceptance/chain:", pt.acceptance_rates.round(3))
    print("  swap acceptance:    ", pt.swap_rates.round(3))
