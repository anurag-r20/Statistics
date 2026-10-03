"""
Simulated tempering built on Metropolis-Hastings.

A single chain whose state is (x, k), where k indexes an inverse temperature
from the ladder 1 = beta_0 > beta_1 > ... > beta_{K-1} > 0. The joint target is

    p(x, k) proportional to pi(x)^beta_k * exp(g_k),

where g_k are log weights. Each iteration does an MH move in x at the current
temperature, then proposes k -> k +/- 1, accepted with probability

    min(1, exp[(beta_k' - beta_k) * log pi(x) + g_k' - g_k]).

Samples with k = 0 are draws from pi. The ideal weights are g_k = -log Z_k
(Z_k the normaliser of pi^beta_k), which gives equal time at every level.
They are usually unknown, so by default they are learned during burn-in with
a Wang-Landau style update and then frozen, so post-burn-in samples are valid.
"""

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Union

import numpy as np

from Metropolis_Hastings import gaussian_random_walk, mh_step


ProposalFn = Callable[[np.ndarray, np.random.Generator], np.ndarray]


def geometric_ladder(n_levels: int, beta_min: float = 0.01) -> np.ndarray:
    """Inverse temperatures spaced geometrically from 1 down to beta_min."""
    if n_levels == 1:
        return np.ones(1)
    return np.geomspace(1.0, beta_min, n_levels)


@dataclass
class STResult:
    samples: np.ndarray            # draws with k = 0 (beta = 1), shape (n_cold, dim)
    x_trace: np.ndarray            # x at every stored iteration, shape (n_samples, dim)
    k_trace: np.ndarray            # temperature index at every stored iteration
    betas: np.ndarray
    log_weights: np.ndarray        # final g_k (fixed after burn-in)
    level_counts: np.ndarray       # post-burn-in visits to each level
    acceptance_rate: float         # MH acceptance in x
    level_move_rate: float         # acceptance of k -> k +/- 1 moves


def simulated_tempering(
    log_target: Callable[[np.ndarray], float],
    x0,
    n_samples: int,
    betas: Optional[Sequence[float]] = None,
    log_weights: Optional[Sequence[float]] = None,
    propose: Optional[Union[ProposalFn, Sequence[ProposalFn]]] = None,
    log_proposal: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    burn_in: int = 0,
    thin: int = 1,
    adapt_rate: float = 1.0,
    seed=None,
) -> STResult:
    """
    Run simulated tempering.

    Parameters
    ----------
    log_target : callable
        x -> log pi(x), up to an additive constant.
    x0 : array_like
        Initial state (starts at k = 0).
    n_samples : int
        Number of stored iterations after burn-in and thinning. Only those at
        k = 0 end up in `samples`, so expect about n_samples / K of them.
    betas : sequence of float, optional
        Inverse temperatures with betas[0] == 1. Defaults to geometric_ladder(8).
    log_weights : sequence of float, optional
        Fixed log weights g_k. If None, they are adapted during burn-in
        (needs a reasonably long burn-in) and then frozen.
    propose : callable or list of callables, optional
        MH proposal in x, shared or one per level. Defaults to a unit Gaussian random walk.
    log_proposal : callable, optional
        log q(x_to | x_from) for asymmetric proposals; None if symmetric.
    burn_in, thin : int
        As in metropolis_hastings.
    adapt_rate : float
        Initial step size of the Wang-Landau weight update (decays as 1/t).
    seed : int or np.random.Generator, optional
    """
    if n_samples < 1 or burn_in < 0 or thin < 1:
        raise ValueError("Require n_samples >= 1, burn_in >= 0, thin >= 1.")

    rng = np.random.default_rng(seed)
    betas = geometric_ladder(8) if betas is None else np.asarray(betas, dtype=float)
    if betas[0] != 1.0 or np.any(betas <= 0) or np.any(np.diff(betas) >= 0):
        raise ValueError("betas must be strictly decreasing, start at 1 and stay positive.")
    K = betas.size

    adapt = log_weights is None
    g = np.zeros(K) if adapt else np.asarray(log_weights, dtype=float).copy()
    if g.size != K:
        raise ValueError("Need one log weight per level.")

    if propose is None:
        propose = gaussian_random_walk(1.0)
    proposals = list(propose) if isinstance(propose, (list, tuple)) else [propose] * K
    if len(proposals) != K:
        raise ValueError("Need one proposal per level.")

    x = np.atleast_1d(np.asarray(x0, dtype=float)).copy()
    logp_x = log_target(x)
    if not np.isfinite(logp_x):
        raise ValueError("log_target(x0) must be finite.")
    k = 0

    x_trace = np.empty((n_samples, x.size))
    k_trace = np.empty(n_samples, dtype=int)
    level_counts = np.zeros(K, dtype=int)
    n_accepted = 0
    n_level_accepted = 0

    total_iters = burn_in + n_samples * thin
    k_out = 0

    for i in range(total_iters):
        # 1. MH move in x at the current temperature.
        x, logp_x, accepted = mh_step(x, logp_x, log_target, proposals[k], rng, log_proposal, beta=betas[k])

        # 2. Propose a neighbouring temperature (symmetric: out-of-range is rejected).
        k_new = k + (1 if rng.uniform() < 0.5 else -1)
        level_accepted = False
        if 0 <= k_new < K:
            log_alpha = (betas[k_new] - betas[k]) * logp_x + g[k_new] - g[k]
            if np.log(rng.uniform()) < log_alpha:
                k = k_new
                level_accepted = True

        if i < burn_in:
            # Wang-Landau: penalise the level just visited so the chain is pushed
            # towards less-visited levels. The decaying step makes g converge.
            if adapt:
                g[k] -= adapt_rate / (1.0 + i / K)
            continue

        n_accepted += accepted
        n_level_accepted += level_accepted
        level_counts[k] += 1

        if (i - burn_in) % thin == thin - 1:
            x_trace[k_out] = x
            k_trace[k_out] = k
            k_out += 1

    g -= g[0]   # only differences matter
    n_post = n_samples * thin

    return STResult(
        samples=x_trace[k_trace == 0],
        x_trace=x_trace,
        k_trace=k_trace,
        betas=betas,
        log_weights=g,
        level_counts=level_counts,
        acceptance_rate=n_accepted / n_post,
        level_move_rate=n_level_accepted / n_post,
    )


if __name__ == "__main__":
    # Bimodal target: equal mixture of N(-5, 1) and N(5, 1), started in the left mode.
    def log_bimodal(x):
        return np.logaddexp(-0.5 * (x[0] + 5) ** 2, -0.5 * (x[0] - 5) ** 2)

    betas = geometric_ladder(6, beta_min=0.02)
    proposals = [gaussian_random_walk(1.0 / np.sqrt(b)) for b in betas]
    st = simulated_tempering(log_bimodal, x0=-5.0, n_samples=60000, betas=betas,
                             propose=proposals, burn_in=20000, seed=0)

    # For this target, Z_k is proportional to beta_k^(-1/2), so the ideal weights are
    # g_k ~ 0.5 * log(beta_k) (approximate once the modes overlap at small beta).
    print("Bimodal target 0.5 N(-5,1) + 0.5 N(5,1), started at x = -5")
    print("  true:                mean = 0,     P(x > 0) = 0.5")
    print(f"  simulated tempering: mean = {st.samples.mean():6.3f}, P(x > 0) = {(st.samples > 0).mean():.3f}"
          f"  ({len(st.samples)} cold samples)")
    print("  betas:              ", st.betas.round(3))
    print("  learned log weights:", st.log_weights.round(2))
    print("  ~ideal log weights: ", (0.5 * np.log(betas)).round(2))
    print("  level visit fraction:", (st.level_counts / st.level_counts.sum()).round(3))
    print("  x acceptance:", round(st.acceptance_rate, 3), "  level-move acceptance:", round(st.level_move_rate, 3))
