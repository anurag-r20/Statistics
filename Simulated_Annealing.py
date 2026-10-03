"""
Simulated annealing built on Metropolis-Hastings.

Minimises an energy f(x) by running MH on the Boltzmann distribution
pi_T(x) proportional to exp(-f(x) / T) while the temperature T is lowered
towards zero. At high T the chain explores freely; as T -> 0 it concentrates
on low-energy states.
"""

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from Metropolis_Hastings import gaussian_random_walk, mh_step


# Cooling schedules: (iteration t, initial temperature T0, total iterations n) -> T_t

def geometric_cooling(alpha: float = 0.999):
    """T_t = T0 * alpha^t."""
    return lambda t, T0, n: T0 * alpha ** t


def linear_cooling(T_min: float = 1e-3):
    """T_t decreases linearly from T0 to T_min."""
    return lambda t, T0, n: T0 + (T_min - T0) * t / max(n - 1, 1)


def logarithmic_cooling():
    """T_t = T0 / log(t + e). Slow, but has asymptotic convergence guarantees."""
    return lambda t, T0, n: T0 / np.log(t + np.e)


@dataclass
class SAResult:
    x_best: np.ndarray           # lowest-energy state visited
    f_best: float
    x_final: np.ndarray          # state at the end of the run
    energies: np.ndarray         # energy of the current state at each iteration
    temperatures: np.ndarray     # temperature used at each iteration
    acceptance_rate: float


def simulated_annealing(
    energy: Callable[[np.ndarray], float],
    x0,
    n_iters: int,
    T0: float = 1.0,
    schedule: Optional[Callable[[int, float, int], float]] = None,
    propose: Optional[Callable[[np.ndarray, np.random.Generator], np.ndarray]] = None,
    log_proposal: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    seed=None,
) -> SAResult:
    """
    Minimise `energy` by simulated annealing.

    Parameters
    ----------
    energy : callable
        x -> f(x), the objective to minimise. May return +inf for infeasible x.
    x0 : array_like
        Starting point (must have finite energy).
    n_iters : int
        Number of MH steps.
    T0 : float
        Initial temperature.
    schedule : callable, optional
        (t, T0, n_iters) -> T_t. Defaults to geometric cooling reaching about
        1e-3 * T0 at the final iteration.
    propose, log_proposal :
        Proposal passed through to the MH step (see Metropolis_Hastings.metropolis_hastings).
    seed : int or np.random.Generator, optional
    """
    if n_iters < 1 or T0 <= 0:
        raise ValueError("Require n_iters >= 1 and T0 > 0.")

    rng = np.random.default_rng(seed)
    if propose is None:
        propose = gaussian_random_walk(1.0)
    if schedule is None:
        schedule = geometric_cooling(1e-3 ** (1.0 / max(n_iters - 1, 1)))

    # pi_T(x) = exp(-f(x)/T): untempered log target is -f, and beta = 1/T.
    log_target = lambda x: -energy(x)

    x = np.atleast_1d(np.asarray(x0, dtype=float)).copy()
    logp_x = log_target(x)
    if not np.isfinite(logp_x):
        raise ValueError("energy(x0) must be finite.")

    x_best, f_best = x.copy(), -logp_x
    energies = np.empty(n_iters)
    temperatures = np.empty(n_iters)
    n_accepted = 0

    for t in range(n_iters):
        T = schedule(t, T0, n_iters)
        x, logp_x, accepted = mh_step(x, logp_x, log_target, propose, rng, log_proposal, beta=1.0 / T)
        n_accepted += accepted

        if -logp_x < f_best:
            x_best, f_best = x.copy(), -logp_x
        energies[t] = -logp_x
        temperatures[t] = T

    return SAResult(
        x_best=x_best,
        f_best=f_best,
        x_final=x,
        energies=energies,
        temperatures=temperatures,
        acceptance_rate=n_accepted / n_iters,
    )


if __name__ == "__main__":
    # Rastrigin function in 2D: global minimum f(0, 0) = 0, surrounded by many local minima.
    def rastrigin(x):
        return 10 * x.size + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))

    x0 = np.array([4.5, -3.7])
    res = simulated_annealing(
        rastrigin, x0, n_iters=50000, T0=10.0,
        propose=gaussian_random_walk(0.3), seed=0,
    )
    print("Simulated annealing on 2D Rastrigin")
    print("  start:          ", x0, " f =", round(rastrigin(x0), 3))
    print("  best found:     ", res.x_best.round(4), " f =", round(res.f_best, 5))
    print("  true minimum:    [0. 0.]  f = 0")
    print("  acceptance rate:", round(res.acceptance_rate, 3))
