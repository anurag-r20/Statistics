"""
Stochastic approximation Monte Carlo (SAMC) and simulated stochastic
approximation annealing (SAA), built on Metropolis-Hastings.

SAMC targets psi(x) = exp(-U(x) / tau) after partitioning the sample space into
subregions E_1, ..., E_m (usually by energy U). It runs MH on the working density

    p_theta(x) proportional to  sum_i  psi(x) exp(-theta_i) 1{x in E_i}

and after every step updates the log weights

    theta <- theta + gamma_t (e_J - pi),

where J is the subregion of the new state, e_J is the J-th unit vector, pi is the
desired fraction of time in each subregion and gamma_t is a decreasing gain.
A subregion the chain stays in has its weight raised, which pushes the chain to
other subregions, so it cannot get trapped in a local mode. In the long run
theta_i - log(pi_i) converges, up to a common constant, to log of the integral of
psi over E_i. This gives

  - subregion probabilities: P(E_i) proportional to pi_i exp(theta_i), over non-empty subregions;
  - importance weights exp(theta_{J(x_t)}) for estimating expectations under psi.

SAA is SAMC whose temperature tau is lowered during the run by a square-root
cooling schedule. It is used for optimisation.
"""

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Union

import numpy as np
from scipy.special import logsumexp

from Metropolis_Hastings import gaussian_random_walk, mh_step


RegionFn = Callable[[np.ndarray, float], int]


def energy_regions(levels: Sequence[float]):
    """
    Partition by energy with increasing levels u_1 < ... < u_{m-1}:

        E_1 = {U <= u_1},  E_i = {u_{i-1} < U <= u_i},  E_m = {U > u_{m-1}}.

    Returns (region, m), where region(x, u) gives the 0-based subregion index.
    """
    levels = np.asarray(levels, dtype=float)
    if levels.ndim != 1 or levels.size == 0 or np.any(np.diff(levels) <= 0):
        raise ValueError("levels must be a non-empty, strictly increasing sequence.")
    return (lambda x, u: int(np.searchsorted(levels, u, side="left"))), levels.size + 1


def sqrt_cooling(T_high: float, T_low: float = 0.0, t0: int = 1):
    """Square-root cooling: tau_t = T_low + T_high * sqrt(t0 / max(t0, t))."""
    return lambda t: T_low + T_high * np.sqrt(t0 / max(t0, t))


@dataclass
class SAMCResult:
    samples: np.ndarray          # stored states (after burn-in, every thin-th), shape (n_stored, dim)
    energies: np.ndarray         # energy of each stored state
    regions: np.ndarray          # subregion index of each stored state
    log_weights: np.ndarray      # theta_{J(x_t)} at each stored state (log importance weight + const)
    theta: np.ndarray            # final log weights
    desired: np.ndarray          # desired sampling distribution pi
    visited: np.ndarray          # whether each subregion was visited at any point in the run
    region_counts: np.ndarray    # visits to each subregion after burn-in
    x_best: np.ndarray           # lowest-energy state visited
    f_best: float
    acceptance_rate: float

    def region_probabilities(self) -> np.ndarray:
        """Estimated P(E_i) under psi, normalised over the visited subregions (0 for the rest)."""
        log_p = np.where(self.visited, np.log(self.desired) + self.theta, -np.inf)
        return np.exp(log_p - logsumexp(log_p))

    def importance_weights(self) -> np.ndarray:
        """Normalised importance weights of the stored states for expectations under psi."""
        w = np.exp(self.log_weights - self.log_weights.max())
        return w / w.sum()


def samc(
    energy: Callable[[np.ndarray], float],
    x0,
    n_iters: int,
    region: RegionFn,
    n_regions: int,
    tau: Union[float, Callable[[int], float]] = 1.0,
    desired: Optional[Sequence[float]] = None,
    propose: Optional[Callable[[np.ndarray, np.random.Generator], np.ndarray]] = None,
    log_proposal: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    t0: float = 1000.0,
    eta: float = 1.0,
    burn_in: int = 0,
    thin: int = 1,
    seed=None,
) -> SAMCResult:
    """
    Run SAMC.

    Parameters
    ----------
    energy : callable
        x -> U(x). The target is psi(x) = exp(-U(x) / tau); use U = 0 for a uniform target.
    x0 : array_like
        Initial state (finite energy, inside a subregion).
    n_iters : int
        Total number of iterations.
    region : callable
        (x, U(x)) -> subregion index in [0, n_regions), or -1 if x is outside the
        sample space (such proposals are rejected). See energy_regions.
    n_regions : int
        Number of subregions m.
    tau : float or callable
        Temperature, or a schedule t -> tau_t for t = 1, 2, ... (this gives SAA).
    desired : sequence of float, optional
        Desired sampling distribution pi over subregions (normalised internally).
        Defaults to uniform.
    propose, log_proposal :
        MH proposal and its log density (None if symmetric), as in metropolis_hastings.
    t0, eta : float
        Gain factor gamma_t = (t0 / max(t0, t))^eta, with eta in (0.5, 1].
    burn_in, thin : int
        Iterations before storing starts, and storage interval after that.
    seed : int or np.random.Generator, optional
    """
    if n_iters < 1 or burn_in < 0 or thin < 1 or n_regions < 1:
        raise ValueError("Require n_iters >= 1, burn_in >= 0, thin >= 1, n_regions >= 1.")
    if t0 <= 0 or not 0.5 < eta <= 1:
        raise ValueError("Require t0 > 0 and 0.5 < eta <= 1.")

    rng = np.random.default_rng(seed)
    if propose is None:
        propose = gaussian_random_walk(1.0)
    schedule = tau if callable(tau) else (lambda t: tau)

    pi = np.full(n_regions, 1.0 / n_regions) if desired is None else np.asarray(desired, dtype=float)
    if pi.shape != (n_regions,) or np.any(pi <= 0):
        raise ValueError("desired must have one positive entry per subregion.")
    pi = pi / pi.sum()

    x = np.atleast_1d(np.asarray(x0, dtype=float)).copy()
    u_x = energy(x)
    j_x = region(x, u_x) if np.isfinite(u_x) else -1
    if not 0 <= j_x < n_regions:
        raise ValueError("x0 must have finite energy and lie in one of the subregions.")

    theta = np.zeros(n_regions)
    state = {"tau": schedule(1)}
    last = {}

    def log_working_density(y):
        u = energy(y)
        j = region(y, u) if np.isfinite(u) else -1
        last["u"], last["j"] = u, j
        if not 0 <= j < n_regions:
            return -np.inf
        return -u / state["tau"] - theta[j]

    n_store = max(0, (n_iters - burn_in) // thin)
    samples = np.empty((n_store, x.size))
    energies = np.empty(n_store)
    regions = np.empty(n_store, dtype=int)
    log_weights = np.empty(n_store)
    visited = np.zeros(n_regions, dtype=bool)
    visited[j_x] = True
    counts = np.zeros(n_regions, dtype=int)
    x_best, f_best = x.copy(), u_x
    n_accepted = 0
    k = 0

    for t in range(1, n_iters + 1):
        state["tau"] = schedule(t)
        logp_x = -u_x / state["tau"] - theta[j_x]
        x, _, accepted = mh_step(x, logp_x, log_working_density, propose, rng, log_proposal)
        if accepted:
            u_x, j_x = last["u"], last["j"]
            n_accepted += 1
            if u_x < f_best:
                x_best, f_best = x.copy(), u_x

        # Weight update: theta <- theta + gamma_t (e_J - pi).
        gamma = (t0 / max(t0, t)) ** eta
        theta -= gamma * pi
        theta[j_x] += gamma
        visited[j_x] = True

        if t > burn_in:
            counts[j_x] += 1
            if (t - burn_in) % thin == 0 and k < n_store:
                samples[k] = x
                energies[k] = u_x
                regions[k] = j_x
                log_weights[k] = theta[j_x]
                k += 1

    return SAMCResult(
        samples=samples,
        energies=energies,
        regions=regions,
        log_weights=log_weights,
        theta=theta,
        desired=pi,
        visited=visited,
        region_counts=counts,
        x_best=x_best,
        f_best=f_best,
        acceptance_rate=n_accepted / n_iters,
    )


def stochastic_approximation_annealing(
    energy: Callable[[np.ndarray], float],
    x0,
    n_iters: int,
    levels: Sequence[float],
    T_high: float,
    T_low: float = 0.0,
    t0_cool: int = 1,
    **samc_kwargs,
) -> SAMCResult:
    """
    Simulated stochastic approximation annealing: SAMC on an energy partition with
    temperature tau_t = T_low + T_high * sqrt(t0_cool / max(t0_cool, t)).
    The lowest-energy state visited is returned as x_best, f_best.
    Other keyword arguments are passed to samc.
    """
    region, m = energy_regions(levels)
    return samc(energy, x0, n_iters, region, m, tau=sqrt_cooling(T_high, T_low, t0_cool), **samc_kwargs)


if __name__ == "__main__":
    # Bimodal target 0.5 N(-5, 1) + 0.5 N(5, 1), started in the left mode. SAMC on an
    # energy partition crosses between the modes even with a small random-walk step.
    def U(x):
        return -np.logaddexp(-0.5 * (x[0] + 5) ** 2, -0.5 * (x[0] - 5) ** 2)

    region, m = energy_regions(np.arange(0.5, 15.0, 0.5))
    res = samc(U, x0=-5.0, n_iters=200000, region=region, n_regions=m,
               propose=gaussian_random_walk(1.0), burn_in=20000, seed=0)
    w = res.importance_weights()
    print("SAMC on 0.5 N(-5,1) + 0.5 N(5,1), started at x = -5")
    print(f"  weighted mean = {np.sum(w * res.samples[:, 0]):.3f} (true 0)")
    print(f"  weighted P(x > 0) = {np.sum(w * (res.samples[:, 0] > 0)):.3f} (true 0.5)")
    print(f"  weighted variance = {np.sum(w * res.samples[:, 0] ** 2):.3f} (true 26)")
