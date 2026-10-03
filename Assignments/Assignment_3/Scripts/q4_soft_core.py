"""
STAT546 HW3, Problem 4: double Metropolis-Hastings for the very-soft-core model.

    f(x | theta) = exp{ -sum_{i<j} phi(||x_i - x_j||, theta) } / Z(theta),
    phi(t, theta) = -log(1 - exp(-n t^2 / (theta |A|))),      prior f(theta) ∝ 1/theta.

Z(theta) is intractable, so theta is updated with the double MH sampler: each
step proposes theta', generates an auxiliary point pattern y from f(. | theta')
with one MH sweep started at the observed pattern, and accepts with a ratio in
which Z(theta) cancels.

No data set is given, so the observed pattern is simulated from the model at a
known theta on A = [0, 1]^2. Point patterns are stored as flat vectors
(x_1, y_1, x_2, y_2, ...).

The prior 1/theta is restricted to [THETA_MIN, THETA_MAX]: as theta -> 0 or
theta -> infinity the likelihood tends to a positive constant, so the
unrestricted 1/theta prior gives a posterior that cannot be normalised.
"""

import numpy as np
from scipy.spatial.distance import pdist

import common
from common import ORANGE, SERIES, TEXT, plt, save_fig, save_results
from Metropolis_Hastings import double_metropolis_hastings, metropolis_hastings

N_POINTS, THETA_TRUE, AREA = 50, 1.0, 1.0
THETA_MIN, THETA_MAX = 0.001, 10.0
N_DRAWS, BURN_IN, STEP, DATA_SWEEPS, SEED = 10000, 1000, 0.6, 500, 546


def log_unnorm_lik(points, theta):
    """log g(x, theta) = sum_{i<j} log(1 - exp(-n d_ij^2 / (theta |A|)))."""
    pts = points.reshape(-1, 2)
    d2 = pdist(pts, "sqeuclidean")
    return np.sum(np.log(-np.expm1(-len(pts) * d2 / (theta[0] * AREA))))


def move_one_point(points, rng):
    """Move one random point to a uniform location in A (symmetric proposal)."""
    new = points.copy()
    i = rng.integers(points.size // 2)
    new[2 * i:2 * i + 2] = rng.uniform(0, 1, 2)
    return new


def log_prior(theta):
    return -np.log(theta[0]) if THETA_MIN <= theta[0] <= THETA_MAX else -np.inf


def log_normal_walk(sigma):
    """theta' = theta * exp(sigma Z): keeps theta > 0. Asymmetric, q(theta' | theta) ∝ 1/theta'."""
    propose = lambda theta, rng: theta * np.exp(sigma * rng.standard_normal(theta.size))
    log_q = lambda to, frm: -np.log(to[0])   # the Gaussian part is symmetric in log space and cancels
    return propose, log_q


def simulate_pattern(rng):
    """Approximate draw from f(. | THETA_TRUE) by a long MH run over point moves."""
    res = metropolis_hastings(lambda x: log_unnorm_lik(x, np.array([THETA_TRUE])),
                              x0=rng.uniform(0, 1, 2 * N_POINTS), n_samples=1,
                              propose=move_one_point, burn_in=DATA_SWEEPS * N_POINTS, seed=rng)
    return res.samples[0]


def plot(th, mean):
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    ax.hist(th, bins=50, density=True, color=SERIES[0], alpha=0.75, edgecolor="white", linewidth=0.6)
    ax.axvline(THETA_TRUE, color=ORANGE, lw=2, label=f"Generating value: {THETA_TRUE}")
    ax.axvline(mean, color=TEXT, lw=1.5, ls="--", label=f"Posterior mean: {mean:.3f}")
    ax.set(xlabel=r"$\theta$", ylabel="Density", title=r"Posterior of $\theta$ from double MH")
    ax.legend(loc="upper right")
    save_fig(fig, "q4_soft_core")


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    data = simulate_pattern(rng)
    print(f"Q4: {N_POINTS} points on the unit square, simulated at theta = {THETA_TRUE}")

    propose, log_q = log_normal_walk(STEP)
    res = double_metropolis_hastings(
        log_prior, log_unnorm_lik, data, theta0=0.5, n_samples=N_DRAWS,
        propose=propose, log_proposal=log_q,
        aux_propose=move_one_point, n_aux_steps=N_POINTS,      # one sweep per auxiliary draw
        burn_in=BURN_IN, seed=rng,
    )

    th = res.samples[:, 0]
    lo, hi = np.quantile(th, [0.025, 0.975])
    print(f"  posterior mean {th.mean():.3f}, 95% interval ({lo:.3f}, {hi:.3f}), "
          f"acceptance {res.acceptance_rate:.3f}")

    save_results("q4", {
        "settings": {"points": N_POINTS, "theta_true": THETA_TRUE, "area": AREA,
                     "prior_range": [THETA_MIN, THETA_MAX], "draws": N_DRAWS, "burn_in": BURN_IN,
                     "log_step_sd": STEP, "inner_sweeps": 1, "data_sweeps": DATA_SWEEPS, "seed": SEED},
        "data": data.reshape(-1, 2).tolist(),
        "posterior_mean": float(th.mean()),
        "ci95": [float(lo), float(hi)],
        "acceptance": res.acceptance_rate,
    })
    plot(th, th.mean())
