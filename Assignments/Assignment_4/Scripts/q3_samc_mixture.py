"""
STAT546 HW4, Problem 3: SAMC for the mean and covariance matrix of

    f(x) = (1/3) N(mu1, Sigma1) + (1/3) N(mu2, Sigma2) + (1/3) N(mu3, Sigma3),

with mu1 = (-8, -8), mu2 = (6, 6), mu3 = (0, 0) and correlations 0.9, -0.9 and 0.

The energy is U(x) = -log f(x), and the sample space is partitioned by energy into
unit-width subregions with levels 2.5, 3.5, ..., 14.5 (14 subregions, the first and last open).
Expectations under f are importance-weighted averages of the SAMC samples, with
weight exp(theta_J) for a sample in subregion J. The exact mean and covariance are
known, which gives a direct check.
"""

import numpy as np
from scipy.special import logsumexp

import common
from common import ORANGE, SERIES, TEXT_2, plt, save_fig, save_results
from Metropolis_Hastings import gaussian_random_walk
from SAMC import energy_regions, samc

MUS = [np.array([-8.0, -8.0]), np.array([6.0, 6.0]), np.array([0.0, 0.0])]
SIGMAS = [np.array([[1.0, 0.9], [0.9, 1.0]]), np.array([[1.0, -0.9], [-0.9, 1.0]]), np.eye(2)]
LEVELS = np.arange(2.5, 15.01, 1.0)
N_ITERS, BURN_IN, THIN, STEP, T0, SEED = 4_000_000, 400_000, 10, 1.5, 1000, 546

PRECISIONS = [np.linalg.inv(S) for S in SIGMAS]
LOG_CONSTS = [np.log(1 / 3) - np.log(2 * np.pi) - 0.5 * np.log(np.linalg.det(S)) for S in SIGMAS]


def log_f(x):
    return logsumexp([c - 0.5 * (x - m) @ P @ (x - m) for m, P, c in zip(MUS, PRECISIONS, LOG_CONSTS)])


def energy(x):
    return -log_f(x)


def exact_moments():
    mean = sum(MUS) / 3
    cov = sum(S + np.outer(m, m) for m, S in zip(MUS, SIGMAS)) / 3 - np.outer(mean, mean)
    return mean, cov


def plot(samples):
    fig, ax = plt.subplots(figsize=(5.0, 4.4))
    g = np.linspace(-16, 14, 300)
    X, Y = np.meshgrid(g, g)
    Z = np.array([[log_f(np.array([a, b])) for a in g] for b in g])
    pts = samples[::40]
    ax.scatter(pts[:, 0], pts[:, 1], s=2, color=SERIES[0], alpha=0.35, rasterized=True, label="SAMC samples")
    ax.contour(X, Y, Z, levels=-np.array([12, 9, 6, 4, 3]), colors=ORANGE, linewidths=0.9, linestyles="solid")
    ax.plot([], [], color=ORANGE, lw=0.9, label="Contours of log f")
    ax.set(xlabel="x1", ylabel="x2", aspect="equal", title="SAMC samples for the three-component mixture",
           xlim=(-16, 14), ylim=(-16, 14))
    ax.legend(loc="upper left", fontsize=7)
    save_fig(fig, "q3_samc_mixture")


if __name__ == "__main__":
    region, m = energy_regions(LEVELS)
    res = samc(energy, np.zeros(2), N_ITERS, region, m, propose=gaussian_random_walk(STEP),
               t0=T0, burn_in=BURN_IN, thin=THIN, seed=SEED)

    w = res.importance_weights()
    mean = w @ res.samples
    d = res.samples - mean
    cov = (w[:, None] * d).T @ d
    mean_exact, cov_exact = exact_moments()
    print(f"Q3: SAMC, {N_ITERS} iterations, {m} subregions, acceptance {res.acceptance_rate:.3f}")
    print(f"  mean estimate {mean.round(3)}, exact {mean_exact.round(3)}")
    print(f"  covariance estimate {cov.round(3).tolist()}, exact {cov_exact.round(3).tolist()}")

    save_results("q3", {
        "settings": {"levels": [LEVELS[0], LEVELS[-1], 1.0], "regions": m, "iterations": N_ITERS,
                     "burn_in": BURN_IN, "thin": THIN, "proposal_sd": STEP, "gain_t0": T0, "seed": SEED},
        "mean": mean.tolist(), "mean_exact": mean_exact.tolist(),
        "cov": cov.tolist(), "cov_exact": cov_exact.tolist(),
        "acceptance": res.acceptance_rate,
    })
    plot(res.samples)
