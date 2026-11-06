"""
STAT546 HW6, Problem 4: EM for a finite mixture of univariate normal distributions.

Simulated data: n = 1000 draws from 0.3 N(-2, 0.7^2) + 0.5 N(1, 1) + 0.2 N(5, 1.5^2).
EM is run from 10 random starting values; the fit with the highest observed-data
log-likelihood is reported. The convergence of the log-likelihood is examined through
its monotone increase and the geometric decay of its distance to the final value.
"""

import numpy as np
from scipy.stats import norm

import common
from common import ORANGE, SERIES, TEXT, TEXT_2, plt, save_fig, save_results
from Expectation_Maximization import normal_mixture_em, normal_mixture_loglik

N, SEED, N_STARTS, MAX_ITER, TOL = 1000, 546, 10, 5000, 1e-12
WEIGHTS = np.array([0.3, 0.5, 0.2])
MEANS = np.array([-2.0, 1.0, 5.0])
SDS = np.array([0.7, 1.0, 1.5])


def mixture_density(grid, w, m, v):
    return np.sum(w * norm.pdf(grid[:, None], m, np.sqrt(v)), axis=1)


GAP_FLOOR = 1e-6   # below this the gap is distorted by l* itself being the last iterate


def linear_rate(gaps):
    """Median ratio of successive gaps, over the stretch where the gap lies in (GAP_FLOOR, 1e-1)."""
    ok = (gaps[:-1] > GAP_FLOOR) & (gaps[:-1] < 1e-1) & (gaps[1:] > GAP_FLOOR)
    return float(np.median(gaps[1:][ok] / gaps[:-1][ok]))


def plot(x, fits, best, ll_star):
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.5), gridspec_kw={"wspace": 0.34})
    ax = axes[0]
    for f in fits:
        ax.plot(np.arange(f.log_likelihood.size), f.log_likelihood, color=SERIES[0], alpha=0.5, lw=1.0)
    ax.plot(np.arange(fits[best].log_likelihood.size), fits[best].log_likelihood, color=ORANGE, lw=1.4,
            label="Highest final value")
    ax.set(xscale="symlog", xlabel="Iteration", ylabel="Observed-data log-likelihood",
           title=f"{len(fits)} random starts", ylim=(ll_star - 250, ll_star + 20))
    ax.legend(loc="lower right", fontsize=7)

    ax = axes[1]
    gaps = ll_star - fits[best].log_likelihood
    keep = gaps > GAP_FLOOR
    ax.plot(np.arange(gaps.size)[keep], gaps[keep], color=ORANGE, lw=1.4)
    ax.set(yscale="log", xlabel="Iteration", ylabel=r"$\ell^* - \ell(\theta^{(t)})$",
           title="Distance to the final log-likelihood")

    ax = axes[2]
    grid = np.linspace(x.min() - 1, x.max() + 1, 600)
    f = fits[best]
    ax.hist(x, bins=50, density=True, color=SERIES[0], alpha=0.6, edgecolor="white", linewidth=0.6, label="Data")
    ax.plot(grid, mixture_density(grid, f.weights, f.means, f.variances), color=ORANGE, lw=1.8, label="EM fit")
    ax.plot(grid, mixture_density(grid, WEIGHTS, MEANS, SDS ** 2), color=TEXT, lw=1.0, ls="--", label="True")
    ax.set(xlabel="x", ylabel="Density", title="Fitted mixture")
    ax.legend(loc="upper right", fontsize=7)
    save_fig(fig, "q4_em_mixture")


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    z = rng.choice(3, size=N, p=WEIGHTS)
    x = rng.normal(MEANS[z], SDS[z])

    fits = [normal_mixture_em(x, 3, max_iter=MAX_ITER, tol=TOL, seed=s) for s in range(N_STARTS)]
    finals = np.array([f.log_likelihood[-1] for f in fits])
    best = int(np.argmax(finals))
    f, ll_star = fits[best], finals[best]
    min_increase = min(float(np.min(np.diff(g.log_likelihood))) for g in fits)
    rate = linear_rate(ll_star - f.log_likelihood)

    print(f"Q4: EM, n = {N}, {N_STARTS} starts; best final log-likelihood {ll_star:.4f} "
          f"after {f.n_iter} iterations (converged = {f.converged})")
    print(f"  starts reaching it (within 1e-6): {int(np.sum(ll_star - finals < 1e-6))} / {N_STARTS}")
    print(f"  smallest increase over all iterations and starts: {min_increase:.2e}")
    print(f"  linear convergence rate of the log-likelihood gap: {rate:.3f}")
    print(f"  weights {f.weights.round(3)}, means {f.means.round(3)}, sds {np.sqrt(f.variances).round(3)}")

    save_results("q4", {
        "settings": {"n": N, "seed": SEED, "starts": N_STARTS, "max_iter": MAX_ITER, "tol": TOL,
                     "true_weights": WEIGHTS.tolist(), "true_means": MEANS.tolist(), "true_sds": SDS.tolist()},
        "estimate": {"weights": f.weights.tolist(), "means": f.means.tolist(),
                     "sds": np.sqrt(f.variances).tolist()},
        "final_log_likelihood": ll_star,
        "log_likelihood_at_true_parameters": normal_mixture_loglik(x, WEIGHTS, MEANS, SDS ** 2),
        "final_log_likelihood_per_start": finals.tolist(),
        "iterations_per_start": [g.n_iter for g in fits],
        "converged_per_start": [g.converged for g in fits],
        "best_start": best,
        "smallest_increase": min_increase,
        "linear_rate": rate,
        "best_log_likelihood_path": f.log_likelihood.tolist(),
        "data": x.tolist(),
    })
    plot(x, fits, best, ll_star)
