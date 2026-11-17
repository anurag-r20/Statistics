"""
STAT546 HW7, Problem 2: SGHMC compared with SGLD on the Bayesian logistic regression of Problem 1.

All runs use minibatch size 100 and 50,000 iterations (the first 5,000 discarded), and each
setting is repeated with 5 seeds. SGHMC uses unit mass, learning rate eta and friction alpha;
"corrected" runs set the gradient-noise estimate beta_hat = eta V / 2, where V is the
per-coordinate variance of the minibatch gradient at the posterior mode.
"""

import time

import arviz as az
import numpy as np
from scipy.stats import gaussian_kde

import common
from common import LINESTYLES, SERIES, TEXT_2, plt, save_fig, save_results
from data import D, N_TRAIN, PRIOR_VAR, laplace, predictive_accuracy, reference_mcmc, simulate
from Stochastic_Gradient_MCMC import (gaussian_prior_grad, gradient_noise_variance, logistic_loglik_grad,
                                      sghmc, sgld)

N_ITER, BURN_IN, BATCH, SEEDS, COEF = 50_000, 5_000, 100, [11, 12, 13, 14, 15], 1
RUNS = {
    "SGLD, eps = 2e-5": {"alg": "sgld", "step": 2e-5},
    "SGLD, eps = 1e-4": {"alg": "sgld", "step": 1e-4},
    "SGHMC, eta = 1e-5, alpha = 0.3": {"alg": "sghmc", "eta": 1e-5, "alpha": 0.3, "corrected": False},
    "SGHMC, eta = 1e-5, alpha = 0.3, corrected": {"alg": "sghmc", "eta": 1e-5, "alpha": 0.3, "corrected": True},
    "SGHMC, eta = 2e-6, alpha = 0.05, corrected": {"alg": "sghmc", "eta": 2e-6, "alpha": 0.05, "corrected": True},
}


def run(config, grads, V, seed):
    args = (*grads, N_TRAIN, np.zeros(D), N_ITER, BATCH)
    if config["alg"] == "sgld":
        return sgld(*args, config["step"], burn_in=BURN_IN, seed=seed)
    beta_hat = 0.5 * config["eta"] * V if config["corrected"] else 0.0
    return sghmc(*args, config["eta"], config["alpha"], noise_estimate=beta_hat, burn_in=BURN_IN, seed=seed)


def measures(res, m, sd, Xt, yt):
    ess = np.array([float(az.ess(res.samples[None, :, j])) for j in range(D)])
    return {
        "cpu_time": res.cpu_time,
        "ess_min": float(ess.min()), "ess_median": float(np.median(ess)),
        "ess_per_second": float(np.median(ess) / res.cpu_time),
        "mean_error": float(np.sqrt(np.mean(((res.samples.mean(0) - m) / sd) ** 2))),
        "sd_ratio": float(np.mean(res.samples.std(0) / sd)),
        "test_accuracy": predictive_accuracy(res.samples[::50], Xt, yt),
    }


def plot(ref, examples, m, sd):
    # One fixed colour and line style per run, in every panel.
    style = {name: (SERIES[i], LINESTYLES[i]) for i, name in enumerate(RUNS)}
    label = {
        "SGLD, eps = 2e-5": r"SGLD, $\epsilon=2\times10^{-5}$",
        "SGLD, eps = 1e-4": r"SGLD, $\epsilon=10^{-4}$",
        "SGHMC, eta = 1e-5, alpha = 0.3": r"SGHMC, $\eta=10^{-5}$, $\alpha=0.3$, $\hat\beta=0$",
        "SGHMC, eta = 1e-5, alpha = 0.3, corrected": r"SGHMC, $\eta=10^{-5}$, $\alpha=0.3$, corrected",
        "SGHMC, eta = 2e-6, alpha = 0.05, corrected": r"SGHMC, $\eta=2\times10^{-6}$, $\alpha=0.05$, corrected",
    }
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), gridspec_kw={"wspace": 0.3})
    shown = ["SGLD, eps = 2e-5", "SGLD, eps = 1e-4", "SGHMC, eta = 2e-6, alpha = 0.05, corrected"]

    ax = axes[0]
    for name in shown:
        c, ls = style[name]
        ax.plot(np.arange(3000), examples[name][:3000, COEF], color=c, ls=ls, lw=0.8)
    ax.axhspan(m[COEF] - 2 * sd[COEF], m[COEF] + 2 * sd[COEF], color=TEXT_2, alpha=0.15, lw=0)
    ax.set(xlabel="Iteration after burn-in", ylabel=rf"$\theta_{COEF}$", title="Trace (first seed)")

    ax = axes[1]
    lags = np.arange(0, 801)
    for name in shown:
        c, ls = style[name]
        x = examples[name][:, COEF] - examples[name][:, COEF].mean()
        acf = np.correlate(x, x, mode="full")[len(x) - 1:][: lags.size] / (x @ x)
        ax.plot(lags, acf, color=c, ls=ls, lw=1.4, label=label[name])
    ax.axhline(0, color=TEXT_2, lw=0.8)
    ax.set(xlabel="Lag (iterations)", ylabel="Autocorrelation", title=rf"Autocorrelation of $\theta_{COEF}$")
    ax.legend(loc="upper right", fontsize=6.5, title="Left and middle panels", title_fontsize=6.5)

    ax = axes[2]
    grid = np.linspace(m[COEF] - 5 * sd[COEF], m[COEF] + 5 * sd[COEF], 400)
    ax.hist(ref[:, COEF], bins=60, density=True, color=TEXT_2, alpha=0.25, label="Full-data MCMC")
    for name in ["SGLD, eps = 1e-4", "SGHMC, eta = 1e-5, alpha = 0.3", "SGHMC, eta = 1e-5, alpha = 0.3, corrected"]:
        c, ls = style[name]
        ax.plot(grid, gaussian_kde(examples[name][:, COEF])(grid), color=c, ls=ls, lw=1.4, label=label[name])
    ax.set(xlabel=rf"$\theta_{COEF}$", ylabel="Density", title="Effect of the noise correction", ylim=(0, 14.5))
    ax.legend(loc="upper left", fontsize=6)
    save_fig(fig, "q2_sghmc")


if __name__ == "__main__":
    X, y, Xt, yt = simulate()
    ref, _, ref_cpu = reference_mcmc(X, y)
    m, sd = ref.mean(axis=0), ref.std(axis=0)
    grads = (gaussian_prior_grad(PRIOR_VAR), logistic_loglik_grad(X, y))

    start = time.process_time()
    mode, _ = laplace(X, y)
    V = gradient_noise_variance(*grads, N_TRAIN, BATCH, mode, n_draws=2000, seed=10)
    v_cpu = time.process_time() - start
    print(f"Q2: gradient-noise variance at the mode, median {np.median(V):.3g} (CPU {v_cpu:.2f} s)")

    summary, per_seed, examples = {}, {}, {}
    for name, config in RUNS.items():
        rows = []
        for seed in SEEDS:
            res = run(config, grads, V, seed)
            rows.append(measures(res, m, sd, Xt, yt))
            if seed == SEEDS[0]:
                examples[name] = res.samples
        per_seed[name] = rows
        summary[name] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        summary[name].update({f"{k}_sd": float(np.std([r[k] for r in rows], ddof=1)) for k in rows[0]})
        s = summary[name]
        print(f"  {name:45s} CPU {s['cpu_time']:.2f} s, ESS min/median {s['ess_min']:.0f}/{s['ess_median']:.0f}, "
              f"ESS/s {s['ess_per_second']:.0f}, mean error {s['mean_error']:.3f}, SD ratio {s['sd_ratio']:.3f}",
              flush=True)

    save_results("q2", {
        "settings": {"iterations": N_ITER, "burn_in": BURN_IN, "batch_size": BATCH, "seeds": SEEDS,
                     "runs": RUNS, "noise_variance_draws": 2000},
        "noise_variance": V.tolist(), "noise_variance_cpu_time": v_cpu,
        "beta_hat_max": {"eta=1e-5": float(0.5 * 1e-5 * V.max()), "eta=2e-6": float(0.5 * 2e-6 * V.max())},
        "reference_cpu_time": ref_cpu,
        "summary": summary, "per_seed": per_seed,
    })
    plot(ref, examples, m, sd)
