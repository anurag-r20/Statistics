"""
STAT546 HW7, Problem 1: SGLD for Bayesian logistic regression with a Gaussian prior,
compared with full-data random-walk Metropolis-Hastings (data.py).

Two studies, each changing one factor (50,000 iterations, the first 5,000 discarded):
  - minibatch size b in {10, 100, 1000, 5000} with constant step 1e-4 (b = 5000 uses
    all the data, i.e. unadjusted Langevin with the exact gradient);
  - step-size sequence with b = 100: constant 2e-5, 1e-4, 5e-4, and the decreasing
    sequence 5e-4 (1 + t / 1000)^(-0.55).
Decreasing-step runs are summarised with step-size-weighted averages.

For a constant step eps, the SD ratio is compared with the prediction
sqrt((1 + eps V_j / 4) / (1 - eps lambda_j / 4)), averaged over coordinates, where V_j is the
variance of the minibatch gradient at the posterior mode and lambda_j = 1 / sd_j^2.
"""

import numpy as np
from scipy.stats import gaussian_kde

import common
from common import LINESTYLES, SERIES, TEXT, TEXT_2, plt, save_fig, save_results
from data import D, N_TRAIN, PRIOR_VAR, laplace, predictive_accuracy, reference_mcmc, simulate
from Stochastic_Gradient_MCMC import (gaussian_prior_grad, gradient_noise_variance, logistic_loglik_grad, sgld,
                                      weighted_moments)

N_ITER, BURN_IN, SEED, COEF = 50_000, 5_000, 548, 1      # COEF: coefficient shown in the figure
BATCH_RUNS = {f"b = {b}": {"batch_size": b, "step": 1e-4} for b in [10, 100, 1000, 5000]}
STEP_RUNS = {
    "constant 2e-5": {"batch_size": 100, "step": 2e-5},
    "constant 1e-4": {"batch_size": 100, "step": 1e-4},
    "constant 5e-4": {"batch_size": 100, "step": 5e-4},
    "decreasing": {"batch_size": 100, "step": "decreasing"},
}


def decreasing(t):
    return 5e-4 * (1 + t / 1000) ** (-0.55)


def iterations_to_posterior(path, m, sd):
    """First iteration at which every coordinate is within 3 posterior SDs of the posterior mean."""
    inside = np.all(np.abs(path - m) < 3 * sd, axis=1)
    return int(np.argmax(inside)) if inside.any() else None


def predicted_sd_ratio(config, V100, sd):
    """Stationary SD inflation of constant-step SGLD on a locally Gaussian posterior (see report)."""
    if config["step"] == "decreasing":
        return None
    eps, V = config["step"], V100 * 100 / config["batch_size"]
    if config["batch_size"] == N_TRAIN:
        V = np.zeros_like(V100)                     # all the data: no gradient noise
    return float(np.mean(np.sqrt((1 + eps * V / 4) / (1 - eps / sd ** 2 / 4))))


def run(config, X, y, seed):
    step = decreasing if config["step"] == "decreasing" else config["step"]
    return sgld(gaussian_prior_grad(PRIOR_VAR), logistic_loglik_grad(X, y), N_TRAIN, np.zeros(D), N_ITER,
                config["batch_size"], step, seed=seed)


def summarise(res, m, sd, Xt, yt):
    keep = slice(BURN_IN, None)
    mean, s = weighted_moments(res.samples[keep], res.step_sizes[keep])
    return {
        "mean_error": float(np.sqrt(np.mean(((mean - m) / sd) ** 2))),
        "sd_ratio": float(np.mean(s / sd)),
        "iterations_to_posterior": iterations_to_posterior(res.samples, m, sd),
        "test_accuracy": predictive_accuracy(res.samples[keep][::50], Xt, yt),
        "cpu_time": res.cpu_time,
    }


def plot(ref, batch_paths, step_paths, m, sd):
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), gridspec_kw={"wspace": 0.3, "width_ratios": [1.1, 1, 1]})
    ax = axes[0]
    grid = np.linspace(m[COEF] - 6 * sd[COEF], m[COEF] + 6 * sd[COEF], 400)
    ax.hist(ref[:, COEF], bins=60, density=True, color=TEXT_2, alpha=0.25, label="Full-data MCMC")
    for i, (name, path) in enumerate(batch_paths.items()):
        ax.plot(grid, gaussian_kde(path[BURN_IN:, COEF])(grid), color=SERIES[i], ls=LINESTYLES[i], lw=1.4,
                label=f"SGLD, {name}")
    ax.set(xlabel=rf"$\theta_{COEF}$", ylabel="Posterior density", title="Minibatch size (step $10^{-4}$)")
    ax.legend(loc="upper left", fontsize=7)

    ax = axes[1]
    t = np.arange(1, 2001)
    for i, (name, path) in enumerate(step_paths.items()):
        ax.plot(t, path[:2000, COEF], color=SERIES[i], ls=LINESTYLES[i], lw=1.0, label=name)
    ax.axhspan(m[COEF] - 2 * sd[COEF], m[COEF] + 2 * sd[COEF], color=TEXT_2, alpha=0.15, lw=0)
    ax.set(xscale="log", xlabel="Iteration", ylabel=rf"$\theta_{COEF}$", ylim=(0, 1.3),
           title="Step size: start of the run (b = 100)")
    ax.legend(loc="lower right", fontsize=7)

    ax = axes[2]
    names = list(step_paths)
    parts = ax.violinplot([step_paths[n][BURN_IN:, COEF] for n in names], showextrema=False, widths=0.8)
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(SERIES[i])
        body.set_alpha(0.6)
    ax.axhspan(m[COEF] - 2 * sd[COEF], m[COEF] + 2 * sd[COEF], color=TEXT_2, alpha=0.15, lw=0)
    ax.set_xticks(range(1, len(names) + 1), [n.replace("constant ", "") for n in names], fontsize=8)
    ax.set(ylabel=rf"$\theta_{COEF}$ after burn-in", title="Step size: spread of the samples")
    ax.grid(axis="x", visible=False)
    fig.text(0.5, -0.04, r"Shaded bands: full-data MCMC mean $\pm$ 2 SD.", ha="center", color=TEXT_2, fontsize=8)
    save_fig(fig, "q1_sgld")


if __name__ == "__main__":
    X, y, Xt, yt = simulate()
    ref, acceptance, ref_cpu = reference_mcmc(X, y)
    m, sd = ref.mean(axis=0), ref.std(axis=0)
    ref_accuracy = predictive_accuracy(ref[::50], Xt, yt)
    print(f"Q1: reference MCMC acceptance {acceptance:.3f}, CPU {ref_cpu:.1f} s, test accuracy {ref_accuracy:.4f}")

    grads = (gaussian_prior_grad(PRIOR_VAR), logistic_loglik_grad(X, y))
    V100 = gradient_noise_variance(*grads, N_TRAIN, 100, laplace(X, y)[0], n_draws=2000, seed=10)

    results, paths = {}, {}
    for study, runs in [("batch_size", BATCH_RUNS), ("step_size", STEP_RUNS)]:
        results[study], paths[study] = {}, {}
        for name, config in runs.items():
            res = run(config, X, y, SEED)
            results[study][name] = {"config": config, **summarise(res, m, sd, Xt, yt),
                                    "predicted_sd_ratio": predicted_sd_ratio(config, V100, sd)}
            paths[study][name] = res.samples
            r = results[study][name]
            print(f"  {study:10s} {name:14s}: mean error {r['mean_error']:.3f}, SD ratio {r['sd_ratio']:.3f} "
                  f"(predicted {r['predicted_sd_ratio']}), "
                  f"reached posterior at {r['iterations_to_posterior']}, CPU {r['cpu_time']:.1f} s", flush=True)

    save_results("q1", {
        "settings": {"n_train": N_TRAIN, "d": D, "prior_var": PRIOR_VAR, "iterations": N_ITER,
                     "burn_in": BURN_IN, "seed": SEED, "decreasing": "5e-4 (1 + t/1000)^-0.55"},
        "reference": {"acceptance": acceptance, "cpu_time": ref_cpu, "test_accuracy": ref_accuracy,
                      "mean": m.tolist(), "sd": sd.tolist(), "samples": len(ref)},
        "gradient_noise_variance_b100": V100.tolist(),
        "results": results,
    })
    plot(ref, paths["batch_size"], paths["step_size"], m, sd)
