"""
STAT546 HW7, Problem 3: numerical study of SGLD and SGHMC on a moderately large Bayesian
linear regression (large_model.py: n = 100,000, d = 100, exact Gaussian posterior).

Baseline: minibatch 1,000, 20,000 iterations (the first 10% discarded), SGLD step 6e-7,
SGHMC learning rate 3e-9 and friction 0.01 with beta_hat = 0. Each study changes one factor:
minibatch size, learning rate, friction (SGHMC), and number of iterations. Runs are sequential
on one BLAS thread, so their CPU times are comparable.
"""

import numpy as np

import common
from common import SERIES, TEXT, TEXT_2, plt, save_fig, save_results
from large_model import D, N_TRAIN, exact_posterior, gradients, score, simulate
from Stochastic_Gradient_MCMC import sghmc, sgld

SEED, N_ITER, BATCH = 21, 20_000, 1000
SGLD_STEP, SGHMC_ETA, SGHMC_ALPHA = 6e-7, 3e-9, 0.01
ITER_GRID = [2_500, 5_000, 10_000, 20_000, 50_000]


def run(alg, grads, batch=BATCH, n_iter=N_ITER, step=None, eta=None, alpha=None, keep_all=False):
    burn = 0 if keep_all else n_iter // 10
    args = (*grads, N_TRAIN, np.zeros(D), n_iter, batch)
    if alg == "SGLD":
        return sgld(*args, step or SGLD_STEP, burn_in=burn, seed=SEED)
    return sghmc(*args, eta or SGHMC_ETA, alpha or SGHMC_ALPHA, burn_in=burn, seed=SEED)


def plot(res):
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.7), gridspec_kw={"wspace": 0.32})
    colors = {"SGLD": SERIES[0], "SGHMC": SERIES[1]}
    markers = {"SGLD": "o", "SGHMC": "s"}

    ax = axes[0]
    for alg in ["SGLD", "SGHMC"]:
        rows = res["iterations"][alg]
        ax.plot([r["iterations"] for r in rows], [r["mean_error"] for r in rows], marker=markers[alg],
                color=colors[alg], lw=1.4, label=alg)
    ax.set(xscale="log", yscale="log", xlabel="Iterations", ylabel="Posterior mean error (SD units)",
           title="Number of iterations")
    ax.set_xticks(ITER_GRID, ["2,500", "5,000", "10,000", "20,000", "50,000"], fontsize=7)
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[1]
    for alg in ["SGLD", "SGHMC"]:
        rows = res["batch_size"][alg]
        ax.plot([r["batch_size"] for r in rows], [r["sd_ratio"] for r in rows], marker=markers[alg],
                color=colors[alg], lw=1.4, label=alg)
    ax.axhline(1.0, color=TEXT_2, lw=0.9, ls=":")
    ax.set(xscale="log", xlabel="Minibatch size", ylabel="Estimated SD / exact SD", title="Minibatch size")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[2]
    points = [("SGLD", r, rf"$\epsilon$ = {r['step']:.0e}") for r in res["learning_rate"]["SGLD"]]
    points += [("SGHMC", r, rf"$\eta$ = {r['eta']:.0e}") for r in res["learning_rate"]["SGHMC"]
               if r["eta"] != SGHMC_ETA]
    points += [("SGHMC", r, rf"$\alpha$ = {r['alpha']}") for r in res["friction"]["SGHMC"] if r["alpha"] != SGHMC_ALPHA]
    base = next(r for r in res["friction"]["SGHMC"] if r["alpha"] == SGHMC_ALPHA)
    points.append(("SGHMC", base, rf"baseline $\eta$ = {SGHMC_ETA:.0e}, $\alpha$ = {SGHMC_ALPHA}"))
    nudge = {rf"$\alpha$ = 0.03": (5, 4), rf"$\eta$ = 1e-09": (5, -9)}
    for alg, r, text in points:
        x, yv = r["sd_ratio"], r["ess_min"] / r["cpu_time"]
        ax.scatter(x, yv, color=colors[alg], marker=markers[alg], s=36, zorder=3)
        ax.annotate(text, (x, yv), xytext=nudge.get(text, (5, 3)), textcoords="offset points", fontsize=6.5,
                    color=TEXT)
    ax.scatter([], [], color=colors["SGLD"], marker="o", label="SGLD")
    ax.scatter([], [], color=colors["SGHMC"], marker="s", label="SGHMC")
    ax.set(yscale="log", xlabel="Estimated SD / exact SD", ylabel="Minimum ESS per CPU second",
           title="Learning rate and friction", xlim=(1.05, 2.35))
    ax.legend(loc="lower right", fontsize=7)
    save_fig(fig, "q3_study")


if __name__ == "__main__":
    X, y, Xt, yt, beta = simulate()
    m, S = exact_posterior(X, y)
    sd = np.sqrt(np.diag(S))
    rmse_exact = float(np.sqrt(np.mean((yt - Xt @ m) ** 2)))
    grads = gradients(X, y)
    measure = lambda r: score(r.samples, r.cpu_time, m, sd, Xt, yt, rmse_exact)
    print(f"Q3: exact posterior SDs {sd.min():.4f}-{sd.max():.4f}; test RMSE of the exact posterior mean "
          f"{rmse_exact:.5f}")

    def report(study, alg, row):
        print(f"  {study:13s} {alg:5s} {row['label']:14s}: CPU {row['cpu_time']:6.1f} s, "
              f"ESS min/median {row['ess_min']:5.0f}/{row['ess_median']:5.0f}, "
              f"mean error {row['mean_error']:.3f}, SD ratio {row['sd_ratio']:.3f}, "
              f"RMSE excess {row['test_rmse_excess']:.1e}", flush=True)

    res = {"batch_size": {}, "learning_rate": {}, "friction": {}, "iterations": {}}
    for alg in ["SGLD", "SGHMC"]:
        res["batch_size"][alg] = []
        for b in [100, 1000, 10_000]:
            row = {"batch_size": b, "label": f"b = {b}", **measure(run(alg, grads, batch=b))}
            res["batch_size"][alg].append(row)
            report("batch size", alg, row)

    res["learning_rate"]["SGLD"] = []
    for step in [2e-7, 6e-7, 2e-6]:
        row = {"step": step, "label": f"{step:.0e}", **measure(run("SGLD", grads, step=step))}
        res["learning_rate"]["SGLD"].append(row)
        report("learning rate", "SGLD", row)
    res["learning_rate"]["SGHMC"] = []
    for eta in [1e-9, 3e-9, 1e-8]:
        row = {"eta": eta, "alpha": SGHMC_ALPHA, "label": f"{eta:.0e}", **measure(run("SGHMC", grads, eta=eta))}
        res["learning_rate"]["SGHMC"].append(row)
        report("learning rate", "SGHMC", row)
    res["friction"]["SGHMC"] = []
    for alpha in [0.003, 0.01, 0.03, 0.1]:
        row = {"eta": SGHMC_ETA, "alpha": alpha, "label": f"a = {alpha}",
               **measure(run("SGHMC", grads, alpha=alpha))}
        res["friction"]["SGHMC"].append(row)
        report("friction", "SGHMC", row)

    for alg in ["SGLD", "SGHMC"]:
        full = run(alg, grads, n_iter=ITER_GRID[-1], keep_all=True)
        res["iterations"][alg] = []
        for T in ITER_GRID:
            part = full.samples[T // 10:T]
            row = {"iterations": T, "label": f"T = {T}",
                   **score(part, full.cpu_time * T / ITER_GRID[-1], m, sd, Xt, yt, rmse_exact)}
            res["iterations"][alg].append(row)
            report("iterations", alg, row)

    save_results("q3", {
        "settings": {"n_train": N_TRAIN, "d": D, "seed": SEED, "baseline_iterations": N_ITER,
                     "baseline_batch": BATCH, "sgld_step": SGLD_STEP, "sghmc_eta": SGHMC_ETA,
                     "sghmc_alpha": SGHMC_ALPHA, "beta_hat": 0.0, "burn_in": "first 10% of each run",
                     "iteration_grid": ITER_GRID,
                     "iteration_study_cpu": "CPU time of the 50,000-iteration run, prorated to each prefix"},
        "exact_sd_range": [float(sd.min()), float(sd.max())], "exact_test_rmse": rmse_exact,
        "results": res,
    })
    plot(res)
