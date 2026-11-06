"""
STAT546 HW6, Problem 1: Robbins-Monro for the root of h(theta) = E(X - theta) = 0,
with X ~ N(mu, sigma^2). The root is theta* = mu.

The update is theta_k = theta_{k-1} + a_k (X_k - theta_{k-1}). Because h is linear, the
error e_k = theta_k - mu satisfies e_k = (1 - a_k) e_{k-1} + a_k (X_k - mu), so the mean
squared error has the exact recursion

    MSE_k = (1 - a_k)^2 MSE_{k-1} + a_k^2 sigma^2,    MSE_0 = (theta_0 - mu)^2.

Five step-size schedules are compared over independent replicates, and the simulated
mean squared errors are checked against this recursion.
"""

import numpy as np

import common
from common import LINESTYLES, SERIES, TEXT, TEXT_2, plt, save_fig, save_results
from Stochastic_Approximation import constant_step, polynomial_step, robbins_monro

MU, SIGMA, THETA0, N_ITER, N_REP, SEED = 5.0, 2.0, 0.0, 10_000, 1000, 546
SCHEDULES = {
    "a_k = 1/k": polynomial_step(1.0),
    "a_k = 0.2/k": polynomial_step(0.2),
    "a_k = 2/k": polynomial_step(2.0),
    "a_k = k^(-0.6)": polynomial_step(1.0, power=0.6),
    "a_k = 0.05": constant_step(0.05),
}
LABELS = {
    "a_k = 1/k": r"$a_k = 1/k$",
    "a_k = 0.2/k": r"$a_k = 0.2/k$",
    "a_k = 2/k": r"$a_k = 2/k$",
    "a_k = k^(-0.6)": r"$a_k = k^{-0.6}$",
    "a_k = 0.05": r"$a_k = 0.05$",
}
REPORT_AT = [10, 100, 1000, 10_000]


def exact_mse(step, n_iter):
    mse = np.empty(n_iter + 1)
    mse[0] = (THETA0 - MU) ** 2
    for k in range(1, n_iter + 1):
        a = step(k)
        mse[k] = (1 - a) ** 2 * mse[k - 1] + a ** 2 * SIGMA ** 2
    return mse


def plot(runs, record):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.9), gridspec_kw={"wspace": 0.28})
    k = record[1:]                      # k = 0 cannot be shown on a log axis
    for i, (name, (res, mse)) in enumerate(runs.items()):
        path = res.path[1:]
        ax1.plot(k, path[:, 0], color=SERIES[i], ls=LINESTYLES[i], lw=1.2, label=LABELS[name])
        ax2.plot(k, np.mean((path - MU) ** 2, axis=1), color=SERIES[i], ls=LINESTYLES[i], lw=1.4,
                 label=LABELS[name])
        ax2.plot(np.arange(1, N_ITER + 1), mse[1:], color=TEXT, lw=0.6, ls=":")
    ax1.axhline(MU, color=TEXT_2, lw=0.9)
    ax1.text(0.97, 0.05, r"horizontal line: root $\mu = 5$", transform=ax1.transAxes, ha="right",
             color=TEXT_2, fontsize=8)
    ax1.set(xscale="log", ylim=(-0.5, 8.5), xlabel="Iteration k", ylabel=r"$\theta_k$",
            title="One sample path per schedule")
    ax2.plot([], [], color=TEXT, lw=0.6, ls=":", label="Exact MSE")
    ax2.set(xscale="log", yscale="log", xlabel="Iteration k", ylabel=r"Mean squared error of $\theta_k$",
            title=f"MSE over {N_REP} replicates")
    ax2.legend(loc="lower left", fontsize=7)
    save_fig(fig, "q1_robbins_monro")


if __name__ == "__main__":
    record = np.unique(np.r_[0, np.round(np.logspace(0, np.log10(N_ITER), 120)).astype(int), REPORT_AT])
    H = lambda theta, rng: rng.normal(MU, SIGMA, theta.shape) - theta
    rng = np.random.default_rng(SEED)

    runs, table = {}, {}
    print(f"Q1: Robbins-Monro, mu = {MU}, sigma = {SIGMA}, theta_0 = {THETA0}, {N_REP} replicates")
    for name, step in SCHEDULES.items():
        res = robbins_monro(H, np.full(N_REP, THETA0), N_ITER, step, record_at=record, seed=rng)
        mse = exact_mse(step, N_ITER)
        runs[name] = (res, mse)
        sim = {int(k): float(np.mean((res.path[record == k][0] - MU) ** 2)) for k in REPORT_AT}
        table[name] = {"simulated_mse": sim, "exact_mse": {int(k): float(mse[k]) for k in REPORT_AT}}
        print(f"  {name:15s} MSE at k = 10^4: simulated {sim[N_ITER]:.5f}, exact {mse[N_ITER]:.5f}")

    save_results("q1", {
        "settings": {"mu": MU, "sigma": SIGMA, "theta0": THETA0, "iterations": N_ITER,
                     "replicates": N_REP, "seed": SEED},
        "mse": table,
    })
    plot(runs, record)
