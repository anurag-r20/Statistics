"""
STAT546 HW4, Problem 4: SAMC for the p-value of the two-sample t-test.

The p-value is the permutation p-value of the pooled two-sample t-statistic:
the fraction of all ways of splitting the combined data into groups of sizes
n1 and n2 whose |t| is at least the observed |t|. Under the null hypothesis
every split is equally likely, so the target is uniform on the splits (energy 0).

SAMC moves between splits by swapping one member of each group, and the space
of splits is partitioned by |t| into subregions, the last one being
{|t| >= |t_obs|}. Because SAMC spends a set share of time in every subregion,
the tail is visited often even when the p-value is very small. The p-value is
the importance-weighted fraction of states in the last subregion.

The data are simulated. With n1 = n2 = 12 there are C(24, 12) = 2,704,156 splits,
few enough to enumerate, which gives the exact p-value as a check.
"""

import itertools

import numpy as np

import common
from common import ORANGE, SERIES, plt, save_fig, save_results
from SAMC import samc

N1 = N2 = 12
SHIFT, SEED = 2.0, 546
N_REGIONS, N_ITERS, BURN_IN, THIN, T0 = 24, 1_000_000, 100_000, 10, 300


def make_t_statistic(data, n1):
    """Pooled t-statistic of a split, as a function of the group-1 sum (the totals are fixed)."""
    n2 = data.size - n1
    S, Q = data.sum(), np.sum(data ** 2)

    def t_from_group1_sum(s1):
        m1, m2 = s1 / n1, (S - s1) / n2
        ss_within = Q - s1 ** 2 / n1 - (S - s1) ** 2 / n2
        return (m1 - m2) / np.sqrt(ss_within / (data.size - 2) * (1 / n1 + 1 / n2))

    return t_from_group1_sum


def swap_one_pair(labels, rng):
    """Swap a random member of group 1 with a random member of group 2 (symmetric proposal)."""
    new = labels.copy()
    new[rng.choice(np.flatnonzero(labels == 1))] = 0.0
    new[rng.choice(np.flatnonzero(labels == 0))] = 1.0
    return new


def exact_abs_t_values(data, t_stat):
    """|t| for every split, by enumeration."""
    combos = np.fromiter(itertools.chain.from_iterable(itertools.combinations(range(data.size), N1)),
                         dtype=np.int8).reshape(-1, N1)
    s1 = np.concatenate([data[combos[i:i + 500_000]].sum(axis=1) for i in range(0, len(combos), 500_000)])
    return np.abs(t_stat(s1))


def plot(edges, p_samc, p_exact):
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    idx = np.arange(p_exact.size)
    ax.bar(idx[:-1], p_exact[:-1], color=SERIES[0], alpha=0.75, edgecolor="white", linewidth=1.2,
           label="Exact (enumeration)")
    ax.bar(idx[-1], p_exact[-1], color=SERIES[2], edgecolor="white", linewidth=1.2,
           label=r"Exact, $|t| \geq |t_{obs}|$ (the p-value)")
    ax.plot(idx, p_samc, "o", color=ORANGE, ms=5, label="SAMC estimate")
    ax.set_yscale("log")
    labels = [f"{e:.1f}" for e in np.r_[0.0, edges]]
    ax.set_xticks(idx[::3], labels[::3])
    ax.set(xlabel="Lower edge of the |t| subregion", ylabel="Probability of the subregion",
           title="Permutation distribution of |t|")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right")
    save_fig(fig, "q4_samc_pvalue")


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    x, y = rng.normal(0.0, 1.0, N1), rng.normal(SHIFT, 1.0, N2)
    data = np.concatenate([x, y])
    t_stat = make_t_statistic(data, N1)
    t_obs = abs(t_stat(x.sum()))

    # Subregions: equal widths of |t| up to t_obs, then {|t| >= t_obs}. The last edge sits a hair
    # below t_obs so that the observed split, recomputed in another order, falls in the tail.
    edges = np.append(np.linspace(0, t_obs, N_REGIONS)[1:-1], t_obs * (1 - 1e-9))

    def region(labels, u):
        return int(np.searchsorted(edges, abs(t_stat(data @ labels)), side="right"))

    labels0 = np.r_[np.ones(N1), np.zeros(N2)]
    res = samc(lambda labels: 0.0, labels0, N_ITERS, region, N_REGIONS, propose=swap_one_pair,
               t0=T0, burn_in=BURN_IN, thin=THIN, seed=rng)
    w = res.importance_weights()
    p_regions = np.array([w[res.regions == i].sum() for i in range(N_REGIONS)])
    p_value = p_regions[-1]

    abs_t = exact_abs_t_values(data, t_stat)
    exact_regions = np.bincount(np.searchsorted(edges, abs_t, side="right"), minlength=N_REGIONS) / abs_t.size
    p_exact = exact_regions[-1]
    print(f"Q4: |t_obs| = {t_obs:.3f}; SAMC p-value = {p_value:.3e}; exact permutation p-value = {p_exact:.3e}")

    save_results("q4", {
        "settings": {"n1": N1, "n2": N2, "shift": SHIFT, "seed": SEED, "regions": N_REGIONS,
                     "iterations": N_ITERS, "burn_in": BURN_IN, "thin": THIN, "gain_t0": T0},
        "x": x.tolist(), "y": y.tolist(),
        "t_obs": float(t_obs),
        "p_value_samc": float(p_value),
        "p_value_exact": float(p_exact),
        "splits": int(abs_t.size),
    })
    plot(edges, p_regions, exact_regions)
