"""
STAT546 HW4, Problem 1: SAMC for the travelling salesman problem with 100 cities
uniformly distributed on a 100 x 100 mile square.

The energy is the tour length L, the target is psi(tour) = exp(-L / TAU), and the
sample space is partitioned by tour length with levels 700, 705, ..., 995 miles
(61 subregions, the first and last open). Moves are 2-opt.
The average shortest-tour length is estimated over independent random city sets.
"""

import numpy as np

import common
from common import save_results
from SAMC import energy_regions, samc
from tsp import N_CITIES, N_SETS, SEED, SIDE, make_tour_length, plot_tour, solve_all_sets, two_opt

N_ITERS, TAU, T0 = 1_000_000, 2.0, 5000
LEVELS = np.arange(700.0, 1000.0, 5.0)


def solve_one(cities, rng):
    region, m = energy_regions(LEVELS)
    x0 = rng.permutation(len(cities)).astype(float)
    return samc(make_tour_length(cities), x0, N_ITERS, region, m, tau=TAU, propose=two_opt,
                t0=T0, thin=N_ITERS, seed=rng)


if __name__ == "__main__":
    print(f"Q1: SAMC for the TSP, {N_CITIES} cities, {N_SETS} city sets, {N_ITERS} iterations each")
    runs = solve_all_sets(solve_one)
    lengths = np.array([r[2] for r in runs])
    for k, L in enumerate(lengths):
        print(f"  set {k + 1:2d}: shortest tour found {L:6.1f}")
    print(f"  average shortest tour: {lengths.mean():.1f} miles")

    save_results("q1", {
        "settings": {"cities": N_CITIES, "side": SIDE, "city_sets": N_SETS, "iterations": N_ITERS,
                     "tau": TAU, "gain_t0": T0, "levels": [LEVELS[0], LEVELS[-1], 5.0],
                     "desired": "uniform", "proposal": "2-opt", "seed": SEED},
        "shortest_lengths": lengths.tolist(),
        "average": float(lengths.mean()),
    })
    cities, tour, length = runs[0]
    plot_tour(cities, tour, length, "SAMC, city set 1", "q1_samc_tsp")
