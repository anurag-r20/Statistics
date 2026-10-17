"""
STAT546 HW4, Problem 2: simulated stochastic approximation annealing (SAA) for the
travelling salesman problem of Problem 1.

SAA is SAMC on the same tour-length partition, with the temperature lowered by the
square-root schedule tau_t = T_HIGH * sqrt(T0_COOL / max(T0_COOL, t)).
"""

import numpy as np

import common
from common import save_results
from SAMC import stochastic_approximation_annealing
from tsp import N_CITIES, N_SETS, SEED, SIDE, make_tour_length, plot_tour, solve_all_sets, two_opt

N_ITERS, T_HIGH, T0_COOL, T0 = 1_000_000, 10.0, 1000, 5000
LEVELS = np.arange(700.0, 1000.0, 5.0)


def solve_one(cities, rng):
    x0 = rng.permutation(len(cities)).astype(float)
    return stochastic_approximation_annealing(make_tour_length(cities), x0, N_ITERS, LEVELS,
                                              T_high=T_HIGH, t0_cool=T0_COOL, propose=two_opt,
                                              t0=T0, thin=N_ITERS, seed=rng)


if __name__ == "__main__":
    print(f"Q2: SAA for the TSP, {N_CITIES} cities, {N_SETS} city sets, {N_ITERS} iterations each")
    runs = solve_all_sets(solve_one)
    lengths = np.array([r[2] for r in runs])
    for k, L in enumerate(lengths):
        print(f"  set {k + 1:2d}: shortest tour found {L:6.1f}")
    print(f"  average shortest tour: {lengths.mean():.1f} miles")

    save_results("q2", {
        "settings": {"cities": N_CITIES, "side": SIDE, "city_sets": N_SETS, "iterations": N_ITERS,
                     "T_high": T_HIGH, "t0_cool": T0_COOL, "gain_t0": T0,
                     "levels": [LEVELS[0], LEVELS[-1], 5.0], "desired": "uniform", "proposal": "2-opt",
                     "seed": SEED},
        "shortest_lengths": lengths.tolist(),
        "average": float(lengths.mean()),
    })
    cities, tour, length = runs[0]
    plot_tour(cities, tour, length, "SAA, city set 1", "q2_saa_tsp")
