"""
STAT546 HW3, Problem 2: simulated annealing for the travelling salesman problem
with 100 cities uniformly distributed on a 100 x 100 mile square.

The state is a tour (a permutation of the cities) and the energy is its length.
Moves are 2-opt: reverse the segment of the tour between two random positions.
This proposal is symmetric, so plain Metropolis acceptance applies.

The average shortest-tour length is estimated by solving independent random
city sets and averaging the shortest tour found for each.
"""

import numpy as np

import common
from common import SERIES, TEXT, plt, save_fig, save_results
from Simulated_Annealing import geometric_cooling, simulated_annealing

N_CITIES, SIDE = 100, 100.0
N_SETS, N_ITERS, T0, T_FINAL, SEED = 20, 500000, 10.0, 0.05, 546


def make_tour_length(cities):
    D = np.sqrt(((cities[:, None, :] - cities[None, :, :]) ** 2).sum(-1))

    def tour_length(tour):
        t = tour.astype(int)
        return D[t, np.roll(t, -1)].sum()

    return tour_length


def two_opt(tour, rng):
    i, j = np.sort(rng.choice(tour.size, size=2, replace=False))
    new = tour.copy()
    new[i:j + 1] = new[i:j + 1][::-1]
    return new


def solve_tsp(cities, rng):
    """One annealing run from a random tour."""
    schedule = geometric_cooling((T_FINAL / T0) ** (1.0 / (N_ITERS - 1)))
    x0 = rng.permutation(len(cities)).astype(float)
    return simulated_annealing(make_tour_length(cities), x0, N_ITERS, T0=T0, schedule=schedule,
                               propose=two_opt, seed=rng)


def plot(cities, res):
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    tour = res.x_best.astype(int)
    loop = np.append(tour, tour[0])
    ax.plot(cities[loop, 0], cities[loop, 1], color=SERIES[0], lw=1.0, zorder=1)
    ax.scatter(cities[:, 0], cities[:, 1], s=12, color=TEXT, zorder=2)
    ax.set(xlabel="Miles", ylabel="Miles", aspect="equal", xlim=(-2, 102), ylim=(-2, 102),
           title=f"Shortest tour found for city set 1: {res.f_best:.1f} miles")
    ax.grid(False)
    save_fig(fig, "q2_tsp")


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    print(f"Q2: TSP, {N_CITIES} cities on a {SIDE:.0f} x {SIDE:.0f} square, {N_SETS} city sets")

    lengths = []
    for k in range(N_SETS):
        cities = rng.uniform(0, SIDE, size=(N_CITIES, 2))
        res = solve_tsp(cities, rng)
        lengths.append(res.f_best)
        if k == 0:
            cities_1, res_1 = cities, res
        print(f"  set {k + 1:2d}: shortest tour found {res.f_best:6.1f}", flush=True)
    lengths = np.array(lengths)
    print(f"  average shortest tour: {lengths.mean():.1f} miles")

    save_results("q2", {
        "settings": {"cities": N_CITIES, "side": SIDE, "city_sets": N_SETS, "moves_per_run": N_ITERS,
                     "T0": T0, "T_final": T_FINAL, "cooling": "geometric", "proposal": "2-opt", "seed": SEED},
        "shortest_lengths": lengths.tolist(),
        "average": float(lengths.mean()),
    })
    plot(cities_1, res_1)
