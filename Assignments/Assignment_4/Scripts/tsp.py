"""
Travelling salesman helpers shared by Problems 1 and 2.

A tour is stored as a float vector holding a permutation of the city indices.
Its energy is the tour length, including the return to the first city.
"""

from concurrent.futures import ProcessPoolExecutor

import numpy as np

from common import SERIES, TEXT, plt, save_fig

N_CITIES, SIDE, N_SETS, SEED = 100, 100.0, 20, 546


def make_tour_length(cities):
    D = np.sqrt(((cities[:, None, :] - cities[None, :, :]) ** 2).sum(-1))

    def tour_length(tour):
        t = tour.astype(int)
        return D[t, np.roll(t, -1)].sum()

    return tour_length


def two_opt(tour, rng):
    """Reverse the segment between two random positions. Symmetric, and keeps every city exactly once."""
    i, j = np.sort(rng.choice(tour.size, size=2, replace=False))
    new = tour.copy()
    new[i:j + 1] = new[i:j + 1][::-1]
    return new


def solve_all_sets(solve_one):
    """Run solve_one(cities, rng) on N_SETS independent city sets, in parallel.
    Each set has its own random stream, so the results do not depend on scheduling."""
    streams = np.random.SeedSequence(SEED).spawn(N_SETS)
    with ProcessPoolExecutor(max_workers=10) as pool:
        return list(pool.map(_solve_set, [solve_one] * N_SETS, streams))


def _solve_set(solve_one, stream):
    rng = np.random.default_rng(stream)
    cities = rng.uniform(0, SIDE, size=(N_CITIES, 2))
    res = solve_one(cities, rng)
    return cities, res.x_best.astype(int), float(res.f_best)


def plot_tour(cities, tour, length, title, name):
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    loop = np.append(tour, tour[0])
    ax.plot(cities[loop, 0], cities[loop, 1], color=SERIES[0], lw=1.0, zorder=1)
    ax.scatter(cities[:, 0], cities[:, 1], s=12, color=TEXT, zorder=2)
    ax.set(xlabel="Miles", ylabel="Miles", aspect="equal", xlim=(-2, 102), ylim=(-2, 102),
           title=f"{title}: {length:.1f} miles")
    ax.grid(False)
    save_fig(fig, name)
