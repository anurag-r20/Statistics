"""
STAT546 HW3, Problem 1: simulated annealing for

    H(x, y) = -(x sin(20y) + y sin(20x))^2 cosh(sin(10x) x)
              -(x cos(10y) - y sin(10x))^2 cosh(cos(20y) y),   (x, y) in [-1.1, 1.1]^2.

Known global minimum: H = -8.12465 at (-1.0445, -1.0084) and (1.0445, -1.0084).
"""

import numpy as np

import common
from common import BLUE_RAMP, ORANGE, TEXT, plt, save_fig, save_results
from Metropolis_Hastings import gaussian_random_walk
from Simulated_Annealing import geometric_cooling, simulated_annealing

N_RUNS, N_ITERS, T0, T_FINAL, STEP, P_JUMP, SEED = 20, 200000, 5.0, 1e-3, 0.05, 0.1, 546


def H(z):
    x, y = z
    return (-(x * np.sin(20 * y) + y * np.sin(20 * x)) ** 2 * np.cosh(np.sin(10 * x) * x)
            - (x * np.cos(10 * y) - y * np.sin(10 * x)) ** 2 * np.cosh(np.cos(20 * y) * y))


def energy(z):
    # Infinite energy outside the square, so MH never accepts a move out of it.
    return H(z) if np.all(np.abs(z) <= 1.1) else np.inf


def make_proposal():
    """Normal step with probability 1 - P_JUMP, otherwise a uniform point in the square.
    Both parts are symmetric, so no Hastings correction is needed."""
    walk = gaussian_random_walk(STEP)

    def propose(z, rng):
        return rng.uniform(-1.1, 1.1, size=2) if rng.uniform() < P_JUMP else walk(z, rng)

    return propose


def plot(x_best):
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    g = np.linspace(-1.1, 1.1, 500)
    X, Y = np.meshgrid(g, g)
    cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    im = ax.pcolormesh(X, Y, H((X, Y)), cmap=cmap, shading="auto", rasterized=True)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).set_label("H(x, y)", color=TEXT)
    ax.scatter(*x_best.T, s=110, marker="*", color=ORANGE, edgecolor="white", linewidth=0.6, zorder=3,
               label="Minimum found by each run")
    ax.set(xlabel="x", ylabel="y", aspect="equal", title="Objective H and minima found",
           xticks=[-1, -0.5, 0, 0.5, 1], yticks=[-1, -0.5, 0, 0.5, 1])
    ax.grid(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13))
    save_fig(fig, "q1_annealing")


if __name__ == "__main__":
    schedule = geometric_cooling((T_FINAL / T0) ** (1.0 / (N_ITERS - 1)))
    rng = np.random.default_rng(SEED)

    print(f"Q1: simulated annealing, {N_RUNS} runs x {N_ITERS} moves, T: {T0} -> {T_FINAL}")
    runs = [simulated_annealing(energy, rng.uniform(-1.1, 1.1, size=2), N_ITERS, T0=T0, schedule=schedule,
                                propose=make_proposal(), seed=rng)
            for _ in range(N_RUNS)]

    f_best = np.array([r.f_best for r in runs])
    x_best = np.array([r.x_best for r in runs])
    best = int(np.argmin(f_best))
    print(f"  minimum found: H = {f_best[best]:.5f} at {x_best[best].round(4)}")
    print(f"  per-run minima range from {f_best.min():.5f} to {f_best.max():.5f}")

    save_results("q1", {
        "settings": {"runs": N_RUNS, "moves_per_run": N_ITERS, "T0": T0, "T_final": T_FINAL,
                     "cooling": "geometric", "proposal_sd": STEP, "jump_probability": P_JUMP, "seed": SEED},
        "best": {"H": float(f_best[best]), "point": x_best[best].tolist()},
        "f_best_per_run": f_best.tolist(),
        "x_best_per_run": x_best.tolist(),
    })
    plot(x_best)
