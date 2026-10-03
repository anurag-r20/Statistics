"""
STAT546 HW3, Problem 3: simulated tempering for the simplified witch's hat
distribution on [0, 1]^d with d = 30,

    f(x) proportional to g(x) = (3^d - 1)/2  if x in B = [0, 1/3]^d  (the "hat"),
                                1            otherwise.

The hat has volume v = 3^-d and height a = (3^d - 1)/2, so its probability is
v a / (v a + 1 - v) = 1/3. This exact value is used to check the simulation.

Tempered levels are g^beta. Their normalising constants Z_beta = v a^beta + (1 - v)
are known exactly, so the weights 1/Z_beta are supplied directly.

The position proposal draws uniformly from the whole cube or uniformly from the
hat, each with probability 1/2, and the Hastings correction q(x)/q(y) is applied.
A proposal from the whole cube alone would almost never enter the hat (volume 3^-30).
"""

import numpy as np

import common
from common import ORANGE, SERIES, plt, save_fig, save_results
from Simulated_Tempering import simulated_tempering

D, N_LEVELS, N_SAMPLES, THIN, BURN_IN, SEED = 30, 20, 250000, 4, 100000, 546


def make_witch_hat(d):
    log_a = np.log((3.0 ** d - 1) / 2)
    log_v = -d * np.log(3)

    def log_g(x):
        return log_a if np.all(x <= 1 / 3) else 0.0

    def log_Z(beta):
        return np.logaddexp(log_v + beta * log_a, np.log1p(-3.0 ** -d))

    return log_g, log_Z


def make_cube_or_hat_proposal(d):
    """Uniform on [0,1]^d or uniform on the hat, each with probability 1/2.
    Density q(y) = 0.5 + 0.5 / v inside the hat and 0.5 outside."""
    log_q_in = np.log(0.5) + np.log1p(3.0 ** d)
    log_q_out = np.log(0.5)

    def propose(x, rng):
        return rng.uniform(0, 1 / 3, d) if rng.uniform() < 0.5 else rng.uniform(0, 1, d)

    def log_q(to, frm):
        return log_q_in if np.all(to <= 1 / 3) else log_q_out

    return propose, log_q


def plot(x1, p_hat):
    # Exact marginal density of the first coordinate: 5/3 on [0, 1/3] and 2/3 on (1/3, 1].
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    ax.hist(x1, bins=np.linspace(0, 1, 31), density=True, color=SERIES[0], alpha=0.75,
            edgecolor="white", linewidth=0.8, label="Simulated")
    ax.plot([0, 0, 1 / 3, 1 / 3, 1], [0, 5 / 3, 5 / 3, 2 / 3, 2 / 3], color=ORANGE, lw=2, label="Exact")
    ax.set(xlabel="First coordinate", ylabel="Density",
           title=f"Samples at the target temperature (P(hat) = {p_hat:.3f})")
    ax.legend(loc="upper right")
    save_fig(fig, "q3_witch_hat")


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    log_g, log_Z = make_witch_hat(D)
    betas = np.linspace(1.0, 0.05, N_LEVELS)
    propose, log_q = make_cube_or_hat_proposal(D)

    print(f"Q3: simulated tempering, d = {D}, {N_LEVELS} temperatures, "
          f"{BURN_IN + N_SAMPLES * THIN} iterations ({BURN_IN} discarded)")
    res = simulated_tempering(log_g, rng.uniform(0, 1, D), N_SAMPLES, betas=betas,
                              log_weights=-np.array([log_Z(b) for b in betas]),
                              propose=propose, log_proposal=log_q,
                              burn_in=BURN_IN, thin=THIN, seed=rng)

    in_hat = np.all(res.samples <= 1 / 3, axis=1)
    print(f"  P(hat) = {in_hat.mean():.4f} (exact 1/3) from {len(res.samples)} target-temperature samples")

    save_results("q3", {
        "settings": {"d": D, "levels": N_LEVELS, "betas": betas.tolist(), "burn_in": BURN_IN,
                     "iterations": BURN_IN + N_SAMPLES * THIN, "thin": THIN, "seed": SEED},
        "p_hat": float(in_hat.mean()),
        "p_hat_exact": 1 / 3,
        "target_samples": int(len(res.samples)),
    })
    plot(res.samples[:, 0], in_hat.mean())
