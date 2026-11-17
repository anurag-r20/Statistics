"""
Data, model and full-data MCMC reference shared by Problems 1 and 2.

Bayesian logistic regression: Y_i | x_i ~ Bernoulli(sigmoid(x_i'theta)), theta ~ N(0, PRIOR_VAR I).
The predictors are an intercept and 9 independent N(0, 1) columns. Training and test sets
are simulated independently.

The full-data reference is random-walk Metropolis-Hastings on the exact posterior, with a
normal proposal whose covariance is (2.38^2 / d) times the inverse Hessian of the negative
log posterior at its mode. It is computed once and saved, so both problems use the same chain.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

import common
from common import RESULTS_DIR
from Metropolis_Hastings import gaussian_random_walk, metropolis_hastings

N_TRAIN, N_TEST, D, PRIOR_VAR, DATA_SEED = 5000, 5000, 10, 10.0, 546
BETA = np.array([-0.5, 1.0, -1.0, 0.5, -0.5, 0.25, -0.25, 0.0, 0.0, 0.0])
REF_ITERS, REF_BURN_IN, REF_SEED = 200_000, 10_000, 547
REFERENCE_FILE = RESULTS_DIR / "q1_reference_mcmc.npz"


def simulate():
    rng = np.random.default_rng(DATA_SEED)

    def draw(n):
        X = np.column_stack([np.ones(n), rng.standard_normal((n, D - 1))])
        return X, (rng.uniform(size=n) < expit(X @ BETA)).astype(float)

    (X, y), (Xt, yt) = draw(N_TRAIN), draw(N_TEST)
    return X, y, Xt, yt


def log_posterior(theta, X, y):
    z = X @ theta
    return float(np.sum(y * z - np.logaddexp(0.0, z)) - 0.5 * theta @ theta / PRIOR_VAR)


def laplace(X, y):
    """Posterior mode and inverse Hessian of the negative log posterior there."""
    res = minimize(lambda t: -log_posterior(t, X, y), np.zeros(D),
                   jac=lambda t: -(X.T @ (y - expit(X @ t)) - t / PRIOR_VAR), method="BFGS",
                   options={"gtol": 1e-10})
    p = expit(X @ res.x)
    H = X.T @ (X * (p * (1 - p))[:, None]) + np.eye(D) / PRIOR_VAR
    return res.x, np.linalg.inv(H)


def reference_mcmc(X, y):
    """Full-data random-walk MH samples (cached in Results/)."""
    if REFERENCE_FILE.exists():
        f = np.load(REFERENCE_FILE)
        return f["samples"], float(f["acceptance"]), float(f["cpu_time"])
    import time
    mode, cov = laplace(X, y)
    start = time.process_time()
    res = metropolis_hastings(lambda t: log_posterior(t, X, y), mode, REF_ITERS - REF_BURN_IN,
                              propose=gaussian_random_walk(2.38 ** 2 / D * cov), burn_in=REF_BURN_IN,
                              seed=REF_SEED)
    cpu = time.process_time() - start
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(REFERENCE_FILE, samples=res.samples, acceptance=res.acceptance_rate, cpu_time=cpu)
    return res.samples, res.acceptance_rate, cpu


def predictive_accuracy(samples, Xt, yt):
    """Test accuracy of the posterior predictive rule: average P(Y = 1) over samples > 1/2."""
    p = expit(Xt @ samples.T).mean(axis=1)
    return float(np.mean((p > 0.5) == (yt == 1)))
