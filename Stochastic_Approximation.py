"""
Robbins-Monro stochastic approximation and minibatch stochastic gradient descent.

Robbins-Monro finds a root of h(theta) = E[H(theta, X)] using only noisy values of h:

    theta_{k+1} = theta_k + a_k H(theta_k, X_{k+1}),    k = 1, 2, ...

Minibatch SGD minimises an average loss by stepping against the gradient of a
random minibatch. For logistic regression with log-likelihood
l_i(theta) = y_i x_i'theta - log(1 + exp(x_i'theta)), the objective is the penalised
empirical risk

    U(theta) = -(1/n) sum_i l_i(theta) + lam ||theta||^2,

and one minibatch step is

    theta <- (1 - 2 eta lam) theta + eta (1/|B|) sum_{i in B} (y_i - sigmoid(x_i'theta)) x_i.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
from scipy.special import expit


# ---------------------------------------------------------------- step sizes


def constant_step(a: float) -> Callable[[int], float]:
    """a_k = a."""
    return lambda k: a


def polynomial_step(a: float, power: float = 1.0, offset: float = 0.0) -> Callable[[int], float]:
    """a_k = a / (k + offset)^power."""
    return lambda k: a / (k + offset) ** power


# ---------------------------------------------------------------- Robbins-Monro


@dataclass
class RMResult:
    theta: np.ndarray        # final iterate
    iterations: np.ndarray   # iteration numbers k at which the path was recorded (0 = start)
    path: np.ndarray         # recorded iterates, shape (len(iterations),) + theta.shape


def robbins_monro(
    H: Callable[[np.ndarray, np.random.Generator], np.ndarray],
    theta0,
    n_iter: int,
    step_size: Callable[[int], float],
    record_at: Optional[Sequence[int]] = None,
    seed=None,
) -> RMResult:
    """
    Run theta_{k+1} = theta_k + a_k H(theta_k, X_{k+1}) for k = 1, ..., n_iter.

    Parameters
    ----------
    H : callable
        (theta, rng) -> noisy value of h(theta), with the same shape as theta. theta may
        hold several independent replicates (e.g. shape (R,)); H then draws independent
        noise for each, and the replicates run side by side.
    theta0 : array_like
        Starting value theta_0 (or one per replicate).
    n_iter : int
        Number of updates.
    step_size : callable
        k -> a_k for k = 1, 2, ...
    record_at : sequence of int, optional
        Iteration numbers in [0, n_iter] at which to store theta_k. Defaults to all.
    seed : int or np.random.Generator, optional
    """
    if n_iter < 1:
        raise ValueError("Require n_iter >= 1.")
    rng = np.random.default_rng(seed)
    theta = np.array(theta0, dtype=float)
    record = np.arange(n_iter + 1) if record_at is None else np.unique(np.asarray(record_at, dtype=int))
    if record.size == 0 or record[0] < 0 or record[-1] > n_iter:
        raise ValueError("record_at must lie in [0, n_iter].")

    path = np.empty((record.size,) + theta.shape)
    j = 0
    if record[0] == 0:
        path[0] = theta
        j = 1
    for k in range(1, n_iter + 1):
        theta = theta + step_size(k) * H(theta, rng)
        if j < record.size and record[j] == k:
            path[j] = theta
            j += 1
    return RMResult(theta=theta, iterations=record, path=path)


# ---------------------------------------------------------------- minibatch SGD


@dataclass
class SGDResult:
    theta: np.ndarray
    epochs: np.ndarray                        # 0 = before the first update
    history: dict = field(default_factory=dict)   # metric name -> values recorded at each epoch


def minibatch_sgd(
    grad: Callable[[np.ndarray, np.ndarray], np.ndarray],
    theta0,
    n: int,
    batch_size: int,
    step_size: Callable[[int], float],
    n_epochs: int,
    callback: Optional[Callable[[np.ndarray], dict]] = None,
    seed=None,
) -> SGDResult:
    """
    Minimise an average loss over n observations by minibatch SGD.

    Each epoch shuffles the observations and splits them into consecutive minibatches of
    batch_size (the last one may be smaller), so every observation is used once per epoch.

    Parameters
    ----------
    grad : callable
        (theta, idx) -> gradient of the objective estimated from the observations idx.
    step_size : callable
        t -> eta_t, where t = 0, 1, ... counts updates over all epochs.
    callback : callable, optional
        theta -> {name: value}, called before the first update and after every epoch.
    """
    if n < 1 or not 1 <= batch_size <= n or n_epochs < 1:
        raise ValueError("Require n >= 1, 1 <= batch_size <= n and n_epochs >= 1.")
    rng = np.random.default_rng(seed)
    theta = np.array(theta0, dtype=float)
    history: dict = {}

    def log(values):
        for name, v in values.items():
            history.setdefault(name, []).append(v)

    if callback is not None:
        log(callback(theta))
    t = 0
    for _ in range(n_epochs):
        order = rng.permutation(n)
        for start in range(0, n, batch_size):
            theta = theta - step_size(t) * grad(theta, order[start:start + batch_size])
            t += 1
        if callback is not None:
            log(callback(theta))
    return SGDResult(theta=theta, epochs=np.arange(n_epochs + 1),
                     history={k: np.array(v) for k, v in history.items()})


# ---------------------------------------------------------------- logistic regression


def logistic_loss(theta, X, y, lam: float = 0.0) -> float:
    """U(theta) = (1/n) sum_i [log(1 + exp(x_i'theta)) - y_i x_i'theta] + lam ||theta||^2."""
    z = X @ theta
    return float(np.mean(np.logaddexp(0.0, z) - y * z) + lam * theta @ theta)


def logistic_gradient(theta, X, y, lam: float = 0.0) -> np.ndarray:
    """Gradient of logistic_loss: (1/n) X'(sigmoid(X theta) - y) + 2 lam theta."""
    return X.T @ (expit(X @ theta) - y) / len(y) + 2.0 * lam * theta


def classification_error(theta, X, y) -> float:
    """Fraction of observations misclassified by the rule x'theta > 0."""
    return float(np.mean((X @ theta > 0) != (y == 1)))


def sgd_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    step_size: Callable[[int], float],
    n_epochs: int,
    lam: float = 0.0,
    theta0=None,
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    seed=None,
) -> SGDResult:
    """
    Minibatch SGD for (ridge-penalised) logistic regression with labels in {0, 1}.
    X should include a column of ones if an intercept is wanted; lam penalises every
    coefficient. history holds 'train_loss' (the objective U on the training data) and,
    if test data are given, 'test_error' (misclassification rate), at every epoch.
    """
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("y must contain only 0 and 1.")
    theta0 = np.zeros(X.shape[1]) if theta0 is None else theta0

    def callback(theta):
        out = {"train_loss": logistic_loss(theta, X, y, lam)}
        if X_test is not None:
            out["test_error"] = classification_error(theta, X_test, y_test)
        return out

    return minibatch_sgd(lambda theta, idx: logistic_gradient(theta, X[idx], y[idx], lam),
                         theta0, len(y), batch_size, step_size, n_epochs, callback, seed)


if __name__ == "__main__":
    # Robbins-Monro for E(X - theta) = 0 with X ~ N(5, 4): a_k = 1/k gives the running mean.
    res = robbins_monro(lambda th, rng: rng.normal(5.0, 2.0, th.shape) - th, np.zeros(1000),
                        10000, polynomial_step(1.0), record_at=[10, 100, 1000, 10000], seed=0)
    print("Robbins-Monro, a_k = 1/k, 1000 replicates; MSE vs exact 4/k:")
    for k, th in zip(res.iterations, res.path):
        print(f"  k = {k:5d}: MSE = {np.mean((th - 5.0) ** 2):.5f}, exact {4.0 / k:.5f}")

    # Minibatch SGD for logistic regression on simulated data.
    rng = np.random.default_rng(1)
    X = np.column_stack([np.ones(5000), rng.standard_normal((5000, 3))])
    beta = np.array([-0.5, 1.0, -1.0, 0.5])
    y = (rng.uniform(size=5000) < expit(X @ beta)).astype(float)
    fit = sgd_logistic_regression(X, y, batch_size=32, step_size=constant_step(0.1), n_epochs=20, seed=2)
    print("SGD logistic regression estimate:", fit.theta.round(3), " true:", beta)
