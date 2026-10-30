"""
Gaussian design shared by both Assignment 5 questions.

The concentration matrix has diagonal 1, first off-diagonals 0.5 and second
off-diagonals 0.25. Both questions use the same n = 300 observations of p = 400
variables. A triangular solve generates covariance Theta^{-1} without forming
that inverse. The mean is assumed to be zero.
"""

import numpy as np
from scipy.linalg import solve_triangular

N, P, SEED = 300, 400, 546


def concentration_matrix(p):
    distance = np.abs(np.arange(p)[:, None] - np.arange(p)[None, :])
    return np.where(distance == 0, 1.0,
                    np.where(distance == 1, 0.5, np.where(distance == 2, 0.25, 0.0)))


def simulate_design():
    theta = concentration_matrix(P)
    L = np.linalg.cholesky(theta)
    rng = np.random.default_rng(SEED)
    X = solve_triangular(L.T, rng.standard_normal((P, N)), lower=False).T
    return X, theta
