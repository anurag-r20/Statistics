"""
STAT546 HW5, Problem 1: learn a Gaussian graph with n = 300 and p = 400.

Nodewise Lasso regresses each standardized variable on the other variables.
An undirected edge is included when both regressions select it (the AND rule).
The penalty sqrt(2 log(p) / n) is fixed before fitting; the true graph is used
only to evaluate the result. The OR rule is also reported for comparison.
"""

import numpy as np
from matplotlib.colors import ListedColormap
from threadpoolctl import threadpool_limits

import common
from common import BLUE, RESULTS_DIR, plt, save_fig, save_results
from data import N, P, SEED, simulate_design
from High_Dimensional_Regression import nodewise_lasso

ALPHA, TOL, MAX_ITER, EDGE_TOL = np.sqrt(2 * np.log(P) / N), 1e-8, 10000, 1e-8


def graph_metrics(estimated, truth):
    upper = np.triu(np.ones_like(truth, dtype=bool), 1)
    tp = int(np.sum(estimated & truth & upper))
    fp = int(np.sum(estimated & ~truth & upper))
    fn = int(np.sum(~estimated & truth & upper))
    tn = int(np.sum(~estimated & ~truth & upper))
    return {"edges": tp + fp, "true_positives": tp, "false_positives": fp,
            "false_negatives": fn, "true_negatives": tn,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn), "f1": 2 * tp / (2 * tp + fp + fn),
            "false_positive_rate": fp / (fp + tn)}


def edge_list(adjacency):
    # Use the assignment's 1-based variable numbers in saved results.
    return (np.argwhere(np.triu(adjacency, 1)) + 1).tolist()


def plot(truth, estimated):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), layout="constrained")
    cmap = ListedColormap(["white", BLUE])
    size = 40   # A fixed excerpt makes individual edges visible; metrics use all 400 nodes.
    for ax, adjacency, title in zip(axes, [truth, estimated], ["True graph", "Nodewise Lasso (AND)"]):
        ax.imshow(adjacency[:size, :size], cmap=cmap, vmin=0, vmax=1, interpolation="nearest",
                  extent=(0.5, size + 0.5, size + 0.5, 0.5), rasterized=True)
        ax.set(xlabel="Variable", ylabel="Variable", title=f"{title}: first {size} variables")
        ax.grid(False)
        ax.tick_params(labelsize=8)
    save_fig(fig, "q1_gaussian_graph")


if __name__ == "__main__":
    X, theta = simulate_design()
    with threadpool_limits(limits=1):
        res = nodewise_lasso(X, ALPHA, max_iter=MAX_ITER, tol=TOL)
    truth = (theta != 0) & ~np.eye(P, dtype=bool)
    graphs = {rule: res.adjacency(rule=rule, threshold=EDGE_TOL) for rule in ["and", "or"]}
    metrics = {rule: graph_metrics(graph, truth) for rule, graph in graphs.items()}
    eigenvalues = np.linalg.eigvalsh(theta)
    Z = (X - res.mean) / res.scale
    inverse_error = res.precision @ (Z.T @ Z / N) - np.eye(P)

    print(f"Q1: nodewise Lasso, n = {N}, p = {P}, alpha = {ALPHA:.5f}")
    for rule, m in metrics.items():
        print(f"  {rule.upper()}: {m['edges']} edges, precision {m['precision']:.3f}, recall {m['recall']:.3f}")

    save_results("q1", {
        "settings": {"n": N, "p": P, "seed": SEED, "mean": 0.0, "method": "nodewise Lasso",
                     "alpha": float(ALPHA), "alpha_rule": "sqrt(2 log(p) / n)", "edge_rule": "and",
                     "standardization": "center and divide by standard deviation with ddof=0",
                     "solver": "sklearn.linear_model.Lasso", "solver_tolerance": TOL,
                     "max_iterations": MAX_ITER, "edge_tolerance": EDGE_TOL},
        "true_edges": int(np.sum(np.triu(truth, 1))),
        "concentration_eigenvalue_range": [float(eigenvalues[0]), float(eigenvalues[-1])],
        "metrics": metrics,
        "estimated_edges": {rule: edge_list(graph) for rule, graph in graphs.items()},
        "max_inverse_diagonal_error": float(np.max(np.abs(np.diag(inverse_error)))),
    })
    np.savez_compressed(RESULTS_DIR / "q1_data.npz", X=X, theta=theta,
                        nodewise_coefficients=res.coefficients)
    print("  saved Results/q1_data.npz")
    plot(truth, graphs["and"])
