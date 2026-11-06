"""
STAT546 HW6, Problem 2: minibatch SGD for logistic regression.

Simulated data: an intercept and 20 standard normal predictors, with
P(Y = 1 | x) = 1 / (1 + exp(-x'beta)). Ten predictors have nonzero coefficients.
Training and test sets are independent.

Starting from a baseline (batch size 32, constant learning rate 0.1), three comparisons
change one factor at a time: batch size, learning rate, and step-size decay
eta_t = eta_0 (1 + t / T)^(-p), where t counts updates and T is the number of updates per
epoch. Training loss is the average negative log-likelihood; test error is the
misclassification rate of the rule x'theta > 0.
"""

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

import common
from common import LINESTYLES, SERIES, TEXT_2, plt, save_fig, save_results
from Stochastic_Approximation import classification_error, logistic_loss, sgd_logistic_regression

N_TRAIN, N_TEST, D, N_EPOCHS, SEED = 10_000, 10_000, 20, 30, 546
BETA = np.r_[-0.5, 1.0, -1.0, 1.0, -1.0, 1.0, 0.5, -0.5, 0.5, -0.5, 0.5, np.zeros(10)]
BASE = {"batch_size": 32, "eta0": 0.1, "power": 0.0}
COMPARISONS = {
    "Batch size": ("batch_size", [1, 8, 64, 512], "b = {}"),
    "Learning rate": ("eta0", [0.01, 0.1, 1.0, 5.0], r"$\eta_0$ = {}"),
    "Step-size decay": ("power", [0.0, 0.5, 1.0], "p = {}"),
}
DECAY_ETA0 = 1.0     # the decay comparison starts from a larger rate, so decay has something to tame


def simulate(rng):
    def draw(n):
        X = np.column_stack([np.ones(n), rng.standard_normal((n, D))])
        return X, (rng.uniform(size=n) < expit(X @ BETA)).astype(float)
    return draw(N_TRAIN), draw(N_TEST)


def run(config, data, seed):
    (X, y), (Xt, yt) = data
    T = int(np.ceil(N_TRAIN / config["batch_size"]))
    eta0, p = config["eta0"], config["power"]
    return sgd_logistic_regression(X, y, config["batch_size"], lambda t: eta0 * (1 + t / T) ** (-p),
                                   N_EPOCHS, X_test=Xt, y_test=yt, seed=seed)


def plot(results, min_loss, true_error):
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.0), sharex=True, gridspec_kw={"hspace": 0.12, "wspace": 0.3})
    for col, (title, (key, values, fmt)) in enumerate(COMPARISONS.items()):
        for i, v in enumerate(values):
            h = results[title][str(v)]
            label = fmt.format(v)
            axes[0, col].plot(h["epochs"], np.array(h["train_loss"]) - min_loss, color=SERIES[i],
                              ls=LINESTYLES[i], lw=1.4, label=label)
            axes[1, col].plot(h["epochs"], h["test_error"], color=SERIES[i], ls=LINESTYLES[i], lw=1.4)
        axes[1, col].axhline(true_error, color=TEXT_2, lw=0.9, ls=":")
        axes[0, col].set(yscale="log", title=title)
        axes[1, col].set(ylim=(true_error - 0.02, true_error + 0.15), xlabel="Epoch")
        axes[1, col].legend(*axes[0, col].get_legend_handles_labels(), loc="upper center",
                            bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=8)
    axes[0, 0].set_ylabel("Training loss minus its minimum")
    axes[1, 0].set_ylabel("Test error")
    axes[1, 2].annotate("true coefficients", (N_EPOCHS, true_error), xytext=(0, -10), textcoords="offset points",
                        ha="right", color=TEXT_2, fontsize=7)
    save_fig(fig, "q2_sgd_logistic")


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    data = simulate(rng)
    (X, y), (Xt, yt) = data

    # Reference values: the minimum training loss (unpenalised maximum likelihood) and the
    # test error of the true coefficients.
    mle = LogisticRegression(C=np.inf, fit_intercept=False, tol=1e-12, max_iter=10_000).fit(X, y).coef_.ravel()
    min_loss = logistic_loss(mle, X, y)
    true_error = classification_error(BETA, Xt, yt)
    print(f"Q2: minimum training loss {min_loss:.5f}; test error of true coefficients {true_error:.4f}")

    results, summary = {}, {}
    for title, (key, values, fmt) in COMPARISONS.items():
        results[title], summary[title] = {}, {}
        for v in values:
            config = {**BASE, key: v}
            if title == "Step-size decay":
                config["eta0"] = DECAY_ETA0
            res = run(config, data, seed=SEED + 1)
            h = res.history
            results[title][str(v)] = {"epochs": res.epochs.tolist(), "train_loss": h["train_loss"].tolist(),
                                      "test_error": h["test_error"].tolist()}
            summary[title][str(v)] = {"config": config, "final_loss_gap": float(h["train_loss"][-1] - min_loss),
                                      "final_test_error": float(h["test_error"][-1])}
            print(f"  {title:15s} {key} = {v:<6}: final loss gap {h['train_loss'][-1] - min_loss:.2e}, "
                  f"test error {h['test_error'][-1]:.4f}", flush=True)

    save_results("q2", {
        "settings": {"n_train": N_TRAIN, "n_test": N_TEST, "predictors": D, "beta": BETA.tolist(),
                     "epochs": N_EPOCHS, "baseline": BASE, "decay_eta0": DECAY_ETA0, "data_seed": SEED,
                     "sgd_seed": SEED + 1},
        "min_training_loss": min_loss, "true_coefficient_test_error": true_error,
        "mle_test_error": classification_error(mle, Xt, yt),
        "summary": summary, "curves": results,
    })
    plot(results, min_loss, true_error)
