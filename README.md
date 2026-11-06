# Statistics

[![CI](https://github.com/anurag-r20/Statistics/actions/workflows/ci.yml/badge.svg)](https://github.com/anurag-r20/Statistics/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/downloads/release/python-3120/)
[![License](https://img.shields.io/github/license/anurag-r20/Statistics)](LICENSE)

General Markov chain Monte Carlo samplers written from scratch, high-dimensional regression methods,
stochastic approximation and EM, and the STAT 546 assignments solved with them.

Includes:
- Metropolis-Hastings, with symmetric or asymmetric proposals
- Double Metropolis-Hastings, for models whose likelihood has an intractable normalising constant
- Simulated annealing
- Parallel tempering
- Simulated tempering
- Stochastic approximation Monte Carlo (SAMC), and simulated stochastic approximation annealing (SAA)
- Nodewise Lasso for Gaussian graphs, and desparsified Lasso for regression inference
- Robbins-Monro stochastic approximation, and minibatch SGD for (ridge-penalised) logistic regression
- EM for finite mixtures of univariate normal distributions

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/anurag-r20/Statistics.git
cd Statistics
```

### 2. Create a virtual environment with python 3.12
```bash
py -3.12 -m venv .venv
.venv\Scripts\activate  # On Windows
source .venv/bin/activate  # On macOS/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run all tests to verify the repo runs ok
```bash
pytest
```

## What's Here

**Samplers** — One module per algorithm in the repository root. Each one ends with a small example that runs when the file is executed directly, e.g. `python Parallel_Tempering.py`.

| Module | Functions |
|---|---|
| [Metropolis_Hastings.py](Metropolis_Hastings.py) | `metropolis_hastings`, `mh_step`, `double_metropolis_hastings`, `gaussian_random_walk` |
| [Simulated_Annealing.py](Simulated_Annealing.py) | `simulated_annealing`, `geometric_cooling`, `linear_cooling`, `logarithmic_cooling` |
| [Parallel_Tempering.py](Parallel_Tempering.py) | `parallel_tempering`, `geometric_ladder` |
| [Simulated_Tempering.py](Simulated_Tempering.py) | `simulated_tempering`, `geometric_ladder` |
| [SAMC.py](SAMC.py) | `samc`, `stochastic_approximation_annealing`, `energy_regions`, `sqrt_cooling` |

All samplers work with log densities and take a `seed` for reproducible runs. The annealing, tempering and SAMC methods are built on `mh_step`, a single Metropolis-Hastings step.

```python
import numpy as np
from Metropolis_Hastings import metropolis_hastings, gaussian_random_walk

log_target = lambda x: -0.5 * x @ x          # standard normal, up to a constant
res = metropolis_hastings(log_target, x0=[0.0, 0.0], n_samples=10000,
                          propose=gaussian_random_walk(1.0), burn_in=1000, seed=0)
print(res.samples.mean(axis=0), res.acceptance_rate)
```

**High-dimensional regression** — [High_Dimensional_Regression.py](High_Dimensional_Regression.py)
provides `nodewise_lasso` and `desparsified_lasso`. It uses scikit-learn's Lasso solver and computes
the graph, bias correction, standard errors, confidence intervals and adjusted p-values explicitly.

**Stochastic approximation and EM** — [Stochastic_Approximation.py](Stochastic_Approximation.py)
provides `robbins_monro`, `minibatch_sgd`, `sgd_logistic_regression` and step-size schedules.
[Expectation_Maximization.py](Expectation_Maximization.py) provides `normal_mixture_em` and
`normal_mixture_loglik`.

**Tests** — [tests](tests) checks each sampler against targets with known answers, such as moments of a normal or gamma distribution, equal mass in the two modes of a bimodal mixture, a conjugate posterior for double Metropolis-Hastings, and exact subregion probabilities for SAMC. Regression tests check analytical Lasso solutions, bias correction, standard errors and changes of units. Stochastic approximation and EM tests check the running-mean form of Robbins-Monro, the penalised SGD step, agreement with scikit-learn's ridge logistic regression, hand-computed EM steps and the monotone EM log-likelihood.

**Assignments** — Each assignment folder holds its scripts, figures, saved results and report.

| Assignment | Topics | Report |
|---|---|---|
| [Assignment 3](Assignments/Assignment_3) | Simulated annealing (function minimisation, travelling salesman), simulated tempering (witch's hat), double Metropolis-Hastings (very-soft-core model) | [STAT546_HW3_Report.pdf](Assignments/Assignment_3/STAT546_HW3_Report.pdf) |
| [Assignment 4](Assignments/Assignment_4) | SAMC and SAA (travelling salesman), SAMC for the moments of a normal mixture, SAMC for permutation p-values of the two-sample t-test | [STAT546_HW4_Report.pdf](Assignments/Assignment_4/STAT546_HW4_Report.pdf) |
| [Assignment 5](Assignments/Assignment_5) | Gaussian graph learning by nodewise Lasso, high-dimensional inference by desparsified Lasso | [STAT546_HW5_Report.pdf](Assignments/Assignment_5/STAT546_HW5_Report.pdf) |
| [Assignment 6](Assignments/Assignment_6) | Robbins-Monro step-size schedules, minibatch SGD for logistic regression, penalised SGD (derivation), EM for normal mixtures, IRO for missing covariates (formulation) | [STAT546_HW6_Report.pdf](Assignments/Assignment_6/STAT546_HW6_Report.pdf) |

To reproduce an assignment, run its scripts from the repository root, for example:
```bash
python Assignments/Assignment_3/Scripts/q1_annealing.py
```
Each script saves its figure in `Figures/` and its numbers in `Results/`.

For Assignment 5, run both questions and compile the report:
```bash
python Assignments/Assignment_5/Scripts/q1_gaussian_graph.py
python Assignments/Assignment_5/Scripts/q2_desparsified_lasso.py
cd Assignments/Assignment_5
pdflatex -interaction=nonstopmode -halt-on-error STAT546_HW5_Report.tex
pdflatex -interaction=nonstopmode -halt-on-error STAT546_HW5_Report.tex
```
Both scripts regenerate the same predictor matrix from seed 546. The simulated predictors and
concentration matrix are saved in `Results/q1_data.npz`; the response, true coefficients and errors
are saved in `Results/q2_data.npz`. The JSON results include the learned edges and inference for
all 400 slopes. The report is maintained in LaTeX, as in the earlier assignments; if settings are
changed and scripts rerun, update its tables from the saved JSON before compiling.
