# Statistics

[![CI](https://github.com/anurag-r20/Statistics/actions/workflows/ci.yml/badge.svg)](https://github.com/anurag-r20/Statistics/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/downloads/release/python-3120/)
[![License](https://img.shields.io/github/license/anurag-r20/Statistics)](LICENSE)

General Markov chain Monte Carlo samplers written from scratch, and the STAT 546 assignments solved with them.

Includes:
- Metropolis-Hastings, with symmetric or asymmetric proposals
- Double Metropolis-Hastings, for models whose likelihood has an intractable normalising constant
- Simulated annealing
- Parallel tempering
- Simulated tempering

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

All samplers work with log densities and take a `seed` for reproducible runs. The annealing and tempering methods are built on `mh_step`, a single Metropolis-Hastings step at a chosen inverse temperature.

```python
import numpy as np
from Metropolis_Hastings import metropolis_hastings, gaussian_random_walk

log_target = lambda x: -0.5 * x @ x          # standard normal, up to a constant
res = metropolis_hastings(log_target, x0=[0.0, 0.0], n_samples=10000,
                          propose=gaussian_random_walk(1.0), burn_in=1000, seed=0)
print(res.samples.mean(axis=0), res.acceptance_rate)
```

**Tests** — [tests](tests) checks each sampler against targets with known answers, such as moments of a normal or gamma distribution, equal mass in the two modes of a bimodal mixture, and a conjugate posterior for double Metropolis-Hastings.

**Assignments** — Each assignment folder holds its scripts, figures, saved results and report.

| Assignment | Topics | Report |
|---|---|---|
| [Assignment 3](Assignments/Assignment_3) | Simulated annealing (function minimisation, travelling salesman), simulated tempering (witch's hat), double Metropolis-Hastings (very-soft-core model) | [STAT546_HW3_Report.pdf](Assignments/Assignment_3/STAT546_HW3_Report.pdf) |

To reproduce an assignment, run its scripts from the repository root, for example:
```bash
python Assignments/Assignment_3/Scripts/q1_annealing.py
```
Each script saves its figure in `Figures/` and its numbers in `Results/`.
