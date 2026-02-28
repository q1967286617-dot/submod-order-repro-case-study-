# Submodular Order Reproduction for MNL Assortment Optimization

This repository reproduces core ideas from the paper **"Submodular Order Functions and Assortment Optimization"** under the Multinomial Logit (MNL) model.

The codebase focuses on:

- Modeling MNL revenue and the induced ordered submodular objective.
- Implementing **Algorithm 1** (threshold-add under cardinality constraint).
- Implementing an exact **OPT** solver (Dinkelbach-based).
- Running end-to-end benchmarks and exporting reproducible results.

The reference PDF used in this repo is:

- `Submodular Order Functions and Assortment Optimization.pdf`

## Problem Setup

For item set `S`, MNL revenue is:

`R(S) = (sum_{i in S} r_i v_i) / (v0 + sum_{i in S} v_i)`

where:

- `r_i` is item price/revenue.
- `v_i` is item utility weight.
- `v0` is no-purchase utility.

The ordered submodular objective used here is:

`f_phi(S) = max_{X subseteq S} R(X)`

In this MNL setting, when items are scanned in non-increasing price order, the maximization over subsets reduces to a max over prefixes. This structure enables an `O(1)` incremental marginal oracle.

## Repository Structure

```text
.
|-- src/
|   |-- mnl.py                    # MNL instance generation, objective helpers, incremental oracle
|   |-- alg1_cardinality.py       # Algorithm 1 for cardinality constraint
|   |-- opt_mnl_cardinality.py    # Exact OPT via Dinkelbach iterations
|   `-- utils/
|       |-- repro.py              # deterministic seeding helpers
|       `-- io.py                 # run directory + config dumping helpers
|-- experiments/
|   |-- pipeline_sanity_check.py
|   |-- alg1_quick_benchmark.py
|   |-- opt_quick_benchmark.py
|   |-- estimate_benchmark_runtime.py
|   |-- alg1_vs_opt_benchmark.py
|   `-- plot_from_summary.ipynb
|-- tests/
|   |-- test_mnl_oracle.py
|   |-- test_alg1.py
|   |-- test_opt_mnl_cardinality.py
|   `-- test_repro.py
|-- results/                      # generated outputs
|-- requirements.txt
`-- README.md
```

## Environment Setup

1. Create and activate a Python environment (recommended Python 3.10+).
2. Install dependencies:

```powershell
pip install -r requirements.txt
```

## Quick Verification

Run tests:

```powershell
pytest -q
```

Expected outcome: all tests pass.

## Experiments

Run all scripts from repository root using `python -m ...`.

### 1) Pipeline Sanity Check

Purpose:

- Validate the result pipeline (`config.json`, `raw.csv`, `summary.csv`, plot) with placeholder ALG/OPT values.

Command:

```powershell
python -m experiments.pipeline_sanity_check
```

Output:

- `results/pipeline_sanity_check_*`
- `alg_vs_opt_ratio_by_trial.png`

### 2) ALG1 Quick Benchmark

Purpose:

- Benchmark Algorithm 1 only on moderate settings.

Command:

```powershell
python -m experiments.alg1_quick_benchmark
```

Output:

- `results/alg1_quick_benchmark_*`
- `raw.csv`, `summary.csv`, `config.json`

### 3) OPT Quick Benchmark

Purpose:

- Benchmark exact OPT solver only.

Command:

```powershell
python -m experiments.opt_quick_benchmark
```

Output:

- `results/opt_quick_benchmark_*`
- `raw.csv`, `config.json`

### 4) Runtime Projection

Purpose:

- Estimate per-trial runtime and project serial total runtime for a larger target number of trials.

Command:

```powershell
python -m experiments.estimate_benchmark_runtime
```

Output:

- `results/runtime_projection_*`
- `raw.csv`, `summary.csv`, `config.json`

### 5) End-to-End ALG1 vs OPT Benchmark

Purpose:

- Compare ALG and OPT across `k` values, compute summary metrics, and generate publication-style figures.

Command:

```powershell
python -m experiments.alg1_vs_opt_benchmark
```

Output:

- `results/alg1_vs_opt_benchmark_*`
- `raw.csv`, `summary.csv`, `config.json`
- `alg1_vs_opt_ratio_of_averages.png`
- `alg1_vs_opt_average_of_ratios.png`

## Key Metrics in `summary.csv`

Common columns include:

- `mean_ALG`, `mean_OPT`: average objective values.
- `ratio_of_averages`: `mean_ALG / mean_OPT`.
- `mean_ratio`: average of per-trial `ALG/OPT`.
- `pass_rate_ALG_le_OPT`: fraction of trials satisfying `ALG <= OPT` (up to tolerance).
- `mean_ALG_ms`, `mean_OPT_ms`: runtime per trial in milliseconds.

## Core Implementation Notes

### `src/mnl.py`

- `generate_section5_instance(...)`: draws synthetic instances.
- `order_by_price_desc(...)`: stable descending-price order (deterministic tie handling).
- `MNLAppendOracle`: `O(1)` incremental marginal gain oracle for ordered scan.
- `f_phi_of_set(...)`: reference implementation for correctness checks.

### `src/alg1_cardinality.py`

- `threshold_add_mnl(...)`: single-threshold pass with cardinality cap `k`.
- `alg1_cardinality_mnl(...)`: threshold grid search over `tau`.
- `sanity_check_solution_value(...)`: slow reference check helper.

### `src/opt_mnl_cardinality.py`

- `opt_mnl_cardinality(...)`: exact solver using Dinkelbach iterations.
- `_select_topk_positive(...)`: deterministic subproblem solver.
- `brute_force_opt_mnl_cardinality(...)`: small-scale exact enumerator for tests.

## Reproducibility

- Use `src/utils/repro.py:set_global_seed(seed)` in experiments.
- Stable sorting and deterministic tie-breaking are used in both ALG and OPT paths.
- Each run writes its full configuration to `config.json`.

## Testing Coverage

- Oracle correctness against reference objective (`tests/test_mnl_oracle.py`).
- ALG output correctness and determinism (`tests/test_alg1.py`).
- OPT correctness against brute-force on small instances (`tests/test_opt_mnl_cardinality.py`).
- Random seed reproducibility (`tests/test_repro.py`).

## Practical Tips

- Start with small `n`, fewer `k`, and fewer trials to validate pipeline quickly.
- Increase `n` and trials only after confirming outputs and runtime.
- Keep old runs in `results/` for traceability; run directories are timestamped.

## Troubleshooting

- `ModuleNotFoundError: No module named 'src'`
- Run scripts as modules from repo root.
- Example command: `python -m experiments.alg1_vs_opt_benchmark`
- Existing run name collision (`FileExistsError`)
- Use a new `run_name` or rerun later (timestamps normally avoid collisions).

## License and Citation

This repository is for coursework/reproduction use. If you publish results based on this code, cite the original paper.
