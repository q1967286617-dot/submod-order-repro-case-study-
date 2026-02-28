from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.io import dump_config, make_run_dir
from src.utils.repro import set_global_seed


def run_one_trial(trial_id: int, seed: int) -> dict:
    rng = set_global_seed(seed)
    t0 = time.perf_counter()

    # Placeholder outputs to verify the experiment pipeline end-to-end.
    alg = float(rng.normal(loc=10.0, scale=1.0))
    opt = alg + float(rng.uniform(0.0, 2.0))

    dt_ms = (time.perf_counter() - t0) * 1000.0
    return {"trial_id": trial_id, "seed": seed, "ALG": alg, "OPT": opt, "runtime_ms": dt_ms}


def main():
    config = {
        "experiment_name": "pipeline_sanity_check",
        "base_seed": 20260227,
        "trials": 30,
        "note": "Pipeline sanity check with placeholder ALG/OPT values.",
    }
    run_name = (
        f"{config['experiment_name']}_trials{config['trials']}_"
        f"seed{config['base_seed']}_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    run_dir = make_run_dir("results", run_name=run_name)
    dump_config(run_dir, config)

    rows = []
    for trial_id in range(config["trials"]):
        seed = config["base_seed"] + trial_id
        rows.append(run_one_trial(trial_id, seed))

    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "raw.csv", index=False)

    summary = {
        "mean_ALG": df["ALG"].mean(),
        "mean_OPT": df["OPT"].mean(),
        "ratio_of_averages": df["ALG"].mean() / df["OPT"].mean(),
        "std_ALG": df["ALG"].std(ddof=1),
        "std_OPT": df["OPT"].std(ddof=1),
        "mean_runtime_ms": df["runtime_ms"].mean(),
    }
    pd.DataFrame([summary]).to_csv(run_dir / "summary.csv", index=False)

    y = (df["ALG"] / df["OPT"]).to_numpy()
    plt.figure()
    plt.plot(np.arange(len(y)), y)
    plt.xlabel("trial")
    plt.ylabel("ALG/OPT")
    plt.title("Pipeline Sanity Check: ALG/OPT per Trial")
    plt.tight_layout()
    plt.savefig(run_dir / "alg_vs_opt_ratio_by_trial.png", dpi=150)

    print(f"[OK] wrote: {run_dir}")


if __name__ == "__main__":
    main()
