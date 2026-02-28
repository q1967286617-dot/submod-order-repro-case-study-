# experiments/smoke_test.py
from __future__ import annotations
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.utils.repro import set_global_seed
from src.utils.io import make_run_dir, dump_config

def run_one_trial(trial_id: int, seed: int) -> dict:
    rng = set_global_seed(seed)
    t0 = time.perf_counter()

    # 占位：用一个“假算法/假OPT”先打通流水线
    alg = float(rng.normal(loc=10.0, scale=1.0))
    opt = alg + float(rng.uniform(0.0, 2.0))  # opt >= alg

    dt_ms = (time.perf_counter() - t0) * 1000.0
    return {"trial_id": trial_id, "seed": seed, "ALG": alg, "OPT": opt, "runtime_ms": dt_ms}

def main():
    config = {
        "base_seed": 20260227,
        "trials": 30,
        "note": "stage0 smoke test pipeline",
    }
    run_dir = make_run_dir("results", run_name=None)
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
    sdf = pd.DataFrame([summary])
    sdf.to_csv(run_dir / "summary.csv", index=False)

    # 简单画个图，验证 matplotlib 输出没问题
    y = (df["ALG"] / df["OPT"]).to_numpy()
    plt.figure()
    plt.plot(np.arange(len(y)), y)
    plt.xlabel("trial")
    plt.ylabel("ALG/OPT")
    plt.title("Smoke Test: ALG/OPT per trial")
    plt.tight_layout()
    plt.savefig(run_dir / "smoke.png", dpi=150)

    print(f"[OK] wrote: {run_dir}")

if __name__ == "__main__":
    main()