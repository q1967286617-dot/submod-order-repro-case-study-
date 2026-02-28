from __future__ import annotations

import time

import pandas as pd

from src.alg1_cardinality import alg1_cardinality_mnl
from src.mnl import generate_section5_instance, order_by_price_desc
from src.utils.io import dump_config, make_run_dir
from src.utils.repro import set_global_seed


def main():
    config = {
        "experiment_name": "alg1_quick_benchmark",
        "n": 5000,
        "k_list": [10, 80, 600],
        "eps": 0.1,
        "trials": 5,
        "base_seed": 20260227,
    }
    run_name = (
        f"{config['experiment_name']}_n{config['n']}_"
        f"t{config['trials']}_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    run_dir = make_run_dir("results", run_name=run_name)
    dump_config(run_dir, config)

    rows = []
    for k in config["k_list"]:
        for t in range(config["trials"]):
            seed = config["base_seed"] + 1000 * k + t
            rng = set_global_seed(seed)

            inst = generate_section5_instance(rng, n=config["n"])
            r, v, rv, v0 = inst["r"], inst["v"], inst["rv"], float(inst["v0"])

            order = order_by_price_desc(r)

            t0 = time.perf_counter()
            out = alg1_cardinality_mnl(r=r, v=v, rv=rv, v0=v0, k=k, eps=config["eps"], order=order)
            dt_ms = (time.perf_counter() - t0) * 1000.0

            rows.append(
                {
                    "k": k,
                    "trial": t,
                    "seed": seed,
                    "ALG": out.best_value,
                    "ALG_set_size": len(out.best_S),
                    "best_tau": out.best_tau,
                    "n_thresholds": len(out.candidates),
                    "runtime_ms": dt_ms,
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "raw.csv", index=False)

    summary = df.groupby("k", as_index=False).agg(
        mean_ALG=("ALG", "mean"),
        std_ALG=("ALG", "std"),
        mean_runtime_ms=("runtime_ms", "mean"),
        mean_set_size=("ALG_set_size", "mean"),
        n_trials=("trial", "count"),
    )
    summary.to_csv(run_dir / "summary.csv", index=False)

    print(f"[OK] wrote results to {run_dir}")


if __name__ == "__main__":
    main()
