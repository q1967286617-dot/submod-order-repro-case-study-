from __future__ import annotations

import time

import pandas as pd

from src.alg1_cardinality import alg1_cardinality_mnl
from src.mnl import generate_section5_instance, order_by_price_desc
from src.opt_mnl_cardinality import opt_mnl_cardinality
from src.utils.io import dump_config, make_run_dir
from src.utils.repro import set_global_seed

K_LIST = [1, 10, 80, 600, 2000, 10000, 25000, 45000]


def one_trial(n: int, k: int, eps: float, seed: int) -> dict:
    rng = set_global_seed(seed)
    inst = generate_section5_instance(rng, n=n)
    r, v, rv, v0 = inst["r"], inst["v"], inst["rv"], float(inst["v0"])
    order = order_by_price_desc(r)

    t0 = time.perf_counter()
    alg = alg1_cardinality_mnl(r=r, v=v, rv=rv, v0=v0, k=k, eps=eps, order=order)
    alg_ms = (time.perf_counter() - t0) * 1000.0

    t1 = time.perf_counter()
    opt = opt_mnl_cardinality(r=r, v=v, v0=v0, k=k)
    opt_ms = (time.perf_counter() - t1) * 1000.0

    return {
        "k": k,
        "ALG_ms": alg_ms,
        "OPT_ms": opt_ms,
        "ALG": alg.best_value,
        "OPT": opt.opt_value,
        "ratio": alg.best_value / opt.opt_value if opt.opt_value > 0 else 1.0,
    }


def main():
    config = {
        "experiment_name": "runtime_projection",
        "n": 50_000,
        "eps": 0.1,
        "k_list": K_LIST,
        "trials_per_k": 5,
        "projected_trials_per_k": 1000,
        "base_seed": 20260227,
    }

    run_name = (
        f"{config['experiment_name']}_n{config['n']}_"
        f"sample{config['trials_per_k']}_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    run_dir = make_run_dir("results", run_name=run_name)
    dump_config(run_dir, config)

    rows = []
    for k in config["k_list"]:
        if k > config["n"]:
            continue
        for t in range(config["trials_per_k"]):
            seed = config["base_seed"] + k + t
            rows.append(one_trial(n=config["n"], k=k, eps=config["eps"], seed=seed))

    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "raw.csv", index=False)

    summary = df.groupby("k", as_index=False).agg(
        mean_ALG_ms=("ALG_ms", "mean"),
        mean_OPT_ms=("OPT_ms", "mean"),
        mean_ratio=("ratio", "mean"),
    )
    summary["mean_total_ms_per_trial"] = summary["mean_ALG_ms"] + summary["mean_OPT_ms"]

    projected_trials = float(config["projected_trials_per_k"])
    summary["projected_total_seconds"] = summary["mean_total_ms_per_trial"] * projected_trials / 1000.0
    summary.to_csv(run_dir / "summary.csv", index=False)

    projected_total_seconds = float(summary["projected_total_seconds"].sum())

    print(summary)
    print(f"\nProjected total (serial) seconds: {projected_total_seconds:.2f}")
    print(f"[OK] wrote: {run_dir}")


if __name__ == "__main__":
    main()
