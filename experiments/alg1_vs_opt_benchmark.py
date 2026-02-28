from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.alg1_cardinality import alg1_cardinality_mnl
from src.mnl import generate_section5_instance, order_by_price_desc
from src.opt_mnl_cardinality import opt_mnl_cardinality
from src.utils.io import dump_config, make_run_dir
from src.utils.repro import set_global_seed


def run_one(trial_id: int, base_seed: int, n: int, k: int, eps: float) -> dict:
    seed = base_seed + k + trial_id
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

    ratio = alg.best_value / opt.opt_value if opt.opt_value > 0 else 1.0
    ok = alg.best_value <= opt.opt_value + 1e-9

    return {
        "k": k,
        "trial": trial_id,
        "seed": seed,
        "ALG": alg.best_value,
        "ALG_set_size": len(alg.best_S),
        "ALG_best_tau": alg.best_tau,
        "ALG_n_thresholds": len(alg.candidates),
        "ALG_runtime_ms": alg_ms,
        "OPT": opt.opt_value,
        "OPT_set_size": len(opt.opt_set),
        "OPT_iters": opt.iterations,
        "OPT_gap": opt.final_gap,
        "OPT_runtime_ms": opt_ms,
        "ratio": ratio,
        "ALG_le_OPT": ok,
    }


def main():
    config = {
        "experiment_name": "alg1_vs_opt_benchmark",
        "n": 50_000,
        "k_list": [1, 10, 80, 600, 2000],
        "trials_per_k": 5,
        "eps": 0.1,
        "base_seed": 20260225,
        "note": "End-to-end benchmark: ALG1 vs OPT with plotting.",
    }

    run_name = (
        f"{config['experiment_name']}_n{config['n']}_"
        f"t{config['trials_per_k']}_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    run_dir = make_run_dir("results", run_name=run_name)
    dump_config(run_dir, config)

    rows = []
    for k in config["k_list"]:
        if k > config["n"]:
            continue
        for t in range(config["trials_per_k"]):
            rows.append(run_one(t, config["base_seed"], config["n"], k, config["eps"]))

    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "raw.csv", index=False)

    summary = df.groupby("k", as_index=False).agg(
        mean_ALG=("ALG", "mean"),
        std_ALG=("ALG", "std"),
        mean_OPT=("OPT", "mean"),
        std_OPT=("OPT", "std"),
        ratio_of_averages=("ALG", lambda x: float(x.mean())),
        mean_ratio=("ratio", "mean"),
        std_ratio=("ratio", "std"),
        pass_rate_ALG_le_OPT=("ALG_le_OPT", "mean"),
        mean_ALG_ms=("ALG_runtime_ms", "mean"),
        mean_OPT_ms=("OPT_runtime_ms", "mean"),
        mean_OPT_iters=("OPT_iters", "mean"),
        n=("trial", "count"),
    )

    summary["ratio_of_averages"] = summary["mean_ALG"] / summary["mean_OPT"]
    summary.to_csv(run_dir / "summary.csv", index=False)

    x = np.log2(summary["k"].to_numpy(dtype=float))
    y_roa = summary["ratio_of_averages"].to_numpy(dtype=float) * 100.0
    y_aor = summary["mean_ratio"].to_numpy(dtype=float) * 100.0
    yerr = summary["std_ratio"].to_numpy(dtype=float) * 100.0

    color_main = "#E0892B"
    color_ref = "#2980B9"

    plt.figure()
    plt.errorbar(x, y_roa, yerr=yerr, fmt="o-", capsize=3, color=color_main, ecolor=color_main)
    plt.axhline(y=100, color=color_ref, linestyle="--", alpha=0.8, zorder=0)
    plt.xlabel("log2(k)")
    plt.ylabel("100 * Average(ALG) / Average(OPT)")
    plt.title("ALG1 vs OPT: Ratio of Averages")
    plt.tight_layout()
    plt.savefig(run_dir / "alg1_vs_opt_ratio_of_averages.png", dpi=180)

    plt.figure()
    plt.errorbar(x, y_aor, yerr=yerr, fmt="o-", capsize=3, color=color_main, ecolor=color_main)
    plt.axhline(y=100, color=color_ref, linestyle="--", alpha=0.8, zorder=0)
    plt.xlabel("log2(k)")
    plt.ylabel("100 * Average(ALG/OPT)")
    plt.title("ALG1 vs OPT: Average of Ratios")
    plt.tight_layout()
    plt.savefig(run_dir / "alg1_vs_opt_average_of_ratios.png", dpi=180)

    print(f"[OK] wrote: {run_dir}")
    print(summary[["k", "ratio_of_averages", "mean_ratio", "pass_rate_ALG_le_OPT", "mean_ALG_ms", "mean_OPT_ms"]])


if __name__ == "__main__":
    main()
