# experiments/estimate_runtime_section5.py
from __future__ import annotations
import time
import pandas as pd
import numpy as np

from src.utils.repro import set_global_seed
from src.mnl import generate_section5_instance, order_by_price_desc
from src.alg1_cardinality import alg1_cardinality_mnl
from src.opt_mnl_cardinality import opt_mnl_cardinality

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
    n = 50_000
    eps = 0.1
    trials_per_k = 5          # 先用5，想更稳改成10或20
    base_seed = 20260227

    rows = []
    for k in K_LIST:
        if k > n:
            continue
        for t in range(trials_per_k):
            seed = base_seed + k + t
            rows.append(one_trial(n=n, k=k, eps=eps, seed=seed))

    df = pd.DataFrame(rows)
    summary = df.groupby("k", as_index=False).agg(
        mean_ALG_ms=("ALG_ms", "mean"),
        mean_OPT_ms=("OPT_ms", "mean"),
        mean_ratio=("ratio", "mean"),
    )
    summary["mean_total_ms_per_trial"] = summary["mean_ALG_ms"] + summary["mean_OPT_ms"]

    # 外推到论文设定：每个k 1000 trials（串行）
    summary["projected_total_seconds_for_1000_trials"] = summary["mean_total_ms_per_trial"] * 1000.0 / 1000.0
    projected_total_seconds = summary["projected_total_seconds_for_1000_trials"].sum()

    print(summary)
    print("\nProjected total (serial) seconds for Section5:", projected_total_seconds)

if __name__ == "__main__":
    main()