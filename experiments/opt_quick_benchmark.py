from __future__ import annotations

import time

import pandas as pd

from src.mnl import generate_section5_instance
from src.opt_mnl_cardinality import opt_mnl_cardinality
from src.utils.io import dump_config, make_run_dir
from src.utils.repro import set_global_seed


def main():
    config = {
        "experiment_name": "opt_quick_benchmark",
        "n": 5000,
        "k_list": [10, 80, 600],
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
            r, v, v0 = inst["r"], inst["v"], float(inst["v0"])

            t0 = time.perf_counter()
            out = opt_mnl_cardinality(r=r, v=v, v0=v0, k=k)
            dt_ms = (time.perf_counter() - t0) * 1000.0

            rows.append(
                {
                    "k": k,
                    "trial": t,
                    "OPT": out.opt_value,
                    "opt_set_size": len(out.opt_set),
                    "iters": out.iterations,
                    "final_gap": out.final_gap,
                    "runtime_ms": dt_ms,
                }
            )

    pd.DataFrame(rows).to_csv(run_dir / "raw.csv", index=False)
    print(f"[OK] wrote {run_dir}")


if __name__ == "__main__":
    main()
