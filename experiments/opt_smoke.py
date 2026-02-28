import time
import numpy as np
import pandas as pd

from src.utils.repro import set_global_seed
from src.utils.io import make_run_dir, dump_config
from src.mnl import generate_section5_instance
from src.opt_mnl_cardinality import opt_mnl_cardinality


def main():
    config = {"n": 5000, "k_list": [10, 80, 600], "trials": 5, "base_seed": 20260227}
    run_dir = make_run_dir("results", run_name="stage4_opt_smoke")
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

    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "raw.csv", index=False)
    print(f"[OK] wrote {run_dir}")


if __name__ == "__main__":
    main()