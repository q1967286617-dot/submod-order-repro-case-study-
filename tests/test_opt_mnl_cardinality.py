import numpy as np

from src.mnl import generate_section5_instance
from src.opt_mnl_cardinality import opt_mnl_cardinality, brute_force_opt_mnl_cardinality


def test_opt_matches_bruteforce_small_instances():
    # 多组随机小实例对拍
    for seed in range(10):
        rng = np.random.default_rng(seed)
        n = 14
        k = 5
        inst = generate_section5_instance(rng, n=n)
        r, v, v0 = inst["r"], inst["v"], float(inst["v0"])

        opt = opt_mnl_cardinality(r=r, v=v, v0=v0, k=k, tol=1e-14, max_iter=500)
        bf_val, bf_set = brute_force_opt_mnl_cardinality(r=r, v=v, v0=v0, k=k)

        assert abs(opt.opt_value - bf_val) < 1e-10


def test_opt_respects_k_and_is_deterministic():
    rng = np.random.default_rng(123)
    inst = generate_section5_instance(rng, n=30)
    r, v, v0 = inst["r"], inst["v"], float(inst["v0"])

    out1 = opt_mnl_cardinality(r=r, v=v, v0=v0, k=7)
    out2 = opt_mnl_cardinality(r=r, v=v, v0=v0, k=7)

    assert len(out1.opt_set) <= 7
    assert out1.opt_set == out2.opt_set
    assert abs(out1.opt_value - out2.opt_value) < 1e-15