import numpy as np

from src.mnl import generate_section5_instance, order_by_price_desc, f_phi_of_set
from src.alg1_cardinality import alg1_cardinality_mnl, threshold_add_mnl


def test_threshold_add_value_matches_reference():
    rng = np.random.default_rng(123)
    inst = generate_section5_instance(rng, n=400)
    r, v, rv, v0 = inst["r"], inst["v"], inst["rv"], float(inst["v0"])
    order = order_by_price_desc(r)

    k = 40
    tau = 0.01  # small threshold to accept many
    res = threshold_add_mnl(order=order, v=v, rv=rv, v0=v0, k=k, tau=tau)

    assert len(res.S) <= k
    # Check computed value equals reference evaluation for the returned set
    val_ref = f_phi_of_set(res.S, r=r, v=v, rv=rv, v0=v0)
    assert abs(res.value - val_ref) < 1e-12


def test_alg1_returns_best_among_candidates_and_is_valid():
    rng = np.random.default_rng(7)
    inst = generate_section5_instance(rng, n=500)
    r, v, rv, v0 = inst["r"], inst["v"], inst["rv"], float(inst["v0"])
    order = order_by_price_desc(r)

    k = 60
    eps = 0.1
    out = alg1_cardinality_mnl(r=r, v=v, rv=rv, v0=v0, k=k, eps=eps, order=order)

    assert len(out.best_S) <= k
    assert len(out.candidates) >= 1

    # best_value should be max of candidate values
    cand_vals = [c.value for c in out.candidates]
    assert abs(out.best_value - max(cand_vals)) < 1e-12

    # best_value matches reference evaluation
    best_ref = f_phi_of_set(out.best_S, r=r, v=v, rv=rv, v0=v0)
    assert abs(out.best_value - best_ref) < 1e-12


def test_alg1_deterministic_given_same_instance():
    rng = np.random.default_rng(0)
    inst = generate_section5_instance(rng, n=300)
    r, v, rv, v0 = inst["r"], inst["v"], inst["rv"], float(inst["v0"])
    order = order_by_price_desc(r)

    out1 = alg1_cardinality_mnl(r=r, v=v, rv=rv, v0=v0, k=50, eps=0.1, order=order)
    out2 = alg1_cardinality_mnl(r=r, v=v, rv=rv, v0=v0, k=50, eps=0.1, order=order)

    assert out1.best_S == out2.best_S
    assert abs(out1.best_value - out2.best_value) < 1e-15