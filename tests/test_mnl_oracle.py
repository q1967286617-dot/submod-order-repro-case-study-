import numpy as np

from src.mnl import (
    generate_section5_instance,
    order_by_price_desc,
    f_phi_from_ordered_indices,
    f_phi_of_set,
    MNLAppendOracle,
)


def test_f_phi_ordered_equals_generic():
    rng = np.random.default_rng(0)
    inst = generate_section5_instance(rng, n=300)
    r, v, rv, v0 = inst["r"], inst["v"], inst["rv"], float(inst["v0"])

    order = order_by_price_desc(r)

    # build a random subset but keep it in scan order
    chosen = []
    for idx in order:
        if rng.random() < 0.15:
            chosen.append(int(idx))

    val1 = f_phi_from_ordered_indices(chosen, v=v, rv=rv, v0=v0)
    val2 = f_phi_of_set(chosen, r=r, v=v, rv=rv, v0=v0)
    assert abs(val1 - val2) < 1e-12


def test_append_oracle_matches_reference_each_step():
    rng = np.random.default_rng(1)
    inst = generate_section5_instance(rng, n=200)
    r, v, rv, v0 = inst["r"], inst["v"], inst["rv"], float(inst["v0"])

    order = order_by_price_desc(r)

    oracle = MNLAppendOracle(v0=v0)
    chosen = []

    for idx in order:
        idx = int(idx)

        # Peek marginal via O(1) oracle
        m, _, _ = oracle.peek_marginal(v[idx], rv[idx])

        # Reference marginal via full evaluation
        f_before = f_phi_from_ordered_indices(chosen, v=v, rv=rv, v0=v0)
        f_after = f_phi_from_ordered_indices(chosen + [idx], v=v, rv=rv, v0=v0)
        m_ref = f_after - f_before

        assert abs(m - m_ref) < 1e-12

        # randomly accept to evolve state (still respects scan order)
        if rng.random() < 0.2:
            oracle.accept(v[idx], rv[idx])
            chosen.append(idx)

    # final value check
    assert abs(oracle.value() - f_phi_from_ordered_indices(chosen, v=v, rv=rv, v0=v0)) < 1e-12