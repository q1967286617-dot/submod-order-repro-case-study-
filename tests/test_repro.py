# tests/test_repro.py
from src.utils.repro import set_global_seed

def test_rng_reproducible():
    rng1 = set_global_seed(123)
    a1 = rng1.normal(size=5)

    rng2 = set_global_seed(123)
    a2 = rng2.normal(size=5)

    assert (a1 == a2).all()