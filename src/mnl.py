from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


def generate_section5_instance(
    rng: np.random.Generator,
    n: int,
    r_low: float = 15.0,
    r_high: float = 25.0,
    mu_low: float = 15.0,
    mu_high: float = 25.0,
    v0: float = 1.0,
    dtype=np.float64,
) -> dict:
    """
    Section 5 instance generator (paper):
      r_i ~ Uniform[15,25]
      mu_i ~ Uniform[15,25]
      v_i = exp(mu_i - r_i)
      v0 = 1
    Returns:
      dict with r, mu, v, rv, v0
    """
    r = rng.uniform(r_low, r_high, size=n).astype(dtype)
    mu = rng.uniform(mu_low, mu_high, size=n).astype(dtype)
    v = np.exp(mu - r).astype(dtype)
    rv = (r * v).astype(dtype)
    return {"r": r, "mu": mu, "v": v, "rv": rv, "v0": dtype(v0)}


def order_by_price_desc(r: np.ndarray) -> np.ndarray:
    """
    Submodular order for MNL: descending price.
    We use a stable sort to make tie-breaking deterministic.
    """
    # mergesort is stable; ties keep original index order
    return np.argsort(-r, kind="mergesort")


def revenue_from_sums(sum_v: float, sum_rv: float, v0: float = 1.0) -> float:
    """R(S) = sum_{i in S} r_i v_i / (v0 + sum_{i in S} v_i)."""
    denom = v0 + sum_v
    if denom <= 0:
        return 0.0
    return float(sum_rv / denom)


def singleton_f(r: np.ndarray, v: np.ndarray, v0: float = 1.0) -> np.ndarray:
    """
    f_phi({i}) = max_{X⊆{i}} R(X) = R({i}) (since R(∅)=0 and R({i})>=0).
    """
    rv = r * v
    return rv / (v0 + v)


def f_phi_from_ordered_indices(
    ordered_indices: Sequence[int],
    v: np.ndarray,
    rv: np.ndarray,
    v0: float = 1.0,
) -> float:
    """
    Compute f_phi(S)=max_{prefix of S} R(prefix),
    assuming `ordered_indices` are already in descending price order.
    (This is the structure exploited by the incremental oracle.)
    """
    sum_v = 0.0
    sum_rv = 0.0
    best = 0.0
    for idx in ordered_indices:
        sum_v += float(v[idx])
        sum_rv += float(rv[idx])
        best = max(best, revenue_from_sums(sum_v, sum_rv, v0))
    return best


def f_phi_of_set(
    indices: Sequence[int],
    r: np.ndarray,
    v: np.ndarray,
    rv: np.ndarray,
    v0: float = 1.0,
) -> float:
    """
    Debug/reference: compute f_phi(S) for an arbitrary set S.
    Sort by descending price inside S, then take best prefix revenue.
    """
    if len(indices) == 0:
        return 0.0
    idx = np.array(indices, dtype=np.int64)
    # sort by price desc, stable tie-break by original index
    order = np.argsort(-r[idx], kind="mergesort")
    ordered = idx[order].tolist()
    return f_phi_from_ordered_indices(ordered, v=v, rv=rv, v0=v0)


@dataclass
class MNLAppendOracle:
    """
    Incremental oracle specialized for the scan in descending price order.

    Invariant:
      Items are considered in non-increasing price order.
      When we accept an item, it is appended at the end of the selected sequence.
      Then f_phi(S) = max( previous_best, R(S) ) (only one new prefix appears).
    """
    v0: float = 1.0
    sum_v: float = 0.0
    sum_rv: float = 0.0
    best_f: float = 0.0

    def value(self) -> float:
        return self.best_f

    def peek_marginal(self, v_i: float, rv_i: float) -> tuple[float, float, float]:
        """
        Return (marginal, new_revenue_full, new_best).
        new_revenue_full = R(S ∪ {i}) for the appended item.
        new_best = max(best_f, new_revenue_full)
        marginal = new_best - best_f = max(0, new_revenue_full - best_f)
        """
        new_sum_v = self.sum_v + float(v_i)
        new_sum_rv = self.sum_rv + float(rv_i)
        new_rev = revenue_from_sums(new_sum_v, new_sum_rv, self.v0)
        new_best = new_rev if new_rev > self.best_f else self.best_f
        marginal = new_best - self.best_f
        return marginal, new_rev, new_best

    def accept(self, v_i: float, rv_i: float) -> float:
        """
        Commit append of item i, update internal sums and best value.
        Returns new best_f.
        """
        self.sum_v += float(v_i)
        self.sum_rv += float(rv_i)
        cur_rev = revenue_from_sums(self.sum_v, self.sum_rv, self.v0)
        if cur_rev > self.best_f:
            self.best_f = cur_rev
        return self.best_f