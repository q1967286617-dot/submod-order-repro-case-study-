from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import List, Sequence, Tuple

import numpy as np

from src.mnl import revenue_from_sums, singleton_f


@dataclass
class OptResult:
    opt_value: float
    opt_set: List[int]
    iterations: int
    final_gap: float


def _select_topk_positive(weights: np.ndarray, k: int) -> np.ndarray:
    """
    Select up to k indices with the largest positive weights.
    Deterministic tie-break: sort by (weight desc, index asc).
    """
    n = weights.shape[0]
    if k <= 0 or n == 0:
        return np.empty(0, dtype=np.int64)

    if k >= n:
        cand = np.arange(n, dtype=np.int64)
    else:
        # get k largest (may include negatives if <k positives exist)
        cand = np.argpartition(weights, -k)[-k:].astype(np.int64)

    cand = cand[weights[cand] > 0.0]
    if cand.size == 0:
        return cand

    # deterministic sort: weight desc, index asc
    order = np.lexsort((cand, -weights[cand]))
    return cand[order]


def opt_mnl_cardinality(
    r: np.ndarray,
    v: np.ndarray,
    v0: float,
    k: int,
    tol: float = 1e-12,
    max_iter: int = 200,
) -> OptResult:
    """
    Compute OPT = max_{|S|<=k} R(S) under MNL:
      R(S) = sum_{i in S} r_i v_i / (v0 + sum_{i in S} v_i)

    We solve exactly via Dinkelbach iterations:
      For current t, maximize A(S) - t*(v0 + B(S)),
      where A(S)=sum r_i v_i, B(S)=sum v_i.
    Subproblem becomes: pick up to k items with positive weight
      w_i(t) = v_i*(r_i - t).

    Returns opt value and one optimal set (indices).
    """
    n = r.shape[0]
    if k <= 0 or n == 0:
        return OptResult(opt_value=0.0, opt_set=[], iterations=0, final_gap=0.0)

    rv = r * v

    # Good deterministic start: best singleton revenue
    t = float(np.max(singleton_f(r=r, v=v, v0=v0)))

    best_val = 0.0
    best_set: List[int] = []

    final_gap = np.inf
    it_used = 0

    for it in range(1, max_iter + 1):
        # weights for subproblem: w = v*(r - t) = rv - t*v
        w = rv - t * v

        idx = _select_topk_positive(w, k=k)
        if idx.size == 0:
            # no profitable items at this t -> best is empty set with value 0
            best_val = 0.0
            best_set = []
            final_gap = -t * float(v0)
            it_used = it
            if abs(final_gap) <= tol:
                break
            t = 0.0
            continue

        sum_v = float(v[idx].sum())
        sum_rv = float(rv[idx].sum())
        val = revenue_from_sums(sum_v=sum_v, sum_rv=sum_rv, v0=v0)

        # Dinkelbach gap: A(S) - t*(v0 + B(S))
        gap = sum_rv - t * (float(v0) + sum_v)

        best_val = val
        best_set = idx.tolist()
        final_gap = float(gap)
        it_used = it

        if abs(gap) <= tol:
            break
        t = val

    return OptResult(
        opt_value=float(best_val),
        opt_set=best_set,
        iterations=int(it_used),
        final_gap=float(final_gap),
    )


# -------------------------
# Reference (tests only)
# -------------------------
def brute_force_opt_mnl_cardinality(
    r: np.ndarray,
    v: np.ndarray,
    v0: float,
    k: int,
) -> Tuple[float, List[int]]:
    """
    Exact brute force OPT for small n (for tests).
    Enumerate all subsets of size <=k and return max R(S).
    """
    n = r.shape[0]
    rv = r * v

    best_val = 0.0
    best_set: List[int] = []

    for sz in range(1, min(k, n) + 1):
        for comb in combinations(range(n), sz):
            idx = np.fromiter(comb, dtype=np.int64)
            sum_v = float(v[idx].sum())
            sum_rv = float(rv[idx].sum())
            val = revenue_from_sums(sum_v=sum_v, sum_rv=sum_rv, v0=v0)
            if val > best_val:
                best_val = val
                best_set = list(map(int, comb))

    return float(best_val), best_set