from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.mnl import MNLAppendOracle, singleton_f, order_by_price_desc, f_phi_of_set


@dataclass
class ThresholdAddResult:
    S: List[int]
    value: float
    tau: float
    scanned: int
    accepted: int


def threshold_add_mnl(
    order: Sequence[int],
    v: np.ndarray,
    rv: np.ndarray,
    v0: float,
    k: int,
    tau: float,
) -> ThresholdAddResult:
    """
    Threshold Add for MNL-based f_phi with scan order = descending price.

    Uses the O(1) append oracle:
      marginal(i|S) = max(0, R(S∪{i}) - current_best_f)
    """
    oracle = MNLAppendOracle(v0=v0)
    S: List[int] = []
    scanned = 0

    for idx in order:
        scanned += 1
        if len(S) >= k:
            break

        m, _, _ = oracle.peek_marginal(v_i=v[idx], rv_i=rv[idx])
        if m >= tau:
            oracle.accept(v_i=v[idx], rv_i=rv[idx])
            S.append(int(idx))

    return ThresholdAddResult(
        S=S,
        value=oracle.value(),
        tau=float(tau),
        scanned=scanned,
        accepted=len(S),
    )


@dataclass
class Alg1Result:
    best_S: List[int]
    best_value: float
    best_tau: float
    candidates: List[ThresholdAddResult]


def alg1_cardinality_mnl(
    r: np.ndarray,
    v: np.ndarray,
    rv: np.ndarray,
    v0: float,
    k: int,
    eps: float = 0.1,
    order: Optional[np.ndarray] = None,
    ensure_order: bool = True,
) -> Alg1Result:
    """
    Algorithm 1 (Cardinality) specialized for the reproduction setting:
      - objective: f_phi(S) = max_{X⊆S} R(X) under MNL
      - submodular order: descending price
      - constraint: |S| <= k

    Returns the best solution among threshold grid runs.
    """
    if k <= 0:
        return Alg1Result(best_S=[], best_value=0.0, best_tau=0.0, candidates=[])

    if order is None:
        order = order_by_price_desc(r)

    if ensure_order:
        # Safety: ensure scan is non-increasing in price
        rr = r[order]
        if np.any(rr[1:] > rr[:-1] + 1e-15):
            raise ValueError("Provided order is not non-increasing in price.")

    # tau0 = (1/k) * max_e f({e}); here f({e}) = R({e})
    f_single = singleton_f(r=r, v=v, v0=v0)
    tau0 = float(np.max(f_single) / float(k))

    # number of thresholds: ceil(log_{1+eps}(k)), but ensure >=1 (k=1 case)
    if k == 1:
        L = 1
    else:
        L = int(np.ceil(np.log(k) / np.log(1.0 + eps)))
        L = max(L, 1)

    candidates: List[ThresholdAddResult] = []
    best_val = -1.0
    best_S: List[int] = []
    best_tau = tau0

    tau = tau0
    for _ in range(L):
        res = threshold_add_mnl(order=order, v=v, rv=rv, v0=v0, k=k, tau=tau)
        candidates.append(res)
        if res.value > best_val:
            best_val = res.value
            best_S = res.S
            best_tau = res.tau
        tau *= (1.0 + eps)

    return Alg1Result(best_S=best_S, best_value=float(best_val), best_tau=float(best_tau), candidates=candidates)


def sanity_check_solution_value(
    S: Sequence[int],
    r: np.ndarray,
    v: np.ndarray,
    rv: np.ndarray,
    v0: float,
) -> float:
    """
    Debug helper: compute f_phi(S) by sorting within S (slow but correct).
    Useful for tests / sanity checks.
    """
    return float(f_phi_of_set(indices=list(S), r=r, v=v, rv=rv, v0=v0))