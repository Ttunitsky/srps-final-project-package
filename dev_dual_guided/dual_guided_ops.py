from __future__ import annotations

import random
from typing import Callable, Dict, Iterable, Tuple


def fair_split_mu(inst) -> Dict[Tuple[int, int], float]:
    """Initialize multipliers with b_j / |K_j| fair split."""
    mu: Dict[Tuple[int, int], float] = {}
    for j, ks in inst.Kj.items():
        share = inst.profits[j] / max(len(ks), 1)
        for k in ks:
            mu[(j, k)] = share
    return mu


def reduced_profit(inst, mu: Dict[Tuple[int, int], float], j: int) -> float:
    """Reduced profit b_j - sum_k mu_{j,k}; positive means under-penalized."""
    return float(inst.profits[j] - sum(mu.get((j, k), 0.0) for k in inst.Kj.get(j, [])))


def dual_pressure(inst, mu: Dict[Tuple[int, int], float], j: int) -> float:
    """Penalty mass sum_k mu_{j,k}; larger values indicate stronger dual pressure."""
    return float(sum(mu.get((j, k), 0.0) for k in inst.Kj.get(j, [])))


def make_dual_guided_destroy(mu_provider: Callable[[], Dict[Tuple[int, int], float]],
                             noise: float = 1e-6):
    """Create destroy(sol, d) that removes jobs least supported by reduced profit.

    Ranking uses low reduced profit first, then high dual pressure.
    """

    def _destroy(solution, d: int) -> None:
        removable = solution.get_removable()
        if not removable:
            return
        mu = mu_provider()
        inst = solution.inst

        ranked = sorted(
            removable,
            key=lambda j: (
                reduced_profit(inst, mu, j) + random.uniform(-noise, noise),
                -dual_pressure(inst, mu, j),
            ),
        )
        for j in ranked[: min(d, len(ranked))]:
            solution.remove(j)

    return _destroy


def make_dual_guided_repair(mu_provider: Callable[[], Dict[Tuple[int, int], float]],
                            positive_first: bool = True,
                            eps: float = 1e-9):
    """Create repair(sol) that prioritizes insertions with strong dual support.

    Score for a feasible insertion is reduced_profit / (1 + insertion_time_increase).
    """

    def _repair(solution) -> None:
        mu = mu_provider()
        inst = solution.inst

        while solution.has_unserved():
            best_score = -float("inf")
            best_item = None
            best_pos = None

            for j in solution.get_unserved():
                cost, pos = solution.best_insertion_cost(j)
                if cost >= float("inf") or pos is None:
                    continue

                rp = reduced_profit(inst, mu, j)
                if positive_first and rp <= 0:
                    # Keep negative reduced-profit jobs as fallback only.
                    score = rp - 10.0
                else:
                    score = rp / (1.0 + max(cost, eps))

                if score > best_score:
                    best_score = score
                    best_item = j
                    best_pos = pos

            if best_item is None:
                break
            solution.insert(best_item, best_pos)

    return _repair


# ---------------------------------------------------------------------------
# Corrected operators (v2).
#
# The v1 operators above are degenerate under the fair-split initialisation
# mu_{j,k} = b_j / |K_j|, which makes sum_k mu_{j,k} = b_j exactly and therefore
# reduced_profit(j) == 0 for every job. Consequences:
#   - v1 destroy sorts on (0 + uniform noise, ...), so noise dominates and the
#     operator degrades to random removal;
#   - v1 repair sends every job to the flat `rp - 10.0` branch, so the score is
#     constant and insertion follows iteration order.
# The v2 operators keep the dual signal as a *modulation* of a competent base
# heuristic, so that when the signal is uninformative they degrade to profit
# density rather than to noise.
# ---------------------------------------------------------------------------


def make_dual_guided_destroy_v2(mu_provider: Callable[[], Dict[Tuple[int, int], float]],
                                noise_frac: float = 0.01):
    """Remove jobs with the weakest dual support; noise scaled to the rp spread."""

    def _destroy(solution, d: int) -> None:
        removable = solution.get_removable()
        if not removable:
            return
        mu = mu_provider()
        inst = solution.inst

        rps = {j: reduced_profit(inst, mu, j) for j in removable}
        spread = max(rps.values()) - min(rps.values()) if rps else 0.0
        jitter = noise_frac * spread  # never allowed to dominate the signal

        ranked = sorted(
            removable,
            key=lambda j: (
                rps[j] + (random.uniform(-jitter, jitter) if jitter > 0 else 0.0),
                -dual_pressure(inst, mu, j),
                -inst.profits[j],
            ),
        )
        for j in ranked[: min(d, len(ranked))]:
            solution.remove(j)

    return _destroy


def make_dual_guided_repair_v2(mu_provider: Callable[[], Dict[Tuple[int, int], float]],
                               weight: float = 1.0,
                               eps: float = 1e-9):
    """Insert by profit density modulated by reduced profit.

    score = (b_j + weight * rp_j) / (1 + insertion_cost)

    With a degenerate mu (rp == 0) this reduces exactly to profit-density greedy
    rather than collapsing to a constant, so the arm measures the dual signal
    instead of measuring iteration order.
    """

    def _repair(solution) -> None:
        inst = solution.inst

        while solution.has_unserved():
            mu = mu_provider()
            best_score = -float("inf")
            best_item = None
            best_pos = None

            for j in solution.get_unserved():
                cost, pos = solution.best_insertion_cost(j)
                if cost >= float("inf") or pos is None:
                    continue

                rp = reduced_profit(inst, mu, j)
                score = (inst.profits[j] + weight * rp) / (1.0 + max(cost, eps))

                if score > best_score:
                    best_score = score
                    best_item = j
                    best_pos = pos

            if best_item is None:
                break
            solution.insert(best_item, best_pos)

    return _repair
