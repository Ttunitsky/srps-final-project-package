"""
Problem-agnostic destroy and repair operator skeletons.

Destroy signature:  destroy(solution, d: int) -> None   (modifies in place)
Repair signature:   repair(solution) -> None             (modifies in place)

The OPS adapter implements a concrete Solution class that exposes:
  solution.get_removable()              -> List of removable items
  solution.remove(item)                 -> None
  solution.relatedness(a, b)            -> float (lower = more related)
  solution.marginal_cost(item)          -> float
  solution.has_unserved()               -> bool
  solution.get_unserved()               -> List
  solution.best_insertion_cost(item)    -> (cost: float, position: Any)
  solution.top_k_insertion_costs(item, k) -> List[(cost, position)]
  solution.insert(item, position)       -> None
"""
from __future__ import annotations

import random
from typing import Any, List


# --- Destroy Operators ---

def destroy_random(solution: Any, d: int) -> None:
    """Remove d randomly selected elements."""
    removable = solution.get_removable()
    if not removable:
        return
    for item in random.sample(removable, min(d, len(removable))):
        solution.remove(item)


def destroy_worst(solution: Any, d: int) -> None:
    """Remove d elements with the highest marginal cost contribution."""
    removable = solution.get_removable()
    if not removable:
        return
    ranked = sorted(removable, key=lambda x: solution.marginal_cost(x), reverse=True)
    for item in ranked[: min(d, len(ranked))]:
        solution.remove(item)


def destroy_shaw(solution: Any, d: int) -> None:
    """Shaw removal: remove a relatedness-cluster anchored on a random seed element."""
    removable = solution.get_removable()
    if not removable:
        return
    seed = random.choice(removable)
    ranked = sorted(removable, key=lambda x: solution.relatedness(seed, x))
    for item in ranked[: min(d, len(ranked))]:
        solution.remove(item)


# --- Repair Operators ---

def repair_greedy(solution: Any) -> None:
    """Best-cost insertion until all mandatory customers are served."""
    while solution.has_unserved():
        unserved = solution.get_unserved()
        best_cost = float('inf')
        best_item = None
        best_pos = None
        for item in unserved:
            cost, pos = solution.best_insertion_cost(item)
            if cost < best_cost:
                best_cost = cost
                best_item = item
                best_pos = pos
        if best_item is None:
            break
        solution.insert(best_item, best_pos)


def repair_random(solution: Any) -> None:
    """Random-order insertion — breaks the profit-greedy topology bias."""
    unserved = solution.get_unserved()
    random.shuffle(unserved)
    for item in unserved:
        cost, pos = solution.best_insertion_cost(item)
        if cost < float("inf"):
            solution.insert(item, pos)


def repair_regret(solution: Any, k: int = 2) -> None:
    """Regret-k insertion: prioritise elements with the highest cost of delay."""
    while solution.has_unserved():
        unserved = solution.get_unserved()
        best_regret = -1.0
        best_item = None
        best_pos = None
        for item in unserved:
            top = solution.top_k_insertion_costs(item, k)
            if not top:
                continue
            regret = (top[1][0] - top[0][0]) if len(top) > 1 else 1e9
            if regret > best_regret:
                best_regret = regret
                best_item = item
                best_pos = top[0][1]
        if best_item is None:
            break
        solution.insert(best_item, best_pos)
