"""
post_process.py — Ejection-chain post-processor for OPS solutions.

Purpose
-------
Applied after ALNS completes to close the gap on instances where the
ALNS solution is below the BKS / reported optimal.  The production ALNS
ran all 500 iters × 3 seeds to completion with no early stop, yet still
missed 1-2 profit units on ~50 instances.  The failure mode is a topology
trap: the current route configuration makes the "missing" job infeasible
to insert directly, but it becomes feasible once a blocking job is evicted.

Algorithm
---------
Depth-1 ejection (handles most "off by 1" cases):
  For each unserved job j (profit-descending):
    (a) Try direct insertion — free improvement if ALNS left any gap.
    (b) For each selected job j' in j's required processor routes
        (profit-ascending — eject cheapest first):
          trial = copy(solution)
          remove j' from trial
          if j can be inserted into trial:
            try to re-insert j' elsewhere in trial
            if re-insertion succeeds → gain = profit[j]          (commit)
            elif profit[j] > profit[j'] → gain = profit[j]-profit[j'] (commit)
            else → discard trial

Depth-2 ejection (catches "off by 2 via 1 insert" cases):
  Same structure but ejects a pair (j'1, j'2) before inserting j.
  Limited to the cheapest-8 eject candidates to keep runtime tractable.

Loop
----
Each successful improvement restarts the round (routes changed → new
insertions may become feasible).  Terminates when a full round produces
no improvement or max_rounds is reached.

Runtime estimate:
  n=150, depth=1: ~0.5–2 s per ejection_chain() call
  n=150, depth=2: ~5–20 s per call (limited to top-8 eject pairs)
  50 GAP instances, depth=2: < 20 min total

Usage
-----
    from core.post_process import ejection_chain

    n_gained = ejection_chain(solution, depth=2, verbose=True)
    new_obj = solution.objective()
"""
from __future__ import annotations

import time
from itertools import combinations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from adapters.ops_adapter import OPSSolution


# Maximum number of eject candidates to consider for depth-2/3 (performance cap)
_D2_MAX_EJECT = 8
_D3_MAX_EJECT = 5   # C(5,3)=10 triples; keeps depth-3 tractable


def ejection_chain(
    solution: "OPSSolution",
    max_rounds: int = 30,
    depth: int = 2,
    verbose: bool = False,
) -> int:
    """Apply ejection-chain post-processing in-place.

    Parameters
    ----------
    solution   : OPSSolution — modified in place if improvement found.
    max_rounds : maximum improvement rounds before giving up.
    depth      : 1 = eject one job; 2 = also try ejecting pairs.
    verbose    : print round-by-round progress.

    Returns
    -------
    n_gained : int — total profit units gained (0 = no improvement).
    """
    t0 = time.perf_counter()
    initial_obj = solution.objective()
    n_gained = 0

    for round_idx in range(max_rounds):
        improved, delta = _one_round(solution, depth)
        if not improved:
            break
        n_gained += delta
        if verbose:
            elapsed = time.perf_counter() - t0
            print(
                f"  ejection round {round_idx+1}: +{delta}  "
                f"obj={solution.objective():.0f}  ({elapsed:.2f}s)"
            )

    if verbose and n_gained == 0:
        print(f"  ejection: no improvement found  ({time.perf_counter()-t0:.2f}s)")

    return n_gained


def _one_round(solution: "OPSSolution", depth: int) -> tuple[bool, int]:
    """Try one improvement pass.  Returns (improved: bool, profit_gain: int)."""
    inst = solution.inst

    # Sort unserved jobs by profit descending — try most valuable first
    unserved = sorted(
        solution.get_unserved(),
        key=lambda j: inst.profits[j],
        reverse=True,
    )

    # ── (a) Direct insertion sweep ─────────────────────────────────────────
    for j in unserved:
        res = solution._enumerate_insertions(j, top_k=1)
        if res:
            solution.insert(j, res[0][1])
            return True, int(inst.profits[j])

    # ── (b) Depth-1: eject one job, insert j, try to reinsert evicted ─────
    for j in unserved:
        required = inst.Kj.get(j, [])
        if not required:
            continue

        # Jobs currently occupying j's required processor routes
        eject_pool = _eject_candidates(solution, inst, j, required)

        # Sort ascending profit: eject cheapest first
        for j_prime in sorted(eject_pool, key=lambda x: inst.profits[x]):
            trial = solution.copy()
            trial.remove(j_prime)

            res_j = trial._enumerate_insertions(j, top_k=1)
            if not res_j:
                continue
            trial.insert(j, res_j[0][1])

            res_jp = trial._enumerate_insertions(j_prime, top_k=1)
            if res_jp:
                # j_prime can be re-inserted elsewhere: net gain = profit[j]
                trial.insert(j_prime, res_jp[0][1])
                gain = int(inst.profits[j])
                _apply(solution, trial)
                return True, gain
            elif inst.profits[j] > inst.profits[j_prime]:
                # j_prime is lost but j is worth more
                gain = int(inst.profits[j] - inst.profits[j_prime])
                _apply(solution, trial)
                return True, gain

    if depth < 2:
        return False, 0

    # ── (c) Depth-2: eject a pair (j'1, j'2), insert j, reinsert both ─────
    for j in unserved:
        required = inst.Kj.get(j, [])
        if not required:
            continue

        eject_pool = _eject_candidates(solution, inst, j, required)
        # Limit to _D2_MAX_EJECT cheapest to keep runtime tractable
        eject_list = sorted(eject_pool, key=lambda x: inst.profits[x])[:_D2_MAX_EJECT]

        for j1p, j2p in combinations(eject_list, 2):
            trial = solution.copy()
            trial.remove(j1p)
            trial.remove(j2p)

            res_j = trial._enumerate_insertions(j, top_k=1)
            if not res_j:
                continue
            trial.insert(j, res_j[0][1])

            # Try to recover both evicted jobs
            res_j1p = trial._enumerate_insertions(j1p, top_k=1)
            recovered_j1p = False
            if res_j1p:
                trial.insert(j1p, res_j1p[0][1])
                recovered_j1p = True

            res_j2p = trial._enumerate_insertions(j2p, top_k=1)
            recovered_j2p = False
            if res_j2p:
                trial.insert(j2p, res_j2p[0][1])
                recovered_j2p = True

            new_obj = trial.objective()
            old_obj = solution.objective()
            if new_obj > old_obj:
                gain = int(new_obj - old_obj)
                _apply(solution, trial)
                return True, gain

    if depth < 3:
        return False, 0

    # ── (d) Depth-3: eject triple (j'1,j'2,j'3), insert j, reinsert all ──
    for j in unserved:
        required = inst.Kj.get(j, [])
        if not required:
            continue

        eject_pool = _eject_candidates(solution, inst, j, required)
        eject_list = sorted(eject_pool, key=lambda x: inst.profits[x])[:_D3_MAX_EJECT]

        for j1p, j2p, j3p in combinations(eject_list, 3):
            trial = solution.copy()
            trial.remove(j1p)
            trial.remove(j2p)
            trial.remove(j3p)

            res_j = trial._enumerate_insertions(j, top_k=1)
            if not res_j:
                continue
            trial.insert(j, res_j[0][1])

            # Try to recover all three evicted jobs (order: cheapest first)
            for jp in (j1p, j2p, j3p):
                res = trial._enumerate_insertions(jp, top_k=1)
                if res:
                    trial.insert(jp, res[0][1])

            new_obj = trial.objective()
            if new_obj > solution.objective():
                gain = int(new_obj - solution.objective())
                _apply(solution, trial)
                return True, gain

    return False, 0


def _eject_candidates(solution, inst, j: int, required: list) -> set:
    """Return the set of selected jobs occupying j's required processor routes."""
    cands: set = set()
    start, end = inst.start, inst.end
    for k in required:
        for node in solution.routes[k]:
            if node != start and node != end and node in solution.selected:
                cands.add(node)
    return cands


def _apply(target: "OPSSolution", source: "OPSSolution") -> None:
    """Copy source routes and selected into target in-place."""
    target.routes = [list(r) for r in source.routes]
    target.selected = set(source.selected)
