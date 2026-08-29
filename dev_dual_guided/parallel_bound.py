"""Parallel evaluation of the Lagrangian per-processor subproblems.

Profiling shows ~84% of `lagrangian_bound` wall time is spent inside
`orienteering_dp_with_selection`, called once per processor per subgradient
iteration. Those solves are independent given mu -- that independence is
exactly what the per-processor relaxation buys -- so the loop is
embarrassingly parallel.

This module reimplements the subgradient loop with the per-processor solves
farmed out to a worker pool. Everything else (threshold decision, subgradient,
Polyak / diminishing step rule, stall handling) is reproduced exactly from
`core.ops_bounds.lagrangian_bound`, so the parallel bound must equal the
sequential bound bit-for-bit. `benchmark_parallel.py` asserts that it does.

Nothing in `core/` is modified; this is an opt-in alternative path.
"""
from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from adapters.ops_adapter import OPSInstance
from core.ops_bounds import orienteering_dp_with_selection

STALL_THRESHOLD = 30

_G = {}


def _worker_init(instance_path):
    """Load the instance once per worker so it is never pickled per task."""
    _G["inst"] = OPSInstance.from_instance_file(instance_path)


def _solve_chunk(payload):
    """Solve the orienteering subproblem for a contiguous block of processors."""
    ks, mu = payload
    inst = _G["inst"]
    profits = inst.profits
    out = []
    for k in ks:
        mod = list(profits)
        for j in inst.Jk[k]:
            mod[j] = mu.get((j, k), 0.0)
        feasible = [j for j in inst.Jk[k] if mod[j] > 0.0]
        if feasible:
            ok, selected = orienteering_dp_with_selection(
                feasible, inst.T, inst.start, inst.end, mod, inst.L
            )
            out.append((k, ok, tuple(sorted(selected))))
        else:
            out.append((k, 0.0, ()))
    return out


def _chunks(inst, w):
    """Partition processors into w blocks balanced by estimated DP cost.

    The bitmask DP is O(2^m * m^2) in the number of candidate jobs, so cost is
    wildly skewed: on some instances a single processor carries more than half
    the total work. Contiguous chunking therefore produces severe imbalance.
    We weight each processor by 2^|J_k| and assign longest-first to the
    currently lightest block (LPT), which is the standard 4/3-approximation for
    makespan on identical machines.

    Note this cannot beat the largest single subproblem: no partition splits one
    DP, so speedup is bounded by total_cost / max_single_cost regardless of w.
    """
    n = inst.num_processors
    w = max(1, min(w, n))
    cost = [2 ** len(inst.Jk[k]) for k in range(n)]
    order = sorted(range(n), key=lambda k: -cost[k])
    blocks = [[] for _ in range(w)]
    loads = [0] * w
    for k in order:
        i = loads.index(min(loads))
        blocks[i].append(k)
        loads[i] += cost[k]
    return [b for b in blocks if b]


def lagrangian_bound_parallel(
    inst,
    instance_path,
    max_iter=100,
    lower_bound=0.0,
    max_time=None,
    mu_init=None,
    workers=4,
    executor=None,
):
    """Parallel-subproblem version of core.ops_bounds.lagrangian_bound."""
    Kj = inst.Kj
    profits = inst.profits

    if mu_init is not None:
        mu = dict(mu_init)
        for j, ks in Kj.items():
            for k in ks:
                if (j, k) not in mu:
                    mu[(j, k)] = profits[j] / max(len(ks), 1)
    else:
        mu = {}
        for j, ks in Kj.items():
            share = profits[j] / max(len(ks), 1)
            for k in ks:
                mu[(j, k)] = share

    blocks = _chunks(inst, workers)

    own_executor = executor is None
    if own_executor:
        executor = ProcessPoolExecutor(
            max_workers=len(blocks),
            initializer=_worker_init,
            initargs=(instance_path,),
        )

    t0 = time.time()
    best_ub = float("inf")
    best_mu = dict(mu)
    bound_history = []
    converged = False
    it = 0
    stall_count = 0

    try:
        for it in range(max_iter):
            if max_time is not None and (time.time() - t0) >= max_time:
                break

            # ---- y_j threshold decision (sequential; negligible cost) --------
            y_contrib = 0.0
            y = {}
            for j, ks in Kj.items():
                total_mu = sum(mu.get((j, k2), 0.0) for k2 in ks)
                residual = profits[j] - total_mu
                if residual > 0.0:
                    y[j] = 1
                    y_contrib += residual
                else:
                    y[j] = 0

            # ---- per-processor orienteering, farmed out ----------------------
            total_ok = 0.0
            x_jk = {}
            results = executor.map(_solve_chunk, [(b, mu) for b in blocks])
            for block in results:
                for k, ok, selected in block:
                    total_ok += ok
                    sel = set(selected)
                    for j in inst.Jk[k]:
                        x_jk[(j, k)] = 1 if j in sel else 0

            lag_val = y_contrib + total_ok
            bound_history.append(lag_val)
            if lag_val < best_ub:
                best_ub = lag_val
                best_mu = dict(mu)
                stall_count = 0
            else:
                stall_count += 1

            # ---- subgradient -------------------------------------------------
            g_sq = 0.0
            g = {}
            for (j, k) in mu:
                g_val = x_jk.get((j, k), 0) - y.get(j, 0)
                g[(j, k)] = g_val
                g_sq += g_val * g_val

            if g_sq < 1e-10:
                converged = True
                break

            # ---- step size ---------------------------------------------------
            if stall_count < STALL_THRESHOLD:
                alpha = (lag_val - lower_bound) / g_sq
                alpha = max(1e-8, min(alpha, 2.0))
            else:
                c = max(1e-4, (best_ub - lower_bound) / max(g_sq ** 0.5, 1e-8))
                alpha = c / ((it + 1) ** 0.5)
                alpha = max(1e-8, min(alpha, 1.0))

            for (j, k) in mu:
                mu[(j, k)] = max(0.0, mu[(j, k)] - alpha * g[(j, k)])
    finally:
        if own_executor:
            executor.shutdown(wait=True)

    return {
        "upper_bound": best_ub,
        "iterations": it + 1,
        "converged": converged,
        "bound_history": bound_history,
        "best_mu": best_mu,
    }
