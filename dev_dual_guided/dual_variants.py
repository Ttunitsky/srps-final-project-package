"""Pluggable multiplier initialisation and step rules for the Lagrangian dual.

Two untested coupling directions from the syllabus:

  (1) Primal -> dual seeding (Week IX, "tie-breaking can prefer solutions closer
      to a target primal"). The solver currently chains dual -> dual via
      mu_init=prev_mu; the incumbent enters only as the Polyak target, never as
      structure. Here we seed mu from the incumbent's own selection so that the
      relaxation's y agrees with the incumbent at iteration 0.

  (2) Dual stabilisation (Week XI, p14: box / trust region, proximal /
      smoothing). The solver uses Polyak with a diminishing fallback after 30
      stalls. Here we add a trust region on the step and Wentges-style smoothing
      toward a stability centre.

Everything else reproduces core.ops_bounds.lagrangian_bound exactly, so any
difference in the bound trajectory is attributable to the variant under test.
"""
from __future__ import annotations

import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.ops_bounds import orienteering_dp_with_selection

STALL_THRESHOLD = 30


# --------------------------------------------------------------- mu init ----
def mu_fair(inst, **kw):
    """Baseline: fair profit split b_j / |K_j|."""
    mu = {}
    for j, ks in inst.Kj.items():
        share = inst.profits[j] / max(len(ks), 1)
        for k in ks:
            mu[(j, k)] = share
    return mu


def mu_incumbent(inst, selected=None, eps=0.10, **kw):
    """Seed mu from a primal incumbent's selection.

    A job the incumbent takes should look attractive to the relaxation, i.e.
    have positive reduced profit, so its multipliers are scaled *down*. A job the
    incumbent rejects is scaled *up* so its reduced profit is non-positive. At
    eps=0 this degenerates to the fair split.
    """
    selected = selected or set()
    mu = {}
    for j, ks in inst.Kj.items():
        share = inst.profits[j] / max(len(ks), 1)
        factor = (1.0 - eps) if j in selected else (1.0 + eps)
        for k in ks:
            mu[(j, k)] = share * factor
    return mu


# ------------------------------------------------------------- step rules ---
def step_polyak(lag_val, lower_bound, g_sq, it, stall, best_ub, **kw):
    """Baseline: Polyak, with diminishing fallback once the bound stalls."""
    if stall < STALL_THRESHOLD:
        a = (lag_val - lower_bound) / g_sq
        return max(1e-8, min(a, 2.0))
    c = max(1e-4, (best_ub - lower_bound) / max(g_sq ** 0.5, 1e-8))
    return max(1e-8, min(c / ((it + 1) ** 0.5), 1.0))


def step_trust(lag_val, lower_bound, g_sq, it, stall, best_ub, trust=0.5, **kw):
    """Polyak clipped into a trust region that tightens as the bound stalls."""
    a = step_polyak(lag_val, lower_bound, g_sq, it, stall, best_ub)
    radius = trust / (1.0 + 0.1 * stall)
    return max(1e-8, min(a, radius))


# ------------------------------------------------------------ main driver ---
def lagrangian_variant(
    inst,
    max_iter=200,
    lower_bound=0.0,
    max_time=None,
    mu_init_fn=mu_fair,
    mu_init_kw=None,
    step_fn=step_polyak,
    step_kw=None,
    smoothing=0.0,
):
    """Subgradient loop with pluggable init, step rule, and dual smoothing.

    smoothing in [0,1): Wentges-style convex combination of the new multipliers
    with the stability centre (the mu at the best bound seen). 0 disables it.
    """
    mu_init_kw = mu_init_kw or {}
    step_kw = step_kw or {}
    profits = inst.profits
    Kj = inst.Kj

    mu = mu_init_fn(inst, **mu_init_kw)

    t0 = time.time()
    best_ub = float("inf")
    best_mu = dict(mu)
    history = []
    converged = False
    it = 0
    stall = 0

    for it in range(max_iter):
        if max_time is not None and (time.time() - t0) >= max_time:
            break

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

        total_ok = 0.0
        x_jk = {}
        for k in range(inst.num_processors):
            mod = list(profits)
            for j in inst.Jk[k]:
                mod[j] = mu.get((j, k), 0.0)
            feasible = [j for j in inst.Jk[k] if mod[j] > 0.0]
            if feasible:
                ok, sel = orienteering_dp_with_selection(
                    feasible, inst.T, inst.start, inst.end, mod, inst.L
                )
                total_ok += ok
                for j in feasible:
                    x_jk[(j, k)] = 1 if j in sel else 0
            else:
                for j in inst.Jk[k]:
                    x_jk[(j, k)] = 0

        lag_val = y_contrib + total_ok
        history.append(lag_val)
        if lag_val < best_ub:
            best_ub = lag_val
            best_mu = dict(mu)
            stall = 0
        else:
            stall += 1

        g_sq = 0.0
        g = {}
        for (j, k) in mu:
            gv = x_jk.get((j, k), 0) - y.get(j, 0)
            g[(j, k)] = gv
            g_sq += gv * gv

        if g_sq < 1e-10:
            converged = True
            break

        alpha = step_fn(lag_val, lower_bound, g_sq, it, stall, best_ub, **step_kw)

        for (j, k) in mu:
            nxt = max(0.0, mu[(j, k)] - alpha * g[(j, k)])
            if smoothing > 0.0:
                nxt = (1.0 - smoothing) * nxt + smoothing * best_mu.get((j, k), nxt)
            mu[(j, k)] = nxt

    return {
        "upper_bound": best_ub,
        "iterations": it + 1,
        "converged": converged,
        "bound_history": history,
        "best_mu": best_mu,
    }
