"""
Upper bounds for OPS via relaxation-based bounding.

Three methods are implemented:

1. Independent orienteering (fast, loose)
   Drop sync entirely. Each of the 55 processors independently solves an
   orienteering problem over its job set. Dividing by min|Kj| gives a
   valid upper bound.

2. Group decomposition (data-adaptive)
   Derive unique processor groups from Kj; solve one orienteering per group.
   Valid but typically looser than (1) when groups overlap.

3. Lagrangian relaxation (slower, tightest)
   Relax the coupling constraint y_j <= x_{j,k} (profit earned only when
   ALL required processors serve j) using dual variables mu_{j,k} >= 0.
   The Lagrangian decomposes into 55 independent sub-problems (orienteering
   with modified profits mu_{j,k}) plus a scalar term for y_j. Sub-gradient
   descent on mu minimizes the dual, approaching the LP-relaxation bound.

   L(mu) = sum_j max(0, b[j] - sum_{k in Kj[j]} mu_{j,k})
           + sum_k O_k(mu)
   where O_k(mu) = orienteering DP for k with profit mu_{j,k} for each j.

   Sub-gradient: d L / d mu_{j,k} = x_{j,k}* - y_j*
   Update:       mu_{j,k} <- max(0, mu_{j,k} - alpha * (x_{j,k}* - y_j*))
   Step size:    Polyak  alpha = (L(mu) - LB) / ||g||^2

best_upper_bound() returns min(all methods); wires Lagrangian in when
include_lagrangian=True (default False to keep existing scripts fast).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from adapters.ops_adapter import OPSInstance

def orienteering_dp(jobs: list, T: list, start: int, end: int,
                    profits: list, L: float) -> float:
    """Exact orienteering DP for a single processor.

    Finds the maximum profit route starting at `start`, visiting any
    subset of `jobs` in any order, and ending at `end`, within time L.

    Uses bitmask DP: dp[mask][i] = minimum time to visit exactly the
    jobs in `mask`, ending at local index i.

    Parameters
    ----------
    jobs    : list of job indices (global, from Jk[k])
    T       : full transition-time matrix (indexed by global job index)
    start   : global index of dummy start (0)
    end     : global index of dummy end (n+1)
    profits : profit list indexed by global job index
    L       : time budget

    Returns
    -------
    Maximum achievable profit (float).
    """
    m = len(jobs)
    if m == 0:
        return 0.0

    INF = float("inf")
    # dp[mask][i] = min time to visit jobs-in-mask ending at local index i
    dp = [[INF] * m for _ in range(1 << m)]

    # Base: visit single job i directly from start
    for i in range(m):
        t = float(T[start][jobs[i]])
        if t <= L:
            dp[1 << i][i] = t

    # Fill in increasing mask size
    for mask in range(1, 1 << m):
        for i in range(m):
            if not (mask >> i & 1):
                continue
            if dp[mask][i] == INF:
                continue
            # Extend to job j not yet in mask
            for j in range(m):
                if mask >> j & 1:
                    continue
                new_mask = mask | (1 << j)
                new_t = dp[mask][i] + float(T[jobs[i]][jobs[j]])
                if new_t < dp[new_mask][j]:
                    dp[new_mask][j] = new_t

    # Find maximum profit over all feasible (mask, i) pairs
    best = 0.0
    for mask in range(1, 1 << m):
        mask_profit = sum(profits[jobs[i]] for i in range(m) if mask >> i & 1)
        for i in range(m):
            if not (mask >> i & 1):
                continue
            if dp[mask][i] == INF:
                continue
            if dp[mask][i] + float(T[jobs[i]][end]) <= L:
                if mask_profit > best:
                    best = mask_profit

    return best


def orienteering_dp_with_selection(jobs: list, T: list, start: int, end: int,
                                   profits: list, L: float) -> tuple:
    """Like orienteering_dp but also returns the set of selected job indices.

    Returns
    -------
    (profit, selected_set)  where selected_set is a set of global job indices.
    """
    m = len(jobs)
    if m == 0:
        return 0.0, set()

    INF = float("inf")
    dp = [[INF] * m for _ in range(1 << m)]

    for i in range(m):
        t = float(T[start][jobs[i]])
        if t <= L:
            dp[1 << i][i] = t

    for mask in range(1, 1 << m):
        for i in range(m):
            if not (mask >> i & 1) or dp[mask][i] == INF:
                continue
            for j in range(m):
                if mask >> j & 1:
                    continue
                new_t = dp[mask][i] + float(T[jobs[i]][jobs[j]])
                new_mask = mask | (1 << j)
                if new_t < dp[new_mask][j]:
                    dp[new_mask][j] = new_t

    best_profit = 0.0
    best_mask = 0
    for mask in range(1, 1 << m):
        mask_profit = sum(profits[jobs[i]] for i in range(m) if mask >> i & 1)
        for i in range(m):
            if not (mask >> i & 1) or dp[mask][i] == INF:
                continue
            if dp[mask][i] + float(T[jobs[i]][end]) <= L:
                if mask_profit > best_profit:
                    best_profit = mask_profit
                    best_mask = mask

    selected = {jobs[i] for i in range(m) if best_mask >> i & 1}
    return best_profit, selected


def lagrangian_bound(inst: "OPSInstance",
                     max_iter: int = 100,
                     lower_bound: float = 0.0,
                     max_time: float = None,
                     mu_init: dict = None) -> dict:
    """Lagrangian relaxation upper bound for OPS.

    Relaxes the coupling constraint y_j <= x_{j,k} for all j, k in Kj[j]
    using dual variables mu_{j,k} >= 0.  At each iteration the Lagrangian
    decomposes into:
      - A trivial y_j decision: y_j = 1 iff b[j] > sum_{k} mu_{j,k}
      - 55 independent orienteering sub-problems with job profit mu_{j,k}

    Sub-gradient descent on mu minimises L(mu) and converges to the LP
    relaxation value (which is <= independent orienteering bound).

    Parameters
    ----------
    inst        : OPSInstance
    max_iter    : number of sub-gradient iterations (default 100)
    lower_bound : known feasible solution value for Polyak step size;
                  use 0 if unknown.
    max_time    : optional wall-clock limit in seconds.
    mu_init     : optional dict of {(j,k): value} to warm-start multipliers
                  from a previous run. If None, initialises from b[j]/|Kj[j]|.
                  Pass result['best_mu'] from a prior call to chain runs.

    Returns
    -------
    dict with keys:
        'upper_bound'   : best L(mu) found (valid UB on P*)
        'iterations'    : sub-gradient iterations run
        'converged'     : True if sub-gradient vanished (LP optimal reached)
        'bound_history' : list of L(mu) value at each iteration
        'best_mu'       : multiplier dict at the best UB — pass as mu_init
                          to a subsequent call to continue without restarting.
    """
    Kj = inst.Kj
    Jk = inst.Jk
    profits = inst.profits

    # ---- Initialise or warm-start multipliers --------------------------------
    if mu_init is not None:
        # Deep copy so the caller's dict is not mutated
        mu: dict = dict(mu_init)
        # Ensure all (j,k) pairs are present (guard against schema mismatch)
        for j, ks in Kj.items():
            for k in ks:
                if (j, k) not in mu:
                    mu[(j, k)] = profits[j] / max(len(ks), 1)
    else:
        # Cold start: fair profit split across required processors.
        # L(mu_init) = independent-orienteering bound; descent can only reduce.
        mu = {}
        for j, ks in Kj.items():
            share = profits[j] / max(len(ks), 1)
            for k in ks:
                mu[(j, k)] = share

    import time as _time
    _t0 = _time.time()

    _T = inst.T

    best_ub = float("inf")
    best_mu: dict = dict(mu)   # multipliers at the best UB seen so far
    bound_history: list = []
    converged = False
    it = 0
    stall_count = 0           # consecutive iterations without UB improvement
    STALL_THRESHOLD = 30      # switch to diminishing step after this many stalls

    for it in range(max_iter):
        if max_time is not None and (_time.time() - _t0) >= max_time:
            break

        # ---- y_j threshold decision -----------------------------------------
        y_contrib = 0.0
        y: dict = {}
        for j, ks in Kj.items():
            total_mu = sum(mu.get((j, k2), 0.0) for k2 in ks)
            residual = profits[j] - total_mu
            if residual > 0.0:
                y[j] = 1
                y_contrib += residual
            else:
                y[j] = 0

        # ---- Per-processor orienteering with profits mu_{j,k} ---------------
        total_ok = 0.0
        x_jk: dict = {}

        for k in range(inst.num_processors):
            mod_profits = list(profits)
            for j in Jk[k]:
                mod_profits[j] = mu.get((j, k), 0.0)

            feasible_jobs = [j for j in Jk[k] if mod_profits[j] > 0.0]

            if feasible_jobs:
                ok, selected = orienteering_dp_with_selection(
                    feasible_jobs, _T, inst.start, inst.end,
                    mod_profits, inst.L,
                )
                total_ok += ok
                for j in feasible_jobs:
                    x_jk[(j, k)] = 1 if j in selected else 0
            else:
                for j in Jk[k]:
                    x_jk[(j, k)] = 0

        lag_val = y_contrib + total_ok
        bound_history.append(lag_val)
        if lag_val < best_ub:
            best_ub = lag_val
            best_mu = dict(mu)   # record multipliers at the new best
            stall_count = 0
        else:
            stall_count += 1

        # ---- Sub-gradient g_{j,k} = x_{j,k}* - y_j* -----------------------
        g_sq = 0.0
        g: dict = {}
        for (j, k) in mu:
            g_val = x_jk.get((j, k), 0) - y.get(j, 0)
            g[(j, k)] = g_val
            g_sq += g_val * g_val

        if g_sq < 1e-10:
            converged = True
            break

        # ---- Step size: Polyak primary, diminishing fallback on stall -------
        # Polyak: alpha = (L(mu) - LB) / ||g||^2
        # When the bound stalls, Polyak can oscillate (large alpha gets clamped
        # at 2.0 while ||g|| is small). Switch to diminishing c/sqrt(t+1) after
        # STALL_THRESHOLD consecutive non-improving iterations.
        if stall_count < STALL_THRESHOLD:
            alpha = (lag_val - lower_bound) / g_sq
            alpha = max(1e-8, min(alpha, 2.0))
        else:
            # Diminishing step: guaranteed convergence, slower but stable near
            # the dual optimum. c calibrated to match Polyak magnitude at iter 0.
            c = max(1e-4, (best_ub - lower_bound) / max(g_sq ** 0.5, 1e-8))
            alpha = c / ((it + 1) ** 0.5)
            alpha = max(1e-8, min(alpha, 1.0))

        # Sub-gradient descent: mu <- max(0, mu - alpha * g)
        for (j, k) in mu:
            mu[(j, k)] = max(0.0, mu[(j, k)] - alpha * g[(j, k)])

    return {
        "upper_bound": best_ub,
        "iterations": it + 1,
        "converged": converged,
        "bound_history": bound_history,
        "best_mu": best_mu,
    }


def evaluate_lagrangian(inst: "OPSInstance", mu: dict) -> dict:
    """Recompute the Lagrangian dual value L(mu) at a *given* multiplier vector.

    This is the single-point evaluation underlying ``lagrangian_bound``'s inner
    loop, exposed on its own so a certificate verifier can independently
    reconstruct the dual bound from stored multipliers (rather than re-running
    sub-gradient descent). For any ``mu`` with mu_{j,k} >= 0,

        L(mu) = sum_j max(0, b[j] - sum_{k in Kj[j]} mu_{j,k})  +  sum_k O_k(mu)

    is a valid upper bound on the SRPS optimum P*. Because all profits are
    integer, ``floor(L(mu))`` is a valid integer upper bound.

    Parameters
    ----------
    inst : OPSInstance
    mu   : dict {(j, k): value} of dual multipliers (e.g. ``best_mu`` returned
           by ``lagrangian_bound`` or loaded from a saved certificate).

    Returns
    -------
    dict with keys:
        'lagrangian_value' : L(mu) (real-valued valid UB on P*)
        'upper_bound'      : floor(L(mu)) (valid integer UB)
        'y_contrib'        : the relaxed-profit term sum_j max(0, b_j - sum mu)
        'route_contrib'    : the orienteering term sum_k O_k(mu)
    """
    Kj = inst.Kj
    Jk = inst.Jk
    profits = inst.profits
    _T = inst.T

    # ---- y_j threshold decision (identical to lagrangian_bound) -------------
    y_contrib = 0.0
    for j, ks in Kj.items():
        total_mu = sum(mu.get((j, k), 0.0) for k in ks)
        residual = profits[j] - total_mu
        if residual > 0.0:
            y_contrib += residual

    # ---- Per-processor orienteering with profits mu_{j,k} -------------------
    route_contrib = 0.0
    for k in range(inst.num_processors):
        mod_profits = list(profits)
        for j in Jk[k]:
            mod_profits[j] = mu.get((j, k), 0.0)
        feasible_jobs = [j for j in Jk[k] if mod_profits[j] > 0.0]
        if feasible_jobs:
            route_contrib += orienteering_dp(
                feasible_jobs, _T, inst.start, inst.end, mod_profits, inst.L,
            )

    lag_val = y_contrib + route_contrib
    import math as _math
    return {
        "lagrangian_value": lag_val,
        "upper_bound": _math.floor(lag_val),
        "y_contrib": y_contrib,
        "route_contrib": route_contrib,
    }


def relaxed_orienteering_bound(inst: "OPSInstance") -> dict:
    """Independent-orienteering upper bound: 55 processors solved separately.

    Valid because any OPS-feasible selection S* can be achieved by each
    processor independently (ignoring sync), so O_k >= contribution of k
    to S*. Summing and dividing by min|Kj| recovers a valid UB on P*.

    Returns
    -------
    dict with keys:
        'processor_bounds' : list of O_k for each processor k
        'sum_ok'           : sum of all O_k
        'min_kj'           : minimum |Kj[j]| across all jobs
        'upper_bound'      : sum_ok / min_kj  (valid OPS upper bound)
    """
    processor_bounds = []
    for k in range(inst.num_processors):
        ok = orienteering_dp(
            jobs=inst.Jk[k],
            T=inst.T,
            start=inst.start,
            end=inst.end,
            profits=inst.profits,
            L=inst.L,
        )
        processor_bounds.append(ok)

    sum_ok = sum(processor_bounds)

    # min |Kj[j]| across all jobs (= 4 for all Class 4, 1 for Class 1, etc.)
    min_kj = min(len(ks) for ks in inst.Kj.values()) if inst.Kj else 1

    upper_bound = sum_ok / min_kj if min_kj > 0 else sum_ok

    return {
        "processor_bounds": processor_bounds,
        "sum_ok": sum_ok,
        "min_kj": min_kj,
        "upper_bound": upper_bound,
    }


def group_decomposition_bound(inst: "OPSInstance") -> dict:
    """Group-decomposition upper bound: solve one orienteering per unique group.

    Each job j belongs to exactly one processor group g = Kj[j] (the set of
    processors that must serve j). Groups are read directly from the instance —
    no structural assumption (sliding-window or otherwise) is made.

    For any OPS optimal S*, each job j in S* belongs to some group g(j), and
    the jobs in S* ∩ group g form a feasible synchronized schedule for g.
    Therefore OPS_g >= sum_{j in S*, Kj[j]=g} b[j], and summing over all
    groups gives UB_group >= P*.  Valid for any instance structure.

    When groups heavily overlap (processors appear in many groups), this bound
    tends to be looser than relaxed_orienteering_bound because each group gets
    an independent full time budget L. When groups are disjoint or nearly so,
    it can be tighter.  Always take min(both) — see best_upper_bound().

    Returns
    -------
    dict with keys:
        'group_bounds' : {group_tuple: OPS_g profit}
        'num_groups'   : number of unique processor groups
        'upper_bound'  : sum of OPS_g across all groups
    """
    # Derive unique groups from Kj (fully data-adaptive, no structural assumption)
    from collections import defaultdict
    groups: dict = defaultdict(list)
    for j, ks in inst.Kj.items():
        groups[tuple(sorted(ks))].append(j)

    group_bounds = {}
    for g, jobs_g in groups.items():
        # With shared T matrix all processors in g can follow the same route,
        # so mini-OPS for group g = single-processor orienteering on jobs_g.
        ok = orienteering_dp(
            jobs=jobs_g,
            T=inst.T,
            start=inst.start,
            end=inst.end,
            profits=inst.profits,
            L=inst.L,
        )
        group_bounds[g] = ok

    ub = sum(group_bounds.values())
    return {
        "group_bounds": group_bounds,
        "num_groups": len(groups),
        "upper_bound": ub,
    }


def best_upper_bound(inst: "OPSInstance",
                     include_lagrangian: bool = False,
                     lagrangian_lower_bound: float = 0.0,
                     lagrangian_max_iter: int = 100) -> dict:
    """Compute the tightest valid upper bound available for this instance.

    Takes the minimum across all valid bounding methods.  Each method is
    valid independently; the minimum is always valid and never looser than
    any individual bound.

    Parameters
    ----------
    inst                   : OPSInstance
    include_lagrangian     : if True, also run lagrangian_bound() and
                             include it in the min; default False because
                             Lagrangian takes ~100× longer than the others.
    lagrangian_lower_bound : passed to lagrangian_bound() as lower_bound
                             (best known feasible value; use 0 if unknown).
    lagrangian_max_iter    : sub-gradient iterations for Lagrangian.

    Returns
    -------
    dict with keys:
        'upper_bound'   : tightest UB found
        'method'        : name of the winning method
        'ub_indep'      : independent-orienteering bound value
        'ub_group'      : group-decomposition bound value
        'ub_lagrangian' : Lagrangian bound value (or None if not computed)
    """
    r_indep = relaxed_orienteering_bound(inst)
    r_group = group_decomposition_bound(inst)

    ub_indep = r_indep["upper_bound"]
    ub_group = r_group["upper_bound"]

    ub_lag = None
    r_lag = None
    if include_lagrangian:
        r_lag = lagrangian_bound(inst,
                                 max_iter=lagrangian_max_iter,
                                 lower_bound=lagrangian_lower_bound)
        ub_lag = r_lag["upper_bound"]

    candidates = {
        "independent_orienteering": ub_indep,
        "group_decomposition":      ub_group,
    }
    if ub_lag is not None:
        candidates["lagrangian"] = ub_lag

    winner = min(candidates, key=candidates.__getitem__)
    ub = candidates[winner]

    return {
        "upper_bound":   ub,
        "method":        winner,
        "ub_indep":      ub_indep,
        "ub_group":      ub_group,
        "ub_lagrangian": ub_lag,
    }


def certified_gap(obj: float, inst: "OPSInstance") -> dict:
    """Certified optimality gap for a solution with value `obj`.

    Uses best_upper_bound() — the tightest valid bound available —
    so the certified gap is as small as possible.

    Returns
    -------
    dict with keys:
        'obj'            : our solution value
        'upper_bound'    : tightest valid UB
        'bound_method'   : which bound won
        'ub_indep'       : independent-orienteering bound
        'ub_group'       : group-decomposition bound
        'gap_abs'        : upper_bound - obj
        'gap_pct'        : gap as % of upper_bound
        'proven_optimal' : True if gap_abs < 1e-6
    """
    result = best_upper_bound(inst)
    ub = result["upper_bound"]
    gap_abs = ub - obj
    gap_pct = (gap_abs / ub * 100) if ub > 0 else 0.0

    return {
        "obj": obj,
        "upper_bound": ub,
        "bound_method": result["method"],
        "ub_indep": result["ub_indep"],
        "ub_group": result["ub_group"],
        "gap_abs": gap_abs,
        "gap_pct": gap_pct,
        "proven_optimal": gap_abs < 1e-6,
    }
