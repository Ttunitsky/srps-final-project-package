from __future__ import annotations

import math
import time
from itertools import combinations
from typing import Any

from core.ops_bounds import orienteering_dp, orienteering_dp_with_selection


def build_projected_cut_data(
    inst: Any,
    include_capacity: bool = True,
    include_pairs: bool = True,
    include_triples: bool = False,
    normalize_capacity: bool = True,
    skip_dominated_subsets: bool = True,
) -> dict:
    """Build the projected y-space cuts used by the cut-augmented LR.

    Capacity cuts:
        sum_{j in J_k} (b_j / O_k) y_j <= 1

    Incompatibility cuts:
        sum_{j in S} y_j <= |S| - 1

    The capacity cuts are normalized by default so that their subgradients
    are numerically comparable to the pair/triple cut subgradients.
    """

    Kj = inst.Kj
    Jk = inst.Jk
    profits = inst.profits

    cap_cuts: dict[int, dict] = {}
    incomp_cuts: dict[tuple[int, ...], dict] = {}

    if include_capacity:
        for k in range(inst.num_processors):
            jobs_k = list(Jk[k])
            if not jobs_k:
                continue

            ok = orienteering_dp(
                jobs=jobs_k,
                T=inst.T,
                start=inst.start,
                end=inst.end,
                profits=profits,
                L=inst.L,
            )

            if ok <= 1e-9:
                continue

            if normalize_capacity:
                rhs = 1.0
                coeffs = {j: float(profits[j]) / float(ok) for j in jobs_k}
            else:
                rhs = float(ok)
                coeffs = {j: float(profits[j]) for j in jobs_k}

            cap_cuts[k] = {
                "rhs": rhs,
                "coeffs": coeffs,
                "ok": float(ok),
                "normalized": bool(normalize_capacity),
            }

    def subset_route_feasible(subset: tuple[int, ...]) -> bool:
        unit_profit = [0.0] * len(profits)
        for j in subset:
            unit_profit[j] = 1.0

        val = orienteering_dp(
            jobs=list(subset),
            T=inst.T,
            start=inst.start,
            end=inst.end,
            profits=unit_profit,
            L=inst.L,
        )

        return val >= len(subset) - 1e-9

    if include_pairs or include_triples:
        max_size = 3 if include_triples else 2
        generated: set[tuple[int, ...]] = set()

        for k in range(inst.num_processors):
            jobs_k = sorted(j for j in Jk[k] if j in Kj)

            for size in range(2, max_size + 1):
                if size == 2 and not include_pairs:
                    continue
                if size == 3 and not include_triples:
                    continue
                if len(jobs_k) < size:
                    continue

                for subset in combinations(jobs_k, size):
                    subset_key = tuple(sorted(subset))

                    if subset_key in generated:
                        continue

                    if skip_dominated_subsets and size > 2:
                        dominated = False
                        for pair in combinations(subset_key, 2):
                            if tuple(sorted(pair)) in generated:
                                dominated = True
                                break
                        if dominated:
                            continue

                    if not subset_route_feasible(subset_key):
                        incomp_cuts[subset_key] = {
                            "rhs": float(size - 1),
                            "jobs": subset_key,
                            "size": size,
                        }
                        generated.add(subset_key)

    return {
        "capacity": cap_cuts,
        "incompatibility": incomp_cuts,
    }


def cut_augmented_lagrangian_bound(
    inst: Any,
    max_iter: int = 100,
    lower_bound: float = 0.0,
    max_time: float | None = None,
    mu_init: dict | None = None,
    gamma_init: dict | None = None,
    nu_init: dict | None = None,
    include_capacity: bool = True,
    include_pairs: bool = True,
    include_triples: bool = False,
    normalize_capacity: bool = True,
    cut_step_scale: float = 1.0,
    projected_g_sq: bool = True,
) -> dict:
    """Cut-augmented Lagrangian upper bound for SRPS.

    Starting from the existing LR:
        relax y_j <= x_{j,k} with mu_{j,k} >= 0.

    Add projected y-space valid inequalities and relax them too:

        capacity:
            sum_j a_{k,j} y_j <= rhs_k

        incompatibility:
            sum_{j in S} y_j <= |S| - 1

    For maximization, each valid inequality is written as rhs - lhs >= 0
    and added to the relaxed objective with a nonnegative multiplier.
    This preserves the decomposition:

      1. y_j remains a threshold decision.
      2. each processor subproblem remains a single-processor OP.
    """

    Kj = inst.Kj
    Jk = inst.Jk
    profits = inst.profits

    cuts = build_projected_cut_data(
        inst,
        include_capacity=include_capacity,
        include_pairs=include_pairs,
        include_triples=include_triples,
        normalize_capacity=normalize_capacity,
    )
    cap_cuts = cuts["capacity"]
    incomp_cuts = cuts["incompatibility"]

    # ---- Initialize mu as in the original LR --------------------------------
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

    # ---- Initialize cut multipliers -----------------------------------------
    if gamma_init is not None:
        gamma = {k: float(v) for k, v in gamma_init.items() if k in cap_cuts}
    else:
        gamma = {k: 0.0 for k in cap_cuts}

    if nu_init is not None:
        nu = {tuple(k): float(v) for k, v in nu_init.items() if tuple(k) in incomp_cuts}
    else:
        nu = {s: 0.0 for s in incomp_cuts}

    t0 = time.perf_counter()

    best_ub = float("inf")
    best_mu = dict(mu)
    best_gamma = dict(gamma)
    best_nu = dict(nu)

    bound_history: list[float] = []
    components_history: list[dict] = []

    converged = False
    stall_count = 0
    stall_threshold = 20
    it = -1

    for it in range(max_iter):
        if max_time is not None and (time.perf_counter() - t0) >= max_time:
            break

        # ---- Precompute y penalties from cut multipliers --------------------
        y_cut_penalty = {j: 0.0 for j in Kj}

        for k, c in cap_cuts.items():
            gk = gamma.get(k, 0.0)
            if gk == 0.0:
                continue
            for j, coeff in c["coeffs"].items():
                if j in y_cut_penalty:
                    y_cut_penalty[j] += gk * coeff

        for subset, ns in nu.items():
            if ns == 0.0:
                continue
            for j in subset:
                if j in y_cut_penalty:
                    y_cut_penalty[j] += ns

        # ---- y_j threshold decision -----------------------------------------
        y_contrib = 0.0
        y: dict[int, int] = {}

        for j, ks in Kj.items():
            total_mu = sum(mu.get((j, k2), 0.0) for k2 in ks)
            residual = profits[j] - total_mu - y_cut_penalty.get(j, 0.0)

            if residual > 0.0:
                y[j] = 1
                y_contrib += residual
            else:
                y[j] = 0

        # ---- Per-processor OP subproblems with profits mu_{j,k} -------------
        total_ok = 0.0
        x_jk: dict[tuple[int, int], int] = {}

        for k in range(inst.num_processors):
            mod_profits = list(profits)

            for j in Jk[k]:
                mod_profits[j] = mu.get((j, k), 0.0)

            feasible_jobs = [j for j in Jk[k] if mod_profits[j] > 0.0]

            if feasible_jobs:
                ok, selected = orienteering_dp_with_selection(
                    feasible_jobs,
                    inst.T,
                    inst.start,
                    inst.end,
                    mod_profits,
                    inst.L,
                )
                total_ok += ok

                selected_set = set(selected)
                for j in feasible_jobs:
                    x_jk[(j, k)] = 1 if j in selected_set else 0

                for j in Jk[k]:
                    if (j, k) not in x_jk:
                        x_jk[(j, k)] = 0
            else:
                for j in Jk[k]:
                    x_jk[(j, k)] = 0

        # ---- Constant terms from relaxed cuts --------------------------------
        cap_const = sum(gamma.get(k, 0.0) * c["rhs"] for k, c in cap_cuts.items())
        incomp_const = sum(nu.get(s, 0.0) * c["rhs"] for s, c in incomp_cuts.items())

        lag_val = y_contrib + total_ok + cap_const + incomp_const
        bound_history.append(lag_val)

        components_history.append(
            {
                "it": it,
                "lag_val": lag_val,
                "y_contrib": y_contrib,
                "route_contrib": total_ok,
                "cap_const": cap_const,
                "incomp_const": incomp_const,
                "num_y_selected": sum(y.values()),
            }
        )

        if lag_val < best_ub:
            best_ub = lag_val
            best_mu = dict(mu)
            best_gamma = dict(gamma)
            best_nu = dict(nu)
            stall_count = 0
        else:
            stall_count += 1

        # ---- Subgradients ----------------------------------------------------
        g_mu: dict[tuple[int, int], float] = {}
        g_gamma: dict[int, float] = {}
        g_nu: dict[tuple[int, ...], float] = {}

        g_sq = 0.0

        for (j, k) in mu:
            g_val = x_jk.get((j, k), 0) - y.get(j, 0)
            g_mu[(j, k)] = g_val
            g_sq += g_val * g_val

        for k, c in cap_cuts.items():
            lhs = 0.0
            for j, coeff in c["coeffs"].items():
                lhs += coeff * y.get(j, 0)

            g_val = c["rhs"] - lhs
            g_gamma[k] = g_val

            # Projected scaling: if gamma is zero and the constraint is slack,
            # do not let many inactive cuts shrink the step size.
            if (not projected_g_sq) or gamma.get(k, 0.0) > 0.0 or g_val < 0.0:
                g_sq += (cut_step_scale * g_val) ** 2

        for subset, c in incomp_cuts.items():
            lhs = sum(y.get(j, 0) for j in subset)
            g_val = c["rhs"] - lhs
            g_nu[subset] = g_val

            if (not projected_g_sq) or nu.get(subset, 0.0) > 0.0 or g_val < 0.0:
                g_sq += (cut_step_scale * g_val) ** 2

        if g_sq < 1e-10:
            converged = True
            break

        # ---- Step size -------------------------------------------------------
        if stall_count < stall_threshold:
            alpha = (lag_val - lower_bound) / g_sq
            alpha = max(1e-8, min(alpha, 2.0))
        else:
            c = max(1e-4, (best_ub - lower_bound) / max(math.sqrt(g_sq), 1e-8))
            alpha = c / math.sqrt(it + 1)
            alpha = max(1e-8, min(alpha, 1.0))

        # ---- Projected updates ----------------------------------------------
        for key, gv in g_mu.items():
            mu[key] = max(0.0, mu[key] - alpha * gv)

        for k, gv in g_gamma.items():
            gamma[k] = max(0.0, gamma.get(k, 0.0) - alpha * cut_step_scale * gv)

        for subset, gv in g_nu.items():
            nu[subset] = max(0.0, nu.get(subset, 0.0) - alpha * cut_step_scale * gv)

    positive_gamma = sum(1 for v in best_gamma.values() if v > 1e-9)
    positive_nu = sum(1 for v in best_nu.values() if v > 1e-9)

    return {
        "upper_bound": best_ub,
        "iterations": it + 1 if it >= 0 else 0,
        "converged": converged,
        "bound_history": bound_history,
        "components_history": components_history,
        "best_mu": best_mu,
        "best_gamma": best_gamma,
        "best_nu": best_nu,
        "num_capacity_cuts": len(cap_cuts),
        "num_incompatibility_cuts": len(incomp_cuts),
        "positive_gamma": positive_gamma,
        "positive_nu": positive_nu,
        "include_capacity": include_capacity,
        "include_pairs": include_pairs,
        "include_triples": include_triples,
        "normalize_capacity": normalize_capacity,
    }
