"""Exact SRPS-1 arc-flow model, solved with CPLEX.

Implements the compact formulation stated in the parent paper:

    max  sum_j b_j y_j
    s.t. sum_{l in V_k\\{0}}   z^k_{0l}   = 1                     for all k
         sum_{i in V_k\\{n+1}} z^k_{i,n+1} = 1                    for all k
         sum_{i in V_k\\{j}} z^k_{ij} = x_{j,k}                   for all k, j in J_k
         sum_{l in V_k\\{j}} z^k_{jl} = x_{j,k}                   for all k, j in J_k
         y_j <= x_{j,k}                                          for all j, k in K_j
         s_{l,k} >= s_{i,k} + t_{il} - M(1 - z^k_{il})            for all k, arcs
         s_{j,k} = s_{j,k'}                                       for all j, k,k' in K_j
         s_{n+1,k} - s_{0,k} <= L                                 for all k

The arc-time propagation doubles as subtour elimination: start times strictly
increase along any used arc with positive transition time, so cycles are
infeasible.

Why this matters for the study: the certified gap is (ub - z)/ub, and the ceiling
argument bounds any strengthening by that quantity because the optimum P* is only
known to lie in [z, ub]. Solving instances exactly pins P* down, which converts
the ceiling from an upper bound on achievable gain into a measured decomposition
of the gap into bound looseness (ub - P*) and primal suboptimality (P* - z).

Requires Python 3.8-3.10 and the CPLEX Python API on PYTHONPATH.
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import cplex
from cplex.exceptions import CplexError  # noqa: F401


def build_model(inst, verbose=False):
    """Return (cplex.Cplex, index maps) for the SRPS-1 arc-flow model."""
    n_end = inst.end
    start = inst.start
    T = inst.T
    L = float(inst.L)

    # Big-M: any start time fits inside the horizon plus one transition.
    maxt = 0.0
    for k in range(inst.num_processors):
        nodes = list(inst.Jk[k]) + [start, n_end]
        for i in nodes:
            for l in nodes:
                if i != l:
                    maxt = max(maxt, float(T[i][l]))
    M = L + maxt + 1.0

    mdl = cplex.Cplex()
    mdl.objective.set_sense(mdl.objective.sense.maximize)
    if not verbose:
        mdl.set_log_stream(None)
        mdl.set_results_stream(None)
        mdl.set_warning_stream(None)
        mdl.set_error_stream(None)

    names, objs, lbs, ubs, types = [], [], [], [], []

    y = {}
    for j in inst.Kj:
        v = "y_%d" % j
        y[j] = v
        names.append(v)
        objs.append(float(inst.profits[j]))
        lbs.append(0.0)
        ubs.append(1.0)
        types.append("B")

    x = {}
    z = {}
    s = {}
    for k in range(inst.num_processors):
        jobs = list(inst.Jk[k])
        if not jobs:
            continue
        nodes = jobs + [start, n_end]
        for j in jobs:
            v = "x_%d_%d" % (j, k)
            x[(j, k)] = v
            names.append(v)
            objs.append(0.0)
            lbs.append(0.0)
            ubs.append(1.0)
            types.append("B")
        for i in nodes:
            for l in nodes:
                if i == l or l == start or i == n_end:
                    continue
                v = "z_%d_%d_%d" % (i, l, k)
                z[(i, l, k)] = v
                names.append(v)
                objs.append(0.0)
                lbs.append(0.0)
                ubs.append(1.0)
                types.append("B")
        for i in nodes:
            v = "s_%d_%d" % (i, k)
            s[(i, k)] = v
            names.append(v)
            objs.append(0.0)
            lbs.append(0.0)
            ubs.append(M)
            types.append("C")

    mdl.variables.add(obj=objs, lb=lbs, ub=ubs, types=types, names=names)

    rows, senses, rhs = [], [], []

    for k in range(inst.num_processors):
        jobs = list(inst.Jk[k])
        if not jobs:
            continue
        nodes = jobs + [start, n_end]

        out0 = [z[(start, l, k)] for l in nodes if (start, l, k) in z]
        rows.append([out0, [1.0] * len(out0)])
        senses.append("E")
        rhs.append(1.0)

        inN = [z[(i, n_end, k)] for i in nodes if (i, n_end, k) in z]
        rows.append([inN, [1.0] * len(inN)])
        senses.append("E")
        rhs.append(1.0)

        for j in jobs:
            ins = [z[(i, j, k)] for i in nodes if (i, j, k) in z]
            rows.append([ins + [x[(j, k)]], [1.0] * len(ins) + [-1.0]])
            senses.append("E")
            rhs.append(0.0)

            outs = [z[(j, l, k)] for l in nodes if (j, l, k) in z]
            rows.append([outs + [x[(j, k)]], [1.0] * len(outs) + [-1.0]])
            senses.append("E")
            rhs.append(0.0)

        for (i, l, kk) in list(z.keys()):
            if kk != k:
                continue
            # s_l >= s_i + t_il - M(1 - z)  ->  s_l - s_i - M*z >= t_il - M
            rows.append([[s[(l, k)], s[(i, k)], z[(i, l, k)]],
                         [1.0, -1.0, -M]])
            senses.append("G")
            rhs.append(float(T[i][l]) - M)

        rows.append([[s[(n_end, k)], s[(start, k)]], [1.0, -1.0]])
        senses.append("L")
        rhs.append(L)

    for j, ks in inst.Kj.items():
        for k in ks:
            if (j, k) not in x:
                continue
            rows.append([[y[j], x[(j, k)]], [1.0, -1.0]])
            senses.append("L")
            rhs.append(0.0)
        present = [k for k in ks if (j, k) in s]
        for a, b in zip(present, present[1:]):
            rows.append([[s[(j, a)], s[(j, b)]], [1.0, -1.0]])
            senses.append("E")
            rhs.append(0.0)

    mdl.linear_constraints.add(lin_expr=rows, senses=senses, rhs=rhs)
    return mdl, {"y": y, "x": x, "z": z, "s": s}


def solve(inst, time_limit=300.0, threads=6, verbose=False, mip_gap=0.0):
    mdl, idx = build_model(inst, verbose=verbose)
    mdl.parameters.timelimit.set(float(time_limit))
    mdl.parameters.threads.set(int(threads))
    mdl.parameters.mip.tolerances.mipgap.set(float(mip_gap))
    mdl.solve()

    st = mdl.solution.get_status_string()
    out = {
        "status": st,
        "n_vars": mdl.variables.get_num(),
        "n_cons": mdl.linear_constraints.get_num(),
        "objective": None,
        "best_bound": None,
        "mip_gap": None,
        "proven_optimal": False,
    }
    try:
        out["objective"] = mdl.solution.get_objective_value()
    except Exception:
        pass
    try:
        out["best_bound"] = mdl.solution.MIP.get_best_objective()
        out["mip_gap"] = mdl.solution.MIP.get_mip_relative_gap()
    except Exception:
        pass
    out["proven_optimal"] = "optimal" in st.lower() and "tolerance" not in st.lower()
    return out
