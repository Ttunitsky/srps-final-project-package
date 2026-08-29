"""S0b gate: is the relaxed selection actually unschedulable, or just badly ordered?

S0 found the min-time per-processor ordering yields a cyclic constraint graph on
7/7 instances. That is NOT proof the selection is infeasible: the relaxation fixes
which jobs each processor takes, not the order. Any single global total order
induces per-processor sequences whose arcs all point forward, hence an acyclic
graph. So acyclicity is always achievable and the real question is whether some
ordering also satisfies makespan <= L.

This matters for soundness. A cut asserting "this selection is infeasible" is
valid only if NO ordering works. If a feasible ordering exists and we cut anyway,
we cut off a feasible point and corrupt the bound.

Search is a time-boxed random-restart hill climb over total orders. Semantics of
the result:
  FEASIBLE FOUND  -> selection is schedulable; a no-good cut on it would be UNSOUND.
  NONE FOUND      -> inconclusive (heuristic failure, not a proof of infeasibility),
                     but reported alongside the makespan lower bound so the gap is visible.
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from adapters.ops_adapter import OPSInstance
from core.ops_bounds import lagrangian_bound
from dev_bpc_replica.s0_instrument import replay_relaxation, optimal_order

BENCH = os.path.join(ROOT, "benchmarks", "ops_raw", "OPS-Benchmark-master", "input")
OUT_DIR = os.path.join(ROOT, "results", "bpc_replica_dev")


def _family(label):
    return label.split("_")[0]


def makespan_for_order(inst, job_procs, proc_ids, order):
    """Makespan when every processor visits its jobs in the given global order.

    All arcs point forward in the order, so the graph is acyclic by construction
    and earliest start times follow from a single forward sweep. Cost is linear
    in the number of arcs, not in jobs x processors.
    """
    start, end, T = inst.start, inst.end, inst.T
    est = {}
    prev = dict.fromkeys(proc_ids, None)

    for j in order:
        t = 0.0
        Tj = T[j]
        for k in job_procs[j]:
            p = prev[k]
            cand = float(T[start][j]) if p is None else est[p] + float(T[p][j])
            if cand > t:
                t = cand
        est[j] = t
        for k in job_procs[j]:
            prev[k] = j

    ms = 0.0
    for p in prev.values():
        if p is None:
            continue
        cand = est[p] + float(T[p][end])
        if cand > ms:
            ms = cand
    return ms


def est_keys(proc_jobs):
    s = set()
    for jobs in proc_jobs.values():
        s |= set(jobs)
    return s


def search_schedule(inst, proc_jobs, budget_s, seed=0):
    """Random-restart hill climb over global total orders, minimising makespan."""
    jobs = sorted(est_keys(proc_jobs))
    n = len(jobs)
    if n == 0:
        return 0.0, 0, 0

    job_procs = {j: [] for j in jobs}
    for k, s in proc_jobs.items():
        for j in s:
            job_procs[j].append(k)
    proc_ids = list(proc_jobs.keys())

    rng = random.Random(seed)
    L = float(inst.L)
    t0 = time.perf_counter()

    best = float("inf")
    restarts = 0
    evals = 0

    # Seed order: by transition time from the depot (cheap, usually decent).
    seeds = [sorted(jobs, key=lambda j: float(inst.T[inst.start][j]))]

    while time.perf_counter() - t0 < budget_s:
        if seeds:
            order = seeds.pop(0)
        else:
            order = jobs[:]
            rng.shuffle(order)
        restarts += 1

        cur = makespan_for_order(inst, job_procs, proc_ids, order)
        evals += 1

        improved = True
        while improved and (time.perf_counter() - t0 < budget_s):
            improved = False
            for i in range(n - 1):
                if time.perf_counter() - t0 >= budget_s:
                    break
                order[i], order[i + 1] = order[i + 1], order[i]
                cand = makespan_for_order(inst, job_procs, proc_ids, order)
                evals += 1
                if cand < cur - 1e-9:
                    cur = cand
                    improved = True
                else:
                    order[i], order[i + 1] = order[i + 1], order[i]

        if cur < best:
            best = cur
        if best <= L:
            break  # feasible schedule found; no need to keep searching

    return best, restarts, evals


def analyse(label, args):
    fam = _family(label)
    ipath = os.path.join(BENCH, fam, "instances", label + ".txt")
    inst = OPSInstance.from_instance_file(ipath)

    lag = lagrangian_bound(
        inst, max_iter=args.lag_max_iter, lower_bound=0.0, max_time=args.lag_max_time
    )
    mu = lag["best_mu"]
    y, sel = replay_relaxation(inst, mu)

    proc_jobs = {k: set(s) for k, s in sel.items() if s}

    # Lower bound on makespan: no ordering beats each processor's own optimal
    # Hamiltonian path, so the max over processors bounds the joint makespan.
    lb = 0.0
    for k, s in proc_jobs.items():
        _, length, _ = optimal_order(inst, s)
        if length != float("inf") and length > lb:
            lb = length

    best_ms, restarts, evals = search_schedule(inst, proc_jobs, args.budget_s, seed=args.seed)

    L = float(inst.L)
    feasible = best_ms <= L
    return {
        "instance": label,
        "family": fam,
        "L": L,
        "claimed_jobs": len(est_keys(proc_jobs)),
        "active_processors": len(proc_jobs),
        "makespan_lb": round(lb, 3),
        "best_makespan_found": round(best_ms, 3),
        "excess_over_L": round(best_ms - L, 3),
        "excess_pct_L": round(100.0 * (best_ms - L) / L, 3) if L else "NA",
        "feasible_schedule_found": int(feasible),
        "restarts": restarts,
        "evals": evals,
        "verdict": "SCHEDULABLE - no-good cut would be UNSOUND"
        if feasible
        else "none found (inconclusive, not a proof)",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subset", required=True)
    p.add_argument("--tag", default="s0b")
    p.add_argument("--lag-max-iter", type=int, default=200)
    p.add_argument("--lag-max-time", type=float, default=30.0)
    p.add_argument("--budget-s", type=float, default=60.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    with open(args.subset, encoding="utf-8") as f:
        labels = [
            r["instance"].strip()
            for r in csv.DictReader(f)
            if r.get("instance", "").strip()
        ]

    os.makedirs(OUT_DIR, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M")
    out = os.path.join(OUT_DIR, "bpc_s0b_" + args.tag + "_" + ts + ".csv")

    rows = []
    for i, lb in enumerate(labels, 1):
        print("[" + str(i) + "/" + str(len(labels)) + "] " + lb, flush=True)
        try:
            r = analyse(lb, args)
        except Exception as e:
            print("    FAILED: " + type(e).__name__ + ": " + str(e), flush=True)
            continue
        rows.append(r)
        print(
            "    lb={0}  best_makespan={1}  L={2}  excess={3} ({4}%)  -> {5}".format(
                r["makespan_lb"],
                r["best_makespan_found"],
                r["L"],
                r["excess_over_L"],
                r["excess_pct_L"],
                r["verdict"],
            ),
            flush=True,
        )

    if not rows:
        print("no rows produced")
        return

    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n_feas = sum(r["feasible_schedule_found"] for r in rows)
    print("\n" + "=" * 66)
    print("S0b SCHEDULABILITY GATE  ({0} instances)  -> {1}".format(len(rows), out))
    print("=" * 66)
    print("  feasible schedule FOUND : {0}/{1}".format(n_feas, len(rows)))
    print("  none found              : {0}/{1}".format(len(rows) - n_feas, len(rows)))
    if n_feas:
        print(
            "  => On those {0}, the relaxed selection IS schedulable. A no-good cut".format(n_feas)
        )
        print("     asserting infeasibility would cut off a feasible point: UNSOUND.")
        print("     S1 as conceived does not survive this.")
    else:
        print("  => No ordering found on any instance. Not a proof of infeasibility,")
        print("     but consistent with genuinely unschedulable selections. S1 would")
        print("     need an exact feasibility oracle before any cut is emitted.")


if __name__ == "__main__":
    main()
