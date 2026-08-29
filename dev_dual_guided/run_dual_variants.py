"""Benchmark mu-initialisation and stabilisation variants of the dual.

Reports, per variant: the bound after a fixed iteration budget, and the number
of iterations needed to come within a tolerance of the best bound any variant
achieves on that instance. The second is the operative metric -- stabilisation
targets convergence rate, and a faster dual is a runtime effect, which is the
one axis the ceiling argument does not bound.
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics as st
import sys
import time
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from adapters.ops_adapter import OPSInstance, OPSSolution
from dev_dual_guided.dual_variants import (
    lagrangian_variant,
    mu_fair,
    mu_incumbent,
    step_polyak,
    step_trust,
)

BENCH = os.path.join(ROOT, "benchmarks", "ops_raw", "OPS-Benchmark-master", "input")
OUT_DIR = os.path.join(ROOT, "results", "dual_guided_dev")


def greedy_incumbent(inst):
    """Cheap primal solution: insert by descending profit. This is what the
    solver itself has available at t=0, so seeding from it is realistic."""
    sol = OPSSolution(inst)
    for j in sorted(sol.get_unserved(), key=lambda x: inst.profits[x], reverse=True):
        c, p = sol.best_insertion_cost(j)
        if c < float("inf"):
            sol.insert(j, p)
    return set(sol.selected), sol.objective()


VARIANTS = [
    # (name, mu_init_fn, mu_kw_needs_incumbent, step_fn, step_kw, smoothing)
    ("baseline",          mu_fair,      False, step_polyak, {},              0.0),
    ("seed_eps05",        mu_incumbent, True,  step_polyak, {},              0.0),
    ("seed_eps10",        mu_incumbent, True,  step_polyak, {},              0.0),
    ("seed_eps25",        mu_incumbent, True,  step_polyak, {},              0.0),
    ("trust05",           mu_fair,      False, step_trust,  {"trust": 0.5},  0.0),
    ("trust10",           mu_fair,      False, step_trust,  {"trust": 1.0},  0.0),
    ("smooth03",          mu_fair,      False, step_polyak, {},              0.3),
    ("smooth06",          mu_fair,      False, step_polyak, {},              0.6),
    ("seed10_trust05",    mu_incumbent, True,  step_trust,  {"trust": 0.5},  0.0),
    ("seed10_smooth03",   mu_incumbent, True,  step_polyak, {},              0.3),
]
EPS = {"seed_eps05": 0.05, "seed_eps10": 0.10, "seed_eps25": 0.25,
       "seed10_trust05": 0.10, "seed10_smooth03": 0.10}


def run_instance(payload):
    label, iters = payload
    fam = label.split("_")[0]
    inst = OPSInstance.from_instance_file(
        os.path.join(BENCH, fam, "instances", label + ".txt")
    )
    sel, z = greedy_incumbent(inst)

    out = {}
    for name, mu_fn, needs_inc, step_fn, step_kw, smooth in VARIANTS:
        kw = {}
        if needs_inc:
            kw = {"selected": sel, "eps": EPS.get(name, 0.10)}
        t0 = time.perf_counter()
        r = lagrangian_variant(
            inst,
            max_iter=iters,
            lower_bound=z,
            max_time=None,
            mu_init_fn=mu_fn,
            mu_init_kw=kw,
            step_fn=step_fn,
            step_kw=step_kw,
            smoothing=smooth,
        )
        out[name] = {
            "ub": r["upper_bound"],
            "hist": r["bound_history"],
            "s": time.perf_counter() - t0,
        }
    return label, inst.num_processors, z, out


def iters_to_within(hist, target, tol):
    """First iteration whose running-best bound is within tol of target."""
    best = float("inf")
    for i, v in enumerate(hist, 1):
        best = min(best, v)
        if best <= target * (1.0 + tol):
            return i
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subset", required=True)
    p.add_argument("--tag", default="variants")
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--tol", type=float, default=0.001)
    args = p.parse_args()

    with open(args.subset, encoding="utf-8") as f:
        labels = [
            r["instance"].strip()
            for r in csv.DictReader(f)
            if r.get("instance", "").strip()
        ]

    os.makedirs(OUT_DIR, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M")
    out_csv = os.path.join(OUT_DIR, "dual_variants_" + args.tag + "_" + ts + ".csv")

    tasks = [(lb, args.iters) for lb in labels]
    results = []
    with Pool(processes=args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(run_instance, tasks), 1):
            label, nproc, z, out = res
            results.append(res)
            best = min(v["ub"] for v in out.values())
            base = out["baseline"]["ub"]
            print(
                "[%d/%d] %-24s base_ub=%.4f best_ub=%.4f (%s)"
                % (i, len(tasks), label[:24], base, best,
                   "baseline" if best == base else "variant wins"),
                flush=True,
            )

    rows = []
    for label, nproc, z, out in results:
        target = min(v["ub"] for v in out.values())
        row = {"instance": label, "processors": nproc, "incumbent_z": z,
               "best_ub_any": round(target, 6)}
        for name, _, _, _, _, _ in VARIANTS:
            v = out[name]
            row[name + "_ub"] = round(v["ub"], 6)
            row[name + "_vs_base"] = round(v["ub"] - out["baseline"]["ub"], 6)
            it = iters_to_within(v["hist"], target, args.tol)
            row[name + "_iters_to_tol"] = it if it is not None else ""
            row[name + "_s"] = round(v["s"], 3)
        rows.append(row)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\n" + "=" * 78)
    print("DUAL VARIANTS (%d instances, %d iters, tol=%.3f%%)"
          % (len(rows), args.iters, 100 * args.tol))
    print("=" * 78)
    print("%-18s %12s %12s %14s %10s"
          % ("variant", "mean dUB", "wins/ties", "mean iters@tol", "mean s"))
    for name, _, _, _, _, _ in VARIANTS:
        d = [r[name + "_vs_base"] for r in rows]
        wins = sum(1 for x in d if x < -1e-9)
        ties = sum(1 for x in d if abs(x) <= 1e-9)
        its = [r[name + "_iters_to_tol"] for r in rows if r[name + "_iters_to_tol"] != ""]
        secs = [r[name + "_s"] for r in rows]
        print("%-18s %12.5f %6d/%-5d %14s %10.2f"
              % (name, st.mean(d), wins, ties,
                 ("%.1f" % st.mean(its)) if its else "n/a",
                 st.mean(secs)))
    print("\n  negative dUB = tighter (better) bound than baseline")
    print("  -> %s" % out_csv)


if __name__ == "__main__":
    main()
