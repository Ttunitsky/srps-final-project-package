"""Solve SRPS instances exactly with CPLEX and decompose the certified gap.

The ceiling argument bounds any strengthening by (ub - z)/ub, because the optimum
P* is only known to lie in [z, ub]. Pinning P* down by exact solution converts
that bound into a measurement: how much of the reported gap is bound looseness
(ub - P*), which a tighter dual could recover, versus primal suboptimality
(P* - z), which it could not.

Run under Python 3.8-3.10 with the CPLEX API on PYTHONPATH.
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics as st
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from adapters.ops_adapter import OPSInstance
from dev_exact.cplex_srps import solve

BENCH = os.path.join(ROOT, "benchmarks", "ops_raw", "OPS-Benchmark-master", "input")
MASTER = os.path.join(ROOT, "results", "adaptive_master.csv")
OUT_DIR = os.path.join(ROOT, "results", "exact")


def load_master():
    out = {}
    with open(MASTER, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            def g(key):
                v = (r.get(key) or "").strip()
                try:
                    return float(v)
                except ValueError:
                    return None
            out[r["instance"].strip()] = {
                "z": g("alns_obj"),
                "ub": g("best_ub"),
                "gap": g("final_cert_gap_pct"),
                "n": g("n"),
            }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subset", default=None)
    p.add_argument("--max-n", type=int, default=50)
    p.add_argument("--limit", type=int, default=8)
    p.add_argument("--time-limit", type=float, default=300.0)
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--tag", default="exact")
    args = p.parse_args()

    master = load_master()

    if args.subset:
        with open(args.subset, encoding="utf-8") as f:
            labels = [r["instance"].strip() for r in csv.DictReader(f)
                      if r.get("instance", "").strip()]
    else:
        cand = [(lb, m) for lb, m in master.items()
                if m["n"] and m["n"] <= args.max_n and m["z"] and m["ub"]]
        cand.sort(key=lambda kv: kv[1]["n"])
        labels = [lb for lb, _ in cand[: args.limit]]

    os.makedirs(OUT_DIR, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M")
    out_csv = os.path.join(OUT_DIR, "exact_" + args.tag + "_" + ts + ".csv")

    rows = []
    for i, lb in enumerate(labels, 1):
        fam = lb.split("_")[0]
        ipath = os.path.join(BENCH, fam, "instances", lb + ".txt")
        if not os.path.exists(ipath):
            print("[%d/%d] %s  SKIP (no instance file)" % (i, len(labels), lb), flush=True)
            continue
        inst = OPSInstance.from_instance_file(ipath)
        m = master.get(lb, {})
        t0 = time.perf_counter()
        r = solve(inst, time_limit=args.time_limit, threads=args.threads)
        wall = time.perf_counter() - t0

        z = m.get("z")
        ub = m.get("ub")
        opt = r["objective"]
        row = {
            "instance": lb, "family": fam, "n": int(m.get("n") or 0),
            "vars": r["n_vars"], "cons": r["n_cons"],
            "cplex_obj": opt, "cplex_bound": r["best_bound"],
            "cplex_gap_pct": (100 * r["mip_gap"]) if r["mip_gap"] is not None else "",
            "proven_optimal": int(r["proven_optimal"]),
            "status": r["status"], "solve_s": round(wall, 1),
            "alns_z": z, "lag_ub": ub,
        }
        # The decomposition is only meaningful when CPLEX proved optimality. On a
        # timed-out run `opt` is merely CPLEX's incumbent, which is frequently
        # worse than the ALNS's, and treating it as P* produces nonsense such as
        # a negative primal suboptimality. Leave those cells empty.
        if r["proven_optimal"] and opt is not None and z is not None and ub:
            row["bound_looseness_pct"] = round(100.0 * (ub - opt) / ub, 4)
            row["primal_subopt_pct"] = round(100.0 * (opt - z) / ub, 4)
            row["cert_gap_pct"] = round(100.0 * (ub - z) / ub, 4)
        else:
            row["bound_looseness_pct"] = ""
            row["primal_subopt_pct"] = ""
            row["cert_gap_pct"] = (
                round(100.0 * (ub - z) / ub, 4) if (z is not None and ub) else ""
            )
        rows.append(row)
        print(
            "[%d/%d] %-24s n=%-3s vars=%5d  P*=%s (%s)  z=%s ub=%s  "
            "loose=%s%%  subopt=%s%%  %.0fs"
            % (i, len(labels), lb[:24], row["n"], row["vars"],
               ("%.0f" % opt) if opt is not None else "NA",
               "proven" if r["proven_optimal"] else r["status"][:18],
               ("%.0f" % z) if z else "NA", ("%.0f" % ub) if ub else "NA",
               row.get("bound_looseness_pct", "NA"),
               row.get("primal_subopt_pct", "NA"), wall),
            flush=True,
        )

    if not rows:
        print("no rows")
        return
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    proven = [r for r in rows if r["proven_optimal"] and "bound_looseness_pct" in r]
    print("\n" + "=" * 72)
    print("EXACT DECOMPOSITION (%d solved, %d proven optimal)" % (len(rows), len(proven)))
    print("=" * 72)
    if proven:
        bl = [r["bound_looseness_pct"] for r in proven]
        ps = [r["primal_subopt_pct"] for r in proven]
        cg = [r["cert_gap_pct"] for r in proven]
        print("  mean certified gap        : %.4f%%" % st.mean(cg))
        print("    of which bound looseness: %.4f%%  (recoverable by a tighter dual)"
              % st.mean(bl))
        print("    of which primal subopt  : %.4f%%  (not recoverable by the dual)"
              % st.mean(ps))
        atopt = sum(1 for r in proven if abs(r["primal_subopt_pct"]) < 1e-9)
        print("  ALNS already optimal on   : %d / %d" % (atopt, len(proven)))
    print("  -> %s" % out_csv)


if __name__ == "__main__":
    main()
