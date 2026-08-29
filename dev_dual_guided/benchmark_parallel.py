"""A/B the sequential and parallel Lagrangian bound.

Two things must hold for the parallel path to be usable:
  1. Correctness -- the bound must be identical, not merely close. The
     subproblems are deterministic and reassembled in processor order, so any
     discrepancy indicates a real bug rather than floating-point drift.
  2. Speedup -- wall-clock must actually improve at a realistic worker count.
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
from core.ops_bounds import lagrangian_bound
from dev_dual_guided.parallel_bound import lagrangian_bound_parallel

BENCH = os.path.join(ROOT, "benchmarks", "ops_raw", "OPS-Benchmark-master", "input")
OUT_DIR = os.path.join(ROOT, "results", "dual_guided_dev")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subset", required=True)
    p.add_argument("--tag", default="par")
    p.add_argument("--lag-max-iter", type=int, default=60)
    p.add_argument("--workers", type=int, nargs="+", default=[2, 4, 8])
    args = p.parse_args()

    with open(args.subset, encoding="utf-8") as f:
        labels = [
            r["instance"].strip()
            for r in csv.DictReader(f)
            if r.get("instance", "").strip()
        ]

    os.makedirs(OUT_DIR, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M")
    out = os.path.join(OUT_DIR, "parallel_bench_" + args.tag + "_" + ts + ".csv")

    rows = []
    for i, lb in enumerate(labels, 1):
        fam = lb.split("_")[0]
        ipath = os.path.join(BENCH, fam, "instances", lb + ".txt")
        inst = OPSInstance.from_instance_file(ipath)
        print("[%d/%d] %s" % (i, len(labels), lb), flush=True)

        # no max_time: fix the iteration count so the comparison is like-for-like
        t0 = time.perf_counter()
        seq = lagrangian_bound(
            inst, max_iter=args.lag_max_iter, lower_bound=0.0, max_time=None
        )
        t_seq = time.perf_counter() - t0
        print("    sequential      %7.2fs  ub=%.6f  iters=%d"
              % (t_seq, seq["upper_bound"], seq["iterations"]), flush=True)

        row = {
            "instance": lb,
            "family": fam,
            "processors": inst.num_processors,
            "iters": seq["iterations"],
            "seq_s": round(t_seq, 3),
            "seq_ub": seq["upper_bound"],
        }

        for w in args.workers:
            t0 = time.perf_counter()
            par = lagrangian_bound_parallel(
                inst,
                ipath,
                max_iter=args.lag_max_iter,
                lower_bound=0.0,
                max_time=None,
                workers=w,
            )
            t_par = time.perf_counter() - t0
            identical = par["upper_bound"] == seq["upper_bound"]
            speedup = t_seq / t_par if t_par > 0 else 0.0
            row["w%d_s" % w] = round(t_par, 3)
            row["w%d_speedup" % w] = round(speedup, 3)
            row["w%d_identical" % w] = int(identical)
            row["w%d_ub" % w] = par["upper_bound"]
            print(
                "    parallel w=%-2d   %7.2fs  speedup=%.2fx  ub_identical=%s"
                % (w, t_par, speedup, "YES" if identical else "NO <-- BUG"),
                flush=True,
            )
        rows.append(row)

    if not rows:
        print("no rows")
        return

    with open(out, "w", newline="", encoding="utf-8") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(rows)

    print("\n" + "=" * 66)
    print("PARALLEL DUAL BENCHMARK (%d instances) -> %s" % (len(rows), out))
    print("=" * 66)
    all_ok = True
    for w in args.workers:
        sp = [r["w%d_speedup" % w] for r in rows]
        ident = all(r["w%d_identical" % w] for r in rows)
        all_ok = all_ok and ident
        print("  workers=%-2d  mean speedup %.2fx  median %.2fx  bounds identical: %s"
              % (w, st.mean(sp), st.median(sp), "YES" if ident else "NO"))
    print("  total sequential time: %.1fs" % sum(r["seq_s"] for r in rows))
    best = max(args.workers, key=lambda w: st.mean([r["w%d_speedup" % w] for r in rows]))
    print("  best worker count: %d" % best)
    if not all_ok:
        print("  WARNING: at least one bound differed - parallel path is incorrect")


if __name__ == "__main__":
    main()
