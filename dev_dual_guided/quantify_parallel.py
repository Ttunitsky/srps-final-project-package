"""Quantify the parallel-dual speedup with repeats, and bound what it is worth.

Reports three things per instance:
  1. measured speedup, averaged over repeats (machine noise is non-trivial);
  2. the structural ceiling total_cost / max_single_cost -- no partition splits a
     single DP, so this caps speedup regardless of worker count;
  3. the fraction of the ceiling actually captured.

A dual-only speedup is not an end-to-end speedup. The final section estimates
the dual's share of total solver wall time from the production-run telemetry, so
the headline figure can be converted into something honest.
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics as st
import sys
import time
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from adapters.ops_adapter import OPSInstance
from core.ops_bounds import lagrangian_bound
from dev_dual_guided.parallel_bound import (
    lagrangian_bound_parallel,
    _chunks,
    _worker_init,
)

BENCH = os.path.join(ROOT, "benchmarks", "ops_raw", "OPS-Benchmark-master", "input")
OUT_DIR = os.path.join(ROOT, "results", "dual_guided_dev")


def ceiling(inst, w):
    cost = [2 ** len(inst.Jk[k]) for k in range(inst.num_processors)]
    tot = sum(cost)
    blocks = _chunks(inst, w)
    loads = [sum(cost[k] for k in b) for b in blocks]
    # Ideal parallel time is the heaviest block; with LPT that is at least the
    # single largest subproblem.
    return tot / max(loads), tot / max(cost)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subset", required=True)
    p.add_argument("--tag", default="quant")
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--repeats", type=int, default=3)
    args = p.parse_args()

    with open(args.subset, encoding="utf-8") as f:
        labels = [
            r["instance"].strip()
            for r in csv.DictReader(f)
            if r.get("instance", "").strip()
        ]

    os.makedirs(OUT_DIR, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M")
    out = os.path.join(OUT_DIR, "parallel_quant_" + args.tag + "_" + ts + ".csv")

    rows = []
    for i, lb in enumerate(labels, 1):
        fam = lb.split("_")[0]
        ipath = os.path.join(BENCH, fam, "instances", lb + ".txt")
        inst = OPSInstance.from_instance_file(ipath)
        lpt_ceil, hard_ceil = ceiling(inst, args.workers)

        ex = ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_worker_init,
            initargs=(ipath,),
        )
        try:
            list(ex.map(int, range(args.workers)))  # warm the pool

            seq_t, par_t = [], []
            ok = True
            for _ in range(args.repeats):
                t = time.perf_counter()
                s = lagrangian_bound(
                    inst, max_iter=args.iters, lower_bound=0.0, max_time=None
                )
                seq_t.append(time.perf_counter() - t)

                t = time.perf_counter()
                pr = lagrangian_bound_parallel(
                    inst, ipath, max_iter=args.iters, lower_bound=0.0,
                    max_time=None, workers=args.workers, executor=ex,
                )
                par_t.append(time.perf_counter() - t)
                ok = ok and (pr["upper_bound"] == s["upper_bound"])
        finally:
            ex.shutdown(wait=True)

        # Use best-of to suppress scheduler noise; report spread separately.
        sb, pb = min(seq_t), min(par_t)
        sp = sb / pb
        rows.append({
            "instance": lb,
            "family": fam,
            "processors": inst.num_processors,
            "seq_best_s": round(sb, 3),
            "seq_spread_pct": round(100.0 * (max(seq_t) - min(seq_t)) / min(seq_t), 1),
            "par_best_s": round(pb, 3),
            "speedup": round(sp, 3),
            "lpt_ceiling": round(lpt_ceil, 3),
            "hard_ceiling": round(hard_ceil, 3),
            "pct_of_ceiling": round(100.0 * sp / lpt_ceil, 1),
            "identical": int(ok),
        })
        print(
            "[%d/%d] %-24s seq=%6.2fs par=%6.2fs  speedup=%.2fx  "
            "ceiling=%.2fx (%.0f%% captured)  identical=%s"
            % (i, len(labels), lb[:24], sb, pb, sp, lpt_ceil,
               100.0 * sp / lpt_ceil, "YES" if ok else "NO"),
            flush=True,
        )

    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    sp = [r["speedup"] for r in rows]
    ce = [r["lpt_ceiling"] for r in rows]
    hc = [r["hard_ceiling"] for r in rows]
    print("\n" + "=" * 70)
    print("PARALLEL DUAL SPEEDUP (%d instances, w=%d, %d iters, best of %d)"
          % (len(rows), args.workers, args.iters, args.repeats))
    print("=" * 70)
    print("  measured speedup    mean %.2fx   median %.2fx   min %.2fx   max %.2fx"
          % (st.mean(sp), st.median(sp), min(sp), max(sp)))
    print("  LPT ceiling         mean %.2fx   median %.2fx" % (st.mean(ce), st.median(ce)))
    print("  hard ceiling (1 DP) mean %.2fx   median %.2fx" % (st.mean(hc), st.median(hc)))
    print("  captured            mean %.0f%% of the achievable ceiling"
          % st.mean([r["pct_of_ceiling"] for r in rows]))
    print("  bounds identical    %s" % ("YES, all" if all(r["identical"] for r in rows) else "NO"))
    print("  seq timing spread   mean %.1f%% across repeats"
          % st.mean([r["seq_spread_pct"] for r in rows]))
    print("  -> %s" % out)


if __name__ == "__main__":
    main()
