"""Where does lagrangian_bound actually spend its time?

Before attempting to parallelise the per-processor subproblem loop, measure what
fraction of a subgradient iteration that loop accounts for. If the orienteering
DPs dominate, parallelism has something to work with; if the surrounding
bookkeeping dominates, it does not.
"""
from __future__ import annotations

import argparse
import cProfile
import csv
import os
import pstats
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from adapters.ops_adapter import OPSInstance
from core.ops_bounds import lagrangian_bound, orienteering_dp_with_selection

BENCH = os.path.join(ROOT, "benchmarks", "ops_raw", "OPS-Benchmark-master", "input")


def subproblem_shape(inst, mu=None):
    """Report the per-processor job counts that drive DP cost (2^m)."""
    sizes = []
    for k in range(inst.num_processors):
        feasible = [j for j in inst.Jk[k] if inst.profits[j] > 0.0]
        sizes.append(len(feasible))
    return sizes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subset", required=True)
    p.add_argument("--lag-max-iter", type=int, default=60)
    p.add_argument("--lag-max-time", type=float, default=30.0)
    p.add_argument("--limit", type=int, default=3)
    args = p.parse_args()

    with open(args.subset, encoding="utf-8") as f:
        labels = [
            r["instance"].strip()
            for r in csv.DictReader(f)
            if r.get("instance", "").strip()
        ][: args.limit]

    for lb in labels:
        fam = lb.split("_")[0]
        inst = OPSInstance.from_instance_file(
            os.path.join(BENCH, fam, "instances", lb + ".txt")
        )
        sizes = subproblem_shape(inst)
        nz = [s for s in sizes if s > 0]
        print("=" * 70)
        print(lb)
        print(
            "  processors=%d  with jobs=%d  max jobs/proc=%d  mean=%.1f  sum 2^m=%.3g"
            % (
                inst.num_processors,
                len(nz),
                max(sizes) if sizes else 0,
                (sum(nz) / len(nz)) if nz else 0.0,
                sum(2 ** s for s in nz) if nz else 0,
            )
        )

        pr = cProfile.Profile()
        t0 = time.perf_counter()
        pr.enable()
        lagrangian_bound(
            inst,
            max_iter=args.lag_max_iter,
            lower_bound=0.0,
            max_time=args.lag_max_time,
        )
        pr.disable()
        wall = time.perf_counter() - t0

        st = pstats.Stats(pr)
        total_tt = 0.0
        dp_tt = 0.0
        for (fn, ln, name), (cc, nc, tt, ct, cal) in st.stats.items():
            total_tt += tt
            if "orienteering_dp" in name:
                dp_tt += tt

        print("  wall=%.2fs  profiled tottime=%.2fs" % (wall, total_tt))
        print(
            "  orienteering DP tottime=%.2fs  -> %.1f%% of profiled time"
            % (dp_tt, 100.0 * dp_tt / max(total_tt, 1e-9))
        )
        print("  top functions by tottime:")
        st.sort_stats("tottime")
        rows = 0
        for (fn, ln, name), (cc, nc, tt, ct, cal) in sorted(
            st.stats.items(), key=lambda kv: -kv[1][2]
        ):
            print(
                "    %-34s tottime=%6.2fs  calls=%d"
                % (name[:34], tt, nc)
            )
            rows += 1
            if rows >= 6:
                break


if __name__ == "__main__":
    main()
