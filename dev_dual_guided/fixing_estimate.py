"""Estimate the reach of Lagrangian reduced-cost variable fixing.

The bound decomposes as

    L(mu) = sum_j max(0, r_j) + sum_k O_k(mu),     r_j = b_j - sum_{k in Kj} mu_{j,k}

Forcing a single y_j changes only that job's term, so at the same mu

    L(mu | y_j = 1) = L(mu) - max(0, r_j) + r_j = L(mu) + min(0, r_j)
    L(mu | y_j = 0) = L(mu) - max(0, r_j)

Writing G = L(mu) - z for the absolute gap against a known feasible incumbent z:

    r_j < -G   =>  every solution with j selected scores below z  =>  fix j OUT
    r_j >  G   =>  every solution without j scores below z        =>  fix j IN

Both tests are O(1) per job given r_j, which the dual already computes. This
script reports how many jobs each test would eliminate, without running any
search.

Note the degeneracy connection: under the fair split mu_{j,k} = b_j/|Kj| we have
r_j == 0 for every job, so neither test can ever fire. The same identity that
disables dual-guided steering also disables fixing.
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

BENCH = os.path.join(ROOT, "benchmarks", "ops_raw", "OPS-Benchmark-master", "input")
MASTER = os.path.join(ROOT, "results", "adaptive_master.csv")
OUT_DIR = os.path.join(ROOT, "results", "dual_guided_dev")


def _load_incumbents():
    out = {}
    with open(MASTER, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            v = (r.get("alns_obj") or "").strip()
            if v:
                try:
                    out[r["instance"].strip()] = float(v)
                except ValueError:
                    pass
    return out


def analyse(label, z, args):
    fam = label.split("_")[0]
    inst = OPSInstance.from_instance_file(
        os.path.join(BENCH, fam, "instances", label + ".txt")
    )
    lag = lagrangian_bound(
        inst, max_iter=args.lag_max_iter, lower_bound=z, max_time=args.lag_max_time
    )
    mu = lag["best_mu"]
    Lmu = float(lag["upper_bound"])
    # L(mu) is a valid upper bound, so G >= 0 mathematically; floating point can
    # still yield a tiny negative. Clamp it, otherwise the fix-out and fix-in
    # tests overlap and every job with r_j ~ 0 is counted twice.
    G = max(0.0, Lmu - z)

    r = {}
    for j, ks in inst.Kj.items():
        r[j] = float(inst.profits[j]) - sum(mu.get((j, k), 0.0) for k in ks)

    fix_out = [j for j, rj in r.items() if rj < -G]
    fix_in = [j for j, rj in r.items() if rj > G]
    n = len(r)
    nz = sum(1 for rj in r.values() if abs(rj) > 1e-12)

    # Sensitivity: the final incumbent gives the smallest G and therefore the most
    # optimistic reach. In practice fixing runs mid-search against a weaker z, so
    # report the reach at incumbents degraded by 1% and 5%.
    sens = {}
    for frac, name in ((0.99, "at_z99"), (0.95, "at_z95")):
        zf = z * frac
        Gf = max(0.0, Lmu - zf)
        sens[name] = sum(1 for rj in r.values() if rj < -Gf or rj > Gf)

    return {
        "instance": label,
        "family": fam,
        "jobs": n,
        "incumbent_z": z,
        "L_mu": round(Lmu, 4),
        "abs_gap_G": round(G, 4),
        "nonzero_rj": nz,
        "max_abs_rj": round(max((abs(v) for v in r.values()), default=0.0), 4),
        "fix_out": len(fix_out),
        "fix_in": len(fix_in),
        "fixable_total": len(fix_out) + len(fix_in),
        "fixable_pct": round(100.0 * (len(fix_out) + len(fix_in)) / max(1, n), 2),
        "fixable_at_z99": sens["at_z99"],
        "fixable_at_z99_pct": round(100.0 * sens["at_z99"] / max(1, n), 2),
        "fixable_at_z95": sens["at_z95"],
        "fixable_at_z95_pct": round(100.0 * sens["at_z95"] / max(1, n), 2),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subset", required=True)
    p.add_argument("--tag", default="fixing")
    p.add_argument("--lag-max-iter", type=int, default=200)
    p.add_argument("--lag-max-time", type=float, default=30.0)
    args = p.parse_args()

    with open(args.subset, encoding="utf-8") as f:
        labels = [
            r["instance"].strip()
            for r in csv.DictReader(f)
            if r.get("instance", "").strip()
        ]

    inc = _load_incumbents()
    os.makedirs(OUT_DIR, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M")
    out = os.path.join(OUT_DIR, "fixing_estimate_" + args.tag + "_" + ts + ".csv")

    rows = []
    for i, lb in enumerate(labels, 1):
        if lb not in inc:
            print("[%d/%d] %s  SKIP (no incumbent)" % (i, len(labels), lb), flush=True)
            continue
        print("[%d/%d] %s" % (i, len(labels), lb), flush=True)
        try:
            rr = analyse(lb, inc[lb], args)
        except Exception as e:
            print("    FAILED: %s: %s" % (type(e).__name__, e), flush=True)
            continue
        rows.append(rr)
        print(
            "    G=%.2f  nonzero r_j=%d/%d  fix_out=%d fix_in=%d  -> %.1f%% of jobs"
            % (
                rr["abs_gap_G"],
                rr["nonzero_rj"],
                rr["jobs"],
                rr["fix_out"],
                rr["fix_in"],
                rr["fixable_pct"],
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

    pct = [r["fixable_pct"] for r in rows]
    tot_jobs = sum(r["jobs"] for r in rows)
    tot_fix = sum(r["fixable_total"] for r in rows)
    print("\n" + "=" * 66)
    print("REDUCED-COST FIXING ESTIMATE  (%d instances)  -> %s" % (len(rows), out))
    print("=" * 66)
    print("  jobs fixable overall      : %d / %d  (%.1f%%)"
          % (tot_fix, tot_jobs, 100.0 * tot_fix / max(1, tot_jobs)))
    print("  mean / median per-instance: %.1f%% / %.1f%%" % (st.mean(pct), st.median(pct)))
    print("  instances with >0 fixable : %d / %d"
          % (sum(1 for x in pct if x > 0), len(pct)))
    print("  best instance             : %.1f%%" % max(pct))
    p99 = [r["fixable_at_z99_pct"] for r in rows]
    p95 = [r["fixable_at_z95_pct"] for r in rows]
    print()
    print("  Sensitivity to incumbent quality (fixing runs mid-search, not post-hoc):")
    print("    mean reach at final z   : %.1f%%" % st.mean(pct))
    print("    mean reach at 0.99 x z  : %.1f%%" % st.mean(p99))
    print("    mean reach at 0.95 x z  : %.1f%%" % st.mean(p95))


if __name__ == "__main__":
    main()
