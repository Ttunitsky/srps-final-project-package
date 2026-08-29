"""Does the trust-region / seeding advantage survive a longer iteration budget?

The 60-iteration smoke test showed variants beating baseline, but every bound
there was far above the production bound for the same instance -- i.e. all arms
were under-converged. A treatment that merely converges faster will look strong
against a starved baseline and then lose its edge once the baseline is allowed to
finish. That is exactly the artifact this study already documented for the fast
screen, so it has to be ruled out rather than assumed away.

Each variant is run once at the maximum budget and its bound_history retained;
the running-best at each checkpoint is then read off, so one run yields the whole
convergence curve.
"""
from __future__ import annotations

import argparse
import csv
import os
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
MASTER = os.path.join(ROOT, "results", "adaptive_master.csv")

PROBE = [
    ("baseline",    mu_fair,      False, step_polyak, {},             0.0),
    ("trust05",     mu_fair,      False, step_trust,  {"trust": 0.5}, 0.0),
    ("trust10",     mu_fair,      False, step_trust,  {"trust": 1.0}, 0.0),
    ("seed_eps10",  mu_incumbent, True,  step_polyak, {},             0.0),
]


def greedy_incumbent(inst):
    sol = OPSSolution(inst)
    for j in sorted(sol.get_unserved(), key=lambda x: inst.profits[x], reverse=True):
        c, p = sol.best_insertion_cost(j)
        if c < float("inf"):
            sol.insert(j, p)
    return set(sol.selected), sol.objective()


def running_best_at(hist, ck):
    best = float("inf")
    out = {}
    for i, v in enumerate(hist, 1):
        best = min(best, v)
        if i in ck:
            out[i] = best
    for c in ck:
        out.setdefault(c, best)
    return out


def run_one(payload):
    label, max_iter, checkpoints = payload
    fam = label.split("_")[0]
    inst = OPSInstance.from_instance_file(
        os.path.join(BENCH, fam, "instances", label + ".txt")
    )
    sel, z = greedy_incumbent(inst)
    res = {}
    for name, mu_fn, needs, step_fn, step_kw, smooth in PROBE:
        kw = {"selected": sel, "eps": 0.10} if needs else {}
        t0 = time.perf_counter()
        r = lagrangian_variant(
            inst, max_iter=max_iter, lower_bound=z, max_time=None,
            mu_init_fn=mu_fn, mu_init_kw=kw,
            step_fn=step_fn, step_kw=step_kw, smoothing=smooth,
        )
        res[name] = {
            "curve": running_best_at(r["bound_history"], checkpoints),
            "s": time.perf_counter() - t0,
            "iters": r["iterations"],
        }
    return label, z, res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subset", required=True)
    p.add_argument("--tag", default="probe")
    p.add_argument("--max-iter", type=int, default=1000)
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args()

    checkpoints = [60, 100, 200, 400, 700, args.max_iter]
    checkpoints = sorted(set(c for c in checkpoints if c <= args.max_iter))

    with open(args.subset, encoding="utf-8") as f:
        labels = [
            r["instance"].strip()
            for r in csv.DictReader(f)
            if r.get("instance", "").strip()
        ]

    prod = {}
    with open(MASTER, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            v = (r.get("best_ub") or "").strip()
            if v:
                try:
                    prod[r["instance"].strip()] = float(v)
                except ValueError:
                    pass

    os.makedirs(OUT_DIR, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M")
    out_csv = os.path.join(OUT_DIR, "dual_convergence_" + args.tag + "_" + ts + ".csv")

    tasks = [(lb, args.max_iter, checkpoints) for lb in labels]
    rows = []
    with Pool(processes=args.workers) as pool:
        for i, (label, z, res) in enumerate(pool.imap_unordered(run_one, tasks), 1):
            print("=" * 74, flush=True)
            print("[%d/%d] %s   greedy z=%.0f   production ub=%s"
                  % (i, len(tasks), label, z,
                     ("%.2f" % prod[label]) if label in prod else "n/a"), flush=True)
            print("  %-12s %s" % ("variant",
                                  "".join("%10d" % c for c in checkpoints)), flush=True)
            for name, _, _, _, _, _ in PROBE:
                curve = res[name]["curve"]
                print("  %-12s %s   (%.1fs)"
                      % (name, "".join("%10.2f" % curve[c] for c in checkpoints),
                         res[name]["s"]), flush=True)
                row = {"instance": label, "variant": name,
                       "greedy_z": z, "prod_ub": prod.get(label, "")}
                for c in checkpoints:
                    row["it%d" % c] = round(curve[c], 4)
                row["seconds"] = round(res[name]["s"], 2)
                rows.append(row)
            b = res["baseline"]["curve"][checkpoints[-1]]
            for name, _, _, _, _, _ in PROBE:
                if name == "baseline":
                    continue
                d = res[name]["curve"][checkpoints[-1]] - b
                print("    %-12s vs baseline at %d iters: %+.4f %s"
                      % (name, checkpoints[-1], d,
                         "(still ahead)" if d < -1e-9 else
                         "(caught up / behind)"), flush=True)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\n-> %s" % out_csv)


if __name__ == "__main__":
    main()
