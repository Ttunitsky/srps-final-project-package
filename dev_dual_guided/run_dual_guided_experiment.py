"""
Isolated development runner for direct dual-guided neighborhood coupling.

This script does not modify or import any patched production code paths.
It composes existing baseline modules with dual-guided destroy/repair operators
that consume the current Lagrangian multipliers.

Default scope is the 30-instance sensitivity subset for safe iteration.
"""
from __future__ import annotations

import argparse
import csv
import math
import multiprocessing as mp
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from adapters.ops_adapter import OPSInstance, OPSSolution
from core.ops_bounds import lagrangian_bound
from core.operators import destroy_random, destroy_shaw, destroy_worst, repair_random, repair_regret
from core.post_process import ejection_chain
from core.search_controller import alns_search
from core.solution_validator import validate_solution as _validate_solution
from dev_dual_guided.dual_guided_ops import (
    fair_split_mu,
    make_dual_guided_destroy,
    make_dual_guided_repair,
    make_dual_guided_destroy_v2,
    make_dual_guided_repair_v2,
)

PHASE_RT = 300.0
ABS_CAP = 1800.0
GAP_THRESHOLD = 0.003
DESTROY_FRACS = [0.25, 1 / 3, 0.40]
LAG_MAX_ITER = 200
LAG_MAX_TIME = 60.0
SEEDS = [42, 123, 456, 789, 1337, 2024]

N_WORKERS = max(1, (os.cpu_count() or 1) // 2)
BENCH = os.path.join(ROOT, "benchmarks", "ops_raw", "OPS-Benchmark-master", "input")
MASTER_CSV = os.path.join(ROOT, "results", "adaptive_master.csv")
OUT_DIR = os.path.join(ROOT, "results", "dual_guided_dev")

SENSITIVITY_INSTANCES = [
    "B_n100_001_a25_001", "B_n110_006_a50_017", "B_n120_011_a75_033", "B_n140_021_a25_061", "B_n150_026_a50_077",
    "C_n050_001_a25_001", "C_n060_011_a50_032", "C_n065_016_a75_048", "C_n070_021_a25_061", "C_n080_031_a50_092",
    "D_n040_001_a25_001", "D_n050_011_a50_032", "D_n060_021_a75_063", "D_n070_031_a25_091", "D_n080_041_a50_122",
    "EB_n100_001_a25_001", "EB_n110_006_a50_017", "EB_n120_011_a75_033", "EB_n140_021_a25_061", "EB_n150_026_a50_077",
    "EC_n050_001_a25_001", "EC_n060_011_a50_032", "EC_n065_016_a75_048", "EC_n070_021_a25_061", "EC_n080_031_a50_092",
    "ED_n040_001_a25_001", "ED_n050_011_a50_032", "ED_n060_021_a75_063", "ED_n070_031_a25_091", "ED_n080_041_a50_122",
]


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--subset-csv", default=None, help="Optional CSV with 'instance' column")
    p.add_argument("--workers", type=int, default=N_WORKERS)
    p.add_argument("--tag", default="dev")
    p.add_argument(
        "--dual-destroy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable the dual-guided destroy operator",
    )
    p.add_argument(
        "--dual-repair",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable the dual-guided repair operator",
    )
    p.add_argument(
        "--dual-feedback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable feeding refreshed Lagrangian multipliers back into dual-guided operators",
    )
    p.add_argument(
        "--ops-v2",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use corrected v2 dual-guided operators (non-degenerate under fair-split mu)",
    )
    p.add_argument(
        "--mu-warm-init",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Seed mu from a real lagrangian_bound solve instead of the fair split "
             "(the fair split makes reduced profit identically zero)",
    )
    p.add_argument("--phase-rt", type=float, default=PHASE_RT, help="Per-phase wallclock budget (seconds)")
    p.add_argument("--abs-cap", type=float, default=ABS_CAP, help="Per-instance wallclock cap (seconds)")
    p.add_argument("--gap-threshold", type=float, default=GAP_THRESHOLD, help="Certified gap stop threshold (fraction)")
    p.add_argument("--lag-max-iter", type=int, default=LAG_MAX_ITER, help="Max iterations in lagrangian_bound refresh")
    p.add_argument("--lag-max-time", type=float, default=LAG_MAX_TIME, help="Max seconds per lagrangian_bound refresh")
    return p.parse_args()


def _load_master():
    meta = {}
    with open(MASTER_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            label = r["instance"].strip()
            ub_s = r.get("best_ub", "").strip()
            bks_s = r.get("bks", "").strip()
            base_obj_s = r.get("alns_obj", "").strip()
            base_rt_s = r.get("alns_runtime_s", "").strip()
            base_gap_s = r.get("final_cert_gap_pct", "").strip()
            meta[label] = {
                "instance": label,
                "family": r["family"].strip(),
                "n": int(r["n"]),
                "alpha": r["alpha"].strip(),
                "group_id": r.get("group_id", "").strip(),
                "ref_obj": float(base_obj_s) if base_obj_s else 0.0,
                "ref_rt_s": float(base_rt_s) if base_rt_s else None,
                "ref_gap_pct": float(base_gap_s) if base_gap_s else None,
                "ub": math.floor(float(ub_s)) if ub_s not in ("", "NA") else None,
                "bks": float(bks_s) if bks_s not in ("", "?", "NA") else None,
            }
    return meta


def _repair_profit(sol):
    for j in sorted(sol.get_unserved(), key=lambda x: sol.inst.profits[x], reverse=True):
        c, p = sol.best_insertion_cost(j)
        if c < float("inf"):
            sol.insert(j, p)


def _repair_ratio(sol):
    cands = []
    for j in sol.get_unserved():
        c, _ = sol.best_insertion_cost(j)
        if c < float("inf"):
            cands.append((sol.inst.profits[j] / max(c, 1.0), j))
    for _, j in sorted(cands, reverse=True):
        c, p = sol.best_insertion_cost(j)
        if c < float("inf"):
            sol.insert(j, p)


def _repair_regret2(sol):
    repair_regret(sol, k=2)


def _repair_random(sol):
    repair_random(sol)


def _worker_init():
    sys.stdout = open(os.devnull, "w")


def _read_subset_from_csv(path):
    with open(path, encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        if "instance" not in (rdr.fieldnames or []):
            raise ValueError("subset csv must include 'instance' column")
        return [r["instance"].strip() for r in rdr if r.get("instance", "").strip()]


def _run_one(args_tuple):
    label, meta, cfg = args_tuple

    phase_rt = float(cfg["phase_rt"])
    abs_cap = float(cfg["abs_cap"])
    gap_threshold = float(cfg["gap_threshold"])
    lag_max_iter = int(cfg["lag_max_iter"])
    lag_max_time = float(cfg["lag_max_time"])

    ipath = os.path.join(BENCH, meta["family"], "instances", label + ".txt")
    inst = OPSInstance.from_instance_file(ipath)

    best_sol = OPSSolution(inst)
    _repair_profit(best_sol)
    best_obj = best_sol.objective()

    ub = meta["ub"]
    if cfg.get("mu_warm_init"):
        # The fair split makes sum_k mu_{j,k} == b_j exactly, hence reduced
        # profit == 0 for every job and no usable guidance signal. Seed from a
        # real subgradient solve so the operators see a non-degenerate dual.
        _seed = lagrangian_bound(
            inst,
            max_iter=lag_max_iter,
            lower_bound=best_obj,
            max_time=lag_max_time,
        )
        mu_state = dict(_seed["best_mu"])
    else:
        mu_state = fair_split_mu(inst)
    prev_mu = dict(mu_state)
    use_dual_destroy = cfg["dual_destroy"]
    use_dual_repair = cfg["dual_repair"]
    use_dual_feedback = cfg["dual_feedback"]

    # Dual-guided operator closures read from mutable mu_state.
    def _mu_provider():
        return mu_state

    if cfg.get("ops_v2"):
        dual_destroy = make_dual_guided_destroy_v2(_mu_provider)
        dual_repair = make_dual_guided_repair_v2(_mu_provider)
    else:
        dual_destroy = make_dual_guided_destroy(_mu_provider)
        dual_repair = make_dual_guided_repair(_mu_provider)

    destroy_ops = [destroy_random, destroy_worst, destroy_shaw]
    repair_ops = [_repair_profit, _repair_ratio, _repair_regret2, _repair_random]
    if use_dual_destroy:
        destroy_ops.insert(0, dual_destroy)
    if use_dual_repair:
        repair_ops.insert(0, dual_repair)

    t_start = time.perf_counter()
    phase = 0
    tier = 0
    lag_calls = 0
    lag_iters = 0
    stop_reason = "RT_CAP"

    while True:
        elapsed = time.perf_counter() - t_start
        remaining = abs_cap - elapsed
        if remaining <= 1.0:
            stop_reason = "RT_CAP"
            break

        phase += 1
        phase_budget = min(phase_rt, remaining)
        n_sel = max(1, len(best_sol.selected))
        d_max = max(3, int(n_sel * DESTROY_FRACS[tier]))

        def gap_fn(sol):
            return max(0.0, (ub - sol.objective()) / ub) if ub else 0.0

        phase_t0 = time.perf_counter()
        phase_best = best_obj

        for seed in (s for _ in range(10000) for s in SEEDS):
            rem = phase_budget - (time.perf_counter() - phase_t0)
            if rem <= 1.0:
                break

            returned_best, neg_obj, info = alns_search(
                initial_solution=best_sol.copy(),
                copy_fn=lambda s: s.copy(),
                objective_fn=lambda s: -s.objective(),
                destroy_ops=destroy_ops,
                repair_ops=repair_ops,
                destroy_size_fn=lambda: random.randint(1, d_max),
                max_iterations=500,
                start_temp=100.0,
                cooling_rate=0.985,
                lambda_decay=0.8,
                sigma1=33.0,
                sigma2=20.0,
                sigma3=13.0,
                seed=seed,
                gap_fn=gap_fn,
                gap_threshold=1e-6,
                gap_stop=(ub is not None),
                max_time=rem,
                stall_patience=150,
                stall_max_scale=4.0,
            )

            seed_obj = -neg_obj
            if seed_obj > best_obj:
                best_obj = seed_obj
                best_sol = returned_best

            if ub is not None and (ub - best_obj) < 0.9999:
                stop_reason = "UB"
                break
            if info["stop_reason"] == "gap_threshold":
                stop_reason = "UB"
                break

        if stop_reason == "UB":
            break

        improved = best_obj > phase_best
        if not improved and ub is not None:
            lag = lagrangian_bound(
                inst,
                max_iter=lag_max_iter,
                lower_bound=best_obj,
                max_time=lag_max_time,
                mu_init=prev_mu,
            )
            lag_calls += 1
            lag_iters += lag["iterations"]
            prev_mu = lag["best_mu"]
            if use_dual_feedback:
                mu_state = dict(prev_mu)
            ub = math.floor(lag["upper_bound"])

        gap_pct = ((ub - best_obj) / ub * 100) if ub else None

        if improved:
            tier = 0
        else:
            tier += 1

        if gap_pct is not None and gap_pct < gap_threshold * 100:
            stop_reason = "GAP<0.3%"
            break
        if tier > 2:
            stop_reason = "TIERS_EXHAUSTED"
            break

    ec_gain = ejection_chain(best_sol, depth=3, max_rounds=30, verbose=False)
    final_obj = best_sol.objective()
    total_rt = time.perf_counter() - t_start
    final_gap = ((ub - final_obj) / ub * 100) if ub else None

    _validate_solution(inst, best_sol, claimed_obj=final_obj, label=label)

    delta_ref = final_obj - meta["ref_obj"]
    delta_bks = (final_obj - meta["bks"]) if meta["bks"] is not None else None
    delta_rt = (total_rt - meta["ref_rt_s"]) if meta["ref_rt_s"] is not None else None
    delta_gap = (final_gap - meta["ref_gap_pct"]) if (final_gap is not None and meta["ref_gap_pct"] is not None) else None

    return {
        "instance": label,
        "family": meta["family"],
        "n": meta["n"],
        "alpha": meta["alpha"],
        "group_id": meta["group_id"],
        "ref_obj": meta["ref_obj"],
        "ref_rt_s": meta["ref_rt_s"] if meta["ref_rt_s"] is not None else "",
        "ref_gap_pct": meta["ref_gap_pct"] if meta["ref_gap_pct"] is not None else "",
        "final_obj": final_obj,
        "delta_ref": delta_ref,
        "bks": meta["bks"] if meta["bks"] is not None else "",
        "delta_bks": delta_bks if delta_bks is not None else "",
        "ub": ub if ub is not None else "",
        "final_gap_pct": round(final_gap, 6) if final_gap is not None else "",
        "delta_rt_s": round(delta_rt, 3) if delta_rt is not None else "",
        "delta_gap_pct": round(delta_gap, 6) if delta_gap is not None else "",
        "ec_gain": ec_gain,
        "phases": phase,
        "lag_calls": lag_calls,
        "lag_iters": lag_iters,
        "dual_destroy": int(use_dual_destroy),
        "dual_repair": int(use_dual_repair),
        "dual_feedback": int(use_dual_feedback),
        "stop_reason": stop_reason,
        "total_rt_s": round(total_rt, 1),
    }


def main():
    args = _parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    master = _load_master()
    if args.subset_csv:
        labels = _read_subset_from_csv(args.subset_csv)
    else:
        labels = list(SENSITIVITY_INSTANCES)

    cfg = {
        "dual_destroy": bool(args.dual_destroy),
        "dual_repair": bool(args.dual_repair),
        "dual_feedback": bool(args.dual_feedback),
        "ops_v2": bool(args.ops_v2),
        "mu_warm_init": bool(args.mu_warm_init),
        "phase_rt": float(args.phase_rt),
        "abs_cap": float(args.abs_cap),
        "gap_threshold": float(args.gap_threshold),
        "lag_max_iter": int(args.lag_max_iter),
        "lag_max_time": float(args.lag_max_time),
    }

    tasks = [(lab, master[lab], cfg) for lab in labels if lab in master]

    ts = time.strftime("%Y%m%d_%H%M")
    out_csv = os.path.join(OUT_DIR, f"dual_guided_{args.tag}_{ts}.csv")

    print("=" * 72)
    print("Dual-guided dev run (isolated)")
    print(f"Instances: {len(tasks)} | Workers: {args.workers}")
    print(
        "Switches: "
        f"dual_destroy={cfg['dual_destroy']} "
        f"dual_repair={cfg['dual_repair']} "
        f"dual_feedback={cfg['dual_feedback']}"
    )
    print(f"Output: {out_csv}")
    print("=" * 72)

    t0 = time.perf_counter()
    rows = []
    with mp.Pool(processes=max(1, args.workers), initializer=_worker_init) as pool:
        for r in pool.imap_unordered(_run_one, tasks, chunksize=1):
            rows.append(r)
            gap_s = f"{r['final_gap_pct']:.4f}%" if isinstance(r["final_gap_pct"], float) else "N/A"
            d_obj = f"{r['delta_ref']:+.0f}"
            d_rt = f"{r['delta_rt_s']:+.1f}s" if isinstance(r["delta_rt_s"], float) else "N/A"
            d_gap = f"{r['delta_gap_pct']:+.4f}pp" if isinstance(r["delta_gap_pct"], float) else "N/A"
            print(
                f"  {r['instance']:35s} gap={gap_s:8s} rt={r['total_rt_s']:6.0f}s "
                f"dObj={d_obj:>5s} dRT={d_rt:>8s} dGap={d_gap:>10s} stop={r['stop_reason']}",
                flush=True,
            )

    fields = [
        "instance", "family", "n", "alpha", "group_id",
        "ref_obj", "ref_rt_s", "ref_gap_pct", "final_obj", "delta_ref", "bks", "delta_bks",
        "ub", "final_gap_pct", "delta_rt_s", "delta_gap_pct", "ec_gain", "phases", "lag_calls", "lag_iters",
        "dual_destroy", "dual_repair", "dual_feedback",
        "stop_reason", "total_rt_s",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    wall = time.perf_counter() - t0
    gaps = [r["final_gap_pct"] for r in rows if isinstance(r["final_gap_pct"], float)]
    mean_gap = sum(gaps) / len(gaps) if gaps else 0.0
    d_objs = [r["delta_ref"] for r in rows if isinstance(r["delta_ref"], (int, float))]
    d_rts = [r["delta_rt_s"] for r in rows if isinstance(r["delta_rt_s"], float)]
    d_gaps = [r["delta_gap_pct"] for r in rows if isinstance(r["delta_gap_pct"], float)]
    mean_d_obj = (sum(d_objs) / len(d_objs)) if d_objs else 0.0
    mean_d_rt = (sum(d_rts) / len(d_rts)) if d_rts else 0.0
    mean_d_gap = (sum(d_gaps) / len(d_gaps)) if d_gaps else 0.0
    print("\n" + "=" * 72)
    print(f"Completed {len(rows)} instances in {wall / 60:.1f} min")
    print(f"Mean final gap: {mean_gap:.4f}%")
    print(
        "Mean delta vs baseline paper: "
        f"dObj={mean_d_obj:+.3f}, dRT={mean_d_rt:+.3f}s, dGap={mean_d_gap:+.4f}pp"
    )
    print(f"CSV: {out_csv}")
    print("=" * 72)


if __name__ == "__main__":
    main()
