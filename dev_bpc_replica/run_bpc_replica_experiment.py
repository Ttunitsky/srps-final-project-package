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
from dev_bpc_replica.bpc_cut_core import CutPool, evaluate_violation, separate_cuts_with_diagnostics

PHASE_RT = 120.0
ABS_CAP = 240.0
GAP_THRESHOLD = 0.003
DESTROY_FRACS = [0.25, 1 / 3, 0.40]
LAG_MAX_ITER = 120
LAG_MAX_TIME = 20.0
SEEDS = [42, 123, 456, 789, 1337, 2024]

N_WORKERS = max(1, (os.cpu_count() or 1) // 2)
BENCH = os.path.join(ROOT, "benchmarks", "ops_raw", "OPS-Benchmark-master", "input")
MASTER_CSV = os.path.join(ROOT, "results", "adaptive_master.csv")
OUT_DIR = os.path.join(ROOT, "results", "bpc_replica_dev")

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
    p.add_argument("--subset-csv", default=None)
    p.add_argument("--workers", type=int, default=N_WORKERS)
    p.add_argument("--tag", default="dev")

    p.add_argument("--phase-rt", type=float, default=PHASE_RT)
    p.add_argument("--abs-cap", type=float, default=ABS_CAP)
    p.add_argument("--gap-threshold", type=float, default=GAP_THRESHOLD)
    p.add_argument("--lag-max-iter", type=int, default=LAG_MAX_ITER)
    p.add_argument("--lag-max-time", type=float, default=LAG_MAX_TIME)

    p.add_argument("--sep-freq-phase", type=int, default=2)
    p.add_argument("--max-cuts-per-round", type=int, default=10)
    p.add_argument("--epsilon-violation", type=float, default=1e-5)
    p.add_argument("--epsilon-reject", type=float, default=1e-4)
    p.add_argument("--lambda-cut", type=float, default=0.25)
    p.add_argument("--cut-age-limit", type=int, default=4)
    p.add_argument("--sep-budget-s-per-phase", type=float, default=10.0)

    p.add_argument("--bpc-cut-destroy", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--bpc-cut-repair", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--bpc-cut-hard-filter", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--bpc-cut-soft-penalty", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--require-cut-activation", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--diagnostic-csv", default=None)

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


def _cut_guided_destroy(sol, d: int, cfg, pool: CutPool, telemetry: dict):
    removable = sol.get_removable()
    if not removable:
        return
    scored = []
    for j in removable:
        cand = sol.copy()
        cand.remove(j)
        viol, _ = evaluate_violation(cand, pool.all(), cfg["epsilon_reject"])
        scored.append((viol, -len(sol.inst.Kj.get(j, [])), -sol.inst.profits[j], j))
    scored.sort()
    for _, _, _, j in scored[: min(d, len(scored))]:
        sol.remove(j)
        telemetry["cut_penalty_hits"] += 1


def _cut_guided_repair(sol, cfg, pool: CutPool, telemetry: dict):
    while sol.has_unserved():
        best_score = -float("inf")
        best_item = None
        best_pos = None

        for j in sol.get_unserved():
            cost, pos = sol.best_insertion_cost(j)
            if cost >= float("inf") or pos is None:
                continue

            cand = sol.copy()
            cand.insert(j, pos)
            viol, violated_ids = evaluate_violation(cand, pool.all(), cfg["epsilon_reject"])

            if cfg["bpc_cut_hard_filter"] and violated_ids:
                telemetry["cut_filter_rejections"] += 1
                continue

            score = sol.inst.profits[j] / (1.0 + max(cost, 1e-9))
            if cfg["bpc_cut_soft_penalty"]:
                score -= cfg["lambda_cut"] * viol
                if viol > 0:
                    telemetry["cut_penalty_hits"] += 1

            if score > best_score:
                best_score = score
                best_item = j
                best_pos = pos

        if best_item is None:
            break
        sol.insert(best_item, best_pos)


def _run_one(args_tuple):
    label, meta, cfg = args_tuple

    ipath = os.path.join(BENCH, meta["family"], "instances", label + ".txt")
    inst = OPSInstance.from_instance_file(ipath)

    best_sol = OPSSolution(inst)
    _repair_profit(best_sol)
    best_obj = best_sol.objective()

    ub = meta["ub"]
    t_start = time.perf_counter()
    phase = 0
    tier = 0
    lag_calls = 0
    lag_iters = 0
    stop_reason = "RT_CAP"

    cut_pool = CutPool()
    telemetry = {
        "cut_gen_calls": 0,
        "cuts_added_total": 0,
        "cuts_active_peak": 0,
        "sep_empty_rounds": 0,
        "sep_nonempty_rounds": 0,
        "sep_budget_skips": 0,
        "sep_candidates_total": 0,
        "sep_candidates_circuit": 0,
        "sep_candidates_route_len": 0,
        "sep_candidates_sync": 0,
        "sep_empty_no_cycles": 0,
        "sep_empty_no_route_over": 0,
        "sep_empty_sync_not_acyclic": 0,
        "sep_empty_sync_no_path": 0,
        "sep_empty_sync_not_over_L": 0,
        "sep_diag_nodes_sum": 0,
        "sep_diag_edges_sum": 0,
        "mean_violation_before_sum": 0.0,
        "mean_violation_before_cnt": 0,
        "mean_violation_after_sum": 0.0,
        "mean_violation_after_cnt": 0,
        "cut_filter_rejections": 0,
        "cut_penalty_hits": 0,
    }

    while True:
        elapsed = time.perf_counter() - t_start
        remaining = cfg["abs_cap"] - elapsed
        if remaining <= 1.0:
            stop_reason = "RT_CAP"
            break

        phase += 1
        phase_budget = min(cfg["phase_rt"], remaining)
        n_sel = max(1, len(best_sol.selected))
        d_max = max(3, int(n_sel * DESTROY_FRACS[tier]))

        if phase == 1 or (phase % max(1, cfg["sep_freq_phase"]) == 0):
            sep_t0 = time.perf_counter()
            new_cuts, sep_diag = separate_cuts_with_diagnostics(best_sol, cfg["epsilon_violation"], cfg["max_cuts_per_round"])
            telemetry["sep_candidates_total"] += int(sep_diag.get("total_candidates", 0))
            telemetry["sep_candidates_circuit"] += int(sep_diag.get("circuit_candidates", 0))
            telemetry["sep_candidates_route_len"] += int(sep_diag.get("route_len_candidates", 0))
            telemetry["sep_candidates_sync"] += int(sep_diag.get("sync_candidates", 0))
            telemetry["sep_diag_nodes_sum"] += int(sep_diag.get("nodes_count", 0))
            telemetry["sep_diag_edges_sum"] += int(sep_diag.get("edge_count", 0))

            if int(sep_diag.get("returned_count", 0)) <= 0:
                telemetry["sep_empty_rounds"] += 1
                if int(sep_diag.get("cycle_count", 0)) <= 0:
                    telemetry["sep_empty_no_cycles"] += 1
                if int(sep_diag.get("route_over_count", 0)) <= 0:
                    telemetry["sep_empty_no_route_over"] += 1
                if not bool(sep_diag.get("acyclic", False)):
                    telemetry["sep_empty_sync_not_acyclic"] += 1
                if bool(sep_diag.get("acyclic", False)) and not bool(sep_diag.get("sync_path_found", False)):
                    telemetry["sep_empty_sync_no_path"] += 1
                if bool(sep_diag.get("acyclic", False)) and bool(sep_diag.get("sync_path_found", False)) and float(sep_diag.get("sync_over", 0.0)) <= cfg["epsilon_violation"]:
                    telemetry["sep_empty_sync_not_over_L"] += 1
            else:
                telemetry["sep_nonempty_rounds"] += 1

            if (time.perf_counter() - sep_t0) <= cfg["sep_budget_s_per_phase"]:
                added = cut_pool.add_many(new_cuts)
                telemetry["cuts_added_total"] += added
            else:
                telemetry["sep_budget_skips"] += 1
            telemetry["cut_gen_calls"] += 1
            cut_pool.age_and_prune(cfg["cut_age_limit"])
            telemetry["cuts_active_peak"] = max(telemetry["cuts_active_peak"], cut_pool.size())

        def gap_fn(sol):
            return max(0.0, (ub - sol.objective()) / ub) if ub else 0.0

        phase_t0 = time.perf_counter()
        phase_best = best_obj

        for seed in (s for _ in range(10000) for s in SEEDS):
            rem = phase_budget - (time.perf_counter() - phase_t0)
            if rem <= 1.0:
                break

            start_sol = best_sol.copy()
            if cfg["bpc_cut_destroy"]:
                _cut_guided_destroy(start_sol, max(1, d_max // 3), cfg, cut_pool, telemetry)

            if cfg["bpc_cut_repair"]:
                _cut_guided_repair(start_sol, cfg, cut_pool, telemetry)

            v_before, _ = evaluate_violation(start_sol, cut_pool.all(), cfg["epsilon_reject"])
            telemetry["mean_violation_before_sum"] += v_before
            telemetry["mean_violation_before_cnt"] += 1

            returned_best, neg_obj, info = alns_search(
                initial_solution=start_sol,
                copy_fn=lambda s: s.copy(),
                objective_fn=lambda s: -s.objective(),
                destroy_ops=[destroy_random, destroy_worst, destroy_shaw],
                repair_ops=[_repair_profit, _repair_ratio, _repair_regret2, _repair_random],
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

            v_after, violated_ids = evaluate_violation(returned_best, cut_pool.all(), cfg["epsilon_reject"])
            telemetry["mean_violation_after_sum"] += v_after
            telemetry["mean_violation_after_cnt"] += 1
            cut_pool.record_hits(violated_ids)

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
                max_iter=cfg["lag_max_iter"],
                lower_bound=best_obj,
                max_time=cfg["lag_max_time"],
                mu_init=None,
            )
            lag_calls += 1
            lag_iters += lag["iterations"]
            ub = math.floor(lag["upper_bound"])

        gap_pct = ((ub - best_obj) / ub * 100) if ub else None

        if improved:
            tier = 0
        else:
            tier += 1

        if gap_pct is not None and gap_pct < cfg["gap_threshold"] * 100:
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

    mean_v_before = telemetry["mean_violation_before_sum"] / max(1, telemetry["mean_violation_before_cnt"])
    mean_v_after = telemetry["mean_violation_after_sum"] / max(1, telemetry["mean_violation_after_cnt"])
    activation_pass = int(telemetry["cuts_added_total"] > 0)

    avg_sep_nodes = telemetry["sep_diag_nodes_sum"] / max(1, telemetry["cut_gen_calls"])
    avg_sep_edges = telemetry["sep_diag_edges_sum"] / max(1, telemetry["cut_gen_calls"])

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
        "cut_gen_calls": telemetry["cut_gen_calls"],
        "cuts_added_total": telemetry["cuts_added_total"],
        "cuts_active_peak": telemetry["cuts_active_peak"],
        "sep_empty_rounds": telemetry["sep_empty_rounds"],
        "sep_nonempty_rounds": telemetry["sep_nonempty_rounds"],
        "sep_budget_skips": telemetry["sep_budget_skips"],
        "sep_candidates_total": telemetry["sep_candidates_total"],
        "sep_candidates_circuit": telemetry["sep_candidates_circuit"],
        "sep_candidates_route_len": telemetry["sep_candidates_route_len"],
        "sep_candidates_sync": telemetry["sep_candidates_sync"],
        "sep_empty_no_cycles": telemetry["sep_empty_no_cycles"],
        "sep_empty_no_route_over": telemetry["sep_empty_no_route_over"],
        "sep_empty_sync_not_acyclic": telemetry["sep_empty_sync_not_acyclic"],
        "sep_empty_sync_no_path": telemetry["sep_empty_sync_no_path"],
        "sep_empty_sync_not_over_L": telemetry["sep_empty_sync_not_over_L"],
        "avg_sep_nodes": round(avg_sep_nodes, 3),
        "avg_sep_edges": round(avg_sep_edges, 3),
        "mean_violation_before": round(mean_v_before, 6),
        "mean_violation_after": round(mean_v_after, 6),
        "cut_filter_rejections": telemetry["cut_filter_rejections"],
        "cut_penalty_hits": telemetry["cut_penalty_hits"],
        "activation_pass": activation_pass,
        "bpc_cut_destroy": int(cfg["bpc_cut_destroy"]),
        "bpc_cut_repair": int(cfg["bpc_cut_repair"]),
        "bpc_cut_hard_filter": int(cfg["bpc_cut_hard_filter"]),
        "bpc_cut_soft_penalty": int(cfg["bpc_cut_soft_penalty"]),
        "stop_reason": stop_reason,
        "total_rt_s": round(total_rt, 1),
    }


def main():
    args = _parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    master = _load_master()
    labels = _read_subset_from_csv(args.subset_csv) if args.subset_csv else list(SENSITIVITY_INSTANCES)

    cfg = {
        "phase_rt": float(args.phase_rt),
        "abs_cap": float(args.abs_cap),
        "gap_threshold": float(args.gap_threshold),
        "lag_max_iter": int(args.lag_max_iter),
        "lag_max_time": float(args.lag_max_time),
        "sep_freq_phase": max(1, int(args.sep_freq_phase)),
        "max_cuts_per_round": max(1, int(args.max_cuts_per_round)),
        "epsilon_violation": float(args.epsilon_violation),
        "epsilon_reject": float(args.epsilon_reject),
        "lambda_cut": float(args.lambda_cut),
        "cut_age_limit": max(1, int(args.cut_age_limit)),
        "sep_budget_s_per_phase": float(args.sep_budget_s_per_phase),
        "bpc_cut_destroy": bool(args.bpc_cut_destroy),
        "bpc_cut_repair": bool(args.bpc_cut_repair),
        "bpc_cut_hard_filter": bool(args.bpc_cut_hard_filter),
        "bpc_cut_soft_penalty": bool(args.bpc_cut_soft_penalty),
        "require_cut_activation": bool(args.require_cut_activation),
    }

    tasks = [(lab, master[lab], cfg) for lab in labels if lab in master]

    ts = time.strftime("%Y%m%d_%H%M")
    out_csv = os.path.join(OUT_DIR, f"bpc_replica_{args.tag}_{ts}.csv")

    print("=" * 72)
    print("BPC-replica dev run (isolated)")
    print(f"Instances: {len(tasks)} | Workers: {args.workers}")
    print(
        "Switches: "
        f"destroy={cfg['bpc_cut_destroy']} repair={cfg['bpc_cut_repair']} "
        f"hard={cfg['bpc_cut_hard_filter']} soft={cfg['bpc_cut_soft_penalty']}"
    )
    print(
        "Params: "
        f"sep_freq={cfg['sep_freq_phase']} max_cuts={cfg['max_cuts_per_round']} "
        f"eps_v={cfg['epsilon_violation']} eps_r={cfg['epsilon_reject']} lambda={cfg['lambda_cut']}"
    )
    print(f"Output: {out_csv}")
    print("=" * 72)

    rows = []
    t0 = time.perf_counter()
    with mp.Pool(processes=max(1, args.workers), initializer=_worker_init) as pool:
        for r in pool.imap_unordered(_run_one, tasks, chunksize=1):
            rows.append(r)
            gap_s = f"{r['final_gap_pct']:.4f}%" if isinstance(r["final_gap_pct"], float) else "N/A"
            print(
                f"  {r['instance']:35s} gap={gap_s:8s} rt={r['total_rt_s']:6.0f}s "
                f"dObj={r['delta_ref']:+.0f} dRT={r['delta_rt_s']:+.1f}s dGap={r['delta_gap_pct']:+.4f}pp "
                f"stop={r['stop_reason']}",
                flush=True,
            )

    fields = [
        "instance", "family", "n", "alpha", "group_id",
        "ref_obj", "ref_rt_s", "ref_gap_pct", "final_obj", "delta_ref", "bks", "delta_bks",
        "ub", "final_gap_pct", "delta_rt_s", "delta_gap_pct", "ec_gain", "phases", "lag_calls", "lag_iters",
        "cut_gen_calls", "cuts_added_total", "cuts_active_peak", "mean_violation_before", "mean_violation_after",
        "sep_empty_rounds", "sep_nonempty_rounds", "sep_budget_skips",
        "sep_candidates_total", "sep_candidates_circuit", "sep_candidates_route_len", "sep_candidates_sync",
        "sep_empty_no_cycles", "sep_empty_no_route_over", "sep_empty_sync_not_acyclic", "sep_empty_sync_no_path", "sep_empty_sync_not_over_L",
        "avg_sep_nodes", "avg_sep_edges",
        "cut_filter_rejections", "cut_penalty_hits",
        "activation_pass",
        "bpc_cut_destroy", "bpc_cut_repair", "bpc_cut_hard_filter", "bpc_cut_soft_penalty",
        "stop_reason", "total_rt_s",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    wall = time.perf_counter() - t0
    gaps = [r["final_gap_pct"] for r in rows if isinstance(r["final_gap_pct"], float)]
    d_objs = [r["delta_ref"] for r in rows if isinstance(r["delta_ref"], (int, float))]
    d_rts = [r["delta_rt_s"] for r in rows if isinstance(r["delta_rt_s"], float)]
    d_gaps = [r["delta_gap_pct"] for r in rows if isinstance(r["delta_gap_pct"], float)]

    print("\n" + "=" * 72)
    print(f"Completed {len(rows)} instances in {wall / 60:.1f} min")
    print(f"Mean final gap: {(sum(gaps)/len(gaps)) if gaps else 0.0:.4f}%")
    print(
        "Mean delta vs baseline paper: "
        f"dObj={(sum(d_objs)/len(d_objs)) if d_objs else 0.0:+.3f}, "
        f"dRT={(sum(d_rts)/len(d_rts)) if d_rts else 0.0:+.3f}s, "
        f"dGap={(sum(d_gaps)/len(d_gaps)) if d_gaps else 0.0:+.4f}pp"
    )

    total_cuts_added = sum(int(r.get("cuts_added_total", 0)) for r in rows)
    empty_rounds = sum(int(r.get("sep_empty_rounds", 0)) for r in rows)
    nonempty_rounds = sum(int(r.get("sep_nonempty_rounds", 0)) for r in rows)
    activation_passes = sum(int(r.get("activation_pass", 0)) for r in rows)
    print(
        "Separation diagnostics: "
        f"cuts_added_total={total_cuts_added}, "
        f"sep_nonempty_rounds={nonempty_rounds}, sep_empty_rounds={empty_rounds}, "
        f"activation_pass_instances={activation_passes}/{len(rows)}"
    )

    diag_csv = args.diagnostic_csv
    if not diag_csv:
        diag_csv = os.path.join(OUT_DIR, f"bpc_activation_diag_{args.tag}_{ts}.csv")
    diag_fields = [
        "instance", "family", "cut_gen_calls", "cuts_added_total", "cuts_active_peak",
        "sep_empty_rounds", "sep_nonempty_rounds", "sep_budget_skips",
        "sep_candidates_total", "sep_candidates_circuit", "sep_candidates_route_len", "sep_candidates_sync",
        "sep_empty_no_cycles", "sep_empty_no_route_over", "sep_empty_sync_not_acyclic", "sep_empty_sync_no_path", "sep_empty_sync_not_over_L",
        "avg_sep_nodes", "avg_sep_edges", "activation_pass",
    ]
    with open(diag_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=diag_fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in diag_fields})

    print(f"CSV: {out_csv}")
    print(f"Activation diagnostics CSV: {diag_csv}")
    print("=" * 72)

    if cfg["require_cut_activation"] and total_cuts_added <= 0:
        print("ERROR: cut activation gate failed (cuts_added_total == 0).")
        sys.exit(2)


if __name__ == "__main__":
    main()
