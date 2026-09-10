"""
run_adaptive_full.py — Full 660-instance adaptive ALNS run (beta algo).

Runs all B/C/D/EB/EC/ED instances with the adaptive loop:
  - 300s ALNS phases, warm-starting from best incumbent
  - After each phase (if obj unchanged): recompute Lagrangian UB (floored, warm-mu)
  - Destroy tier escalation on stall: 25% → 33% → 40%; reset on any obj improvement
  - Stop: gap < 0.3%,  total_rt >= 3600s,  or tiers exhausted
  - Per-phase log: phase, tier, obj, UB, gap%, seeds, RT, event

Outputs:
  results/adaptive_full_<timestamp>.csv        — per-instance summary
  results/adaptive_full_<timestamp>_phases.csv — per-phase convergence data
"""
from __future__ import annotations

import argparse
import csv
import math
import multiprocessing as mp
import os
import pickle
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

from adapters.ops_adapter import OPSInstance, OPSSolution
from core.ops_bounds import lagrangian_bound
from dev_cut_lagrangian.cut_augmented_lagrangian import cut_augmented_lagrangian_bound
from core.operators import destroy_random, destroy_shaw, destroy_worst, repair_random, repair_regret
from core.post_process import ejection_chain
from core.search_controller import alns_search
from core.solution_validator import validate_solution as _validate_solution

# ── Config ────────────────────────────────────────────────────────────────────
PHASE_RT       = 300.0
ABS_CAP        = 3600.0
GAP_THRESHOLD  = 0.003            # 0.3%
DESTROY_FRACS  = [0.25, 1/3, 0.40]
LAG_MAX_ITER   = 200
LAG_MAX_TIME   = 60.0

N_WORKERS      = 6

STUDY_FAMILIES = {"B", "C", "D", "EB", "EC", "ED"}   # 660 instances

SEEDS          = [42, 123, 456, 789, 1337, 2024]
BENCH          = os.path.join("benchmarks", "ops_raw", "OPS-Benchmark-master", "input")
MASTER_CSV     = os.path.join("results", "master_results.csv")
INCUMBENTS_DIR = os.path.join("results", "beta_incumbents")
CERTS_DIR      = os.path.join("results", "beta_certificates")
TIMESTAMP      = time.strftime("%Y%m%d_%H%M")


# ── Helpers ───────────────────────────────────────────────────────────────────
def _save_incumbent(label, sol, inc_dir=INCUMBENTS_DIR):
    os.makedirs(inc_dir, exist_ok=True)
    with open(os.path.join(inc_dir, label + ".pkl"), "wb") as f:
        pickle.dump({"routes": [list(r) for r in sol.routes],
                     "selected": list(sol.selected)}, f)


def _save_certificate(label, cert, cert_dir=CERTS_DIR):
    """Persist the dual certificate (multipliers + bound provenance) per instance.

    Stores the multiplier vector ``best_mu`` at the reported upper bound so that
    a verifier can reconstruct L(mu) exactly (see scripts/verify_certificates.py).
    ``mu`` is keyed by "j,k" strings (pickle-safe, json-friendly). The bare
    incumbent PKL remains the primal half; this is the dual half.
    """
    os.makedirs(cert_dir, exist_ok=True)
    payload = dict(cert)
    if payload.get("best_mu") is not None:
        payload["best_mu"] = {f"{j},{k}": v for (j, k), v in payload["best_mu"].items()}
    with open(os.path.join(cert_dir, label + ".pkl"), "wb") as f:
        pickle.dump(payload, f)


def _load_incumbent(label, inst, inc_dir=INCUMBENTS_DIR):
    path = os.path.join(inc_dir, label + ".pkl")
    if not os.path.exists(path):
        return None, 0.0
    with open(path, "rb") as f:
        data = pickle.load(f)
    sol = OPSSolution(inst)
    sol.routes   = [list(r) for r in data["routes"]]
    sol.selected = set(data["selected"])
    return sol, sol.objective()


def _repair_profit(sol):
    for j in sorted(sol.get_unserved(), key=lambda j: sol.inst.profits[j], reverse=True):
        c, p = sol.best_insertion_cost(j)
        if c < float("inf"):
            sol.insert(j, p)


def _repair_ratio(sol):
    cands = [(sol.inst.profits[j] / max(sol.best_insertion_cost(j)[0], 1.0), j)
             for j in sol.get_unserved()
             if sol.best_insertion_cost(j)[0] < float("inf")]
    for _, j in sorted(cands, reverse=True):
        c, p = sol.best_insertion_cost(j)
        if c < float("inf"):
            sol.insert(j, p)


def _repair_regret2(sol): repair_regret(sol, k=2)
def _repair_random(sol):  repair_random(sol)

DESTROY_OPS = [destroy_random, destroy_worst, destroy_shaw]
REPAIR_OPS  = [_repair_profit, _repair_ratio, _repair_regret2, _repair_random]


def _load_master():
    rows = []
    with open(MASTER_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["family"].strip() not in STUDY_FAMILIES:
                continue
            label  = r["instance"].strip()
            ub_s   = r.get("best_ub", "").strip()
            bks_s  = r.get("bks", "").strip()
            rt_s   = r.get("alns_runtime_s", "0").strip()
            rows.append({
                "label":      label,
                "family":     r["family"].strip(),
                "n":          int(r["n"]),
                "alpha":      r["alpha"].strip(),
                "group_id":   r.get("group_id", "").strip(),
                "master_obj": float(r["alns_obj"]),
                "master_rt":  float(rt_s) if rt_s else 0.0,
                "master_cert_gap": r.get("final_cert_gap_pct", "").strip(),
                "master_status":   r.get("alns_status", "").strip(),
                "ub":         math.floor(float(ub_s)) if ub_s and ub_s not in ("", "NA") else None,
                "bks":        float(bks_s) if bks_s not in ("", "?", "NA") else None,
            })
    # Sort longest expected RT first for load balancing
    rows.sort(key=lambda r: r["master_rt"], reverse=True)
    return rows


# ── Per-instance adaptive runner ──────────────────────────────────────────────
def run_instance(args) -> dict:
    meta   = args
    label  = meta["label"]
    family = meta["family"]
    master_obj = meta["master_obj"]
    bks        = meta["bks"]
    no_refresh = meta.get("no_refresh", False)   # H1a arm: skip in-loop re-bound
    save_cert  = meta.get("save_cert", False)    # persist dual certificate (mu)
    read_inc_dir  = meta.get("read_inc_dir", INCUMBENTS_DIR)   # warm-start source
    write_inc_dir = meta.get("write_inc_dir", INCUMBENTS_DIR)  # where to save (tagged in experiments)
    write_cert_dir = meta.get("write_cert_dir", CERTS_DIR)

    ipath = os.path.join(BENCH, family, "instances", label + ".txt")
    inst  = OPSInstance.from_instance_file(ipath)

    # Warm-start: saved incumbent or greedy
    saved_sol, saved_obj = _load_incumbent(label, inst, read_inc_dir)
    init = OPSSolution(inst)
    _repair_profit(init)
    if saved_sol is not None and saved_obj > init.objective():
        best_sol, best_obj = saved_sol, saved_obj
    else:
        best_sol, best_obj = init, init.objective()

    ub       = meta["ub"]   # floored best_ub from master
    prev_mu  = None
    prev_gamma = None
    prev_nu = None
    bound_mode = meta.get("bound_mode", "lag")
    raw_ub   = float(ub) if ub is not None else None   # ungated (pre-validity-guard) UB
    num_ub_refreshes   = 0  # in-loop Lagrangian re-bounds that actually tightened
    num_lag_calls      = 0  # total in-loop Lagrangian re-bound calls
    num_lag_iters_total = 0 # sub-gradient iterations summed over those calls
    t_start  = time.perf_counter()
    phase    = 0
    tier     = 0
    phase_logs  = []
    output_lines = []
    stop_reason  = "RT_CAP"

    output_lines.append(f"\n{'='*80}")
    output_lines.append(
        f"  {label}  |  master={master_obj:.0f}  BKS={bks if bks else 'N/A'}"
        f"  init_UB={ub if ub else 'N/A'}  master_status={meta['master_status']}")
    output_lines.append(
        f"  {'Ph':>2} {'Tier':>4} {'Obj':>8} {'UB':>8} {'Gap%':>6} "
        f"{'Seeds':>5} {'PhRT':>6} {'CumRT':>7}  Event")
    output_lines.append(f"  {'-'*78}")

    # Fast-path: already proven optimal
    if ub is not None and (ub - best_obj) < 0.9999:
        stop_reason = "UB"
        phase = 1
        gap_pct = 0.0
        phase_logs.append({
            "phase": 1, "tier": 0, "obj": best_obj, "ub": ub,
            "gap_pct": 0.0, "seeds": 0, "phase_rt": 0.0, "cumul_rt": 0.0,
            "event": "UB_PROVEN_INIT",
        })
        output_lines.append(
            f"  {1:>2} {0:>4}  {best_obj:>8.0f} {ub:>8.0f} {0.0:>6.3f}% "
            f"{0:>5} {0:>5.0f}s {0:>6.0f}s  UB_PROVEN_INIT")
    else:
        while True:
            elapsed   = time.perf_counter() - t_start
            remaining = ABS_CAP - elapsed
            if remaining <= 1.0:
                stop_reason = "RT_CAP"
                break

            phase += 1
            phase_budget = min(PHASE_RT, remaining)
            n_sel  = max(1, len(best_sol.selected))
            d_max  = max(3, int(n_sel * DESTROY_FRACS[tier]))

            def gap_fn(sol):
                return max(0.0, (ub - sol.objective()) / ub) if ub else 0.0

            phase_t0       = time.perf_counter()
            seeds_run      = 0
            phase_best_obj = best_obj

            for seed in (s for _ in range(10_000) for s in SEEDS):
                rem = phase_budget - (time.perf_counter() - phase_t0)
                if rem <= 1.0:
                    break
                start_sol = best_sol.copy()
                returned_best, neg_obj, info = alns_search(
                    initial_solution=start_sol,
                    copy_fn=lambda s: s.copy(),
                    objective_fn=lambda s: -s.objective(),
                    destroy_ops=DESTROY_OPS,
                    repair_ops=REPAIR_OPS,
                    destroy_size_fn=lambda: random.randint(1, d_max),
                    max_iterations=500,
                    start_temp=100.0,
                    cooling_rate=0.985,
                    lambda_decay=0.8,
                    sigma1=33.0, sigma2=20.0, sigma3=13.0,
                    seed=seed,
                    gap_fn=gap_fn,
                    gap_threshold=1e-6,
                    gap_stop=(ub is not None),
                    max_time=rem,
                    stall_patience=150,
                    stall_max_scale=4.0,
                )
                seeds_run += 1
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

            phase_rt     = time.perf_counter() - phase_t0
            cumul_rt     = time.perf_counter() - t_start
            obj_improved = best_obj > phase_best_obj

            # Recompute UB only if obj did not improve (skipped entirely in the
            # H1a no-refresh arm: the bound stays at its initial value)
            new_ub = ub
            ub_tightened = False
            if not obj_improved and ub is not None and stop_reason != "UB" and not no_refresh:
                if bound_mode == "lag":
                    lag = lagrangian_bound(
                        inst,
                        max_iter=LAG_MAX_ITER,
                        lower_bound=best_obj,
                        max_time=LAG_MAX_TIME,
                        mu_init=prev_mu,
                    )
                elif bound_mode in {"cutlag_cap", "cutlag_pair", "cutlag_triple"}:
                    lag = cut_augmented_lagrangian_bound(
                        inst,
                        max_iter=LAG_MAX_ITER,
                        lower_bound=best_obj,
                        max_time=LAG_MAX_TIME,
                        mu_init=prev_mu,
                        gamma_init=prev_gamma,
                        nu_init=prev_nu,
                        include_capacity=True,
                        include_pairs=bound_mode in {"cutlag_pair", "cutlag_triple"},
                        include_triples=bound_mode == "cutlag_triple",
                        cut_step_scale=1.0,
                        projected_g_sq=True,
                    )
                else:
                    raise ValueError(f"Unknown bound_mode: {bound_mode}")
                num_lag_calls += 1
                num_lag_iters_total += lag["iterations"]
                new_ub = math.floor(lag["upper_bound"])
                prev_mu = lag["best_mu"]
                if bound_mode.startswith("cutlag"):
                    prev_gamma = lag.get("best_gamma")
                    prev_nu = lag.get("best_nu")
                raw_ub = lag["upper_bound"]
                if new_ub < ub:
                    ub_tightened = True
                    num_ub_refreshes += 1
                ub = new_ub

            gap_pct = (ub - best_obj) / ub * 100 if ub else None

            # Determine event and tier update
            if stop_reason == "UB":
                event = "UB_STOP"
            elif obj_improved:
                tier  = 0
                event = f"OBJ+{best_obj - phase_best_obj:.0f} tier->0"
            elif ub_tightened:
                if gap_pct is not None and gap_pct < GAP_THRESHOLD * 100:
                    event = "UB_TIGHT→GAP<0.3%"
                else:
                    tier_before = tier
                    tier += 1
                    if tier > 2:
                        event = "TIERS_EXHAUSTED"
                    else:
                        event = f"STALL→tier{tier_before}→{tier} d_max={d_max}"
            else:
                tier_before = tier
                tier += 1
                if tier > 2:
                    event = "TIERS_EXHAUSTED"
                else:
                    event = f"STALL→tier{tier_before}→{tier} d_max={d_max}"

            gap_s = f"{gap_pct:.3f}%" if gap_pct is not None else "  N/A"
            ub_s  = f"{ub:.0f}" if ub else "  N/A"
            output_lines.append(
                f"  {phase:>2} {tier:>4}  {best_obj:>8.0f} {ub_s:>8} {gap_s:>6} "
                f"{seeds_run:>5} {phase_rt:>5.0f}s {cumul_rt:>6.0f}s  {event}")

            phase_logs.append({
                "phase": phase, "tier": tier, "obj": best_obj, "ub": ub,
                "gap_pct": gap_pct, "seeds": seeds_run,
                "phase_rt": phase_rt, "cumul_rt": cumul_rt, "event": event,
            })

            if stop_reason == "UB":
                break
            if gap_pct is not None and gap_pct < GAP_THRESHOLD * 100:
                stop_reason = "GAP<0.3%"
                break
            if tier > 2:
                stop_reason = "TIERS_EXHAUSTED"
                break

    # Final EC pass
    ec_gain   = ejection_chain(best_sol, depth=3, max_rounds=30, verbose=False)
    final_obj = best_sol.objective()
    total_rt  = time.perf_counter() - t_start  # validation time excluded from RT

    # Independent validation — must pass before result is recorded (RT excluded)
    _validate_solution(inst, best_sol, claimed_obj=final_obj, label=label)

    _save_incumbent(label, best_sol, write_inc_dir)

    if save_cert:
        _save_certificate(label, {
            "instance": label,
            "best_mu": prev_mu,                 # multipliers at the reported UB (None if no refresh)
            "bound_mode": bound_mode,
            "best_gamma": prev_gamma,
            "best_nu": prev_nu,
            "raw_ub": raw_ub,                   # ungated Lagrangian UB (real-valued)
            "reported_ub": ub,                  # floored UB used for the certified gap
            "incumbent_obj": final_obj,
            "num_ub_refreshes": num_ub_refreshes,
            "num_lag_calls": num_lag_calls,
            "num_lag_iters_total": num_lag_iters_total,
            "seed_set": list(SEEDS),
            "no_refresh": no_refresh,
        }, write_cert_dir)

    if ub is not None:
        gap_pct = (ub - final_obj) / ub * 100
    else:
        gap_pct = None

    diff_bks = (final_obj - bks) if bks is not None else None
    if diff_bks is None:
        beta_status = "NA"
    elif diff_bks > 0:
        beta_status = "BEAT"
    elif diff_bks == 0:
        beta_status = "TIE"
    else:
        beta_status = "GAP"

    output_lines.append(f"  {'─'*78}")
    output_lines.append(
        f"  FINAL: obj={final_obj:.0f}  EC+{ec_gain}"
        f"  gap={f'{gap_pct:.3f}%' if gap_pct is not None else 'N/A'}"
        f"  vs_BKS={f'{diff_bks:+.0f}' if diff_bks is not None else 'N/A'}"
        f"  vs_master={final_obj-master_obj:+.0f}"
        f"  RT={total_rt:.0f}s  stop={stop_reason}  phases={phase}")

    return {
        # identifiers
        "instance": label, "family": family,
        "n": meta["n"], "alpha": meta["alpha"], "group_id": meta["group_id"],
        # reference
        "bks": bks if bks is not None else "",
        "master_obj": master_obj, "master_rt_s": meta["master_rt"],
        "master_cert_gap_pct": meta["master_cert_gap"],
        "master_status": meta["master_status"],
        # adaptive results
        "beta_obj": final_obj, "beta_ub": ub if ub is not None else "",
        "beta_gap_pct": f"{gap_pct:.4f}" if gap_pct is not None else "",
        "beta_ec_gain": ec_gain, "beta_phases": phase,
        "beta_stop_reason": stop_reason, "beta_total_rt_s": f"{total_rt:.1f}",
        # certificate provenance
        "beta_raw_ub": f"{raw_ub:.4f}" if raw_ub is not None else "",
        "beta_bound_mode": bound_mode,
        "beta_cut_positive_gamma": sum(1 for v in (prev_gamma or {}).values() if v > 1e-9) if bound_mode.startswith("cutlag") else "",
        "beta_cut_positive_nu": sum(1 for v in (prev_nu or {}).values() if v > 1e-9) if bound_mode.startswith("cutlag") else "",
        "beta_num_ub_refreshes": num_ub_refreshes,
        "beta_num_lag_iters": num_lag_iters_total,
        "beta_no_refresh": int(no_refresh),
        # comparison
        "beta_vs_master": final_obj - master_obj,
        "beta_vs_bks": diff_bks if diff_bks is not None else "",
        "beta_status": beta_status,
        # internal
        "_phase_logs": phase_logs,
        "_output_lines": output_lines,
    }


# ── Worker init ───────────────────────────────────────────────────────────────
def _worker_init():
    sys.stdout = open(os.devnull, "w")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Adaptive ALNS full run (beta algo). Defaults reproduce the "
                    "canonical 660-instance run; flags enable the hard-tail "
                    "refresh experiment and certificate persistence.")
    parser.add_argument("--no-refresh", action="store_true",
                        help="H1a arm: skip the in-loop Lagrangian re-bound "
                             "(bound stays at its initial value).")
    parser.add_argument("--bound-mode", default="lag",
                        choices=["lag", "cutlag_cap", "cutlag_pair", "cutlag_triple"],
                        help="In-loop upper-bound refresh: original Lagrangian or cut-augmented Lagrangian.")
    parser.add_argument("--subset", default=None,
                        help="Path to a CSV with an 'instance' column; restrict "
                             "the run to those instances (e.g. results/hard_tail_70.csv).")
    parser.add_argument("--save-certificates", action="store_true",
                        help="Persist the dual certificate (multipliers + bound "
                             "provenance) per instance to results/beta_certificates/.")
    parser.add_argument("--tag", default=None,
                        help="Suffix for output filenames (e.g. 'hardtail_H0').")
    parser.add_argument("--save-suffix", default=None,
                        help="If set, write incumbents/certificates to "
                             "results/beta_incumbents<suffix>/ and "
                             "results/beta_certificates<suffix>/ instead of the "
                             "canonical dirs, while STILL warm-starting from the "
                             "canonical incumbents. Use for experiment arms so they "
                             "never overwrite the canonical artifact set.")
    cli = parser.parse_args()

    tag      = f"_{cli.tag}" if cli.tag else ""
    OUT_CSV  = os.path.join("results", f"adaptive_full_{TIMESTAMP}{tag}.csv")
    PHASES_CSV = os.path.join("results", f"adaptive_full_{TIMESTAMP}{tag}_phases.csv")

    instances = _load_master()

    # Optional subset restriction (e.g. the 70 hard-tail instances)
    if cli.subset:
        with open(cli.subset, encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            col = "instance" if "instance" in rdr.fieldnames else "instance_id"
            keep = {r[col].strip() for r in rdr}
        instances = [m for m in instances if m["label"] in keep]
        missing = keep - {m["label"] for m in instances}
        if missing:
            print(f"WARNING: {len(missing)} subset instances not in master/study set: "
                  f"{sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}")

    # Inject per-run config into each meta payload (pickled to spawned workers)
    write_inc_dir  = INCUMBENTS_DIR + cli.save_suffix if cli.save_suffix else INCUMBENTS_DIR
    write_cert_dir = CERTS_DIR + cli.save_suffix if cli.save_suffix else CERTS_DIR
    for m in instances:
        m["no_refresh"]     = cli.no_refresh
        m["bound_mode"]     = cli.bound_mode
        m["save_cert"]      = cli.save_certificates
        m["read_inc_dir"]   = INCUMBENTS_DIR          # always warm-start from canonical
        m["write_inc_dir"]  = write_inc_dir
        m["write_cert_dir"] = write_cert_dir

    n_total   = len(instances)
    os.makedirs(os.path.dirname(OUT_CSV) if os.path.dirname(OUT_CSV) else ".", exist_ok=True)

    arm = "H1a (NO in-loop refresh)" if cli.no_refresh else "H0 (full adaptive loop)"
    arm = f"{arm} | bound_mode={cli.bound_mode}"
    print("=" * 80)
    print(f"Adaptive ALNS full run — {n_total} instances  |  {N_WORKERS} workers")
    print(f"Arm     : {arm}")
    print(f"Families: {sorted(STUDY_FAMILIES)}")
    if cli.subset:
        print(f"Subset  : {cli.subset}")
    print(f"Certs   : {'ON → ' + write_cert_dir if cli.save_certificates else 'off'}")
    if cli.save_suffix:
        print(f"Inc dir : write → {write_inc_dir}  (warm-start read ← {INCUMBENTS_DIR})")
    print(f"Phase RT: {PHASE_RT:.0f}s | Abs cap: {ABS_CAP:.0f}s | Gap threshold: {GAP_THRESHOLD*100:.1f}%")
    print(f"Destroy tiers: {[f'{int(f*100)}%' for f in DESTROY_FRACS]}  | EC depth=3")
    print(f"Sorted by master RT desc (longest first) for load balancing")
    print(f"Output: {OUT_CSV}")
    print("=" * 80)

    summary_rows = []
    phase_rows   = []
    results      = []
    n_done       = 0
    t_wall       = time.perf_counter()

    # Open output files in append mode so we can write as results arrive
    summary_fields = [
        "instance", "family", "n", "alpha", "group_id",
        "bks", "master_obj", "master_rt_s", "master_cert_gap_pct", "master_status",
        "beta_obj", "beta_ub", "beta_gap_pct", "beta_ec_gain", "beta_phases",
        "beta_stop_reason", "beta_total_rt_s",
        "beta_raw_ub", "beta_bound_mode", "beta_cut_positive_gamma", "beta_cut_positive_nu", "beta_num_ub_refreshes", "beta_num_lag_iters", "beta_no_refresh",
        "beta_vs_master", "beta_vs_bks", "beta_status",
    ]
    phase_fields = [
        "instance", "family", "n", "alpha",
        "phase", "tier", "obj", "ub", "gap_pct", "seeds",
        "phase_rt", "cumul_rt", "event",
    ]

    with (open(OUT_CSV, "w", newline="", encoding="utf-8") as sf,
          open(PHASES_CSV, "w", newline="", encoding="utf-8") as pf):
        sw = csv.DictWriter(sf, fieldnames=summary_fields)
        pw = csv.DictWriter(pf, fieldnames=phase_fields)
        sw.writeheader()
        pw.writeheader()

        with mp.Pool(processes=N_WORKERS, initializer=_worker_init) as pool:
            for r in pool.imap_unordered(run_instance, instances, chunksize=1):
                n_done += 1
                elapsed_wall = time.perf_counter() - t_wall
                eta_s = (elapsed_wall / n_done) * (n_total - n_done) if n_done > 0 else 0

                # Print buffered instance output
                for line in r["_output_lines"]:
                    print(line, flush=True)

                # Progress line
                print(f"  [{n_done:>3}/{n_total}]  wall={elapsed_wall/60:.1f}min  "
                      f"ETA≈{eta_s/3600:.1f}h  stop={r['beta_stop_reason']}  "
                      f"phases={r['beta_phases']}  gap={r['beta_gap_pct']}%",
                      flush=True)

                # Write summary row
                row = {k: r[k] for k in summary_fields}
                sw.writerow(row)
                sf.flush()

                # Write phase rows
                for p in r["_phase_logs"]:
                    pw.writerow({
                        "instance": r["instance"], "family": r["family"],
                        "n": r["n"], "alpha": r["alpha"],
                        **{k: v for k, v in p.items()},
                    })
                pf.flush()

                results.append(r)

    # Final aggregate summary
    wall = time.perf_counter() - t_wall
    improved  = [r for r in results if r["beta_vs_master"] > 0]
    regressed = [r for r in results if r["beta_vs_master"] < 0]
    below_bks = [r for r in results if r["beta_vs_bks"] != "" and float(r["beta_vs_bks"]) < 0]
    bks_insts = [r for r in results if r["bks"] != ""]

    stop_counts = {}
    for r in results:
        stop_counts[r["beta_stop_reason"]] = stop_counts.get(r["beta_stop_reason"], 0) + 1

    gaps = [float(r["beta_gap_pct"]) for r in results if r["beta_gap_pct"] != ""]
    gaps.sort()

    print("\n" + "=" * 80)
    print(f"  FULL RUN SUMMARY — {n_total} instances  |  wall RT: {wall/3600:.2f}h")
    print(f"  Improved vs master    : {len(improved):>4} / {n_total}")
    print(f"  Regressed vs master   : {len(regressed):>4} / {n_total}")
    print(f"  Still below BKS       : {len(below_bks):>4} / {len(bks_insts)} (with BKS)")
    print(f"  Stop reasons: {stop_counts}")
    if gaps:
        import statistics
        print(f"  Cert gap — mean={statistics.mean(gaps):.3f}%  "
              f"median={statistics.median(gaps):.3f}%  "
              f"max={max(gaps):.3f}%")
        print(f"  Within 0.3%: {sum(1 for g in gaps if g < 0.3)}/{len(gaps)}")
        print(f"  Within 1.0%: {sum(1 for g in gaps if g < 1.0)}/{len(gaps)}")
        print(f"  Within 2.0%: {sum(1 for g in gaps if g < 2.0)}/{len(gaps)}")
    print(f"  Summary → {OUT_CSV}")
    print(f"  Phases  → {PHASES_CSV}")
    print("=" * 80)
