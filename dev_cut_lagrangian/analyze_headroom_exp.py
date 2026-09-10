"""Analyze the headroom experiment: compare LR vs cutLR_triple in the full pipeline
when the initial UB is set to floor(LR_200_cold) (giving real headroom for refreshes).

Usage:
    python dev_cut_lagrangian/analyze_headroom_exp.py \
        --lr-csv  results/adaptive_full_YYYYMMDD_HHMM_headroom_lr.csv \
        --cut-csv results/adaptive_full_YYYYMMDD_HHMM_headroom_cutlag.csv \
        --weak-ub-csv data/headroom_weak_ubs.csv \
        --standalone-csv results/analysis/cut_aug_lr_stratified54.csv
"""
import argparse
import csv
import math
import statistics


def load_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr-csv",  required=True, help="LR arm pipeline output CSV")
    parser.add_argument("--cut-csv", required=True, help="cutLR arm pipeline output CSV")
    parser.add_argument("--weak-ub-csv", default="data/headroom_weak_ubs.csv")
    parser.add_argument("--standalone-csv",
                        default="results/analysis/cut_aug_lr_stratified54.csv")
    args = parser.parse_args()

    lr_rows  = {r["instance"]: r for r in load_csv(args.lr_csv)}
    cut_rows = {r["instance"]: r for r in load_csv(args.cut_csv)}
    weak_ubs = {r["instance"]: int(r["weak_ub"]) for r in load_csv(args.weak_ub_csv)}

    # Load best standalone cutLR bound per instance
    standalone_best = {}
    for r in load_csv(args.standalone_csv):
        iid = r["instance"]
        if r["method"].startswith("cutLR"):
            ub = float(r["upper_bound"])
            if iid not in standalone_best or ub < standalone_best[iid]:
                standalone_best[iid] = ub

    common = sorted(set(lr_rows) & set(cut_rows))
    print(f"\nHeadroom Experiment: {len(common)} instances in common\n")

    # ── Per-instance comparison ───────────────────────────────────────────────
    header = (f"{'Instance':40s} {'Init UB':>8} "
              f"{'LR gap%':>7} {'LR ref':>6} "
              f"{'CUT gap%':>8} {'CUT ref':>7} "
              f"{'Δgap':>8}  Status")
    print(header)
    print("─" * len(header))

    wins_cut = 0
    wins_lr  = 0
    ties     = 0
    cut_refreshes_total = 0
    lr_refreshes_total  = 0
    instances_with_headroom = 0  # floor(best_cutLR_standalone) < weak_ub

    lr_gaps_all  = []
    cut_gaps_all = []
    lr_gaps_headroom  = []
    cut_gaps_headroom = []

    for iid in common:
        lr  = lr_rows[iid]
        cut = cut_rows[iid]
        weak_ub = weak_ubs.get(iid)

        lr_gap  = float(lr["beta_gap_pct"])  if lr["beta_gap_pct"]  else None
        cut_gap = float(cut["beta_gap_pct"]) if cut["beta_gap_pct"] else None
        lr_ref  = int(lr["beta_num_ub_refreshes"])
        cut_ref = int(cut["beta_num_ub_refreshes"])

        lr_refreshes_total  += lr_ref
        cut_refreshes_total += cut_ref

        # Check if this is a headroom instance: standalone cutLR can beat weak_ub
        best_cut_standalone = standalone_best.get(iid)
        has_headroom = (
            best_cut_standalone is not None
            and weak_ub is not None
            and math.floor(best_cut_standalone) < weak_ub
        )
        if has_headroom:
            instances_with_headroom += 1

        if lr_gap is not None:
            lr_gaps_all.append(lr_gap)
        if cut_gap is not None:
            cut_gaps_all.append(cut_gap)
        if has_headroom:
            if lr_gap is not None:
                lr_gaps_headroom.append(lr_gap)
            if cut_gap is not None:
                cut_gaps_headroom.append(cut_gap)

        if lr_gap is None or cut_gap is None:
            status = "?"
        elif cut_gap < lr_gap - 1e-6:
            status = "CUT_WINS"
            wins_cut += 1
        elif lr_gap < cut_gap - 1e-6:
            status = "LR_WINS"
            wins_lr += 1
        else:
            status = "TIE"
            ties += 1

        hdroom_marker = " *" if has_headroom else "  "
        delta = (cut_gap - lr_gap) if (lr_gap is not None and cut_gap is not None) else None
        delta_s = f"{delta:+.4f}" if delta is not None else "    ?"

        print(f"{iid:40s}{hdroom_marker} {weak_ub if weak_ub else '?':>7} "
              f" {lr_gap:>6.4f}%  {lr_ref:>5} "
              f"  {cut_gap:>7.4f}%  {cut_ref:>6} "
              f" {delta_s:>8}  {status}")

    print(f"\n{'─'*80}")
    print(f"(* = headroom instance: floor(best_cutLR_standalone) < initial UB)")
    print()

    # ── Aggregate summary ─────────────────────────────────────────────────────
    print("=" * 60)
    print("AGGREGATE RESULTS")
    print("=" * 60)
    print(f"  Total instances:       {len(common)}")
    print(f"  Headroom instances:    {instances_with_headroom}")
    print()
    print(f"  UB Refreshes (all 54):")
    print(f"    LR  total refreshes:    {lr_refreshes_total}")
    print(f"    CUT total refreshes:    {cut_refreshes_total}")
    print()
    print(f"  Final certified gap (all {len(common)} instances):")
    if lr_gaps_all:
        print(f"    LR   — mean={statistics.mean(lr_gaps_all):.4f}%  "
              f"median={statistics.median(lr_gaps_all):.4f}%  "
              f"max={max(lr_gaps_all):.4f}%")
    if cut_gaps_all:
        print(f"    CUT  — mean={statistics.mean(cut_gaps_all):.4f}%  "
              f"median={statistics.median(cut_gaps_all):.4f}%  "
              f"max={max(cut_gaps_all):.4f}%")
    print()
    print(f"  Final certified gap (headroom {instances_with_headroom} instances):")
    if lr_gaps_headroom:
        print(f"    LR   — mean={statistics.mean(lr_gaps_headroom):.4f}%  "
              f"median={statistics.median(lr_gaps_headroom):.4f}%")
    if cut_gaps_headroom:
        print(f"    CUT  — mean={statistics.mean(cut_gaps_headroom):.4f}%  "
              f"median={statistics.median(cut_gaps_headroom):.4f}%")
    print()
    print(f"  Instance-level outcomes:")
    print(f"    CUT wins: {wins_cut}")
    print(f"    LR  wins: {wins_lr}")
    print(f"    Ties:     {ties}")
    print("=" * 60)


if __name__ == "__main__":
    main()
