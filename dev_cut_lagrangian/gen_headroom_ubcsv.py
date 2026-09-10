"""Generate the two CSVs needed for the headroom experiment:

  data/stratified54_list.csv      — 'instance' column (54 entries)
  data/headroom_weak_ubs.csv      — 'instance,weak_ub' using floor(LR_200_cold)

Run from the repo root:
    python dev_cut_lagrangian/gen_headroom_ubcsv.py
"""
import csv
import math
import os

STRAT54_MANIFEST = os.path.join("data", "relaxation_stratified_54.csv")
STRAT54_ANALYSIS = os.path.join("results", "analysis", "cut_aug_lr_stratified54.csv")
LIST_OUT  = os.path.join("data", "stratified54_list.csv")
WEAK_OUT  = os.path.join("data", "headroom_weak_ubs.csv")


def main():
    # ── 1. Load the 54 instance IDs ──────────────────────────────────────────
    instances = []
    with open(STRAT54_MANIFEST, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            instances.append(row["instance_id"].strip())
    assert len(instances) == 54, f"Expected 54, got {len(instances)}"
    print(f"Loaded {len(instances)} instances from {STRAT54_MANIFEST}")

    # ── 2. Extract LR_200_cold bound per instance ────────────────────────────
    lr200 = {}
    with open(STRAT54_ANALYSIS, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["method"].strip() == "LR_200_cold":
                iid = row["instance"].strip()
                lr200[iid] = float(row["upper_bound"])

    missing = [i for i in instances if i not in lr200]
    if missing:
        raise RuntimeError(f"LR_200_cold missing for: {missing}")
    print(f"Found LR_200_cold bounds for all {len(lr200)} instances")

    # ── 3. Write stratified54_list.csv ──────────────────────────────────────
    with open(LIST_OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["instance"])
        for i in instances:
            w.writerow([i])
    print(f"Written: {LIST_OUT}")

    # ── 4. Write headroom_weak_ubs.csv ───────────────────────────────────────
    with open(WEAK_OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["instance", "weak_ub", "lr200_raw"])
        w.writeheader()
        for i in instances:
            raw = lr200[i]
            floored = math.floor(raw)
            w.writerow({"instance": i, "weak_ub": floored, "lr200_raw": f"{raw:.6f}"})
    print(f"Written: {WEAK_OUT}")

    # ── 5. Quick sanity report ────────────────────────────────────────────────
    gaps_big = [(i, lr200[i]) for i in instances if lr200[i] - math.floor(lr200[i]) > 0.001]
    print(f"\nInstances with LR_200_cold gap > 0 (flooring will differ from raw): {len(gaps_big)}")
    big_gap_instances = [(i, lr200[i]) for i in instances if lr200[i] > math.floor(lr200[i]) + 0.5]
    print(f"Instances where floor(LR_200_cold) < LR_200_cold by >0.5: {len(big_gap_instances)}")

    # Show the 15 hardest instances (highest LR_200_cold value vs lb_incumbent)
    print("\nTop 10 instances by LR_200_cold raw value (hardest dual gap):")
    sorted_insts = sorted(instances, key=lambda i: lr200[i], reverse=True)
    for i in sorted_insts[:10]:
        print(f"  {i:40s}  LR_200_cold={lr200[i]:.4f}  weak_ub={math.floor(lr200[i])}")


if __name__ == "__main__":
    main()
