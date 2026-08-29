from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from adapters.ops_adapter import OPSInstance
from core.ops_bounds import lagrangian_bound
from dev_cut_lagrangian.cut_augmented_lagrangian import cut_augmented_lagrangian_bound


PRIMARY_FAMILIES = ["B", "C", "D", "EB", "EC", "ED"]
BENCH = os.path.join("benchmarks", "ops_raw", "OPS-Benchmark-master")


def gap_pct(lb: float, ub: float) -> float:
    if ub <= 0:
        return float("nan")
    return max(0.0, (ub - lb) / ub * 100.0)


def instance_path(family: str, label: str) -> str:
    return os.path.join(BENCH, "input", family, "instances", label + ".txt")


def timed_call(fn, *args, **kwargs):
    t0 = time.perf_counter()
    res = fn(*args, **kwargs)
    rt = time.perf_counter() - t0
    return res, rt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-csv", default="data/relaxation_stratified_54.csv")
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--out", default="results/analysis/cut_aug_lr_pilot.csv")
    parser.add_argument("--lag-max-time", type=float, default=30.0)
    parser.add_argument("--cut-max-time", type=float, default=30.0)
    parser.add_argument("--base-iters", type=int, default=100)
    parser.add_argument("--extra-iters", type=int, default=100)
    args = parser.parse_args()

    manifest = pd.read_csv("data/instance_manifest.csv")
    adaptive = pd.read_csv("results/adaptive_master.csv")

    if args.subset_csv:
        subset = pd.read_csv(args.subset_csv)
        if "instance_id" not in subset.columns and "instance" in subset.columns:
            subset = subset.rename(columns={"instance": "instance_id"})
        subset_ids = [str(x) for x in subset["instance_id"].tolist()]
        order = {label: i for i, label in enumerate(subset_ids)}
        candidates = manifest[manifest["instance_id"].astype(str).isin(order)].copy()
        candidates["__order"] = candidates["instance_id"].astype(str).map(order)
        candidates = candidates.sort_values("__order").drop(columns=["__order"]).reset_index(drop=True)
    else:
        candidates = manifest[manifest["family"].isin(PRIMARY_FAMILIES)].copy()
        candidates = candidates.sort_values(["family", "n", "alpha", "instance_id"]).reset_index(drop=True)

    if args.start:
        candidates = candidates.iloc[args.start:].reset_index(drop=True)
    if args.limit:
        candidates = candidates.head(args.limit)

    rows = []

    for idx, r in candidates.iterrows():
        label = str(r["instance_id"])
        family = str(r["family"])
        path = instance_path(family, label)

        print(f"[{idx + 1}/{len(candidates)}] {label}")

        inst = OPSInstance.from_instance_file(path)

        match = adaptive[adaptive["instance"] == label]
        if len(match) > 0 and not pd.isna(match.iloc[0].get("alns_obj", float("nan"))):
            lb = float(match.iloc[0]["alns_obj"])
        elif len(match) > 0 and not pd.isna(match.iloc[0].get("master_obj", float("nan"))):
            lb = float(match.iloc[0]["master_obj"])
        else:
            lb = 0.0

        bks = None
        if len(match) > 0 and "bks" in match.columns:
            try:
                bks = float(match.iloc[0]["bks"])
            except Exception:
                bks = None

        def add_row(method: str, ub: float, runtime_s: float, iterations, extra: str = ""):
            invalid_vs_lb = ub + 1e-6 < lb
            invalid_vs_bks = False if bks is None else ub + 1e-6 < bks

            rows.append({
                "instance": label,
                "family": family,
                "n": int(r["n"]),
                "alpha": float(r["alpha"]),
                "lb_incumbent": lb,
                "bks": bks,
                "method": method,
                "upper_bound": float(ub),
                "gap_pct": gap_pct(lb, float(ub)),
                "runtime_s": float(runtime_s),
                "iterations": iterations,
                "invalid_vs_lb": invalid_vs_lb,
                "invalid_vs_bks": invalid_vs_bks,
                "extra": extra,
            })

        # ------------------------------------------------------------------
        # 1. Baseline LR_100
        # ------------------------------------------------------------------
        lr100, rt100 = timed_call(
            lagrangian_bound,
            inst,
            max_iter=args.base_iters,
            lower_bound=lb,
            max_time=args.lag_max_time,
        )

        add_row(
            method=f"LR_{args.base_iters}",
            ub=lr100["upper_bound"],
            runtime_s=rt100,
            iterations=lr100["iterations"],
            extra=f"hist_len={len(lr100.get('bound_history', []))}",
        )

        # ------------------------------------------------------------------
        # 2. Cold LR_200
        # ------------------------------------------------------------------
        lr200, rt200 = timed_call(
            lagrangian_bound,
            inst,
            max_iter=args.base_iters + args.extra_iters,
            lower_bound=lb,
            max_time=2 * args.lag_max_time,
        )

        add_row(
            method=f"LR_{args.base_iters + args.extra_iters}_cold",
            ub=lr200["upper_bound"],
            runtime_s=rt200,
            iterations=lr200["iterations"],
            extra=f"hist_len={len(lr200.get('bound_history', []))}",
        )

        # ------------------------------------------------------------------
        # 3. Continue regular LR from LR_100 best_mu
        # ------------------------------------------------------------------
        lr_cont, rt_cont = timed_call(
            lagrangian_bound,
            inst,
            max_iter=args.extra_iters,
            lower_bound=lb,
            max_time=args.lag_max_time,
            mu_init=lr100["best_mu"],
        )

        lr_cont_ub = min(float(lr100["upper_bound"]), float(lr_cont["upper_bound"]))

        add_row(
            method=f"LR_{args.base_iters}_plus_LR_{args.extra_iters}",
            ub=lr_cont_ub,
            runtime_s=rt100 + rt_cont,
            iterations=lr100["iterations"] + lr_cont["iterations"],
            extra=f"base_ub={lr100['upper_bound']};cont_ub={lr_cont['upper_bound']}",
        )

        # ------------------------------------------------------------------
        # 4. Cut-augmented LR from LR_100 best_mu
        # ------------------------------------------------------------------
        cut_configs = [
            ("cutLR_cap_warm", dict(include_capacity=True, include_pairs=False, include_triples=False)),
            ("cutLR_pair_warm", dict(include_capacity=True, include_pairs=True, include_triples=False)),
            ("cutLR_triple_warm", dict(include_capacity=True, include_pairs=True, include_triples=True)),
        ]

        for name, kwargs in cut_configs:
            cut, rt_cut = timed_call(
                cut_augmented_lagrangian_bound,
                inst,
                max_iter=args.extra_iters,
                lower_bound=lb,
                max_time=args.cut_max_time,
                mu_init=lr100["best_mu"],
                cut_step_scale=1.0,
                projected_g_sq=True,
                **kwargs,
            )

            cut_ub = min(float(lr100["upper_bound"]), float(cut["upper_bound"]))

            add_row(
                method=f"{name}_{args.extra_iters}",
                ub=cut_ub,
                runtime_s=rt100 + rt_cut,
                iterations=lr100["iterations"] + cut["iterations"],
                extra=(
                    f"base_ub={lr100['upper_bound']};"
                    f"cut_ub={cut['upper_bound']};"
                    f"cap_cuts={cut.get('num_capacity_cuts')};"
                    f"incomp_cuts={cut.get('num_incompatibility_cuts')};"
                    f"pos_gamma={cut.get('positive_gamma')};"
                    f"pos_nu={cut.get('positive_nu')}"
                ),
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)

    print()
    print("Wrote", out)
    print()
    print(
        df.groupby("method")
          .agg(
              n=("instance", "count"),
              mean_gap_pct=("gap_pct", "mean"),
              median_gap_pct=("gap_pct", "median"),
              max_gap_pct=("gap_pct", "max"),
              mean_runtime_s=("runtime_s", "mean"),
              invalid_vs_lb=("invalid_vs_lb", "sum"),
              invalid_vs_bks=("invalid_vs_bks", "sum"),
          )
          .sort_values("mean_gap_pct")
          .to_string()
    )

    print()
    print("Invalid rows:")
    bad = df[df["invalid_vs_lb"] | df["invalid_vs_bks"]]
    if len(bad):
        print(bad[["instance", "method", "upper_bound", "lb_incumbent", "bks", "gap_pct"]].to_string(index=False))
    else:
        print("none")


if __name__ == "__main__":
    main()
