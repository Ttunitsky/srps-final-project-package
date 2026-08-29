from __future__ import annotations

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


FAMS = ["B", "C", "D", "EB", "EC", "ED"]
BENCH = os.path.join("benchmarks", "ops_raw", "OPS-Benchmark-master")

manifest = pd.read_csv("data/instance_manifest.csv")
adaptive = pd.read_csv("results/adaptive_master.csv")

row = manifest[manifest["family"].isin(FAMS)].iloc[0]
label = str(row["instance_id"])
family = str(row["family"])

path = os.path.join(BENCH, "input", family, "instances", label + ".txt")
inst = OPSInstance.from_instance_file(path)

match = adaptive[adaptive["instance"] == label]
lb = float(match.iloc[0]["alns_obj"]) if len(match) else 0.0

print("instance:", label)
print("family:", family)
print("LB:", lb)
print()

def gap_pct(ub: float) -> float:
    return max(0.0, (ub - lb) / ub * 100.0) if ub > 0 else float("nan")

def show(name: str, res: dict, rt: float) -> None:
    ub = float(res["upper_bound"])
    print(
        f"{name:>24}: "
        f"UB={ub:.4f}, gap={gap_pct(ub):.4f}%, "
        f"iters={res.get('iterations')}, time={rt:.3f}s, "
        f"cap_cuts={res.get('num_capacity_cuts', '')}, "
        f"incomp_cuts={res.get('num_incompatibility_cuts', '')}, "
        f"pos_gamma={res.get('positive_gamma', '')}, "
        f"pos_nu={res.get('positive_nu', '')}"
    )

# --- Original LR baselines -------------------------------------------------
baseline = {}

for iters in [50, 100]:
    t0 = time.perf_counter()
    res = lagrangian_bound(inst, max_iter=iters, lower_bound=lb, max_time=30)
    rt = time.perf_counter() - t0
    baseline[iters] = res
    show(f"LR_{iters}", res, rt)

print()

# --- Cold cut-augmented LR -------------------------------------------------
configs = [
    ("cutLR_cap_cold", dict(include_capacity=True, include_pairs=False, include_triples=False)),
    ("cutLR_pair_cold", dict(include_capacity=True, include_pairs=True, include_triples=False)),
    ("cutLR_triple_cold", dict(include_capacity=True, include_pairs=True, include_triples=True)),
]

for name, kwargs in configs:
    t0 = time.perf_counter()
    res = cut_augmented_lagrangian_bound(
        inst,
        max_iter=100,
        lower_bound=lb,
        max_time=30,
        mu_init=None,
        cut_step_scale=1.0,
        projected_g_sq=True,
        **kwargs,
    )
    rt = time.perf_counter() - t0
    show(name, res, rt)

print()

# --- Warm cut-augmented LR from original LR_100 best_mu --------------------
warm_mu = baseline[100]["best_mu"]

configs = [
    ("cutLR_cap_warm", dict(include_capacity=True, include_pairs=False, include_triples=False)),
    ("cutLR_pair_warm", dict(include_capacity=True, include_pairs=True, include_triples=False)),
    ("cutLR_triple_warm", dict(include_capacity=True, include_pairs=True, include_triples=True)),
]

for name, kwargs in configs:
    t0 = time.perf_counter()
    res = cut_augmented_lagrangian_bound(
        inst,
        max_iter=100,
        lower_bound=lb,
        max_time=30,
        mu_init=warm_mu,
        cut_step_scale=1.0,
        projected_g_sq=True,
        **kwargs,
    )
    rt = time.perf_counter() - t0
    show(name, res, rt)

print()
print("Validity check note: every UB should be >= LB.")
