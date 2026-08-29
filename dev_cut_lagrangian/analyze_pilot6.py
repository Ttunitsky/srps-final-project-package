import pandas as pd
import re

path = "results/analysis/cut_aug_lr_pilot6.csv"
df = pd.read_csv(path)

pivot_gap = df.pivot_table(
    index=["instance", "family", "n", "alpha"],
    columns="method",
    values="gap_pct",
    aggfunc="first",
).reset_index()

pivot_ub = df.pivot_table(
    index=["instance", "family", "n", "alpha"],
    columns="method",
    values="upper_bound",
    aggfunc="first",
).reset_index()

methods = [
    "LR_100",
    "LR_200_cold",
    "LR_100_plus_LR_100",
    "cutLR_cap_warm_100",
    "cutLR_pair_warm_100",
    "cutLR_triple_warm_100",
]

print("Gap table:")
print(pivot_gap[["instance", "n", "alpha"] + methods].to_string(index=False))

print()
print("Mean gap deltas vs LR_100_plus_LR_100:")
base = "LR_100_plus_LR_100"
for m in ["LR_200_cold", "cutLR_cap_warm_100", "cutLR_pair_warm_100", "cutLR_triple_warm_100"]:
    delta = pivot_gap[base] - pivot_gap[m]
    print(f"{m:24s} mean_delta={delta.mean():.6f}, wins={(delta > 1e-9).sum()}/{len(delta)}")

print()
print("Mean gap deltas vs LR_200_cold:")
base = "LR_200_cold"
for m in ["cutLR_cap_warm_100", "cutLR_pair_warm_100", "cutLR_triple_warm_100"]:
    delta = pivot_gap[base] - pivot_gap[m]
    print(f"{m:24s} mean_delta={delta.mean():.6f}, wins={(delta > 1e-9).sum()}/{len(delta)}")

# Parse raw cut_ub/base_ub from the extra column.
cut = df[df["method"].str.startswith("cutLR")].copy()

def extract_float(text, key):
    if not isinstance(text, str):
        return None
    m = re.search(rf"{key}=([0-9.]+)", text)
    return float(m.group(1)) if m else None

cut["base_ub_raw"] = cut["extra"].apply(lambda s: extract_float(s, "base_ub"))
cut["cut_ub_raw"] = cut["extra"].apply(lambda s: extract_float(s, "cut_ub"))
cut["raw_cut_improves_base"] = cut["cut_ub_raw"] + 1e-6 < cut["base_ub_raw"]

print()
print("Raw cut run improves its LR_100 warm-start base:")
print(
    cut.groupby("method")["raw_cut_improves_base"]
       .agg(["sum", "count", "mean"])
       .to_string()
)

print()
print("Raw cut UB details:")
print(
    cut[["instance", "method", "base_ub_raw", "cut_ub_raw", "upper_bound", "gap_pct", "raw_cut_improves_base"]]
    .sort_values(["instance", "method"])
    .to_string(index=False)
)
