import pandas as pd
import re
from pathlib import Path

IN = Path("results/analysis/cut_aug_lr_stratified54.csv")
OUT_DIR = Path("results/analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(IN)

for col in ["upper_bound", "gap_pct", "runtime_s", "lb_incumbent", "bks"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

cut_methods = [
    "cutLR_cap_warm_100",
    "cutLR_pair_warm_100",
    "cutLR_triple_warm_100",
]

baseline_methods = [
    "LR_100",
    "LR_200_cold",
    "LR_100_plus_LR_100",
]

key_cols = ["instance", "family", "n", "alpha"]

# Keep metadata separately, because bks may be NaN and should not be used
# as a pivot index.
meta = (
    df.groupby(key_cols, as_index=False)
      .agg(
          lb_incumbent=("lb_incumbent", "first"),
          bks=("bks", "first"),
      )
)

gap_pivot = (
    df.pivot_table(
        index=key_cols,
        columns="method",
        values="gap_pct",
        aggfunc="first",
        dropna=False,
    )
    .reset_index()
)

ub_pivot = (
    df.pivot_table(
        index=key_cols,
        columns="method",
        values="upper_bound",
        aggfunc="first",
        dropna=False,
    )
    .reset_index()
)

gap_wide = meta.merge(gap_pivot, on=key_cols, how="left")
ub_wide = meta.merge(ub_pivot, on=key_cols, how="left")

expected_methods = baseline_methods + cut_methods
missing_cols = [m for m in expected_methods if m not in gap_wide.columns]
if missing_cols:
    raise RuntimeError(f"Missing method columns after pivot: {missing_cols}")

# Best cutLR per instance.
gap_wide["best_cutLR_method"] = gap_wide[cut_methods].idxmin(axis=1)
gap_wide["best_cutLR_gap"] = gap_wide[cut_methods].min(axis=1)

ub_wide["best_cutLR_method"] = ub_wide[cut_methods].idxmin(axis=1)
ub_wide["best_cutLR_ub"] = ub_wide[cut_methods].min(axis=1)

# Deltas: positive means cutLR is better.
for base in ["LR_100", "LR_200_cold", "LR_100_plus_LR_100"]:
    gap_wide[f"best_cutLR_delta_vs_{base}"] = gap_wide[base] - gap_wide["best_cutLR_gap"]

for m in cut_methods:
    gap_wide[f"{m}_delta_vs_LR200"] = gap_wide["LR_200_cold"] - gap_wide[m]
    gap_wide[f"{m}_delta_vs_LR100_continue"] = gap_wide["LR_100_plus_LR_100"] - gap_wide[m]

# ---------------------------------------------------------------------
# Raw cut UB from extra column
# ---------------------------------------------------------------------
def extract_float(text, key):
    if not isinstance(text, str):
        return None
    m = re.search(rf"{key}=([0-9.]+)", text)
    return float(m.group(1)) if m else None

def extract_int(text, key):
    if not isinstance(text, str):
        return None
    m = re.search(rf"{key}=([0-9]+)", text)
    return int(m.group(1)) if m else None

cut_df = df[df["method"].isin(cut_methods)].copy()

cut_df["base_ub_raw"] = cut_df["extra"].apply(lambda s: extract_float(s, "base_ub"))
cut_df["cut_ub_raw"] = cut_df["extra"].apply(lambda s: extract_float(s, "cut_ub"))
cut_df["cap_cuts"] = cut_df["extra"].apply(lambda s: extract_int(s, "cap_cuts"))
cut_df["incomp_cuts"] = cut_df["extra"].apply(lambda s: extract_int(s, "incomp_cuts"))
cut_df["pos_gamma"] = cut_df["extra"].apply(lambda s: extract_int(s, "pos_gamma"))
cut_df["pos_nu"] = cut_df["extra"].apply(lambda s: extract_int(s, "pos_nu"))

cut_df["raw_cut_improves_base"] = cut_df["cut_ub_raw"] + 1e-6 < cut_df["base_ub_raw"]

raw_summary = (
    cut_df.groupby("method")
      .agg(
          n=("instance", "count"),
          raw_improves_base_count=("raw_cut_improves_base", "sum"),
          raw_improves_base_pct=("raw_cut_improves_base", "mean"),
          mean_cap_cuts=("cap_cuts", "mean"),
          mean_incomp_cuts=("incomp_cuts", "mean"),
          mean_pos_gamma=("pos_gamma", "mean"),
          mean_pos_nu=("pos_nu", "mean"),
      )
      .reset_index()
)

raw_summary["raw_improves_base_pct"] *= 100

# ---------------------------------------------------------------------
# Pairwise summary
# ---------------------------------------------------------------------
rows = []

def add_row(comparison, method, base, delta_series):
    delta_series = delta_series.dropna()
    rows.append({
        "comparison": comparison,
        "method": method,
        "base": base,
        "n": len(delta_series),
        "wins": int((delta_series > 1e-9).sum()),
        "ties": int((delta_series.abs() <= 1e-9).sum()),
        "losses": int((delta_series < -1e-9).sum()),
        "mean_delta_gap_points": float(delta_series.mean()),
        "median_delta_gap_points": float(delta_series.median()),
        "max_positive_delta": float(delta_series.max()),
        "max_negative_delta": float(delta_series.min()),
    })

for base in ["LR_200_cold", "LR_100_plus_LR_100"]:
    add_row(
        comparison="best_cutLR_vs_base",
        method="best_cutLR",
        base=base,
        delta_series=gap_wide[base] - gap_wide["best_cutLR_gap"],
    )

for m in cut_methods:
    for base in ["LR_200_cold", "LR_100_plus_LR_100"]:
        add_row(
            comparison="single_cutLR_vs_base",
            method=m,
            base=base,
            delta_series=gap_wide[base] - gap_wide[m],
        )

pairwise = pd.DataFrame(rows)

# ---------------------------------------------------------------------
# By family / alpha
# ---------------------------------------------------------------------
by_family = (
    gap_wide.groupby("family")
      .agg(
          n=("instance", "count"),
          mean_LR100=("LR_100", "mean"),
          mean_LR200=("LR_200_cold", "mean"),
          mean_LR100_continue=("LR_100_plus_LR_100", "mean"),
          mean_best_cutLR=("best_cutLR_gap", "mean"),
          wins_best_vs_LR200=("best_cutLR_delta_vs_LR_200_cold", lambda s: int((s > 1e-9).sum())),
          wins_best_vs_LR100_continue=("best_cutLR_delta_vs_LR_100_plus_LR_100", lambda s: int((s > 1e-9).sum())),
          mean_delta_best_vs_LR200=("best_cutLR_delta_vs_LR_200_cold", "mean"),
          mean_delta_best_vs_LR100_continue=("best_cutLR_delta_vs_LR_100_plus_LR_100", "mean"),
      )
      .reset_index()
)

by_alpha = (
    gap_wide.groupby("alpha")
      .agg(
          n=("instance", "count"),
          mean_LR100=("LR_100", "mean"),
          mean_LR200=("LR_200_cold", "mean"),
          mean_LR100_continue=("LR_100_plus_LR_100", "mean"),
          mean_best_cutLR=("best_cutLR_gap", "mean"),
          wins_best_vs_LR200=("best_cutLR_delta_vs_LR_200_cold", lambda s: int((s > 1e-9).sum())),
          wins_best_vs_LR100_continue=("best_cutLR_delta_vs_LR_100_plus_LR_100", lambda s: int((s > 1e-9).sum())),
          mean_delta_best_vs_LR200=("best_cutLR_delta_vs_LR_200_cold", "mean"),
          mean_delta_best_vs_LR100_continue=("best_cutLR_delta_vs_LR_100_plus_LR_100", "mean"),
      )
      .reset_index()
)

# ---------------------------------------------------------------------
# Optional: compare to standalone Cut-LP if previous CSV exists
# ---------------------------------------------------------------------
standalone_path = Path("results/relaxation_suite/stratified_54_full_suite.csv")
standalone_compare = None

if standalone_path.exists():
    st = pd.read_csv(standalone_path)
    for col in ["gap_pct", "upper_bound", "runtime_s"]:
        if col in st.columns:
            st[col] = pd.to_numeric(st[col], errors="coerce")

    st = st[st["method"].isin(["cut_lp_cap_only", "cut_lp_pair", "cut_lp_triple", "lagrangian_100"])]

    st_wide = (
        st.pivot_table(
            index=key_cols,
            columns="method",
            values="gap_pct",
            aggfunc="first",
            dropna=False,
        )
        .reset_index()
    )

    standalone_compare = gap_wide.merge(
        st_wide,
        on=key_cols,
        how="left",
    )

    standalone_compare["standalone_best_cutLP_gap"] = standalone_compare[
        ["cut_lp_cap_only", "cut_lp_pair", "cut_lp_triple"]
    ].min(axis=1)

    standalone_compare["delta_standalone_cutLP_vs_best_cutLR"] = (
        standalone_compare["best_cutLR_gap"] - standalone_compare["standalone_best_cutLP_gap"]
    )

# ---------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------
gap_wide.to_csv(OUT_DIR / "cut_aug_lr_stratified54_per_instance.csv", index=False)
pairwise.to_csv(OUT_DIR / "cut_aug_lr_stratified54_pairwise_summary.csv", index=False)
by_family.to_csv(OUT_DIR / "cut_aug_lr_stratified54_by_family.csv", index=False)
by_alpha.to_csv(OUT_DIR / "cut_aug_lr_stratified54_by_alpha.csv", index=False)
raw_summary.to_csv(OUT_DIR / "cut_aug_lr_stratified54_raw_multiplier_summary.csv", index=False)

if standalone_compare is not None:
    standalone_compare.to_csv(OUT_DIR / "cut_aug_lr_stratified54_vs_standalone_cutlp.csv", index=False)

# ---------------------------------------------------------------------
# Print key outputs
# ---------------------------------------------------------------------
print("Sanity counts:")
print("raw rows:", len(df))
print("unique instances in raw:", df["instance"].nunique())
print("instances after pivot:", len(gap_wide))
print("by_family n sum:", int(by_family["n"].sum()))
print("by_alpha n sum:", int(by_alpha["n"].sum()))

print()
print("Main method summary:")
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
print("Pairwise wins / deltas:")
print(pairwise.to_string(index=False))

print()
print("By family:")
print(by_family.to_string(index=False))

print()
print("By alpha:")
print(by_alpha.to_string(index=False))

print()
print("Raw cut multiplier summary:")
print(raw_summary.to_string(index=False))

if standalone_compare is not None:
    print()
    print("Standalone Cut-LP comparison:")
    print(
        standalone_compare[
            [
                "LR_100",
                "LR_200_cold",
                "LR_100_plus_LR_100",
                "best_cutLR_gap",
                "standalone_best_cutLP_gap",
                "delta_standalone_cutLP_vs_best_cutLR",
            ]
        ].mean(numeric_only=True).to_string()
    )

print()
print("Wrote corrected analysis CSVs under results/analysis.")
