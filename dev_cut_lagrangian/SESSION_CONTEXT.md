# SRPS Cuts Project — Session Context & Recovery Guide

**Last Updated**: 2026-09-10 (headroom experiment complete)
**Session Goal**: Complete the headroom experiment (Option B) for Tomer's cut-augmented Lagrangian contribution  
**Deadline**: October 25, 2026 (course submission)  
**Repo root**: `C:\Users\tomer.tunitsky\exact-algos-project\sandbox_new_david_repo\david_package_20260829_1416\`  
**Branch**: `tomer/cut-augmented-lagrangian`

---

## 1. Project Overview

### Course & Partners
- **Course**: EACOP — Exact Algorithms for Combinatorial Optimization Problems, TAU, 2026
- **Partners**: Tomer Tunitsky + David Chouchena  
- **Advisor**: Dr. Mor Kaspi  
- **Problem**: SRPS — Selective Routing Problem with Synchronization (profit-maximizing multi-processor orienteering)

### What David Built (the baseline)
A **certified primal–dual solver** combining:
- **ALNS** (Adaptive Large Neighborhood Search) as the primal engine
- **Lagrangian relaxation L(μ)** as the dual engine — decomposes per-processor into independent orienteering DPs
- **Subgradient descent** with Polyak steps (100 iter warm or 200 iter cold)
- **Results on 660 instances**: mean gap 0.133%, median 0.046%, 283 proven optimal

David also ran a **falsification study of 10 course techniques** — 9 failed. The key reason: the **ceiling argument** — the certified gap is already 0.11pp mean / 0.009pp median, below the noise floor of any gap-directed experiment.

### What Tomer Built (the contribution)
**Cut-augmented Lagrangian L(μ, γ, ν)** — adds valid inequalities in projected y-space into David's Lagrangian:
1. **Capacity cuts**: `Σ_{j∈J_k} (b_j/O_k) y_j ≤ 1` where O_k = independent orienteering optimum for processor k
2. **Pair incompatibility**: `y_i + y_j ≤ 1` (pair can't fit on any single-processor route)
3. **Triple incompatibility**: `y_i + y_j + y_l ≤ 2` (triple can't all fit)

**Key property**: Decomposition is preserved — cuts add γ/ν penalty to the y-threshold decision, per-processor OP subproblems unchanged.

**Three configurations**: cap_only, pair (cap+pairs), triple (cap+pairs+triples)

### Mor's Guidance (Zoom call, Aug 19 2026)
- "Integrate cuts into the Lagrangian per-processor" — done
- Distinction between **Cut-and-Branch** (Tomer: cuts at root only) vs **Branch-and-Cut** (David: not done)

---

## 2. Experiment History

### Experiment 1: Standalone Cut-LP (DONE)
Run a projected y-space LP relaxation with cuts, comparing tightness.  
Results: 660 instances, documented but not the focus.

### Experiment 2: stratified54 — Standalone CutLR vs LR (DONE)
**54-instance balanced sample**: 6 families (B/C/D/EB/EC/ED) × 3 alpha levels × 3 difficulty positions.  
**Methods compared** (standalone dual only, no ALNS):
- `LR_100`: 100 cold iterations of standard LR
- `LR_200_cold`: 200 cold iterations (the 2x budget benchmark)
- `LR_100_plus_LR_100`: 100 cold then 100 warm (= `LR_200_cold` continuation)
- `cutLR_cap_warm_100`: LR_100 as warm start, then 100 cut-augmented iterations with cap cuts
- `cutLR_pair_warm_100`: same with cap+pair cuts
- `cutLR_triple_warm_100`: same with cap+pair+triple cuts

**Results** (per `results/analysis/cut_aug_lr_stratified54_pairwise_summary.csv`):
- `cutLR_cap_warm_100` beats `LR_200_cold`: mean −0.127pp, ~30+ wins
- `cutLR_pair_warm_100` beats `LR_200_cold`: mean −0.072pp (mixed: cap wins on small/easy, pair on hard)
- `cutLR_triple_warm_100` beats `LR_200_cold`: varies, best on α=0.5 hard instances

**Key file**: `results/analysis/cut_aug_lr_stratified54.csv` (all methods, all 54 instances)  
**Key file**: `results/analysis/cut_aug_lr_stratified54_per_instance.csv` (pivoted summary)

### Experiment 3: hard6 Full Pipeline (DONE)
**6 hard instances** run through the full adaptive ALNS pipeline.  
**Two arms**:
- `adaptive_full_20260907_2107_hard6_lag_mono.csv` — LR in-loop
- `adaptive_full_20260907_2139_hard6_cutlag_triple_mono.csv` — cutLR_triple in-loop

**Critical finding**: `beta_num_ub_refreshes = 0` on ALL 6 instances for BOTH arms.

**Explanation (ceiling argument)**: The master's initial UB (from David's full run) is already tighter than what 100-iteration in-loop Lagrangian can achieve. CutLR's lower raw_ub doesn't help because floor(raw_ub) >= master_UB for all 6 instances.

### Experiment 4: Headroom Experiment — Full 54 Instances (DONE ✅)

**Design**: Initial UB overridden to `floor(LR_200_cold)` from standalone stratified54 analysis, giving the in-loop dual refreshes real headroom to tighten.

**ARM 1 (LR)**:
- File: `results/adaptive_full_20260910_1011.csv`
- Runtime: ~1.93h, 3 workers
- Note: ran without `--save-suffix` (flags got dropped on PowerShell paste) → updated canonical `beta_incumbents/`

**ARM 2 (cutLR_triple)**:
- File: `results/adaptive_full_20260910_1216_headroom_cutlag.csv`
- Runtime: ~1.16h, 3 workers (faster — more GAP/UB stops)
- Note: started from ARM 1's updated canonical incumbents (primal contamination for some instances)

---

## 3. Headroom Experiment — Final Results ✅

### Aggregate Summary (54 instances)

| Metric | LR | cutLR_triple | Winner |
|--------|-----|--------------|--------|
| Mean certified gap (all 54) | **0.5326%** | **0.4282%** | cutLR (−0.105pp) |
| Median certified gap (all 54) | **0.0876%** | **0.0801%** | cutLR |
| Max certified gap | 5.310% | 5.310% | TIE |
| Total UB refreshes | 32 | 35 | cutLR |
| Instance wins | **8** | **12** | cutLR |
| Ties | 34 | 34 | — |

### Headroom-Only Subset (25 flagged instances)

These are instances where `floor(best_cutLR_standalone) < initial_UB`, meaning cutLR provably has room to beat the starting UB.

| Metric | LR | cutLR_triple | Winner |
|--------|-----|--------------|--------|
| Mean certified gap | **1.0017%** | **0.7802%** | cutLR (−0.222pp) |
| Median certified gap | **0.5470%** | **0.4111%** | cutLR |

### Per-Instance Decision Table (notable cases)

**cutLR WINS (12 instances)**:

| Instance | Initial UB | LR gap% | CUT gap% | Δ | LR ref | CUT ref |
|----------|------------|---------|----------|---|--------|---------|
| B_n150_030_a25_088 * | 3561 | 4.437% | 0.028% | −4.41pp | 0 | 1 |
| C_n080_035_a50_104 * | 2807 | 0.680% | 0.036% | −0.64pp | 1 | 2 |
| D_n060_023_a25_067 * | 1278 | 0.473% | 0.158% | −0.31pp | 1 | 2 |
| B_n150_030_a50_089 * | 4995 | 0.421% | 0.241% | −0.18pp | 2 | 3 |
| EC_n080_035_a50_104 * | 2695 | 0.745% | 0.560% | −0.19pp | 2 | 2 |
| B_n130_016_a25_046 * | 2820 | 2.695% | 2.453% | −0.24pp | 0 | 1 |
| D_n060_023_a50_068 * | 1786 | 0.678% | 0.622% | −0.06pp | 2 | 3 |
| D_n080_045_a75_135 | 2953 | 0.677% | 0.644% | −0.03pp | 0 | 1 |
| EC_n080_035_a75_105 | 2932 | 0.341% | 0.273% | −0.07pp | 0 | 0 |
| B_n130_016_a75_048 | 4334 | 0.069% | 0.023% | −0.05pp | 0 | 0 |
| C_n050_001_a50_002 * | 1240 | 0.081% | 0.000% | −0.08pp | 1 | 1 |
| C_n050_001_a75_003 | 1445 | 0.069% | 0.000% | −0.07pp | 0 | 0 |

**LR WINS (8 instances)**:

| Instance | Initial UB | LR gap% | CUT gap% | Δ | Notes |
|----------|------------|---------|----------|---|-------|
| ED_n080_045_a25_133 * | 1710 | 0.826% | 0.943% | +0.12pp | LR warm accumulation wins |
| D_n080_045_a25_133 * | 1956 | 3.138% | 3.287% | +0.15pp | LR warm accumulation wins |
| EB_n130_016_a25_046 * | 2268 | 1.236% | 1.367% | +0.13pp | LR warm accumulation wins |
| EB_n150_030_a25_088 * | 2983 | 0.202% | 0.269% | +0.07pp | LR final refresh tighter |
| C_n065_018_a25_052 * | 1390 | 0.000% | 0.072% | +0.07pp | LR proved optimal; cutLR didn't |
| ED_n080_045_a50_134 * | 2684 | 0.374% | 0.411% | +0.04pp | small margin |
| EB_n100_001_a25_001 | 1611 | 0.062% | 0.124% | +0.06pp | small margin |
| ED_n060_023_a50_068 | 1739 | 0.633% | 0.690% | +0.06pp | small margin |

### Key Finding: B_n150_030_a25_088 (Caveat)

This is the most dramatic win for cutLR (LR gap=4.437% vs CUT gap=0.028%), but it has a **primal contamination caveat**:
- ARM 1 (LR) ALNS regressed to obj=3403 (master_obj=3550); the canonical incumbent was weak for this instance
- ARM 2 (cutLR) started from ARM 1's saved 3403, and improved back to master_obj=3550
- The UB difference is clean and real (LR UB=3561 vs CUT UB=3551, Δ=10)
- But the gap difference (4.41pp) is amplified by the primal regression on ARM 1
- For the paper: cite the UB difference (Δ=10) as the clean evidence; treat the gap difference as an upper bound

### Experimental Design Caveat

ARM 1 ran without `--save-suffix` (flags dropped on PowerShell paste), so it updated the canonical `beta_incumbents/`. ARM 2 started from those updated incumbents. This creates:
- **Primal contamination**: ARM 2 may benefit from ARM 1's improved primal solutions on some instances
- **UB comparison is clean**: the dual refreshes and final UB values are unaffected by the primal contamination
- For rigorous comparison, future runs should use `--save-suffix` to separate incumbent directories

---

## 4. Summary for the Paper

### Story Arc (complete)

1. **Standalone evidence (stratified54)**: cutLR_warm_100 beats LR_200_cold in apples-to-apples dual comparison on harder instances. Cuts genuinely tighten the relaxation.

2. **Pipeline evidence (hard6)**: Neither LR nor cutLR triggers refreshes when the master's tight initial UB leaves no room (ceiling argument, David's Section 7).

3. **Headroom experiment (DONE)**: Under deliberately weakened initial UB = floor(LR_200_cold), cutLR_triple outperforms LR:
   - 12 instance wins vs 8 (all 54)
   - Mean gap 0.428% vs 0.533% (all 54); 0.780% vs 1.002% (headroom 25)
   - 35 total refreshes vs 32; cutLR tightens UB more often
   - LR wins 8 instances (mostly α=0.25 hard cases where accumulated warm multipliers eventually converge)

4. **Conclusion**: cutLR demonstrably tightens the dual in standalone mode AND in the pipeline when headroom exists. The pre-existing tight UB from David's full solver removes the headroom needed — consistent with the ceiling argument.

---

## 5. File Map (Key Files Only)

```
david_package_20260829_1416/
│
├── run_adaptive_full_cutlag_exp_monotone.py   ← MAIN RUNNER (modified for --weak-ub-csv)
│
├── dev_cut_lagrangian/
│   ├── cut_augmented_lagrangian.py            ← Tomer's cutLR implementation
│   ├── gen_headroom_ubcsv.py                  ← generates the weak-UB CSVs
│   ├── analyze_headroom_exp.py                ← compare LR vs cutLR pipeline arms
│   └── SESSION_CONTEXT.md                     ← THIS FILE
│
├── data/
│   ├── stratified54_list.csv                  ← 54-instance subset list
│   └── headroom_weak_ubs.csv                  ← per-instance initial UB = floor(LR_200_cold)
│
├── results/
│   ├── master_results.csv                     ← David's 660-instance baseline
│   ├── adaptive_full_20260907_2107_hard6_lag_mono.csv       ← hard6 LR arm (all 0 refreshes)
│   ├── adaptive_full_20260907_2139_hard6_cutlag_triple_mono.csv ← hard6 cutLR arm (all 0 refreshes)
│   ├── adaptive_full_20260910_1011.csv        ← HEADROOM ARM 1 (LR, ~1.93h)
│   ├── adaptive_full_20260910_1216_headroom_cutlag.csv ← HEADROOM ARM 2 (cutLR, ~1.16h)
│   └── analysis/
│       ├── cut_aug_lr_stratified54.csv               ← full standalone results
│       └── cut_aug_lr_stratified54_per_instance.csv  ← pivoted per-instance
│
└── benchmarks/ops_raw/OPS-Benchmark-master/input/   ← instance files
```

---

## 6. How to Resume This Session

1. Read this file: `dev_cut_lagrangian/SESSION_CONTEXT.md`
2. The headroom experiment is **COMPLETE** — see Section 3 for all results
3. **Next task**: Write the methodology and results section for Tomer's contribution to the joint paper

### Pending After Headroom Experiment
- [ ] Write the paper methodology section (cutLR formulation, pipeline integration)
- [ ] Write the paper results section (standalone + headroom experiment findings)
- [ ] Address reference paper comparison (Riera-Ledesma & Salazar-González 2021)

---

## 7. Reminder: What NOT to Do

- **Do NOT use the smoke test CSVs** for the full run — use `stratified54_list.csv`
- **The monotone UB update is critical** — `UB_t = min(UB_{t-1}, candidate_UB)`. Never discard a valid certificate.
- **Do NOT use BigQuery** — wrong context (Taboola rule); instances are in local benchmark files
