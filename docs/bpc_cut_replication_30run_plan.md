# BPC Cut Replication: Full-30 Run and Calibration Plan

## Goal
Build and evaluate a mathematically grounded replication of BPC cut generation/separation logic (MIS-style synchronization infeasibility cuts) inside the heuristic workflow, then calibrate low-level implementation parameters on the canonical 30-instance subset.

## Scope and constraints
- Scope: 30-instance stratified subset already used in sensitivity experiments.
- Baseline comparator: results/adaptive_master.csv reference columns (objective/runtime/certified gap).
- Required outcome metrics:
  - mean delta objective (vs baseline)
  - mean delta runtime
  - mean delta certified gap
  - max certified gap
  - share with cert gap < 0.2%
  - stop-reason distribution

## Implementation milestones (before calibration)

### M1. Explicit cut model layer
Create a new isolated module tree:
- dev_bpc_replica/cut_model.py
- dev_bpc_replica/separation.py
- dev_bpc_replica/run_bpc_replica_experiment.py

Core responsibilities:
1. Represent active cut set C with metadata:
   - cut_id
   - cut_type (circuit / over-length path)
   - support (nodes/edges)
   - rhs/lhs values
   - violation score
   - age and hit_count
2. Evaluate cut violation for:
   - current incumbent
   - candidate insertions/removals in repair/destroy
3. Maintain deterministic serialization of cuts per instance for reproducibility.

### M2. Separation procedures (mathematical replication intent)
Implement two separation routines aligned to paper logic:
1. Circuit-style infeasibility detection in synchronization support graph.
2. Path-length infeasibility detection (effective route-time over L under synchronization).

Operational behavior per separation call:
- Detect violated cuts with tolerance epsilon_violation.
- Rank by violation magnitude.
- Add top-k cuts (k = max_cuts_per_round).
- De-duplicate by canonical cut signature.

### M3. Heuristic integration policy
Integrate cut layer through two channels:
1. Hard-feasibility filter: reject candidate moves violating active cuts above epsilon_reject.
2. Soft penalty: score -= lambda_cut * violation for near-violating moves.

Add toggles:
- --bpc-cut-destroy / --no-bpc-cut-destroy
- --bpc-cut-repair / --no-bpc-cut-repair
- --bpc-cut-hard-filter / --no-bpc-cut-hard-filter

### M4. Telemetry and artifact schema
Per-run CSV must include:
- cut_gen_calls
- cuts_added_total
- cuts_active_peak
- mean_violation_before
- mean_violation_after
- cut_filter_rejections
- cut_penalty_hits
- all existing delta fields (delta_ref, delta_rt_s, delta_gap_pct)

## Calibration parameter set

Primary parameters to tune:
1. sep_freq_phase: {1, 2, 3}
2. max_cuts_per_round: {5, 10, 20}
3. epsilon_violation: {1e-6, 1e-5, 1e-4}
4. epsilon_reject: {1e-5, 1e-4, 1e-3}
5. lambda_cut: {0.1, 0.25, 0.5, 1.0}
6. cut_age_limit (phases): {2, 4, 8}
7. cut_weight_decay: {0.0, 0.05, 0.1}
8. sep_budget_s_per_phase: {5, 10, 20}

Fixed run controls for comparable calibration:
- phase_rt = 300
- abs_cap = 1800
- gap_threshold = 0.003
- workers = 4

## 30-instance series design

### Stage A: Correctness and stability gate (3 runs on all 30 instances)
Runs:
1. bpc_control (all BPC-cut toggles off)
2. bpc_hard_only (hard filter on, soft penalty off)
3. bpc_soft_only (hard filter off, soft penalty on)

Go/No-Go criteria:
- zero crashes
- zero invalid solutions
- telemetry columns complete
- runtime inflation <= +20% vs control

### Stage B: Coarse calibration (12 runs on all 30 instances)
Use a structured subset of parameter combinations (orthogonal-style coverage):
- 12 configurations spanning low/medium/high for sep_freq_phase, max_cuts_per_round, lambda_cut, and sep_budget.

Selection rule to Stage C:
- keep top 4 by lexicographic objective:
  1) lower mean_delta_gap_pp
  2) lower max_final_gap_pct
  3) lower mean_delta_rt_s

### Stage C: Focused refinement (8 runs on all 30 instances)
- Local neighborhood tuning around top-4 from Stage B.
- Two nearby variants per finalist (tighten epsilon, adjust lambda/age).

Selection rule to Stage D:
- keep top 2 Pareto candidates (gap quality vs runtime).

### Stage D: Robustness confirmation (6 runs on all 30 instances)
- top 2 finalists
- 3 seed-bundles each (seed-set perturbation for robustness)

Final selection:
- choose single recommended configuration by robust mean rank:
  - mean_delta_gap_pp
  - max_final_gap_pct
  - share gap<0.2%
  - runtime delta

Total planned runs: 29 full-30 runs.

## Estimated compute
Assuming ~55 minutes per full-30 run at 4 workers (observed ballpark with 1800s cap):
- 29 runs ~ 26.5 compute-hours wall-clock serial.
- With 2 parallel runners on separate machines/containers: ~13-14 hours.

## Deliverables
1. results/analysis/bpc_replica_manifest_<ts>.csv
2. results/analysis/bpc_replica_summary_<ts>.csv
3. results/analysis/bpc_replica_report_<ts>.txt
4. results/analysis/bpc_replica_pareto_<ts>.csv
5. results/analysis/bpc_replica_recommendation_<ts>.md

## Success criteria (acceptance-oriented)
A candidate is considered acceptable if on the 30-instance set it achieves:
- mean final gap < 0.2%
- max final gap < 2.0%
- no increase in TIERS_EXHAUSTED share vs control
- runtime penalty not worse than +10% vs control (or improved)

## Immediate next implementation order
1. Implement M1 and M2 (cut model + separation).
2. Add M3 integration toggles in run_bpc_replica_experiment.py.
3. Add telemetry fields (M4).
4. Run Stage A and publish first manifest.
