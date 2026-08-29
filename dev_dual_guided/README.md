# Dual-Guided ALNS Dev Project (Isolated)

This folder contains an isolated development runner for direct dual-guided neighborhood coupling.

## Safety guarantees

- No edits to production scripts or modules.
- All experimental code is under `dev_dual_guided/`.
- Outputs are written to `results/dual_guided_dev/`.

## What is added in this dev runner

- A dual-guided destroy operator that removes jobs with weak reduced-profit support.
- A dual-guided repair operator that prioritizes insertions by reduced-profit score adjusted by insertion cost.
- Multipliers are refreshed by the existing in-loop Lagrangian rebound and fed back into these operators.
- Every result row is compared to the baseline paper run (`adaptive_master.csv`) on:
  - objective value,
  - runtime,
  - certified optimality gap.

## Files

- `dev_dual_guided/dual_guided_ops.py`
  - dual-guided operator factories and multiplier utilities.
- `dev_dual_guided/run_dual_guided_experiment.py`
  - isolated experiment runner using baseline ALNS core + dual-guided moves.

## Run (recommended first step)

From repository root:

```powershell
python dev_dual_guided/run_dual_guided_experiment.py --tag smoke --workers 2
```

## Ablation switches (all additions are independently toggleable)

- `--dual-destroy` / `--no-dual-destroy`
- `--dual-repair` / `--no-dual-repair`
- `--dual-feedback` / `--no-dual-feedback`

Examples:

```powershell
# Baseline inside this dev runner (no dual-guided additions)
python dev_dual_guided/run_dual_guided_experiment.py --tag baseline --no-dual-destroy --no-dual-repair --no-dual-feedback

# Destroy only
python dev_dual_guided/run_dual_guided_experiment.py --tag d_only --dual-destroy --no-dual-repair --no-dual-feedback

# Repair only
python dev_dual_guided/run_dual_guided_experiment.py --tag r_only --no-dual-destroy --dual-repair --no-dual-feedback

# Full dual-guided coupling (default)
python dev_dual_guided/run_dual_guided_experiment.py --tag full --dual-destroy --dual-repair --dual-feedback
```

Optional custom subset:

```powershell
python dev_dual_guided/run_dual_guided_experiment.py --subset-csv results/hard_tail_70.csv --tag hardtail --workers 4
```

## Notes

- Default run targets the same 30-instance stratified subset used by sensitivity experiments.
- This is an experimental branch for algorithmic coupling; production behavior remains unchanged.
- CSV output includes switch states (`dual_destroy`, `dual_repair`, `dual_feedback`) for easy ablation tables.
- CSV output also includes baseline-comparison fields:
  - `ref_obj`, `ref_rt_s`, `ref_gap_pct`
  - `delta_ref` (objective delta), `delta_rt_s` (runtime delta), `delta_gap_pct` (gap delta in percentage points).
