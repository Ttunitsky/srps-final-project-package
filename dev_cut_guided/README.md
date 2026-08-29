# Cut-Guided ALNS Dev Project (Isolated)

This folder contains an isolated development runner for cut-informed neighborhood coupling inspired by the published SRPS branch-and-price-and-cut model.

## Safety guarantees

- No edits to production scripts or modules.
- All experimental code is under `dev_cut_guided/`.
- Outputs are written to `results/cut_guided_dev/`.

## What is added in this dev runner

- A cut-guided destroy operator that removes jobs on or near the current critical synchronization path.
- A cut-guided repair operator that prioritizes insertions which reduce critical-path pressure.
- The existing Lagrangian rebound remains available as the underlying ALNS-LR backbone.
- Every result row is compared to the baseline paper run (`adaptive_master.csv`) on:
  - objective value,
  - runtime,
  - certified optimality gap.

## Files

- `dev_cut_guided/cut_guided_ops.py`
  - cut-state extraction and cut-guided operator factories.
- `dev_cut_guided/run_cut_guided_experiment.py`
  - isolated experiment runner using baseline ALNS core + cut-guided moves.

## Run (recommended first step)

From repository root:

```powershell
python dev_cut_guided/run_cut_guided_experiment.py --tag smoke --workers 2 --subset-csv dev_cut_guided/smoke_one_instance.csv
```

## Ablation switches (all additions are independently toggleable)

- `--cut-destroy` / `--no-cut-destroy`
- `--cut-repair` / `--no-cut-repair`

Examples:

```powershell
# Baseline inside this dev runner (no cut-guided additions)
python dev_cut_guided/run_cut_guided_experiment.py --tag baseline --subset-csv dev_cut_guided/smoke_one_instance.csv --no-cut-destroy --no-cut-repair

# Destroy only
python dev_cut_guided/run_cut_guided_experiment.py --tag d_only --subset-csv dev_cut_guided/smoke_one_instance.csv --cut-destroy --no-cut-repair

# Repair only
python dev_cut_guided/run_cut_guided_experiment.py --tag r_only --subset-csv dev_cut_guided/smoke_one_instance.csv --no-cut-destroy --cut-repair

# Full cut-guided coupling (default)
python dev_cut_guided/run_cut_guided_experiment.py --tag full --subset-csv dev_cut_guided/smoke_one_instance.csv --cut-destroy --cut-repair
```

## Notes

- Default run targets the same 30-instance stratified subset used by sensitivity experiments.
- This is an experimental branch for algorithmic coupling; production behavior remains unchanged.
- CSV output includes switch states (`cut_destroy`, `cut_repair`) for easy ablation tables.
- CSV output also includes baseline-comparison fields:
  - `ref_obj`, `ref_rt_s`, `ref_gap_pct`
  - `delta_ref` (objective delta), `delta_rt_s` (runtime delta), `delta_gap_pct` (gap delta in percentage points).
- The cut signal is recomputed from the current synchronization graph; it is not a stored BPC cut pool, but a lightweight operational proxy for the paper’s synchronization infeasibility logic.
