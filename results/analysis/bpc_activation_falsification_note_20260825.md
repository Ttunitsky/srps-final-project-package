# BPC Activation Falsification Note (2026-08-25, revised)

> **Status.** This note is the raw experimental record for the activation sweep.
> The full analysis, including the corrected separation target and the ceiling
> argument, lives in `paper/coupling_falsification.tex`. An earlier revision of
> this note drew conclusions about *effectiveness* that the data do not support;
> see "Correction" below.

## Objective
Test whether BPC-like cuts activate under realistic thresholds, and whether
forced activation changes outcomes.

## Protocol
Common run settings (3-instance hard subset):
- subset: `configs/bpc_activation_subset3.csv`
- phase_rt=20, abs_cap=45, sep_freq_phase=1, max_cuts_per_round=25
- epsilon_reject=1e-6, sep_budget_s_per_phase=20
- gate: `--require-cut-activation` (fails when total `cuts_added_total == 0`)

Threshold sweep on `epsilon_violation`: `1e-7`, `-50`, `-100`, `-200`.

## Results

| label | cuts_added | sep_nonempty | activation | mean_viol_after | filter_rejections | mean_delta_gap_pp | mean_final_gap_pct |
|---|---:|---:|---:|---:|---:|---:|---:|
| realistic_1e-7 | 0 | 0 | 0/3 | 0.0 | 0 | +0.057444 | 0.206910 |
| bridge_-50 | 2 | 1 | 1/3 | 0.0 | 0 | +0.057444 | 0.206910 |
| bridge_-100 | 8 | 4 | 3/3 | 0.0 | 0 | +0.057444 | 0.206910 |
| forced_-200 | 15 | 4 | 3/3 | 0.0 | 0 | +0.057444 | 0.206910 |

Final objectives were bit-identical (4971 / 2285 / 2226) at every point.

## Correction to the earlier revision

The earlier revision concluded that forced activation "does not improve
outcomes" and that "practical value here is not demonstrated." **That is not
supported by this experiment**, for a reason visible in the two columns added
above.

`mean_violation_after = 0.0` and `cut_filter_rejections = 0` hold at *every*
threshold, including with 15 active cuts. The evaluation routine scores each cut
against `c.rhs`, frozen at construction as the true horizon `L`:

```python
viol = max(0.0, max_len - c.rhs)   # c.rhs == inst.L
```

For any feasible candidate `max_len <= L`, so this is identically `0.0`
regardless of `epsilon_violation`, which gates only *admission* into the pool.
The guard `if viol > 0.0` therefore never fires, no candidate is ever rejected,
and the soft penalty subtracts `lambda_cut * 0`.

The outcomes are not unchanged because cuts do not help. They are unchanged
because **no treatment was ever applied**. Four rows identical to six decimals
is an algebraic identity, not a statistical null. The experiment has zero power
by construction and cannot distinguish "cuts do not help" from "cuts are wired
inert."

## What this note does support

1. Under physically meaningful thresholds, cut activation is absent (0 cuts,
   0/3 instances). This is structural: separation runs against the ALNS
   incumbent, which is feasible by construction, so `cycle_count = 0`,
   `route_over_count = 0`, and `sync_over <= 0` on every call.
2. The pipeline can be made to activate, so the absence in (1) is not a plumbing
   failure.

It does **not** support any claim about the effectiveness of BPC-style cuts.

## Related defect

`cut_penalty_hits` is mislabeled: it is incremented once per job removal in the
guided destroy operator regardless of any active cut, which is why runs with an
empty pool still report 8 / 4 / 8.

## Artifacts
- realistic: `results/bpc_replica_dev/bpc_replica_activation_sanity_20260825_1310.csv`
- bridge -50: `results/bpc_replica_dev/bpc_replica_activation_bridge_epsm50_20260825_1320.csv`
- bridge -100: `results/bpc_replica_dev/bpc_replica_activation_bridge_epsm100_20260825_1324.csv`
- forced -200: `results/bpc_replica_dev/bpc_replica_activation_forced_epsneg_20260825_1317.csv`
- (matching `bpc_activation_diag_*` diagnostics alongside each)

## Follow-on work
- `dev_bpc_replica/s0_instrument.py` — separation retargeted at the Lagrangian
  relaxed solution, where violations do exist.
- `dev_bpc_replica/s0b_schedulability.py` — soundness test for the resulting cut.
- `paper/coupling_falsification.tex` — full write-up.
