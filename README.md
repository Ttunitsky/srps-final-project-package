# SRPS Coupling Falsification — Reproduction Package

**Authors:** David Chouchena, Tomer Tunitsky
**Course:** 0572-5330, *Exact Algorithms for Combinatorial Optimization Problems* — Tel-Aviv University

This repository contains everything needed to re-run, verify, or extend a study
that tests **ten techniques from the course** against an existing certified
primal–dual solver for the Selective Routing Problem with Synchronisation (SRPS).

- **The paper:** [`paper/coupling_falsification.tex`](paper/coupling_falsification.tex)
- **The findings, in prose:** [`COURSE_PROJECT.md`](COURSE_PROJECT.md)
- **This file:** how to install, run, and read everything.

---

## 1. Five-minute quick start

```bash
# 1. install Python dependencies
pip install -r requirements.txt

# 2. see what can be run  (benchmark instances are already included — see §3)
python reproduce.py --list

# 3. run everything at reduced budgets (a few minutes)
python reproduce.py --all --quick

# 4. rebuild the paper's tables from whatever results exist
python reproduce.py --fragments
```

If step 3 finishes without errors, your setup is correct and you can run the
full experiments with `python reproduce.py --all` (about 6 hours).

---

## 2. What you need

| Requirement | Needed for | If missing |
|---|---|---|
| Python 3.9+ | everything except the `exact` stage | nothing works |
| `requirements.txt` packages | everything | `pip install -r requirements.txt` |
| Benchmark instances | everything | **already included** — nothing to do (§3) |
| CPLEX 22.1.1 + Python 3.8–3.10 | the `exact` stage only | that one stage auto-skips; all others run |

**About the CPLEX stage.** The CPLEX Python API does not support Python 3.11 or
newer, so that one stage runs under a separate interpreter. `reproduce.py`
handles this automatically and *skips the stage with an explanation* if CPLEX is
not found. Override the defaults with environment variables:

```bash
export CPLEX_PYTHON="/path/to/python3.10.exe"
export CPLEX_API="/path/to/CPLEX_Studio2211/cplex/python/3.10/x64_win64"
```

---

## 3. Benchmark instances — already included

**Nothing to download.** The 71 instance files every experiment touches are
bundled in this repository under
`benchmarks/ops_raw/OPS-Benchmark-master/input/` (5.7 MB, CC0 1.0 public
domain). `python reproduce.py --all` works out of the box.

Verify with:

```bash
find benchmarks/ops_raw/OPS-Benchmark-master/input -name "*.txt" | wc -l
```

You should see **71**.

The full 780-instance benchmark is only needed to re-run the baseline evaluation
of the paper's Section 6. See [`BENCHMARK.md`](BENCHMARK.md) for sources, the
exact directory layout, and which families are worth downloading (the two largest
are not part of the study).

## 4. Repository map

### The solver being studied (do not edit)

| Path | What it is |
|---|---|
| `core/` | the baseline solver: ALNS operators, search controller, Lagrangian bound, validator |
| `adapters/` | instance and solution classes (`OPSInstance`, `OPSSolution`) |
| `validators/` | independent feasibility checker |

**No file in these three directories was modified by this study.** Every
instrument composes them from the outside. This is what makes the comparisons
trustworthy: a difference between an experiment arm and its control cannot be an
artifact of a changed baseline.

The two functions that matter most:

- `core/ops_bounds.py` → `lagrangian_bound()` — subgradient descent producing the
  dual upper bound. This is the object most of the study probes.
- `core/ops_bounds.py` → `orienteering_dp_with_selection()` — the per-processor
  bitmask DP. **84% of dual runtime is here**, which is why it is the target of
  the parallelisation experiment.

### The study's instruments

| Path | Tests which technique | Modifies solver? |
|---|---|---|
| `dev_bpc_replica/s0_instrument.py` | do violations exist in the relaxed solution? | no (read-only) |
| `dev_bpc_replica/s0b_schedulability.py` | is the resulting cut *sound*? | no (read-only) |
| `dev_bpc_replica/bpc_cut_core.py` | cut model + separation routines | no |
| `dev_bpc_replica/run_bpc_replica_experiment.py` | the original 29-run cut calibration | no |
| `dev_dual_guided/dual_guided_ops.py` | dual-guided operators — **v1 (degenerate) and v2 (corrected)** | no |
| `dev_dual_guided/run_dual_guided_experiment.py` | H2 arm matrices | no |
| `dev_dual_guided/dual_variants.py` | pluggable μ-init and step rules | no |
| `dev_dual_guided/run_dual_variants.py` | seeding + stabilisation sweep | no |
| `dev_dual_guided/convergence_probe.py` | does a verdict survive a longer budget? | no |
| `dev_dual_guided/fixing_estimate.py` | reduced-cost variable fixing reach | no |
| `dev_dual_guided/parallel_bound.py` | parallel dual (LPT-balanced) | no |
| `dev_dual_guided/quantify_parallel.py` | speedup benchmark with repeats | no |
| `dev_dual_guided/profile_dual.py` | where dual time goes | no |
| `dev_exact/cplex_srps.py` | the SRPS-1 arc-flow MILP | no |
| `dev_exact/run_exact.py` | uncoupled CPLEX reference | no |

### Configuration, results, paper

| Path | Contents |
|---|---|
| `configs/*.csv` | instance subsets — one `instance` column per file |
| `results/bpc_replica_dev/` | H1 outputs (S0, S0b, cut suite, activation sweep) |
| `results/dual_guided_dev/` | H2, variants, fixing, parallel outputs |
| `results/exact/` | CPLEX outputs |
| `results/analysis/` | notes and cross-run summaries |
| `paper/coupling_falsification.tex` | the paper |
| `paper/fragments/*.tex` | **generated** results tables — do not edit by hand |
| `build_*_fragments.py` | regenerate those fragments from the CSVs |
| `reproduce.py` | run any or all experiments |

---

## 5. Running experiments

```bash
python reproduce.py --list          # every stage, its command, its runtime
python reproduce.py --stage s0      # one stage at full budget
python reproduce.py --stage s0 --quick
python reproduce.py --all           # everything, ~6 hours
python reproduce.py --all --quick   # everything, a few minutes
```

| Stage | What it answers | ~Runtime |
|---|---|---|
| `s0` | do violations exist in the relaxed solution? | 15 min |
| `s0b` | would a cut on them be sound? | 50 min |
| `h2` | does dual guidance help? (9 arms) | 150 min |
| `fixing` | how many variables can reduced-cost fixing eliminate? | 15 min |
| `variants` | do seeding or stabilisation tighten the bound? | 30 min |
| `convergence` | does the verdict change at 1000 iterations? | 10 min |
| `parallel` | how much faster is the parallel dual? | 10 min |
| `exact` | what is the true optimum? (CPLEX) | 90 min |
| `fragments` | rebuild the paper's tables | 1 min |

**Results never overwrite.** Every run writes a timestamped file, so repeated
runs accumulate and the fragment builders pick up the most recent.

---

## 6. Reading the results

### Universal columns

| Column | Meaning |
|---|---|
| `final_obj` / `alns_z` / `z` | the incumbent (primal solution value) |
| `ub` / `lag_ub` | the Lagrangian upper bound — the certificate |
| `final_gap_pct` | `(ub − z)/ub × 100` — the certified gap |
| `delta_gap_pct` | change vs the reference run; **negative is better** |
| `delta_ref` | change in objective vs reference; **positive is better** |
| `stop_reason` | `GAP<0.3%`, `UB`, `RT_CAP`, or `TIERS_EXHAUSTED` |

### The one trap to know about

**`ub` is floored to an integer** before the gap is computed. A variant can
improve the real-valued bound by 0.8 and change the certified gap by *nothing*.
Several techniques in this study look like improvements until you apply the
floor. When comparing bounds, always check `floor(ub)`, not `ub`.

### Stage-specific columns

- **S0:** `A_partial_jobs` (consistency violations), `B_graph_acyclic`,
  `B_cycle_count`, `B_sync_excess_over_L`
- **S0b:** `feasible_schedule_found` — **1 means a cut on this instance would be
  unsound**, because a feasible schedule provably exists
- **fixing:** `fixable_pct` at the final incumbent, plus `fixable_at_z99_pct`
  and `fixable_at_z95_pct` — the realistic mid-search figures
- **exact:** `proven_optimal` — **when 0, `cplex_obj` is not the optimum**, it is
  just CPLEX's incumbent, and the decomposition columns are deliberately blank

---

## 7. Rebuilding the paper

```bash
python reproduce.py --fragments     # regenerate tables from the CSVs
```

Then compile `paper/coupling_falsification.tex` with any LaTeX toolchain
(Overleaf works; upload the whole `paper/` directory including `fragments/`).

Never edit `paper/fragments/*.tex` by hand — they are overwritten on every
rebuild. To change a number, change the experiment.

---

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `FileNotFoundError: ...instances/X.txt` | asking for an instance outside the bundled 71 | download the full set — [`BENCHMARK.md`](BENCHMARK.md) |
| `ModuleNotFoundError: cplex` | CPLEX API absent or wrong Python | §2 — or ignore; only `exact` needs it |
| `exact` stage says SKIP | CPLEX not found | expected; every other stage still runs |
| A stage seems to hang | most run 10–90 min with sparse output | check `results/` for new files |
| Fragments show "pending" | no CSV for that stage yet | run the stage, then `--fragments` |
| Different numbers than the paper | timing varies; bounds should not | bounds must match exactly; report if not |

---

## 9. What the study found

Ten techniques, one summary: **every technique targeting the certified gap was
absorbed by a ceiling that was computable in advance; the two targeting runtime
were not bounded by it, and one delivered.**

| Technique | Target | Outcome |
|---|---|---|
| BPC-style cut separation | gap | falsified on three grounds |
| Dual-guided destroy/repair | gap | +0.0037 pp — nothing |
| Warm-μ dual scheduling | gap | −0.0086 pp at full budget (a screen artifact) |
| Primal→dual seeding | gap | 9 tighter / 8 looser after flooring |
| Dual stabilisation | gap | budget-dependent; no transfer |
| Lagrangian decomposition | gap | **provably equal** to the existing bound |
| Reduced-cost fixing | runtime | 24.8% → 0.3% reach at a 1%-suboptimal incumbent |
| **Parallel subproblems** | **runtime** | **1.42× dual, bit-identical bounds** |
| Compact MILP (CPLEX) | reference | Lagrangian bound tighter on 22/30 |

Full reasoning is in [`COURSE_PROJECT.md`](COURSE_PROJECT.md) and the paper.

---

## 10. AI assistance

Claude (Anthropic) helped write the instruments, run the experiment suites, and
draft the paper. The questions, the design calls, and the conclusions are the
authors'; every number here reproduces from the committed artifacts.
