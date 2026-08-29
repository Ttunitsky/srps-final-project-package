# EACOP End-Term Project — Falsification Study

**Authors:** David Chouchena, Tomer Tunitsky

**Course:** 0572-5330, *Exact Algorithms for Combinatorial Optimization
Problems* -- Tel-Aviv University.

**Paper:** [`paper/coupling_falsification.tex`](paper/coupling_falsification.tex)

This directory tree is the reproduction package for a study that tests two
strengthening hypotheses against an existing certified primal–dual solver for
the Selective Routing Problem with Synchronisation (SRPS), and falsifies both.

---

## The two hypotheses

| | hypothesis | course topic | verdict |
|---|---|---|---|
| **H1** | Import BPC-style cut separation into the heuristic | Lectures V–VII (cutting planes, branch-and-cut, branch-and-price) | falsified on three independent grounds |
| **H2** | Steer destroy/repair operators with Lagrangian multipliers | Lectures III, IX (Lagrangian relaxation, decomposition) | falsified; apparent effect traced to a confound |

Both failures are structural rather than parametric, and each is diagnosed to a
specific mechanism rather than attributed to insufficient tuning.

---

## Headline findings

**H1.** Separation against the ALNS incumbent cannot produce a violated cut —
the incumbent is feasible by construction, so every condition the separator
tests is identically unsatisfied. A 29-run calibration suite (128 min, 8
parameters) confirmed this, recommending the all-cuts-disabled control with
`cuts_added_total = 0` in every run. Forcing activation admitted up to 15 cuts
while leaving every outcome column bit-identical — *mechanism inertness*.
Redirecting separation at the Lagrangian relaxed solution finds abundant
violations, but the resulting cut family is **unsound**: on 16 of 30 instances
an explicit feasible schedule exists for the relaxed selection.

**Ceiling.** Independently of design, the certificate is already exact on 48.5%
of 780 instances, capping *any* strengthening at **0.1136 pp mean / 0.0087 pp
median** gap reduction — below the ±0.01–0.09 pp arm-to-arm noise of the
experiments built to detect it. This bounds H2 as well as H1, since
`z ≤ P* ≤ ub` makes both levers subject to the same limit.

**H2.** The operator design is degenerate under its own initialisation: the fair
split `μ_{j,k} = b_j/|K_j|` forces `Σ_k μ_{j,k} = b_j` exactly, so the reduced
profit is **identically zero for every job** (verified: 0/80 and 0/150 nonzero).
Destroy degenerates to random removal, repair to insertion-order greedy. After
correcting it, a warm-μ control arm shows the apparent −0.13 pp improvement is
**entirely the extra dual solve**, not the guidance: guidance isolated against
that baseline spans −0.004 to +0.021 pp, with 2 of 4 arms worse.

---

## Layout

```
dev_bpc_replica/       H1 instruments
  bpc_cut_core.py        cut model + separation (circuit / route-len / sync-path)
  run_bpc_replica_experiment.py
  s0_instrument.py       S0: violations in the Lagrangian relaxed solution
  s0b_schedulability.py  S0b: soundness test for the resulting cut
dev_dual_guided/       H2 instruments
  dual_guided_ops.py     v1 (degenerate) + v2 (corrected) operators
  run_dual_guided_experiment.py
  fixing_estimate.py     reduced-cost variable fixing reach
dev_cut_guided/        earlier cut-guided variant (superseded by dev_bpc_replica)
core/, adapters/       baseline solver (unmodified by this study)
configs/               instance subsets used by each experiment
results/bpc_replica_dev/   H1 artifacts
results/dual_guided_dev/   H2 artifacts
results/analysis/          notes and cross-run summaries
paper/                 coupling_falsification.tex + generated fragments
```

**No instrument in this study modifies the baseline solver.** `core/` and
`adapters/` are untouched; every experiment composes them from outside.

---

## Reproduction

Requires Python 3.10+ and `requirements.txt`. Benchmark instances are **not**
redistributed here — fetch them from
<https://github.com/RieraULL/OPS-Benchmark> into
`benchmarks/ops_raw/OPS-Benchmark-master/input/`.

```bash
# H1 - S0: do violations exist in the relaxed solution?
python dev_bpc_replica/s0_instrument.py \
    --subset configs/bpc_s0_full30.csv --tag full30 \
    --lag-max-iter 200 --lag-max-time 30

# H1 - S0b: is the resulting cut sound?
python dev_bpc_replica/s0b_schedulability.py \
    --subset configs/bpc_s0_full30_a.csv --tag full30a \
    --lag-max-iter 200 --lag-max-time 30 --budget-s 90

# H2 - corrected arm (v2 operators + warm mu)
python dev_dual_guided/run_dual_guided_experiment.py \
    --subset-csv configs/coupling_refine_subset12.csv \
    --phase-rt 120 --abs-cap 240 --lag-max-iter 120 --lag-max-time 20 \
    --ops-v2 --mu-warm-init --dual-destroy --dual-repair --dual-feedback \
    --tag h2_v2_full

# H2 - the isolating control that overturns the naive reading
python dev_dual_guided/run_dual_guided_experiment.py \
    --subset-csv configs/coupling_refine_subset12.csv \
    --phase-rt 120 --abs-cap 240 --lag-max-iter 120 --lag-max-time 20 \
    --mu-warm-init --no-dual-destroy --no-dual-repair --no-dual-feedback \
    --tag h2_control_warm

# Reduced-cost fixing reach (no search runs needed)
python dev_dual_guided/fixing_estimate.py \
    --subset configs/bpc_s0_full30.csv --tag full30

# Rebuild the paper's results tables from the CSVs
python build_falsification_fragments.py
```

To reproduce the *degenerate* v1 behaviour, drop `--ops-v2` and
`--mu-warm-init`; both original operators are retained unmodified behind those
flags.

---

## Methodological guards this study recommends

1. **Gate on effect, not activation.** The ε-sweep passed an explicit
   `--require-cut-activation` gate while remaining completely inert. Assert that
   the treated arm *differs from control* on a smoke subset before spending a
   calibration suite.
2. **Compute the ceiling first.** It costs seconds from data already in hand and
   tells you whether a campaign can possibly succeed.
3. **Verify the signal is non-degenerate.** `Σ_k b_j/|K_j| = b_j` is a one-line
   check that would have caught the H2 defect before any run.
4. **Vary the suspected cause on its own.** Without the warm-μ control arm, H2
   would have been reported as a 0.13 pp success.

---

## Relationship to the parent work

The baseline solver and its 660-instance evaluation belong to a separate
manuscript. That work is unmodified by this study; nothing here changes any
reported baseline result. The one place the two touch is a single sentence added
to the parent paper's Discussion recording the ceiling as the reason BPC-style
separation was not pursued.

---

## AI assistance

Claude (Anthropic) helped write the instruments, run the experiment suites, and
draft the paper. The questions, the design calls, and the conclusions are the
authors'; every number here reproduces from the committed artifacts.
