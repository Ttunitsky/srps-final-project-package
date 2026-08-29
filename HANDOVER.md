# Handover — EACOP End-Term Project

**For:** Tomer Tunitsky
**From:** David Chouchena
**Course:** 0572-5330, *Exact Algorithms for Combinatorial Optimization Problems* — Tel-Aviv University

This package contains the complete project: the paper, the code, the results,
and everything needed to rebuild the PDF and re-run the experiments.

Three things to do, in order: **build the PDF** (§1), **push the repo** (§2),
**submit** (§3). Sections 4–6 are background if you want to go deeper.

---

## 1. Build the PDF on Overleaf

The paper is LaTeX split across a main file and generated result tables, so the
folder structure has to be preserved. Ten minutes, no local install needed.

1. Go to **overleaf.com** → **New Project** → **Upload Project**.
2. Zip the `paper/` folder from this package **on its own** and upload that zip.
   The result in Overleaf must look like:
   ```
   coupling_falsification.tex
   fragments/
       s0_results.tex
       s0b_results.tex
       h2_results.tex
       prod_results.tex
       fixing_results.tex
       variants_results.tex
       convergence_results.tex
       parallel_results.tex
       exact_results.tex
       coverage_results.tex
   ```
   **The `fragments/` folder must stay a subfolder.** If Overleaf flattens it,
   the compile fails with `File 'fragments/s0_results.tex' not found`.
3. Click **Menu** (top left) → set **Main document** to
   `coupling_falsification.tex` and **Compiler** to **pdfLaTeX**.
4. Click **Recompile**. Then **click it a second time** — LaTeX needs two passes
   to resolve the table of contents and all the cross-references. After one pass
   you will see `??` where section numbers should be; that is normal and the
   second pass fixes it.
5. Download the PDF with the **Download PDF** button.

**Expected result:** roughly 30–35 pages, a table of contents, and no `??`
anywhere in the text.

### If something goes wrong

| Error | Fix |
|---|---|
| `File 'fragments/xxx.tex' not found` | the folder got flattened — re-upload keeping `fragments/` as a subfolder |
| `??` in the text after compiling | recompile once more; it needs two passes |
| `Undefined control sequence \pp` | you are compiling a fragment instead of the main file — set Main document correctly |
| Missing package errors | switch Compiler to pdfLaTeX; Overleaf has every package used here |

No figures, no bibliography tool, no custom class — one `\bibitem` inline and
standard packages only. It should compile first time.

---

## 2. Push the repo

The repository already exists as a **private** GitHub repo:

**`github.com/chouchena/SRPS-Coupling-Falsification`**

If David has added you as a collaborator, just clone it — it is already current
and nothing needs pushing:

```bash
git clone https://github.com/chouchena/SRPS-Coupling-Falsification
```

If you would rather submit from your own account, this package is a complete
working tree. From inside the unzipped folder:

```bash
git init -b main
git add -A
git commit -m "EACOP end-term project: SRPS coupling falsification study"
git remote add origin <your-repo-url>
git push -u origin main
```

**Before pushing anywhere public**, note that `README.md` §10 carries an AI
assistance disclosure. That is deliberate and should stay unless the course
policy says otherwise.

---

## 3. What to submit

- The **PDF** built in §1
- The **repository URL** from §2

The paper is self-contained: it states the problem, the ten techniques tested,
how each was tested and calibrated, the results, and the conclusions. A reader
does not need the repo to follow it.

---

## 4. What's in this package

| Path | What it is |
|---|---|
| `paper/` | the paper (upload this to Overleaf) |
| `README.md` | full guide to the repo — start here for anything technical |
| `COURSE_PROJECT.md` | the findings written out in prose |
| `reproduce.py` | re-runs any or all experiments |
| `core/`, `adapters/`, `validators/` | the solver being studied — **not modified by this project** |
| `dev_bpc_replica/` | instruments for the cut-separation hypothesis |
| `dev_dual_guided/` | instruments for the dual-coupling hypothesis |
| `dev_exact/` | the standalone CPLEX reference |
| `configs/` | instance subsets used by each experiment |
| `results/` | every CSV the paper's tables are built from |
| `benchmarks/` | the 71 bundled instance files (CC0) |
| `BENCHMARK.md` | how to obtain the full benchmark, if you want it |

Benchmark instance files **are** included: the 71 the experiments touch, 5.7 MB,
under CC0 1.0 public domain. Nothing needs downloading to re-run the study. The
full 780-instance benchmark is only needed to re-run the baseline evaluation, and
`BENCHMARK.md` explains where to get it and which families are worth the
download.

---

## 5. The one-paragraph summary

We tested ten techniques from the course against an existing certified
primal–dual SRPS solver, expecting at least one to improve it. Nine did not, and
the interesting part is *why*: a ceiling computable in advance from the solver's
own published numbers bounds any gap-directed technique at 0.11 pp on average and
0.009 pp at the median — below the noise of the experiments built to detect it.
Two techniques targeted runtime instead, which the ceiling does not bound, and
one of those delivered. Along the way two measurement traps appeared that are
more transferable than the negative results themselves: a mechanism that fires,
reports firing, and cannot affect the outcome; and verdicts that reverse when the
computational budget changes — in both directions.

---

## 6. If you want to re-run anything

```bash
pip install -r requirements.txt
python reproduce.py --list          # every experiment, its command, its runtime
python reproduce.py --all --quick   # everything at reduced budgets, a few minutes
python reproduce.py --all           # everything at full budget, about 6 hours
```

`README.md` covers setup, the repo map, how to read every result column, and
troubleshooting. One thing worth knowing before reading any result: **the upper
bound is floored to an integer before the gap is computed**, so a technique can
improve the real-valued bound and change the certified gap by nothing. That
catches several of the techniques in this study.

---

## 7. Known open item

The paper's threats-to-validity section pins the configuration the study
measures (a dual refresh capped at 200 iterations). Section 8.4 shows that budget
to be the binding constraint on the dual, and notes that raising it tightens the
certificate by 0.29 pp on average at essentially no runtime cost. The paper states
this explicitly and explains which conclusions strengthen under a tighter
configuration and which would need re-testing. **No action needed for
submission.**
