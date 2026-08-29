# Benchmark data — what's included, and how to get the rest

## Short version

**You do not need to download anything to reproduce this study.** The 71 instance
files every experiment touches are already in this repository, under
`benchmarks/ops_raw/OPS-Benchmark-master/input/`. Run `python reproduce.py --all`
and it works.

You only need the full benchmark if you want to re-run the **660-instance baseline
evaluation** of the paper's Section 6.

---

## What's bundled here

| | |
|---|---|
| Instance files | **71** (5.7 MB) |
| Covers | every subset in `configs/*.csv` |
| Enables | all nine experiment stages in `reproduce.py` |
| Licence | **CC0 1.0 Universal** (public domain) — see `LICENSE.txt` beside the data |

Also bundled: `results/adaptive_master.csv`, the 780-row table of incumbents,
bounds, and certified gaps from the baseline evaluation. Several analyses read it
directly rather than re-running the solver — including the ceiling computation of
Table 6, which needs no instance files at all.

---

## Getting the full benchmark

The benchmark is by Riera-Ledesma and Salazar-González, released under CC0.
Two equivalent sources:

- **Zenodo:** <https://zenodo.org/records/17557917>
- **GitHub:** <https://github.com/RieraULL/OPS-Benchmark>

### Where the files must go

The loader resolves instances by this exact path:

```
benchmarks/ops_raw/OPS-Benchmark-master/input/<FAMILY>/instances/<LABEL>.txt
```

So `B_n140_021_a25_061` must land at:

```
benchmarks/ops_raw/OPS-Benchmark-master/input/B/instances/B_n140_021_a25_061.txt
```

### From the GitHub clone

`git clone` produces a directory named `OPS-Benchmark`, but the loader expects
`OPS-Benchmark-master` (the name GitHub's zip download uses). Copy the `input`
tree across rather than renaming, so the bundled 71 files are not disturbed:

```bash
git clone https://github.com/RieraULL/OPS-Benchmark
cp -r OPS-Benchmark/input/* benchmarks/ops_raw/OPS-Benchmark-master/input/
```

### From the Zenodo download

The archive already unpacks as `OPS-Benchmark-master`:

```bash
unzip OPS-Benchmark-master.zip
cp -r OPS-Benchmark-master/input/* benchmarks/ops_raw/OPS-Benchmark-master/input/
```

### Verify

```bash
find benchmarks/ops_raw/OPS-Benchmark-master/input -name "*.txt" | wc -l
```

- **71** — only the bundled subset; every experiment stage runs
- **841** — the six study families (B, C, D, EB, EC, ED), 75 MB; the full baseline
  can be re-run
- **1191** — everything including families A and EA, 567 MB

---

## Which families matter

| Family | Class | In the 660 study set? |
|---|---|---|
| B, C, D | horizontal distance, `\|K_j\|` ∈ {2,3,4} | yes |
| EB, EC, ED | Euclidean distance, `\|K_j\|` ∈ {2,3,4} | yes |
| A, EA | `\|K_j\|` = 1 — reduces to team orienteering | no |

**A and EA are 493 MB of the 567 MB and are not part of the study.** They carry no
synchronisation, so they are excluded from the primary set. They appear only in
the 780-row ceiling table of Section 6.3, and there only through
`adaptive_master.csv`, which is already bundled — you never need their instance
files.

Downloading only B, C, D, EB, EC, ED gets you everything at 75 MB instead of 567.

---

## If something goes wrong

| Symptom | Cause | Fix |
|---|---|---|
| `FileNotFoundError: ...instances/X.txt` | wrong directory depth | check the path has `input/<FAMILY>/instances/` |
| Files landed in `OPS-Benchmark/` | `git clone` naming | copy `input/*` into the existing `OPS-Benchmark-master/input/` |
| Count is 0 | copied the repo root, not `input` | copy the `input` tree, not its parent |

The instance format and the benchmark's own documentation are described in
`benchmarks/ops_raw/OPS-Benchmark-master/README.md` once the full set is
unpacked; provenance and DOIs are in `benchmarks/DATA_PROVENANCE.md`.
