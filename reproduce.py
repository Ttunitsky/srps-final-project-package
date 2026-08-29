"""Reproduce every experiment in the study.

    python reproduce.py --list           show all stages, budgets, and runtimes
    python reproduce.py --stage s0       run one stage
    python reproduce.py --all            run everything in order
    python reproduce.py --all --quick    reduced budgets, for a smoke check
    python reproduce.py --fragments      rebuild the paper's tables from CSVs

Each stage records the exact command it runs, so `--list` doubles as the
protocol appendix. Results land in results/<area>/ with a timestamped filename;
nothing is overwritten, so repeated runs accumulate rather than clobber.

The `exact` stage needs CPLEX and a Python 3.8-3.10 interpreter (the CPLEX API
does not support 3.11+). It is skipped automatically when unavailable; every
other stage runs on the same interpreter as the solver.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

CPLEX_PY = os.environ.get(
    "CPLEX_PYTHON",
    r"C:\Users\user\AppData\Local\Programs\Python\Python310\python.exe",
)
CPLEX_API = os.environ.get(
    "CPLEX_API",
    r"C:\Program Files\IBM\ILOG\CPLEX_Studio2211\cplex\python\3.10\x64_win64",
)


class Stage:
    def __init__(self, key, title, minutes, args, quick=None, needs_cplex=False):
        self.key = key
        self.title = title
        self.minutes = minutes
        self.args = args
        self.quick = quick
        self.needs_cplex = needs_cplex


STAGES = [
    Stage(
        "s0", "S0: violations in the Lagrangian relaxed solution", 15,
        ["dev_bpc_replica/s0_instrument.py", "--subset", "configs/bpc_s0_full30.csv",
         "--tag", "full30", "--lag-max-iter", "200", "--lag-max-time", "30"],
        quick=["dev_bpc_replica/s0_instrument.py", "--subset",
               "configs/bpc_activation_subset3.csv", "--tag", "quick",
               "--lag-max-iter", "40", "--lag-max-time", "6"],
    ),
    Stage(
        "s0b", "S0b: is the resulting cut sound?", 50,
        ["dev_bpc_replica/s0b_schedulability.py", "--subset",
         "configs/bpc_s0_full30.csv", "--tag", "full30",
         "--lag-max-iter", "200", "--lag-max-time", "30", "--budget-s", "90"],
        quick=["dev_bpc_replica/s0b_schedulability.py", "--subset",
               "configs/bpc_activation_subset3.csv", "--tag", "quick",
               "--lag-max-iter", "40", "--lag-max-time", "6", "--budget-s", "15"],
    ),
    Stage(
        "fixing", "Reduced-cost fixing reach", 15,
        ["dev_dual_guided/fixing_estimate.py", "--subset",
         "configs/bpc_s0_full30.csv", "--tag", "full30fixed",
         "--lag-max-iter", "200", "--lag-max-time", "30"],
        quick=["dev_dual_guided/fixing_estimate.py", "--subset",
               "configs/bpc_activation_subset3.csv", "--tag", "quick",
               "--lag-max-iter", "40", "--lag-max-time", "6"],
    ),
    Stage(
        "variants", "Dual variants: seeding and stabilisation", 30,
        ["dev_dual_guided/run_dual_variants.py", "--subset",
         "configs/bpc_s0_full30.csv", "--tag", "full30",
         "--iters", "200", "--workers", "6"],
        quick=["dev_dual_guided/run_dual_variants.py", "--subset",
               "configs/bpc_activation_subset3.csv", "--tag", "quick",
               "--iters", "60", "--workers", "3"],
    ),
    Stage(
        "convergence", "Convergence probe to 1000 iterations", 10,
        ["dev_dual_guided/convergence_probe.py", "--subset",
         "configs/bpc_s0_hardtail4.csv", "--tag", "hardtail",
         "--max-iter", "1000", "--workers", "6"],
        quick=["dev_dual_guided/convergence_probe.py", "--subset",
               "configs/bpc_activation_subset3.csv", "--tag", "quick",
               "--max-iter", "200", "--workers", "3"],
    ),
    Stage(
        "parallel", "Parallel dual speedup", 10,
        ["dev_dual_guided/quantify_parallel.py", "--subset",
         "configs/bpc_s0_hardtail4.csv", "--tag", "hardtail",
         "--iters", "200", "--workers", "4", "--repeats", "3"],
        quick=["dev_dual_guided/quantify_parallel.py", "--subset",
               "configs/bpc_activation_subset3.csv", "--tag", "quick",
               "--iters", "60", "--workers", "4", "--repeats", "1"],
    ),
    Stage(
        "exact", "Uncoupled CPLEX reference on SRPS-1", 90,
        ["dev_exact/run_exact.py", "--subset", "configs/exact_small_nonzero.csv",
         "--time-limit", "180", "--threads", "6", "--tag", "small30"],
        quick=["dev_exact/run_exact.py", "--max-n", "50", "--limit", "2",
               "--time-limit", "30", "--threads", "4", "--tag", "quick"],
        needs_cplex=True,
    ),
]

# H2 arm matrices are parameterised runs of one driver, listed separately so the
# arm definitions stay visible rather than buried in a loop.
H2_DRIVER = "dev_dual_guided/run_dual_guided_experiment.py"
H2_SCREEN = ["--subset-csv", "configs/coupling_refine_subset12.csv",
             "--phase-rt", "120", "--abs-cap", "240",
             "--lag-max-iter", "120", "--lag-max-time", "20"]
H2_PROD = ["--subset-csv", "configs/coupling_refine_subset12.csv",
           "--phase-rt", "300", "--abs-cap", "1800",
           "--lag-max-iter", "200", "--lag-max-time", "60"]
H2_ARMS_SCREEN = [
    ("h2_control", ["--no-dual-destroy", "--no-dual-repair", "--no-dual-feedback"]),
    ("h2_control_warm", ["--mu-warm-init", "--no-dual-destroy", "--no-dual-repair",
                         "--no-dual-feedback"]),
    ("h2_v1_warm", ["--mu-warm-init", "--dual-destroy", "--dual-repair",
                    "--dual-feedback"]),
    ("h2_v2_destroy", ["--ops-v2", "--mu-warm-init", "--dual-destroy",
                       "--no-dual-repair", "--dual-feedback"]),
    ("h2_v2_repair", ["--ops-v2", "--mu-warm-init", "--no-dual-destroy",
                      "--dual-repair", "--dual-feedback"]),
    ("h2_v2_full", ["--ops-v2", "--mu-warm-init", "--dual-destroy", "--dual-repair",
                    "--dual-feedback"]),
]
H2_ARMS_PROD = [
    ("h2p_control", ["--no-dual-destroy", "--no-dual-repair", "--no-dual-feedback"]),
    ("h2p_control_warm", ["--mu-warm-init", "--no-dual-destroy", "--no-dual-repair",
                          "--no-dual-feedback"]),
    ("h2p_v2_full", ["--ops-v2", "--mu-warm-init", "--dual-destroy", "--dual-repair",
                     "--dual-feedback"]),
]


def cplex_available():
    if not os.path.exists(CPLEX_PY):
        return False, "interpreter not found: %s" % CPLEX_PY
    env = dict(os.environ)
    env["PYTHONPATH"] = CPLEX_API + os.pathsep + ROOT
    try:
        r = subprocess.run([CPLEX_PY, "-c", "import cplex"], env=env,
                           capture_output=True, timeout=60)
        return (r.returncode == 0,
                "ok" if r.returncode == 0 else "cannot import cplex")
    except Exception as e:
        return False, str(e)


def run(cmd_args, use_cplex=False):
    env = dict(os.environ)
    interp = PY
    if use_cplex:
        interp = CPLEX_PY
        env["PYTHONPATH"] = CPLEX_API + os.pathsep + ROOT
    cmd = [interp] + cmd_args
    print("  $ " + " ".join(cmd_args), flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=ROOT, env=env)
    print("    exit=%d  %.1f min" % (r.returncode, (time.time() - t0) / 60.0),
          flush=True)
    return r.returncode


def run_h2(quick=False):
    rc = 0
    common, arms, label = (
        (H2_SCREEN, H2_ARMS_SCREEN[:2], "screen (quick: 2 arms)") if quick
        else (H2_SCREEN, H2_ARMS_SCREEN, "screen (6 arms)")
    )
    print("H2 %s" % label, flush=True)
    for tag, flags in arms:
        rc |= run([H2_DRIVER] + common + flags + ["--tag", tag])
    if not quick:
        print("H2 production (3 arms)", flush=True)
        for tag, flags in H2_ARMS_PROD:
            rc |= run([H2_DRIVER] + H2_PROD + flags + ["--tag", tag])
    return rc


def do_fragments():
    rc = 0
    for b in ("build_falsification_fragments.py", "build_extra_fragments.py",
              "build_exact_fragments.py"):
        if os.path.exists(os.path.join(ROOT, b)):
            rc |= run([b])
    return rc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--list", action="store_true")
    p.add_argument("--stage", default=None)
    p.add_argument("--all", action="store_true")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--fragments", action="store_true")
    a = p.parse_args()

    if a.list:
        ok, why = cplex_available()
        print("Stages (est. runtime at full budget):\n")
        for st in STAGES:
            mark = ""
            if st.needs_cplex:
                mark = "   [CPLEX: %s]" % ("available" if ok else why)
            print("  %-12s %-48s ~%3d min%s" % (st.key, st.title, st.minutes, mark))
            print("      $ python %s" % " ".join(st.args))
        print("  %-12s %-48s ~%3d min" % ("h2", "H2 arm matrices (screen + production)", 150))
        print("      6 screen arms + 3 production arms of %s" % H2_DRIVER)
        print("  %-12s %-48s ~%3d min" % ("fragments", "rebuild paper tables from CSVs", 1))
        print("\n  --all at full budget is roughly 6 hours; --quick is a few minutes.")
        return

    if a.fragments:
        sys.exit(do_fragments())

    todo = STAGES if a.all else [s for s in STAGES if s.key == a.stage]
    if not todo and a.stage != "h2":
        print("unknown stage: %s (use --list)" % a.stage)
        sys.exit(2)

    rc = 0
    ok, why = cplex_available()
    for st in todo:
        if st.needs_cplex and not ok:
            print("SKIP %s - %s" % (st.key, why), flush=True)
            continue
        print("\n=== %s: %s ===" % (st.key, st.title), flush=True)
        args = st.quick if (a.quick and st.quick) else st.args
        rc |= run(args, use_cplex=st.needs_cplex)

    if a.all or a.stage == "h2":
        print("\n=== h2: arm matrices ===", flush=True)
        rc |= run_h2(quick=a.quick)

    if a.all:
        print("\n=== fragments ===", flush=True)
        rc |= do_fragments()

    sys.exit(rc)


if __name__ == "__main__":
    main()
