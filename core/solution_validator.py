"""
core/solution_validator.py — Shared OPS solution validation gate.

Call validate_solution() on every final solution before recording results.
Validation time is excluded from the search RT budget by the caller.

A failure prints a loud red-flag banner and raises, blocking result recording.
Import this in every run script: adaptive, sensitivity, ablation.
"""
from __future__ import annotations

import sys
from validators.validate_ops_solution import validate


_RED   = "\033[91m"
_RESET = "\033[0m"
_BOLD  = "\033[1m"


def validate_solution(inst, sol, claimed_obj: float, label: str = "") -> None:
    """Validate sol against inst and claimed_obj.

    Runs all 7 checks (route bookends, job membership, route/selected
    consistency, acyclicity, schedule feasibility, objective recomputation).
    Validation time is not counted against the search RT budget.

    Parameters
    ----------
    inst        : OPSInstance
    sol         : OPSSolution — the final best solution
    claimed_obj : float — objective the search is reporting
    label       : str — instance name, for error identification

    Raises
    ------
    AssertionError on any violation, after printing a loud red-flag banner.
    """
    try:
        validate(inst, sol, claimed_obj=claimed_obj)
    except AssertionError as e:
        banner = (
            f"\n{_RED}{_BOLD}"
            f"{'!'*70}\n"
            f"  VALIDATION FAILED — {label}\n"
            f"  {e}\n"
            f"  Result will NOT be recorded.\n"
            f"{'!'*70}"
            f"{_RESET}\n"
        )
        print(banner, file=sys.stderr, flush=True)
        print(banner, flush=True)
        raise
