"""
Independent OPS solution validator.

Must be called OUTSIDE the search loop on every candidate best solution
before recording it as a result. A solution that fails validation cannot
be reported, even if the objective value looks attractive.

This module is intentionally independent of the search internals — it
re-derives feasibility from scratch using only the instance data and the
solution routes, so it cannot be fooled by a corrupted internal state.

Checks performed (all must pass):
  1. Every route starts at dummy start and ends at dummy end.
  2. Jobs in each route belong to Jk[k] for that processor.
  3. Every job in any route is in the solution's selected set.
  4. Every selected job appears in every required processor's route.
  5. The constraint graph is acyclic (no synchronization ordering conflict).
  6. Earliest start times computed via longest-path satisfy s[end] <= L.
  7. The recomputed objective matches the claimed value.

Raises AssertionError with a descriptive message on any violation.
"""
from __future__ import annotations

from adapters.ops_adapter import OPSInstance, OPSSolution, compute_schedule


def validate(
    instance: OPSInstance,
    solution: OPSSolution,
    claimed_obj: float,
) -> bool:
    """Independently validate solution against instance, verify claimed objective.

    Parameters
    ----------
    instance    : OPSInstance parsed from the benchmark files
    solution    : OPSSolution to validate
    claimed_obj : the objective value the search is reporting

    Returns True on success. Raises AssertionError on any violation.
    """
    # 1. Route bookends
    for k, route in enumerate(solution.routes):
        assert route[0] == instance.start, (
            f"Route {k}: first element {route[0]} ≠ start {instance.start}"
        )
        assert route[-1] == instance.end, (
            f"Route {k}: last element {route[-1]} ≠ end {instance.end}"
        )

    # 2. Route job membership
    for k, route in enumerate(solution.routes):
        valid = set(instance.Jk[k])
        for job in route:
            if job in (instance.start, instance.end):
                continue
            assert job in valid, (
                f"Route {k}: job {job} not in Jk[{k}]"
            )

    # 3. Route ↔ selected consistency (routes ⊆ selected)
    for k, route in enumerate(solution.routes):
        for job in route:
            if job in (instance.start, instance.end):
                continue
            assert job in solution.selected, (
                f"Route {k}: job {job} in route but not in selected set"
            )

    # 4. Selected ↔ routes consistency (selected ⊆ routes)
    for job in solution.selected:
        for k in instance.Kj.get(job, []):
            assert job in solution.routes[k], (
                f"Selected job {job} missing from required processor {k}'s route"
            )

    # 5–6. Schedule feasibility (cycle check + time budget)
    times, ok = compute_schedule(solution.routes, solution.selected, instance)
    if not ok:
        if times is None:
            raise AssertionError(
                "Infeasible: cycle in constraint graph — "
                "contradictory synchronization ordering between processors"
            )
        end_time = times.get(instance.end, "?")
        raise AssertionError(
            f"Infeasible: schedule end time {end_time} exceeds L = {instance.L}"
        )

    # 7. Objective
    recomputed = sum(instance.profits[j] for j in solution.selected)
    assert abs(recomputed - claimed_obj) < 1e-6, (
        f"Objective mismatch: claimed {claimed_obj:.6f}, "
        f"recomputed {recomputed:.6f}"
    )

    return True
