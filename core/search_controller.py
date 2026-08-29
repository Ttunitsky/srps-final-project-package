from __future__ import annotations

import math
import random
from typing import Any, Callable, List, Optional


def roulette_select(weights: List[float]) -> int:
    """Weighted random selection over operator indices."""
    total = sum(weights)
    if total == 0.0:
        return random.randint(0, len(weights) - 1)
    r = random.uniform(0, total)
    cumulative = 0.0
    for idx, w in enumerate(weights):
        cumulative += w
        if r <= cumulative:
            return idx
    return len(weights) - 1


def two_opt(route: List[Any], dist: Callable[[Any, Any], float]) -> List[Any]:
    """2-opt improvement on a single route [depot, ..., depot].
    Only reorders internal nodes; never changes assignment or fleet size.
    """
    if len(route) <= 4:
        return route
    best = list(route)
    improved = True
    while improved:
        improved = False
        n = len(best)
        for i in range(1, n - 2):
            for j in range(i + 1, n - 1):
                d_old = dist(best[i - 1], best[i]) + dist(best[j], best[j + 1])
                d_new = dist(best[i - 1], best[j]) + dist(best[i], best[j + 1])
                if d_new < d_old - 1e-9:
                    best[i : j + 1] = best[i : j + 1][::-1]
                    improved = True
    return best


def alns_search(
    initial_solution: Any,
    copy_fn: Callable[[Any], Any],
    objective_fn: Callable[[Any], float],
    destroy_ops: List[Callable],
    repair_ops: List[Callable],
    destroy_size_fn: Callable[[], int],
    max_iterations: int = 500,
    start_temp: float = 100.0,
    cooling_rate: float = 0.985,
    lambda_decay: float = 0.8,
    sigma1: float = 33.0,
    sigma2: float = 20.0,
    sigma3: float = 13.0,
    seed: int = 42,
    history: Optional[list] = None,
    gap_fn: Optional[Callable[[Any], float]] = None,
    gap_threshold: Optional[float] = None,
    gap_stop: bool = False,
    max_time: Optional[float] = None,
    stall_patience: int = 0,
    stall_max_scale: float = 4.0,
) -> tuple[Any, float, dict]:
    """Generic ALNS loop with simulated-annealing acceptance and adaptive operator weights.

    The caller supplies:
      copy_fn           — deep copy of a solution
      objective_fn      — lower is better; return float('inf') for infeasible
      destroy_ops       — list of callables: destroy(sol, d) -> None
      repair_ops        — list of callables: repair(sol) -> None
      destroy_size_fn   — callable: () -> int (base number of elements to remove)
      gap_fn            — optional: (best_sol) -> float gap vs upper bound
      gap_threshold     — optional: checkpoint gap (fraction, e.g. 0.02 = 2%)
                          When first crossed, records the event.
                          If gap_stop=True, also terminates the search.
      gap_stop          — if True, terminate when gap_fn(best_sol) <= gap_threshold.
                          Default False (checkpoint-only, backward compatible).
      max_time          — optional: wall-clock time budget in seconds.
                          Terminates at max_time even if max_iterations not reached.
      stall_patience    — iterations with no improvement before escalating destroy size.
                          0 disables escalation (default). When >0, the destroy size
                          returned by destroy_size_fn() is multiplied by a factor that
                          grows gradually during stalls and resets to 1.0 on improvement.
                          Prevents the search getting trapped in deep local optima.
      stall_max_scale   — maximum destroy-size multiplier during stall (default 4.0).

    Termination: soonest of max_iterations reached, max_time exceeded, or
    (if gap_stop=True) gap_fn(best_sol) drops below gap_threshold.

    Returns
    -------
    (best_sol, best_obj, run_info)
    run_info keys:
      'iterations_run'       : actual iterations completed
      'time_elapsed'         : wall-clock seconds
      'stop_reason'          : 'max_iterations' | 'max_time' | 'gap_threshold'
      'gap_threshold_hit'    : bool — was the threshold ever crossed?
      'gap_threshold_iter'   : iteration when first crossed (or None)
      'gap_threshold_time'   : wall-clock seconds when first crossed (or None)
      'gap_threshold_obj'    : best_obj when threshold was first crossed (or None)
      'final_gap'            : gap_fn(best_sol) at termination (or None)
      'max_stall_scale'      : peak destroy-size multiplier reached during run
    """
    import time as _time

    random.seed(seed)
    t_start = _time.time()

    current_sol = initial_solution
    best_sol = copy_fn(current_sol)
    current_obj = objective_fn(current_sol)
    best_obj = current_obj

    d_weights = [1.0] * len(destroy_ops)
    r_weights = [1.0] * len(repair_ops)
    d_scores = [0.0] * len(destroy_ops)
    r_scores = [0.0] * len(repair_ops)
    d_counts = [0] * len(destroy_ops)
    r_counts = [0] * len(repair_ops)

    temp = start_temp

    gap_threshold_hit  = False
    gap_threshold_iter = None
    gap_threshold_time = None
    gap_threshold_obj  = None
    stop_reason = "max_iterations"
    iteration = 0

    # Stall-based destroy escalation
    iters_since_improve = 0
    stall_scale = 1.0
    peak_stall_scale = 1.0

    for iteration in range(max_iterations):
        # --- wall-clock termination ---
        elapsed = _time.time() - t_start
        if max_time is not None and elapsed >= max_time:
            stop_reason = "max_time"
            break

        d_idx = roulette_select(d_weights)
        r_idx = roulette_select(r_weights)
        d_counts[d_idx] += 1
        r_counts[r_idx] += 1

        candidate = copy_fn(current_sol)
        # Adaptive destroy size: escalate during stalls, reset on improvement
        d_base = destroy_size_fn()
        if stall_patience > 0 and iters_since_improve >= stall_patience:
            stall_scale = min(stall_scale * 1.05, stall_max_scale)
        else:
            stall_scale = max(stall_scale * 0.98, 1.0)
        peak_stall_scale = max(peak_stall_scale, stall_scale)
        d = max(1, round(d_base * stall_scale))
        iters_since_improve += 1
        destroy_ops[d_idx](candidate, d)
        repair_ops[r_idx](candidate)

        candidate_obj = objective_fn(candidate)
        score = 0.0

        if candidate_obj < current_obj:
            current_sol = candidate
            current_obj = candidate_obj
            score = sigma2
            if candidate_obj < best_obj:
                best_sol = copy_fn(candidate)
                best_obj = candidate_obj
                score = sigma1
                iters_since_improve = 0
                stall_scale = 1.0
        else:
            delta = candidate_obj - current_obj
            try:
                prob = math.exp(-delta / temp)
            except OverflowError:
                prob = 0.0
            if random.random() < prob:
                current_sol = candidate
                current_obj = candidate_obj
                score = sigma3

        d_scores[d_idx] += score
        r_scores[r_idx] += score

        if iteration > 0 and iteration % 50 == 0:
            for i in range(len(destroy_ops)):
                if d_counts[i] > 0:
                    d_weights[i] = (
                        lambda_decay * d_weights[i]
                        + (1 - lambda_decay) * (d_scores[i] / d_counts[i])
                    )
                    d_scores[i] = 0.0
                    d_counts[i] = 0
            for i in range(len(repair_ops)):
                if r_counts[i] > 0:
                    r_weights[i] = (
                        lambda_decay * r_weights[i]
                        + (1 - lambda_decay) * (r_scores[i] / r_counts[i])
                    )
                    r_scores[i] = 0.0
                    r_counts[i] = 0

        temp *= cooling_rate

        if history is not None:
            gap = gap_fn(best_sol) if gap_fn else None
            history.append({"iteration": iteration, "time_s": _time.time() - t_start,
                            "best_obj": best_obj, "gap": gap})

        # --- gap checkpoint (record; optionally stop) ---
        if (not gap_threshold_hit
                and gap_threshold is not None
                and gap_fn is not None):
            current_gap = gap_fn(best_sol)
            if current_gap <= gap_threshold:
                gap_threshold_hit  = True
                gap_threshold_iter = iteration
                gap_threshold_time = _time.time() - t_start
                gap_threshold_obj  = best_obj
                if gap_stop:
                    stop_reason = "gap_threshold"
                    break

    total_elapsed = _time.time() - t_start
    final_gap = gap_fn(best_sol) if gap_fn is not None else None

    run_info = {
        "iterations_run":     iteration + 1,
        "time_elapsed":       total_elapsed,
        "stop_reason":        stop_reason,
        "gap_threshold_hit":  gap_threshold_hit,
        "gap_threshold_iter": gap_threshold_iter,
        "gap_threshold_time": gap_threshold_time,
        "gap_threshold_obj":  gap_threshold_obj,
        "final_gap":          final_gap,
        "max_stall_scale":    peak_stall_scale,
    }

    return best_sol, best_obj, run_info
