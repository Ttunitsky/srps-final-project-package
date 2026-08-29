from __future__ import annotations

import random
from typing import Callable, Dict, List, Optional, Set, Tuple


def _constraint_graph(solution) -> Tuple[Set[int], Dict[int, List[Tuple[int, float]]]]:
    inst = solution.inst
    nodes: Set[int] = {inst.start, inst.end} | set(solution.selected)
    edges_out: Dict[int, List[Tuple[int, float]]] = {v: [] for v in nodes}
    seen_edges: Set[Tuple[int, int]] = set()

    for route in solution.routes:
        for idx in range(len(route) - 1):
            i, j = route[idx], route[idx + 1]
            if i in nodes and j in nodes and (i, j) not in seen_edges:
                edges_out[i].append((j, float(inst.T[i][j])))
                seen_edges.add((i, j))

    return nodes, edges_out


def _topo_longest_path(inst, nodes: Set[int], edges_out: Dict[int, List[Tuple[int, float]]]):
    in_deg: Dict[int, int] = {v: 0 for v in nodes}
    for u in nodes:
        for v, _ in edges_out[u]:
            in_deg[v] += 1

    queue: List[int] = [v for v in nodes if in_deg[v] == 0]
    topo: List[int] = []
    while queue:
        u = queue.pop(0)
        topo.append(u)
        for v, _ in edges_out[u]:
            in_deg[v] -= 1
            if in_deg[v] == 0:
                queue.append(v)

    if len(topo) < len(nodes):
        return None, None, None, False

    dist: Dict[int, float] = {v: float("-inf") for v in nodes}
    pred: Dict[int, Optional[int]] = {v: None for v in nodes}
    dist[inst.start] = 0.0

    for u in topo:
        if dist[u] == float("-inf"):
            continue
        for v, w in edges_out[u]:
            candidate = dist[u] + w
            if candidate > dist[v]:
                dist[v] = candidate
                pred[v] = u

    return topo, dist, pred, True


def _reconstruct_path(pred: Dict[int, Optional[int]], start: int, end: int) -> List[int]:
    if pred.get(end) is None and start != end:
        return [start, end]

    path: List[int] = [end]
    cur = end
    guard = 0
    while cur != start and pred.get(cur) is not None:
        cur = pred[cur]  # type: ignore[index]
        path.append(cur)
        guard += 1
        if guard > 10_000:
            break

    path.reverse()
    if not path or path[0] != start:
        return [start, end]
    return path


def build_cut_state(solution) -> Dict[str, object]:
    """Summarize the current synchronization cut pressure for a solution.

    The published BPC cuts are circuit/path infeasibility cuts derived from the
    synchronization graph. This helper does not try to reuse a cut pool; it
    recomputes the current critical path and uses it as the structural signal.
    """
    inst = solution.inst
    nodes, edges_out = _constraint_graph(solution)
    topo, dist, pred, acyclic = _topo_longest_path(inst, nodes, edges_out)

    if not acyclic or topo is None or dist is None or pred is None:
        critical_path = [inst.start, inst.end]
        critical_jobs: Set[int] = set(solution.selected)
        return {
            "feasible": False,
            "end_time": float("inf"),
            "slack": float("-inf"),
            "critical_path": critical_path,
            "critical_jobs": critical_jobs,
            "critical_path_len": 0,
            "pressure": float(len(solution.selected) + 10.0),
        }

    end_time = dist.get(inst.end, float("-inf"))
    if end_time == float("-inf"):
        critical_path = [inst.start, inst.end]
    else:
        critical_path = _reconstruct_path(pred, inst.start, inst.end)

    critical_jobs = {j for j in critical_path if j not in (inst.start, inst.end)}
    slack = float(inst.L - end_time) if end_time != float("-inf") else float("-inf")
    pressure = float(len(critical_jobs))
    if slack != float("-inf"):
        pressure += 1.0 / (1.0 + max(slack, 0.0))
    pressure += 0.15 * sum(len(inst.Kj.get(j, [])) for j in critical_jobs)

    return {
        "feasible": end_time <= inst.L,
        "end_time": end_time,
        "slack": slack,
        "critical_path": critical_path,
        "critical_jobs": critical_jobs,
        "critical_path_len": max(0, len(critical_path) - 2),
        "pressure": pressure,
    }


def cut_pressure_for_job(inst, cut_state: Dict[str, object], j: int) -> float:
    critical_jobs = cut_state.get("critical_jobs", set())
    pressure = 0.0
    if isinstance(critical_jobs, set) and j in critical_jobs:
        pressure += 3.0

    pressure += 0.25 * len(inst.Kj.get(j, []))

    slack = cut_state.get("slack", 0.0)
    if isinstance(slack, (int, float)) and slack != float("-inf"):
        pressure += 1.0 / (1.0 + max(float(slack), 0.0))

    return float(pressure)


def make_cut_guided_destroy(
    cut_provider: Callable[[], Dict[str, object]],
    noise: float = 1e-6,
):
    """Destroy jobs that are on or close to the current critical synchronization path."""

    def _destroy(solution, d: int) -> None:
        removable = solution.get_removable()
        if not removable:
            return

        cut_state = cut_provider()
        inst = solution.inst
        ranked = sorted(
            removable,
            key=lambda j: (
                -cut_pressure_for_job(inst, cut_state, j) + random.uniform(-noise, noise),
                -len(inst.Kj.get(j, [])),
                -inst.profits[j],
            ),
        )
        for j in ranked[: min(d, len(ranked))]:
            solution.remove(j)

    return _destroy


def make_cut_guided_repair(
    cut_provider: Callable[[], Dict[str, object]],
    positive_first: bool = True,
    cut_weight: float = 0.35,
    eps: float = 1e-9,
):
    """Repair using profit over insertion cost, penalized by cut pressure."""

    def _repair(solution) -> None:
        while solution.has_unserved():
            base_state = cut_provider()
            best_score = -float("inf")
            best_item = None
            best_pos = None

            for j in solution.get_unserved():
                cost, pos = solution.best_insertion_cost(j)
                if cost >= float("inf") or pos is None:
                    continue

                candidate = solution.copy()
                candidate.insert(j, pos)
                cand_state = build_cut_state(candidate)

                if positive_first and inst_profits(solution, j) <= 0:
                    score = -10.0 - cut_weight * float(cand_state["pressure"])
                else:
                    score = (
                        inst_profits(solution, j) / (1.0 + max(cost, eps))
                        - cut_weight * float(cand_state["pressure"])
                        - 0.1 * float(cand_state["critical_path_len"])
                    )

                # Slight preference for jobs that reduce the current cut pressure.
                base_pressure = float(base_state.get("pressure", 0.0))
                score += 0.05 * (base_pressure - float(cand_state["pressure"]))

                if score > best_score:
                    best_score = score
                    best_item = j
                    best_pos = pos

            if best_item is None:
                break
            solution.insert(best_item, best_pos)

    return _repair


def inst_profits(solution, j: int) -> float:
    return float(solution.inst.profits[j])
