from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple


@dataclass
class BPCCut:
    cut_id: str
    cut_type: str  # circuit | route_len | sync_path
    nodes: Tuple[int, ...]
    edges: Tuple[Tuple[int, int], ...]
    rhs: float
    lhs: float
    violation: float
    age: int = 0
    hit_count: int = 0


class CutPool:
    def __init__(self) -> None:
        self._cuts: Dict[str, BPCCut] = {}

    def add_many(self, cuts: Iterable[BPCCut]) -> int:
        added = 0
        for c in cuts:
            if c.cut_id not in self._cuts:
                self._cuts[c.cut_id] = c
                added += 1
        return added

    def age_and_prune(self, age_limit: int) -> None:
        drop = []
        for cid, c in self._cuts.items():
            c.age += 1
            if c.age > age_limit:
                drop.append(cid)
        for cid in drop:
            self._cuts.pop(cid, None)

    def record_hits(self, violated_ids: Iterable[str]) -> None:
        for cid in violated_ids:
            c = self._cuts.get(cid)
            if c is not None:
                c.hit_count += 1
                c.age = 0

    def all(self) -> List[BPCCut]:
        return list(self._cuts.values())

    def size(self) -> int:
        return len(self._cuts)


def _selected_nodes(solution) -> Set[int]:
    inst = solution.inst
    return {inst.start, inst.end} | set(solution.selected)


def _constraint_graph(solution) -> Tuple[Set[int], Dict[int, List[Tuple[int, float]]]]:
    inst = solution.inst
    nodes = _selected_nodes(solution)
    edges_out: Dict[int, List[Tuple[int, float]]] = {v: [] for v in nodes}
    seen_edges: Set[Tuple[int, int]] = set()

    for route in solution.routes:
        for idx in range(len(route) - 1):
            i, j = route[idx], route[idx + 1]
            if i in nodes and j in nodes and (i, j) not in seen_edges:
                edges_out[i].append((j, float(inst.T[i][j])))
                seen_edges.add((i, j))

    return nodes, edges_out


def _find_cycles(nodes: Set[int], edges_out: Dict[int, List[Tuple[int, float]]]) -> List[List[int]]:
    cycles: List[List[int]] = []
    color: Dict[int, int] = {v: 0 for v in nodes}
    stack: List[int] = []

    def dfs(u: int) -> None:
        color[u] = 1
        stack.append(u)
        for v, _ in edges_out.get(u, []):
            if color.get(v, 0) == 0:
                dfs(v)
            elif color.get(v, 0) == 1:
                # Found back-edge cycle.
                if v in stack:
                    idx = stack.index(v)
                    cyc = stack[idx:] + [v]
                    cycles.append(cyc)
        stack.pop()
        color[u] = 2

    for n in nodes:
        if color[n] == 0:
            dfs(n)
    return cycles


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
            cand = dist[u] + w
            if cand > dist[v]:
                dist[v] = cand
                pred[v] = u

    return topo, dist, pred, True


def _reconstruct_path(pred: Dict[int, Optional[int]], start: int, end: int) -> List[int]:
    if pred.get(end) is None and start != end:
        return [start, end]

    path = [end]
    cur = end
    guard = 0
    while cur != start and pred.get(cur) is not None:
        cur = pred[cur]  # type: ignore[index]
        path.append(cur)
        guard += 1
        if guard > 10000:
            break
    path.reverse()
    if not path or path[0] != start:
        return [start, end]
    return path


def _build_cut_candidates(solution, epsilon_violation: float):
    inst = solution.inst
    nodes, edges_out = _constraint_graph(solution)

    circuit_cuts: List[BPCCut] = []
    route_len_cuts: List[BPCCut] = []
    sync_cut: Optional[BPCCut] = None

    cycle_count = 0
    for cyc in _find_cycles(nodes, edges_out):
        cycle_count += 1
        cyc_nodes = tuple(cyc)
        cyc_edges = tuple((cyc[i], cyc[i + 1]) for i in range(len(cyc) - 1))
        lhs = float(len(cyc_edges))
        rhs = float(len(cyc_edges) - 1)
        viol = lhs - rhs
        if viol > epsilon_violation:
            cid = f"circuit:{','.join(map(str, cyc_nodes))}"
            circuit_cuts.append(BPCCut(cid, "circuit", cyc_nodes, cyc_edges, rhs=rhs, lhs=lhs, violation=viol))

    route_over_count = 0
    route_checked_count = 0
    max_route_over = 0.0
    for route in solution.routes:
        if len(route) < 2:
            continue
        route_checked_count += 1
        length = 0.0
        edges: List[Tuple[int, int]] = []
        for i in range(len(route) - 1):
            u, v = route[i], route[i + 1]
            length += float(inst.T[u][v])
            edges.append((u, v))
        rhs = float(inst.L)
        lhs = float(length)
        viol = lhs - rhs
        if viol > max_route_over:
            max_route_over = viol
        if viol > epsilon_violation:
            route_over_count += 1
            rid = ",".join(map(str, route))
            cid = f"route_len:{rid}"
            route_len_cuts.append(BPCCut(cid, "route_len", tuple(route), tuple(edges), rhs=rhs, lhs=lhs, violation=viol))

    topo, dist, pred, acyclic = _topo_longest_path(inst, nodes, edges_out)
    sync_candidate = False
    sync_path_found = False
    sync_over = 0.0
    if acyclic and dist is not None and pred is not None:
        end_time = dist.get(inst.end, float("-inf"))
        if end_time != float("-inf"):
            sync_path_found = True
            lhs = float(end_time)
            rhs = float(inst.L)
            sync_over = lhs - rhs
            if sync_over > epsilon_violation:
                sync_candidate = True
                path = _reconstruct_path(pred, inst.start, inst.end)
                edges = tuple((path[i], path[i + 1]) for i in range(len(path) - 1))
                cid = f"sync_path:{','.join(map(str, path))}"
                sync_cut = BPCCut(cid, "sync_path", tuple(path), edges, rhs=rhs, lhs=lhs, violation=sync_over)

    all_cuts = list(circuit_cuts) + list(route_len_cuts)
    if sync_cut is not None:
        all_cuts.append(sync_cut)

    diag = {
        "nodes_count": len(nodes),
        "edge_count": sum(len(vs) for vs in edges_out.values()),
        "cycle_count": cycle_count,
        "circuit_candidates": len(circuit_cuts),
        "route_checked_count": route_checked_count,
        "route_over_count": route_over_count,
        "route_len_candidates": len(route_len_cuts),
        "max_route_over": float(max_route_over),
        "acyclic": bool(acyclic),
        "sync_path_found": bool(sync_path_found),
        "sync_over": float(sync_over),
        "sync_candidates": 1 if sync_candidate else 0,
        "total_candidates": len(all_cuts),
    }

    return all_cuts, diag


def separate_cuts(
    solution,
    epsilon_violation: float,
    max_cuts_per_round: int,
) -> List[BPCCut]:
    cuts, _ = _build_cut_candidates(solution, epsilon_violation)
    cuts.sort(key=lambda c: c.violation, reverse=True)
    return cuts[: max(1, int(max_cuts_per_round))]


def separate_cuts_with_diagnostics(
    solution,
    epsilon_violation: float,
    max_cuts_per_round: int,
):
    cuts, diag = _build_cut_candidates(solution, epsilon_violation)
    cuts.sort(key=lambda c: c.violation, reverse=True)
    limited = cuts[: max(1, int(max_cuts_per_round))]
    diag = dict(diag)
    diag["returned_count"] = len(limited)
    return limited, diag


def evaluate_violation(solution, active_cuts: Iterable[BPCCut], epsilon_reject: float) -> Tuple[float, List[str]]:
    inst = solution.inst

    # Build edge and route summaries once.
    route_edges: Set[Tuple[int, int]] = set()
    route_lengths: Dict[int, float] = {}
    for ridx, route in enumerate(solution.routes):
        length = 0.0
        for i in range(len(route) - 1):
            u, v = route[i], route[i + 1]
            route_edges.add((u, v))
            length += float(inst.T[u][v])
        route_lengths[ridx] = length

    nodes, edges_out = _constraint_graph(solution)
    topo, dist, pred, acyclic = _topo_longest_path(inst, nodes, edges_out)
    end_time = dist.get(inst.end, float("-inf")) if (acyclic and dist is not None) else float("inf")

    total = 0.0
    violated_ids: List[str] = []
    for c in active_cuts:
        viol = 0.0
        if c.cut_type == "circuit":
            overlap = sum(1 for e in c.edges if e in route_edges)
            # Approximate violation: full overlap suggests repeated invalid precedence.
            if overlap >= max(1, len(c.edges) - 1):
                viol = 1.0
        elif c.cut_type == "route_len":
            max_len = max(route_lengths.values()) if route_lengths else 0.0
            viol = max(0.0, max_len - c.rhs)
        elif c.cut_type == "sync_path":
            if end_time != float("-inf") and end_time != float("inf"):
                viol = max(0.0, end_time - c.rhs)

        if viol > 0.0:
            total += viol
            if viol > epsilon_reject:
                violated_ids.append(c.cut_id)

    return float(total), violated_ids
