from __future__ import annotations

from typing import Dict, List, Tuple


def floyd_warshall(
    nodes: List[int],
    arc_cost: Dict[Tuple[int, int], float],
) -> Tuple[Dict[int, Dict[int, float]], Dict[int, Dict[int, int]]]:
    """All-pairs shortest paths. Returns (dist, next_node) dicts."""
    dist = {u: {v: float('inf') for v in nodes} for u in nodes}
    nxt = {u: {v: -1 for v in nodes} for u in nodes}
    for u in nodes:
        dist[u][u] = 0.0
    for (u, v), c in arc_cost.items():
        if c < dist[u][v]:
            dist[u][v] = c
            nxt[u][v] = v
    for k in nodes:
        for i in nodes:
            for j in nodes:
                through = dist[i][k] + dist[k][j]
                if through < dist[i][j]:
                    dist[i][j] = through
                    nxt[i][j] = nxt[i][k]
    return dist, nxt


def shortest_path(
    nxt: Dict[int, Dict[int, int]], u: int, v: int
) -> List[int]:
    """Reconstruct shortest path from u to v using next_node table."""
    if nxt[u][v] == -1 and u != v:
        return []
    path = [u]
    while u != v:
        u = nxt[u][v]
        if u == -1:
            return []
        path.append(u)
    return path


def prim_mst(nodes: List[int], dist: Dict[int, Dict[int, float]]) -> float:
    """Minimum spanning tree weight via Prim's algorithm."""
    if len(nodes) <= 1:
        return 0.0
    visited = {nodes[0]}
    unvisited = set(nodes[1:])
    total = 0.0
    while unvisited:
        best = float('inf')
        best_v = -1
        for u in visited:
            for v in unvisited:
                if dist[u][v] < best:
                    best = dist[u][v]
                    best_v = v
        if best_v == -1:
            break
        total += best
        visited.add(best_v)
        unvisited.remove(best_v)
    return total
