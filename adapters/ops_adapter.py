"""
OPS Adapter — Selective Routing Problem with Synchronization.

Reference:
  Riera-Ledesma, J. & Salazar-González, J.-J. (2021).
  Selective Routing Problem with Synchronization.
  Computers & Operations Research, 135, 105465.
  DOI: 10.1016/j.cor.2021.105465

Data source:
  Zenodo: https://zenodo.org/records/17557917
  GitHub: https://github.com/RieraULL/OPS-Benchmark

Strictly identical problem definition — no constraint relaxations.

Classes and functions
---------------------
OPSInstance       — parsed instance: T matrix, Jk, Kj, L, profits
compute_schedule  — longest-path feasibility (System of Difference Constraints)
OPSSolution       — solution state + operator interface for core/operators.py
parse_ops         — parse a (target_file, instance_file) pair
objective         — standalone wrapper
feasible          — standalone wrapper
"""
from __future__ import annotations

import json
from collections import defaultdict, deque
from itertools import product
from typing import Dict, List, Optional, Set, Tuple

import numpy as _np


# ---------------------------------------------------------------------------
# Instance
# ---------------------------------------------------------------------------

class OPSInstance:
    """Parsed OPS benchmark instance.

    Attributes
    ----------
    n               : number of real jobs (targets), indexed 1..n
    start           : dummy start index (= 0)
    end             : dummy end index   (= n + 1)
    T               : (n+2) × (n+2) transition-time matrix.
                      T[i][j] = p_i + c_ij  (processing time of i + cost i→j)
    Jk              : Jk[k] = job indices that processor k can serve
    Kj              : Kj[j] = processor indices that serve job j (derived)
    all_jobs        : sorted list of all unique job indices in the instance
    L               : global time limit
    profits         : profits[j] for j in 0..n+1 (0 for dummy nodes)
    num_processors  : 55 in all published OPS instances
    """

    def __init__(
        self,
        n: int,
        T: List[List[float]],
        Jk: List[List[int]],
        L: float,
        profits: List[float],
    ) -> None:
        self.n = n
        self.start: int = 0
        self.end: int = n + 1
        self.T = T
        self.Jk = Jk
        self.L = L
        self.profits = profits
        self.num_processors: int = len(Jk)

        # Kj[j] = list of processors that require job j
        kj: Dict[int, List[int]] = defaultdict(list)
        for k, jobs in enumerate(Jk):
            for j in jobs:
                kj[j].append(k)
        self.Kj: Dict[int, List[int]] = dict(kj)

        self.all_jobs: List[int] = sorted({j for jobs in Jk for j in jobs})
        self._T_np: Optional["_np.ndarray"] = None  # cached on first access

    @property
    def T_np(self) -> "_np.ndarray":
        """T matrix as a numpy float64 array (cached after first call)."""
        if self._T_np is None:
            self._T_np = _np.asarray(self.T, dtype=_np.float64)
        return self._T_np

    @classmethod
    def from_files(cls, target_path: str, instance_path: str) -> "OPSInstance":
        """Parse a (target file, instance file) pair into an OPSInstance."""
        with open(target_path, encoding="utf-8") as f:
            tgt = json.load(f)
        with open(instance_path, encoding="utf-8") as f:
            inst = json.load(f)

        n = len(tgt["targets"])
        T = inst["T"]
        Jk = inst["Jk"]
        L = float(inst["L"])
        profits = [float(b) for b in inst["b"]]

        return cls(n=n, T=T, Jk=Jk, L=L, profits=profits)

    @classmethod
    def from_instance_file(cls, instance_path: str) -> "OPSInstance":
        """Parse an instance file alone — infers n from T matrix dimensions.

        Used for extended benchmark instances (e.g. n=65-80 in ED family)
        where no separate target file exists.
        """
        with open(instance_path, encoding="utf-8") as f:
            inst = json.load(f)

        T = inst["T"]
        n = len(T) - 2            # T is (n+2) × (n+2)
        Jk = inst["Jk"]
        L = float(inst["L"])
        profits = [float(b) for b in inst["b"]]

        return cls(n=n, T=T, Jk=Jk, L=L, profits=profits)


# ---------------------------------------------------------------------------
# Feasibility: System of Difference Constraints via longest-path
# ---------------------------------------------------------------------------

def compute_schedule(
    routes: List[List[int]],
    selected: Set[int],
    inst: OPSInstance,
) -> Tuple[Optional[Dict[int, float]], bool]:
    """Compute earliest feasible start times for all jobs in the solution.

    Models the solution as a directed constraint graph:
      • Nodes : {start} ∪ selected ∪ {end}
      • Edges : for every consecutive pair (i, j) in any processor route,
                add edge i → j with weight T[i][j].

    Correctness conditions:
      (a) No directed cycle in the constraint graph — if a cycle exists, the
          synchronization ordering is contradictory and the solution is infeasible
          regardless of the time budget.
      (b) Longest path from start to end ≤ L.

    Algorithm: Kahn's topological sort (cycle detection) followed by a
    single longest-path relaxation pass in topological order.

    Returns
    -------
    (start_times, is_feasible)
        start_times : dict {job → earliest start time} or None if a cycle
                      was detected.
        is_feasible : True iff (a) no cycle and (b) path to end ≤ L.
    """
    nodes: Set[int] = {inst.start, inst.end} | selected

    # Build adjacency list; deduplicate edges (same T[i][j] for all processors)
    edges_out: Dict[int, List[Tuple[int, float]]] = {v: [] for v in nodes}
    seen_edges: Set[Tuple[int, int]] = set()

    for route in routes:
        for idx in range(len(route) - 1):
            i, j = route[idx], route[idx + 1]
            if (i, j) not in seen_edges and i in nodes and j in nodes:
                edges_out[i].append((j, float(inst.T[i][j])))
                seen_edges.add((i, j))

    # --- Kahn's topological sort (detects cycles) ---
    in_deg: Dict[int, int] = {v: 0 for v in nodes}
    for u in nodes:
        for v, _ in edges_out[u]:
            in_deg[v] += 1

    queue: deque = deque(v for v in nodes if in_deg[v] == 0)
    topo: List[int] = []

    while queue:
        u = queue.popleft()
        topo.append(u)
        for v, _ in edges_out[u]:
            in_deg[v] -= 1
            if in_deg[v] == 0:
                queue.append(v)

    if len(topo) < len(nodes):
        # Cycle detected — synchronization ordering is contradictory
        return None, False

    # --- Longest-path pass (earliest start times) ---
    dist: Dict[int, float] = {v: float("-inf") for v in nodes}
    dist[inst.start] = 0.0

    for u in topo:
        if dist[u] == float("-inf"):
            continue
        for v, w in edges_out[u]:
            candidate = dist[u] + w
            if candidate > dist[v]:
                dist[v] = candidate

    end_time = dist.get(inst.end, float("-inf"))
    if end_time > inst.L:
        return dist, False

    return dist, True


# ---------------------------------------------------------------------------
# Solution
# ---------------------------------------------------------------------------

class OPSSolution:
    """Solution state for the OPS problem.

    Representation
    --------------
    routes[k] : ordered list [start, j₁, j₂, …, end] for processor k.
                An empty route is [start, end].
    selected  : set of job indices currently chosen.

    Every job in ``selected`` appears in the route of every processor in
    Kj[j]. Conversely, every non-dummy job in any route is in ``selected``.

    Operator interface (required by core/operators.py)
    --------------------------------------------------
    get_removable, remove, relatedness, marginal_cost,
    has_unserved, get_unserved, best_insertion_cost,
    top_k_insertion_costs, insert
    """

    def __init__(self, inst: OPSInstance) -> None:
        self.inst = inst
        self.routes: List[List[int]] = [
            [inst.start, inst.end] for _ in range(inst.num_processors)
        ]
        self.selected: Set[int] = set()

    def copy(self) -> "OPSSolution":
        sol: OPSSolution = OPSSolution.__new__(OPSSolution)
        sol.inst = self.inst
        sol.routes = [list(r) for r in self.routes]
        sol.selected = set(self.selected)
        return sol

    # ------------------------------------------------------------------
    # Objective and schedule
    # ------------------------------------------------------------------

    def objective(self) -> float:
        """Total profit of selected jobs."""
        return sum(self.inst.profits[j] for j in self.selected)

    def schedule(self) -> Tuple[Optional[Dict[int, float]], bool]:
        """Return (start_times, is_feasible) for the current solution."""
        return compute_schedule(self.routes, self.selected, self.inst)

    def is_feasible(self) -> bool:
        _, ok = compute_schedule(self.routes, self.selected, self.inst)
        return ok

    # ------------------------------------------------------------------
    # Operator interface
    # ------------------------------------------------------------------

    def get_removable(self) -> List[int]:
        """Return all currently selected (removable) jobs."""
        return list(self.selected)

    def remove(self, j: int) -> None:
        """Remove job j from the solution and all required processor routes."""
        if j not in self.selected:
            return
        self.selected.discard(j)
        for k in self.inst.Kj.get(j, []):
            route = self.routes[k]
            if j in route:
                route.remove(j)

    def relatedness(self, a: int, b_job: int) -> float:
        """Symmetric travel time between two jobs.

        Lower value = more related (geographically closer).
        Used by Shaw removal to cluster removals geographically.
        """
        T = self.inst.T
        return float(T[a][b_job] + T[b_job][a])

    def marginal_cost(self, j: int) -> float:
        """Cost contribution of job j; higher → removed first by destroy_worst.

        We define this as negative profit so destroy_worst removes the
        lowest-profit jobs first (least valuable to retain).
        """
        return -float(self.inst.profits[j])

    def has_unserved(self) -> bool:
        """True if any job in the instance is not yet selected."""
        return len(self.selected) < len(self.inst.all_jobs)

    def get_unserved(self) -> List[int]:
        """Return all jobs not yet in the solution."""
        return [j for j in self.inst.all_jobs if j not in self.selected]

    def best_insertion_cost(
        self, j: int
    ) -> Tuple[float, Optional[Dict[int, int]]]:
        """Find the cheapest feasible insertion of job j.

        Returns
        -------
        (time_increase, positions_dict)
            time_increase   : increase in s[end] after insertion (0.0 if
                              inserting j does not delay the end time).
            positions_dict  : {processor_k: insertion_index_in_route}
                              for all processors in Kj[j].
        Returns (inf, None) if no feasible insertion exists.
        """
        results = self._enumerate_insertions(j, top_k=1)
        if not results:
            return float("inf"), None
        return results[0]

    def top_k_insertion_costs(
        self, j: int, k: int
    ) -> List[Tuple[float, Dict[int, int]]]:
        """Return up to k cheapest feasible insertions for job j.

        Used by repair_regret in core/operators.py to compute insertion
        regret (difference between best and second-best insertion cost).
        """
        return self._enumerate_insertions(j, top_k=k)

    def insert(self, j: int, positions: Dict[int, int]) -> None:
        """Insert job j into the solution at the given per-processor positions.

        Parameters
        ----------
        j         : job index
        positions : {processor_k: index} as returned by best_insertion_cost
                    or top_k_insertion_costs. Index refers to the position
                    in the route BEFORE this insertion.
        """
        for k in self.inst.Kj.get(j, []):
            self.routes[k].insert(positions[k], j)
        self.selected.add(j)

    # ------------------------------------------------------------------
    # Core insertion enumeration
    # ------------------------------------------------------------------

    def _enumerate_insertions(
        self, j: int, top_k: int = 1
    ) -> List[Tuple[float, Dict[int, int]]]:
        """Evaluate insertion positions for job j across all required processors.

        Strategy
        --------
        For each required processor, rank all insertion gaps by local cost:
            Δlocal = T[prev][j] + T[j][next] − T[prev][next]
        Take the top MAX_LOCAL_CANDS cheapest gaps per processor.
        Enumerate the Cartesian product of per-processor candidates, evaluate
        each combination by recomputing the global schedule, and collect all
        feasible (time_increase, positions) pairs sorted ascending.

        This keeps the search tractable: for K_j processors and
        MAX_LOCAL_CANDS=3 candidates each, at most 3^K_j combinations
        are tried (≤ 81 for Class 4 with K_j = 4).

        Returns
        -------
        List of (time_increase, positions_dict) sorted by time_increase,
        up to top_k entries. Empty list if no feasible insertion exists.
        """
        MAX_LOCAL_CANDS = 5

        inst = self.inst
        T = inst.T
        required: List[int] = inst.Kj.get(j, [])

        if not required:
            return []

        # Current end time (before insertion)
        base_times, _ = compute_schedule(self.routes, self.selected, inst)
        base_end = base_times.get(inst.end, 0.0) if base_times is not None else 0.0

        # Per-processor: sort insertion gaps by local transition cost
        proc_candidates: List[List[Tuple[float, int]]] = []
        for k in required:
            route = self.routes[k]
            local: List[Tuple[float, int]] = []
            for pos in range(1, len(route)):
                prev_node = route[pos - 1]
                next_node = route[pos]
                delta = (
                    float(T[prev_node][j])
                    + float(T[j][next_node])
                    - float(T[prev_node][next_node])
                )
                local.append((delta, pos))
            local.sort()
            proc_candidates.append(local[:MAX_LOCAL_CANDS])

        # Evaluate all position combinations globally
        results: List[Tuple[float, Dict[int, int]]] = []

        for combo in product(*proc_candidates):
            # combo[i] = (local_delta, pos) for required[i]
            positions: Dict[int, int] = {
                k: pos for k, (_, pos) in zip(required, combo)
            }

            # Shallow copy: only deep-copy the K routes that get j inserted.
            # Processors are distinct so insertions are independent.
            test_routes: List[List[int]] = list(self.routes)
            for k, pos in positions.items():
                test_routes[k] = list(self.routes[k])
                test_routes[k].insert(pos, j)

            times, feasible = compute_schedule(
                test_routes, self.selected | {j}, inst
            )
            new_end = times.get(inst.end, float("inf")) if feasible else float("inf")  # type: ignore[union-attr]

            if not feasible:
                continue
            results.append((new_end - base_end, positions))

        results.sort(key=lambda x: x[0])
        return results[:top_k]

    # ------------------------------------------------------------------
    # Independent validator
    # ------------------------------------------------------------------

    def validate(self, claimed_obj: Optional[float] = None) -> bool:
        """Full independent feasibility check.

        Verifies:
          1. Each route starts at dummy start and ends at dummy end.
          2. Each job in a route belongs to Jk[k] for that processor.
          3. Every non-dummy job in any route is in the selected set.
          4. Every selected job appears in the route of every required processor.
          5. The constraint graph is acyclic and the end time does not exceed L.
          6. The computed objective matches claimed_obj (if provided).

        Returns True on success; raises AssertionError on any violation.
        """
        inst = self.inst

        for k, route in enumerate(self.routes):
            assert route[0] == inst.start, (
                f"Route {k}: first element {route[0]} ≠ start {inst.start}"
            )
            assert route[-1] == inst.end, (
                f"Route {k}: last element {route[-1]} ≠ end {inst.end}"
            )

        for k, route in enumerate(self.routes):
            valid = set(inst.Jk[k])
            for job in route:
                if job in (inst.start, inst.end):
                    continue
                assert job in valid, (
                    f"Route {k}: job {job} not in Jk[{k}]"
                )

        for k, route in enumerate(self.routes):
            for job in route:
                if job in (inst.start, inst.end):
                    continue
                assert job in self.selected, (
                    f"Route {k}: job {job} is in the route but not in selected"
                )

        for job in self.selected:
            for k in inst.Kj.get(job, []):
                assert job in self.routes[k], (
                    f"Selected job {job} missing from route of required processor {k}"
                )

        times, ok = compute_schedule(self.routes, self.selected, inst)
        if not ok:
            if times is None:
                raise AssertionError(
                    "Infeasible: cycle in constraint graph "
                    "(contradictory synchronization ordering)"
                )
            raise AssertionError(
                f"Infeasible: schedule end time "
                f"{times.get(inst.end, '?'):.2f} > L = {inst.L}"
            )

        if claimed_obj is not None:
            actual = self.objective()
            assert abs(actual - claimed_obj) < 1e-6, (
                f"Objective mismatch: claimed {claimed_obj}, computed {actual}"
            )

        return True


# ---------------------------------------------------------------------------
# Standalone convenience wrappers (for core/search_controller.py interface)
# ---------------------------------------------------------------------------

def parse_ops(target_path: str, instance_path: str) -> OPSInstance:
    """Parse a (target_file, instance_file) pair into an OPSInstance."""
    return OPSInstance.from_files(target_path, instance_path)


def objective(solution: OPSSolution) -> float:
    """Total profit of selected jobs in solution."""
    return solution.objective()


def feasible(solution: OPSSolution) -> bool:
    """True iff the solution is synchronization-feasible and within budget L."""
    return solution.is_feasible()
