"""S0 gate: does the Lagrangian relaxed solution carry violations a cut could target?

Read-only instrument. Runs lagrangian_bound, replays the relaxation at best_mu,
and measures two independent violation channels:

  A. Consistency  - jobs partially claimed across their required processors
                    (the constraint the Lagrangian dualizes: x_{j,k} = y_j).
  B. Sync-time    - the per-machine orienteering DP enforces route length but
                    ignores synchronization waiting, which counts toward L.
                    Rebuilding the difference-constraint system over the relaxed
                    routes exposes how optimistic the DP view is.

Kill condition for the BPC program: both channels empty on every instance.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from adapters.ops_adapter import OPSInstance
from core.ops_bounds import lagrangian_bound, orienteering_dp_with_selection
from dev_bpc_replica.bpc_cut_core import _find_cycles

BENCH = os.path.join(ROOT, "benchmarks", "ops_raw", "OPS-Benchmark-master", "input")
OUT_DIR = os.path.join(ROOT, "results", "bpc_replica_dev")
ORDER_DP_CAP = 13


def _family(label):
    return label.split("_")[0]


def replay_relaxation(inst, mu):
    """Recompute y_j and per-processor selections x_{j,k} at a given mu."""
    profits = inst.profits
    y = {}
    for j, ks in inst.Kj.items():
        total_mu = sum(mu.get((j, k), 0.0) for k in ks)
        y[j] = 1 if (profits[j] - total_mu) > 0.0 else 0

    sel = {}
    for k in range(inst.num_processors):
        mod = list(profits)
        for j in inst.Jk[k]:
            mod[j] = mu.get((j, k), 0.0)
        feas = [j for j in inst.Jk[k] if mod[j] > 0.0]
        if feas:
            _, s = orienteering_dp_with_selection(
                feas, inst.T, inst.start, inst.end, mod, inst.L
            )
            sel[k] = set(s)
        else:
            sel[k] = set()
    return y, sel


def optimal_order(inst, sel_set):
    """Recover the min-time visiting order for one processor's selected jobs."""
    jobs = sorted(sel_set)
    m = len(jobs)
    if m == 0:
        return [], 0.0, False
    if m > ORDER_DP_CAP:
        # Nearest-neighbour fallback; flagged in output.
        cur = inst.start
        remaining = set(jobs)
        order = []
        total = 0.0
        while remaining:
            nxt = min(remaining, key=lambda j: float(inst.T[cur][j]))
            total += float(inst.T[cur][nxt])
            order.append(nxt)
            remaining.discard(nxt)
            cur = nxt
        total += float(inst.T[cur][inst.end])
        return order, total, True

    INF = float("inf")
    T = inst.T
    start = inst.start
    end = inst.end
    dp = [[INF] * m for _ in range(1 << m)]
    par = [[-1] * m for _ in range(1 << m)]
    for i in range(m):
        dp[1 << i][i] = float(T[start][jobs[i]])
    for mask in range(1, 1 << m):
        for i in range(m):
            if not (mask >> i & 1) or dp[mask][i] == INF:
                continue
            base = dp[mask][i]
            for j in range(m):
                if mask >> j & 1:
                    continue
                nm = mask | (1 << j)
                nt = base + float(T[jobs[i]][jobs[j]])
                if nt < dp[nm][j]:
                    dp[nm][j] = nt
                    par[nm][j] = i
    full = (1 << m) - 1
    best_i = -1
    best_t = INF
    for i in range(m):
        if dp[full][i] == INF:
            continue
        t = dp[full][i] + float(T[jobs[i]][end])
        if t < best_t:
            best_t = t
            best_i = i
    if best_i < 0:
        return jobs, INF, False
    order = []
    mask = full
    i = best_i
    while i != -1:
        order.append(jobs[i])
        pi = par[mask][i]
        mask ^= (1 << i)
        i = pi
    order.reverse()
    return order, best_t, False


def sync_makespan(inst, routes):
    """Earliest sync-feasible completion via the difference-constraint system.

    Arc (i -> j) on any processor forces s_j - s_i >= t_ij. A synchronized job is
    one shared node, so simultaneous start is enforced structurally. Earliest
    start times are the longest path from the depot.
    """
    nodes = {inst.start, inst.end}
    for r in routes.values():
        nodes |= set(r)
    adj = {v: [] for v in nodes}
    indeg = {v: 0 for v in nodes}
    seen = set()
    for k, r in routes.items():
        seq = [inst.start] + list(r) + [inst.end]
        for a, b in zip(seq, seq[1:]):
            if (a, b) in seen:
                continue
            seen.add((a, b))
            adj[a].append((b, float(inst.T[a][b])))
            indeg[b] += 1

    queue = [v for v in nodes if indeg[v] == 0]
    topo = []
    while queue:
        u = queue.pop(0)
        topo.append(u)
        for v, _ in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                queue.append(v)
    if len(topo) < len(nodes):
        # Cyclic: no feasible schedule exists at all. Count the violated circuits
        # with the same routine the cut layer uses for separation.
        cycles = _find_cycles(nodes, adj)
        return None, False, len(seen), len(cycles)

    NEG = float("-inf")
    dist = {v: NEG for v in nodes}
    dist[inst.start] = 0.0
    for u in topo:
        if dist[u] == NEG:
            continue
        for v, w in adj[u]:
            if dist[u] + w > dist[v]:
                dist[v] = dist[u] + w
    ms = dist.get(inst.end, NEG)
    return (None if ms == NEG else ms), True, len(seen), 0


def analyse(label, args):
    fam = _family(label)
    ipath = os.path.join(BENCH, fam, "instances", label + ".txt")
    inst = OPSInstance.from_instance_file(ipath)

    t0 = time.perf_counter()
    lag = lagrangian_bound(
        inst, max_iter=args.lag_max_iter, lower_bound=0.0, max_time=args.lag_max_time
    )
    lag_s = time.perf_counter() - t0
    mu = lag["best_mu"]
    y, sel = replay_relaxation(inst, mu)

    # ---- Channel A: consistency violations -----------------------------------
    multi = 0
    partial = 0
    partial_multi = 0
    under_claimed = 0
    over_claimed = 0
    for j, ks in inst.Kj.items():
        need = len(ks)
        claims = sum(1 for k in ks if j in sel.get(k, ()))
        if need >= 2:
            multi += 1
            if 0 < claims < need:
                partial_multi += 1
        if 0 < claims < need:
            partial += 1
        if y.get(j, 0) == 1 and claims < need:
            under_claimed += 1
        if y.get(j, 0) == 0 and claims > 0:
            over_claimed += 1

    g_nonzero = 0
    g_sq = 0.0
    for (j, k) in mu:
        gv = (1 if j in sel.get(k, ()) else 0) - y.get(j, 0)
        if gv:
            g_nonzero += 1
            g_sq += float(gv * gv)

    # ---- Channel B: synchronization-time optimism ----------------------------
    routes = {}
    dp_lengths = []
    nn_fallbacks = 0
    for k, s in sel.items():
        if not s:
            continue
        order, length, fb = optimal_order(inst, s)
        if fb:
            nn_fallbacks += 1
        routes[k] = order
        dp_lengths.append(length)

    max_route_len = max(dp_lengths) if dp_lengths else 0.0
    ms, acyclic, arc_count, cycle_count = sync_makespan(inst, routes)
    sync_excess = (ms - inst.L) if ms is not None else None

    coupled_nodes = 0
    for j, ks in inst.Kj.items():
        if sum(1 for k in ks if j in sel.get(k, ())) >= 2:
            coupled_nodes += 1

    return {
        "instance": label,
        "family": fam,
        "L": inst.L,
        "num_processors": inst.num_processors,
        "lag_ub": lag["upper_bound"],
        "lag_iters": lag["iterations"],
        "lag_s": round(lag_s, 2),
        "jobs_total": len(inst.Kj),
        "jobs_multiproc": multi,
        "y_selected": sum(y.values()),
        "relaxed_claimed_jobs": sum(
            1 for j in inst.Kj if any(j in sel.get(k, ()) for k in inst.Kj[j])
        ),
        "A_partial_jobs": partial,
        "A_partial_multiproc": partial_multi,
        "A_under_claimed": under_claimed,
        "A_over_claimed": over_claimed,
        "A_subgrad_nonzero": g_nonzero,
        "A_subgrad_sq": round(g_sq, 4),
        "B_max_dp_route_len": round(max_route_len, 3),
        "B_sync_makespan": (round(ms, 3) if ms is not None else "NA"),
        "B_sync_excess_over_L": (round(sync_excess, 3) if sync_excess is not None else "NA"),
        "B_sync_excess_pct_L": (
            round(100.0 * sync_excess / inst.L, 3)
            if sync_excess is not None and inst.L
            else "NA"
        ),
        "B_graph_acyclic": int(acyclic),
        "B_cycle_count": cycle_count,
        "B_arc_count": arc_count,
        "B_coupled_nodes": coupled_nodes,
        "order_dp_fallbacks": nn_fallbacks,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subset", required=True)
    p.add_argument("--tag", default="s0")
    p.add_argument("--lag-max-iter", type=int, default=200)
    p.add_argument("--lag-max-time", type=float, default=30.0)
    args = p.parse_args()

    with open(args.subset, encoding="utf-8") as f:
        labels = [
            r["instance"].strip()
            for r in csv.DictReader(f)
            if r.get("instance", "").strip()
        ]

    os.makedirs(OUT_DIR, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M")
    out = os.path.join(OUT_DIR, "bpc_s0_" + args.tag + "_" + ts + ".csv")

    rows = []
    for i, lb in enumerate(labels, 1):
        print("[" + str(i) + "/" + str(len(labels)) + "] " + lb, flush=True)
        try:
            r = analyse(lb, args)
        except Exception as e:  # keep the sweep alive; record the failure
            print("    FAILED: " + type(e).__name__ + ": " + str(e), flush=True)
            continue
        rows.append(r)
        print(
            "    A: partial={0} (multiproc {1}) |g|^2={2}   "
            "B: acyclic={3} cycles={4} makespan={5} vs L={6} excess={7} ({8}%)".format(
                r["A_partial_jobs"],
                r["A_partial_multiproc"],
                r["A_subgrad_sq"],
                r["B_graph_acyclic"],
                r["B_cycle_count"],
                r["B_sync_makespan"],
                r["L"],
                r["B_sync_excess_over_L"],
                r["B_sync_excess_pct_L"],
            ),
            flush=True,
        )

    if not rows:
        print("no rows produced")
        return

    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    tot_partial = sum(r["A_partial_jobs"] for r in rows)
    tot_partial_mp = sum(r["A_partial_multiproc"] for r in rows)
    exc = [r["B_sync_excess_over_L"] for r in rows if r["B_sync_excess_over_L"] != "NA"]
    pos_exc = [e for e in exc if e > 0]

    print("\n" + "=" * 62)
    print("S0 GATE SUMMARY  ({0} instances)  -> {1}".format(len(rows), out))
    print("=" * 62)
    print(
        "  Channel A  consistency violations : {0} jobs ({1} on multi-processor jobs)".format(
            tot_partial, tot_partial_mp
        )
    )
    cyclic_rows = [r for r in rows if not int(r["B_graph_acyclic"])]
    tot_cycles = sum(int(r["B_cycle_count"]) for r in rows)
    print(
        "  Channel B1 temporally infeasible (cyclic) instances : {0}/{1}"
        "  [{2} circuits total]".format(len(cyclic_rows), len(rows), tot_cycles)
    )
    print(
        "  Channel B2 instances with sync excess > 0 : {0}/{1}"
        "  (schedulable instances only)".format(len(pos_exc), len(exc))
    )
    if pos_exc:
        print(
            "             max excess over L : {0}  mean : {1}".format(
                max(pos_exc), round(sum(pos_exc) / len(pos_exc), 3)
            )
        )
    verdict = (
        "PASS - violations exist, S1 is worth building"
        if (tot_partial or pos_exc or cyclic_rows)
        else "FAIL - no violations on any channel; BPC program is dead"
    )
    print("  VERDICT: " + verdict)


if __name__ == "__main__":
    main()
