"""Fragments for the standalone-CPLEX comparison and the syllabus coverage table.

Uses f-strings throughout: the percent-formatting operator and literal LaTeX
percent signs interact badly, and this file is full of both.
"""
from __future__ import annotations

import csv
import glob
import os
import statistics as st

ROOT = os.path.dirname(os.path.abspath(__file__))
FRAG = os.path.join(ROOT, "paper", "fragments")
EXACT = os.path.join(ROOT, "results", "exact")


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _w(name, text):
    os.makedirs(FRAG, exist_ok=True)
    with open(os.path.join(FRAG, name), "w", encoding="utf-8") as f:
        f.write(text)
    print("wrote", os.path.join("paper", "fragments", name))


def _rows(pattern):
    hits = sorted(glob.glob(os.path.join(EXACT, pattern)))
    if not hits:
        return []
    with open(hits[-1], encoding="utf-8") as f:
        return list(csv.DictReader(f))


# --------------------------------------------------------------- exact ------
def build_exact():
    rows = _rows("exact_small30_*.csv")
    if not rows:
        _w("exact_results.tex", r"\emph{Exact CPLEX comparison pending.}" + "\n")
        return

    n = len(rows)
    proven = [r for r in rows if int(r.get("proven_optimal") or 0)]
    unproven = [r for r in rows if not int(r.get("proven_optimal") or 0)]

    at_opt = [r for r in proven
              if _f(r.get("primal_subopt_pct")) is not None
              and abs(_f(r["primal_subopt_pct"])) < 1e-9]

    tighter = [r for r in rows
               if _f(r.get("cplex_bound")) is not None
               and _f(r.get("lag_ub")) is not None
               and _f(r["lag_ub"]) < _f(r["cplex_bound"]) - 1e-9]

    def cmp_inc(rs, op):
        out = []
        for r in rs:
            a, b = _f(r.get("cplex_obj")), _f(r.get("alns_z"))
            if a is None or b is None:
                continue
            if op == "<" and a < b - 1e-9:
                out.append(r)
            elif op == "=" and abs(a - b) < 1e-9:
                out.append(r)
            elif op == ">" and a > b + 1e-9:
                out.append(r)
        return out

    worse, equal, better = (cmp_inc(unproven, o) for o in ("<", "=", ">"))

    t = []
    t.append(
        "The ceiling argument bounds any strengthening by $(ub-z)/ub$ because the "
        "optimum $P^\\ast$ is known only to lie in $[z,ub]$. Solving instances "
        "exactly would pin $P^\\ast$ down and convert that bound into a "
        "measurement. We therefore solved the compact SRPS-1 arc-flow model "
        "directly in CPLEX~22.1.1---standalone, sharing no state with the ALNS or "
        "the Lagrangian---on the "
        f"{n} instances with $n\\le 50$ that still carry a nonzero certified gap. "
        "Instances already certified optimal are excluded: their decomposition is "
        "trivially zero."
    )
    t.append("")
    t.append(
        f"CPLEX closed {len(proven)} of {n} within a 180-second limit. Only those "
        "admit a valid decomposition; on a timed-out run CPLEX's incumbent is not "
        f"$P^\\ast$, and on {len(worse)} of the {len(unproven)} unproven instances "
        "it was strictly worse than the ALNS incumbent."
    )
    t.append("")
    t.append(r"\begin{table}[htbp]")
    t.append(r"\centering")
    t.append(
        r"\caption{The " + str(len(proven)) + r" instances CPLEX closed. "
        r"\emph{looseness} is $(ub-P^\ast)/ub$, recoverable by a tighter dual; "
        r"\emph{subopt} is $(P^\ast-z)/ub$, which no dual improvement can reach.}"
    )
    t.append(r"\label{tab:exact}")
    t.append(r"\begin{tabular}{lrrrrr}")
    t.append(r"\toprule")
    t.append(r"Instance & $P^\ast$ & $z$ & $ub$ & looseness & subopt \\")
    t.append(r"\midrule")
    for r in proven:
        lbl = r["instance"].replace("_", r"\_")
        t.append(
            rf"\texttt{{{lbl}}} & {_f(r['cplex_obj']):.0f} & {_f(r['alns_z']):.0f} & "
            rf"{_f(r['lag_ub']):.0f} & {_f(r['bound_looseness_pct']):.2f}\% & "
            rf"{_f(r['primal_subopt_pct']):.2f}\% \\"
        )
    t.append(r"\bottomrule")
    t.append(r"\end{tabular}")
    t.append(r"\end{table}")
    t.append("")
    t.append(
        f"The ALNS is \\emph{{provably optimal}} on {len(at_opt)} of these "
        f"{len(proven)}, so on those the entire certified gap is bound looseness "
        "and nothing is left for the primal to recover. Across the unproven "
        f"instances CPLEX matched the ALNS on {len(equal)} and beat it on "
        f"{len(better)}, which is external verification of incumbent quality from "
        "a solver sharing no code with the heuristic."
    )
    t.append("")
    if tighter:
        t.append(
            f"The more useful finding concerns the bound. On {len(tighter)} of {n} "
            "instances the Lagrangian certificate is \\emph{strictly tighter} than "
            "the dual bound CPLEX reaches within its limit. The theory anticipates "
            "this: the Lagrangian subproblem is an orienteering problem solved "
            "exactly by dynamic programming, and orienteering lacks the integrality "
            "property, so its Lagrangian bound may strictly dominate the LP "
            "relaxation of the compact model. It does here. This is also why the "
            "literature attacks SRPS by branch-price-and-cut rather than solving "
            "SRPS-1 directly: the compact relaxation is weak."
        )
        t.append("")
    t.append(
        "The decomposition the experiment was built for is nonetheless limited, "
        "for a structural reason worth recording. A direct MILP reaches only small "
        "instances, and small instances are largely closed already---75\\% of those "
        f"with $n\\le 50$ have a certified gap of exactly zero---leaving {n} "
        f"candidates of which CPLEX closed {len(proven)}. The instances where the "
        "split would matter carry gaps above 1.5\\% at $n=110$--$150$, far beyond "
        "reach when the specialised branch-price-and-cut of the literature closes "
        "only $n\\le 60$. On the hard tail the ceiling remains an analytic bound; "
        "it cannot be replaced by measurement."
    )
    _w("exact_results.tex", "\n".join(t) + "\n")


# ------------------------------------------------------------ coverage ------
# Two context rows first (not interventions), then the numbered techniques.
# Numbering is deliberate: rows 1-10 are the strengthening attempts; the two
# unnumbered rows are the bound being strengthened and the reference solver.
CONTEXT = [
    ("Lagrangian relaxation", "dual $ub$",
     "the method under study", r"\ref{sec:lagbound}"),
    ("Compact MILP (SRPS-1) in CPLEX", "reference",
     "looser bound than the Lagrangian on 22/30", r"\ref{sec:exact}"),
]

COVERAGE = [
    ("BPC-style cut separation", "dual $ub$",
     "falsified on three independent grounds", r"\ref{sec:cuts}"),
    ("Dual-guided destroy / repair", "primal $z$",
     r"$+0.0037\pp$ vs.\ warm control at production budget",
     r"\ref{sec:guided}, \S\ref{sec:prod}"),
    ("Warm-$\\mu$ dual effort scheduling", "dual $ub$",
     r"$-0.0086\pp$ at production budget", r"\ref{sec:prod}"),
    ("Primal-to-dual seeding", "dual $ub$",
     "9 tighter / 8 looser after flooring", r"\ref{sec:seeding}"),
    ("Dual stabilisation (trust region, smoothing)", "dual $ub$",
     "budget-dependent; no transfer at the solver's budget", r"\ref{sec:convergence}"),
    ("Lagrangian decomposition (variable copies)", "dual $ub$",
     "provably equal to the existing bound", r"\ref{sec:ld}"),
    ("Dantzig--Wolfe reformulation / column generation", "dual $ub$",
     "master bound equals the Lagrangian dual optimum", r"\ref{sec:combining}"),
    ("Surrogate relaxation", "dual $ub$",
     "dominates in theory; forfeits separability", r"\ref{sec:combining}"),
    ("Reduced-cost variable fixing", "runtime",
     r"reach $24.8\%\to0.3\%$ at $0.99z$", r"\ref{sec:fixing}"),
    ("Parallel subproblem evaluation", "runtime",
     r"$\mathbf{1.42\times}$ dual, bit-identical bounds", r"\ref{sec:parallel}"),
]


def build_coverage():
    t = []
    t.append(
        "Table~\\ref{tab:coverage} lists every technique evaluated, the metric it "
        "targets, and its outcome. \\textbf{The numbered rows are the ten "
        "strengthening techniques} and are the subject of "
        "Sections~\\ref{sec:dual}--\\ref{sec:runtime}. The two unnumbered rows "
        "above the rule are not interventions: one is the solver's own dual bound, "
        "the object the ten techniques try to strengthen, and the other is an "
        "independent reference solver used for comparison in "
        "Section~\\ref{sec:exact}."
    )
    t.append("")
    t.append(
        "The organising observation is the split by target. Interventions aimed at "
        "the certified gap are bounded in advance by the ceiling of "
        "Table~\\ref{tab:ceiling}, and every one of them was absorbed by it; the "
        "two aimed at runtime carry no such bound, and one of them delivered."
    )
    t.append("")
    t.append(r"\begin{table}[htbp]")
    t.append(r"\centering")
    t.append(r"\small")
    t.append(
        r"\caption{Techniques evaluated in this study. Rows 1--10 are the "
        r"strengthening attempts; the two rows above the rule are the bound being "
        r"strengthened and the reference solver. Section references point to the "
        r"derivation and the supporting measurement.}"
    )
    t.append(r"\label{tab:coverage}")
    t.append(r"\begin{tabular}{clll}")
    t.append(r"\toprule")
    t.append(r"\# & Technique & Target & Outcome \\")
    t.append(r"\midrule")
    for tech, target, outcome, ref in CONTEXT:
        t.append(rf"--- & \emph{{{tech}}} & \emph{{{target}}} & \emph{{{outcome}}} (\S{ref}) \\")
    t.append(r"\midrule")
    for i, (tech, target, outcome, ref) in enumerate(COVERAGE, 1):
        t.append(rf"{i} & {tech} & {target} & {outcome} (\S{ref}) \\")
    t.append(r"\bottomrule")
    t.append(r"\end{tabular}")
    t.append(r"\end{table}")
    _w("coverage_results.tex", "\n".join(t) + "\n")


if __name__ == "__main__":
    build_exact()
    build_coverage()
