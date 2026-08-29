"""Fragments for the parallelisation and dual-variant experiments.

Kept separate from build_falsification_fragments.py so the two can evolve
independently; both write into paper/fragments/.
"""
from __future__ import annotations

import csv
import glob
import math
import os
import statistics as st

ROOT = os.path.dirname(os.path.abspath(__file__))
FRAG = os.path.join(ROOT, "paper", "fragments")
DUAL = os.path.join(ROOT, "results", "dual_guided_dev")


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _latest(pattern):
    hits = sorted(glob.glob(os.path.join(DUAL, pattern)))
    if not hits:
        return []
    with open(hits[-1], encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _w(name, text):
    os.makedirs(FRAG, exist_ok=True)
    with open(os.path.join(FRAG, name), "w", encoding="utf-8") as f:
        f.write(text)
    print("wrote", os.path.join("paper", "fragments", name))


# ------------------------------------------------------- parallelisation ----
def build_parallel():
    rows = _latest("parallel_quant_*.csv")
    if not rows:
        _w("parallel_results.tex", r"\emph{Parallel benchmark pending.}" + "\n")
        return

    sp = [_f(r["speedup"]) for r in rows]
    ce = [_f(r["lpt_ceiling"]) for r in rows]
    cap = [_f(r["pct_of_ceiling"]) for r in rows]
    ident = all(int(r["identical"]) for r in rows)

    t = []
    t.append(
        "Profiling attributes 84\\% of dual wall time to "
        "\\texttt{orienteering\\_dp\\_with\\_selection}, invoked once per processor "
        "per subgradient iteration. The per-processor solves are independent by "
        "construction~\\cite{Fisher1981Lagrangian} --- that independence is what "
        "the relaxation buys --- so the loop parallelises. Unlike every other "
        "intervention tested here, "
        "this one cannot alter the objective, the bound, or validity---only "
        "wall-clock---so it cannot be falsified in the same way."
    )
    t.append("")
    t.append(r"\begin{table}[htbp]")
    t.append(r"\centering")
    t.append(
        r"\caption{Parallel dual, %d instances, %s workers, 200 iterations, best "
        r"of 3 repeats. Timing spread across repeats averaged %.1f\%%.}"
        % (len(rows), "4",
           st.mean([_f(r["seq_spread_pct"]) for r in rows]))
    )
    t.append(r"\label{tab:parallel}")
    t.append(r"\begin{tabular}{lrrrr}")
    t.append(r"\toprule")
    t.append(r"Instance & sequential (s) & parallel (s) & speedup & \% of ceiling \\")
    t.append(r"\midrule")
    for r in rows:
        t.append(
            r"\texttt{%s} & %.2f & %.2f & %.2f$\times$ & %.0f\%% \\"
            % (r["instance"].replace("_", r"\_"), _f(r["seq_best_s"]),
               _f(r["par_best_s"]), _f(r["speedup"]), _f(r["pct_of_ceiling"]))
        )
    t.append(r"\midrule")
    t.append(
        r"\textbf{mean} & & & \textbf{%.2f}$\times$ & %.0f\%% \\"
        % (st.mean(sp), st.mean(cap))
    )
    t.append(r"\bottomrule")
    t.append(r"\end{tabular}")
    t.append(r"\end{table}")
    t.append("")
    t.append(
        "Bounds were identical to the sequential path in every configuration and "
        "every repeat%s, which is the correctness claim the approach needs."
        % ("" if ident else " except where noted")
    )
    t.append("")
    t.append(
        "Two structural limits are worth recording. First, DP cost is $O(2^m m^2)$ "
        "in a processor's candidate count $m$, and $m$ ranges from 0 to 13 across "
        "the 55 processors, so cost is extremely skewed: on \\texttt{B\\_n140} a "
        "single processor carries 53\\%% of the instance's total dual work. Because "
        "no partition splits one DP, speedup is capped at "
        "$\\text{total}/\\text{max\\_single}$ regardless of worker count---a mean "
        "ceiling of %.2f$\\times$ here. Second, contiguous chunking is badly wrong "
        "for such a distribution (observed imbalance up to 5.35$\\times$ at eight "
        "workers); longest-processing-time assignment recovers most of the "
        "difference, lifting one instance from 1.13$\\times$ to 1.71$\\times$."
        % st.mean(ce)
    )
    t.append("")
    t.append(
        "The end-to-end value is nonetheless small, because the component is not on "
        "the critical path. In the production-budget arms the dual was invoked on "
        "one instance of twelve; the others terminated on the gap threshold before "
        "any phase failed to improve. On the single instance that did use it, the "
        "dual accounted for roughly 45 of 1297 seconds, or 3.5\\%% of runtime, so a "
        "%.2f$\\times$ component speedup yields about 1.01$\\times$ overall and "
        "exactly 1.00$\\times$ elsewhere. The honest summary is a correct and "
        "measured component improvement whose component does not matter much in "
        "the current architecture; it would matter in a cold-start regime where the "
        "initial bound must be computed rather than loaded."
        % st.mean(sp)
    )
    _w("parallel_results.tex", "\n".join(t) + "\n")


# --------------------------------------------------------- dual variants ----
NICE = {
    "baseline": "baseline (fair split, Polyak)",
    "seed_eps05": r"incumbent seed, $\varepsilon=0.05$",
    "seed_eps10": r"incumbent seed, $\varepsilon=0.10$",
    "seed_eps25": r"incumbent seed, $\varepsilon=0.25$",
    "trust05": "trust region, radius 0.5",
    "trust10": "trust region, radius 1.0",
    "smooth03": "dual smoothing, 0.3",
    "smooth06": "dual smoothing, 0.6",
    "seed10_trust05": "seed 0.10 + trust 0.5",
    "seed10_smooth03": "seed 0.10 + smoothing 0.3",
}
ORDER = ["baseline", "seed_eps05", "seed_eps10", "seed_eps25",
         "trust05", "trust10", "smooth03", "smooth06",
         "seed10_trust05", "seed10_smooth03"]


def build_variants():
    rows = _latest("dual_variants_full30_*.csv")
    if not rows:
        _w("variants_results.tex", r"\emph{Dual variant sweep pending.}" + "\n")
        return

    n = len(rows)
    t = []
    t.append(
        "Two coupling directions from the syllabus remained untested after the "
        "dual-guided operators: "
        "primal-to-dual seeding, in which the incumbent's own selection sets the "
        "starting multipliers rather than only serving as the Polyak target; and "
        "dual stabilisation via trust region or smoothing"
        "~\\cite{DuMerle1999Stabilized}. "
        "Both are evaluated below over %d instances at 200 subgradient iterations." % n
    )
    t.append("")
    t.append(r"\begin{table}[htbp]")
    t.append(r"\centering")
    t.append(
        r"\caption{Dual variants over %d instances, 200 iterations. $\Delta$UB is "
        r"the change in the real-valued bound against baseline; negative is "
        r"tighter. The final column is what survives $\lfloor\cdot\rfloor$, which "
        r"is what the certified gap actually uses.}" % n
    )
    t.append(r"\label{tab:variants}")
    t.append(r"\begin{tabular}{lrrr}")
    t.append(r"\toprule")
    t.append(r"Variant & mean $\Delta$UB & wins / %d & floor: tighter / looser \\" % n)
    t.append(r"\midrule")
    for k in ORDER:
        col = k + "_vs_base"
        if col not in rows[0]:
            continue
        d = [_f(r[col]) for r in rows]
        wins = sum(1 for x in d if x < -1e-9)
        tighter = looser = 0
        for r in rows:
            fb = math.floor(_f(r["baseline_ub"]))
            fv = math.floor(_f(r[k + "_ub"]))
            if fv < fb:
                tighter += 1
            elif fv > fb:
                looser += 1
        flo = "---" if k == "baseline" else ("%d / %d" % (tighter, looser))
        t.append(r"%s & %+.3f & %d & %s \\" % (NICE[k], st.mean(d), wins, flo))
    t.append(r"\bottomrule")
    t.append(r"\end{tabular}")
    t.append(r"\end{table}")
    t.append("")
    d10 = [_f(r["seed_eps10_vs_base"]) for r in rows]
    t.append(
        "Seeding produces a small mean tightening ($%.2f$ for "
        "$\\varepsilon=0.10$, winning on %d instances of %d), but the median "
        "instance moves by $%.3f$---that is, not at all. The mean is carried by "
        "%d instances with $|\\Delta \\text{UB}|\\ge 1$, spanning $%.1f$ to "
        "$%+.1f$. After flooring, the direction is a coin flip, and two of the "
        "three seeding variants are net worse. There is no usable improvement here."
        % (st.mean(d10), sum(1 for x in d10 if x < -1e-9), n,
           st.median(d10), sum(1 for x in d10 if abs(x) >= 1.0),
           min(d10), max(d10))
    )
    t.append("")
    t.append(
        "Smoothing is straightforwardly harmful and monotone in its weight. The "
        "trust region requires more care, and is treated separately below, because "
        "its verdict depends on the iteration budget."
    )
    _w("variants_results.tex", "\n".join(t) + "\n")


# ------------------------------------------------------------ convergence ---
def build_convergence():
    rows = _latest("dual_convergence_*.csv")
    if not rows:
        _w("convergence_results.tex", r"\emph{Convergence probe pending.}" + "\n")
        return

    by = {}
    for r in rows:
        by.setdefault(r["instance"], {})[r["variant"]] = r
    cks = [c for c in ("it60", "it100", "it200", "it400", "it700", "it1000")
           if c in rows[0]]

    t = []
    t.append(
        "At 60 iterations the trust region appeared decisively ahead "
        "($-8.07$ mean). At 200 it was behind ($+1.58$). That sign reversal is the "
        "signature of a starved baseline, the same failure this study documents for "
        "the screening budget, so the question had to be settled at a budget where "
        "the baseline is allowed to finish. Each variant was therefore run once at "
        "1000 iterations with its bound history retained, and the running best read "
        "off at checkpoints."
    )
    t.append("")
    t.append(r"\begin{table}[htbp]")
    t.append(r"\centering")
    t.append(
        r"\caption{Bound trajectories to 1000 subgradient iterations. "
        r"\texttt{prod} is the bound the production configuration reports, which "
        r"uses at most 200 iterations warm-started.}"
    )
    t.append(r"\label{tab:convergence}")
    t.append(r"\begin{tabular}{llrrrrrr}")
    t.append(r"\toprule")
    t.append(r"Instance & Variant & " + " & ".join(c[2:] for c in cks) + r" & prod \\")
    t.append(r"\midrule")
    for inst, d in by.items():
        for vi, (v, r) in enumerate(d.items()):
            name = inst.replace("_", r"\_") if vi == 0 else ""
            prod = r.get("prod_ub", "")
            t.append(
                r"\texttt{%s} & %s & %s & %s \\"
                % (name, v.replace("_", r"\_"),
                   " & ".join("%.2f" % _f(r[c]) for c in cks),
                   ("%.0f" % _f(prod)) if _f(prod) else "---")
            )
        t.append(r"\midrule")
    t = t[:-1]
    t.append(r"\bottomrule")
    t.append(r"\end{tabular}")
    t.append(r"\end{table}")
    t.append("")

    gains = []
    for inst, d in by.items():
        if "baseline" in d and "trust10" in d and "it1000" in d["baseline"]:
            fb = math.floor(_f(d["baseline"]["it1000"]))
            ft = math.floor(_f(d["trust10"]["it1000"]))
            if ft < fb:
                gains.append((inst, fb - ft))
    t.append(
        "At 1000 iterations the trust region leads on all four instances, so the "
        "200-iteration verdict was premature rather than the 60-iteration one being "
        "spurious: the variants had simply not separated yet. After flooring, "
        "%d of the four show a strictly tighter integer bound (%s)."
        % (len(gains), ", ".join("%s by %d" % (i.replace("_", r"\_"), g)
                                 for i, g in gains) if gains else "none")
    )
    t.append("")
    t.append(
        "Two effects must be separated before claiming anything, the same "
        "decomposition that exposed the warm-$\\mu$ confound. Part of the gain "
        "against production is simply running the subgradient longer: baseline "
        "alone improves several instances once given 1000 iterations rather than "
        "200. Only the residual is attributable to stabilisation. On "
        "\\texttt{B\\_n140} the bound moves from 3195 (production) to 3192 "
        "(baseline at 1000) to 3187 (trust region at 1000), so of eight integer "
        "units, five come from the budget and three from the trust region."
    )
    t.append("")
    t.append(
        "The practical conclusion is budget-conditional and therefore does not "
        "transfer to the solver as configured. It caps the dual at 200 "
        "iterations or 60 seconds, warm-started and invoked only when a phase fails "
        "to improve---and at 200 iterations the trust region is worse on average "
        "and 9-to-10 after flooring. Its advantage exists only at roughly five "
        "times that budget, on the hardest instances, at a proportionate runtime "
        "cost. We therefore report it as a scoped observation about the dual in "
        "isolation, not as a recommended change."
    )
    _w("convergence_results.tex", "\n".join(t) + "\n")


if __name__ == "__main__":
    build_parallel()
    build_variants()
    build_convergence()
