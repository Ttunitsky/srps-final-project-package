# OPS Problem Description
## Selective Routing Problem with Synchronization (SRPS)
*Riera-Ledesma & Salazar-González, Computers & Operations Research, 2021*

---

## The Real-World Setting

The problem comes from managing EMIR, a spectrograph at the Gran Telescopio Canarias (the world's largest optical telescope). The instrument has 55 pairs of sliding bars arranged in parallel horizontal bands. To observe an astronomical object, you physically move one or more bar pairs into position. This reconfiguration takes real time, and observing time is severely limited — the telescope is oversubscribed and each astronomer gets only a few hours.

---

## Problem Definition in Human Terms

Imagine you have:
- **55 parallel processors** (the bar pairs, one per band), each capable of visiting a sequence of customers
- **A list of potential jobs** (astronomical objects), each with a **profit** (scientific priority: values of 1, 10, or 100) and a **processing time** (how long the observation takes)
- **A global time budget L** that cannot be exceeded from start to finish

The catch: some objects span multiple bands. An object in a "dead zone" between bands requires bars from 2, 3, or even 4 contiguous bands to observe it simultaneously. This creates the **synchronization constraint**: all bar pairs assigned to a given object must **start processing that object at exactly the same time** — not approximately, exactly. They can arrive early and wait, but the start must be simultaneous.

**You must decide:**
1. Which subset of objects to observe (selection)
2. In what sequence each bar pair visits its assigned objects (routing)
3. What starting times to assign to each observation (scheduling)

All three decisions interact. Selecting an object forces multiple processors to coordinate. The route of each processor determines transition times. The synchronization forces idle waiting on some processors.

---

## Objective Function

**Maximize the total sum of profits of selected objects.**

No minimization, no cost — pure profit maximization. A profit of 100 is worth 100 times more than a profit of 1. The selection of which objects to observe is itself a decision variable — nothing is mandatory.

---

## Constraints

1. **Global time limit L:** The total elapsed time from the start position (dummy job 0) to the end position (dummy job n+1) cannot exceed L on any processor. This is a global horizon, not a per-route budget.

2. **Route structure:** Each processor must follow an elementary path — starting at a dummy parking position, visiting a sequence of assigned objects in order, and returning to parking. No object may be visited twice by the same processor.

3. **Synchronization (the hard constraint):** If an object requires processors k₁, k₂, ..., kₘ, all of those processors must start processing it at the exact same time sⱼ. The difference constraint system forces: for every arc (i→j) traversed by processor k, the start time of j minus the start time of i must be at least the transition time tᵢⱼ. Processors can wait (idle) before the synchronized start — but this waiting time counts toward the global budget L.

4. **Consistency:** If a processor visits an object, then every other processor required by that object must also visit it. You cannot partially serve a synchronized job.

---

## Instance Sizes

The benchmark has **two families** and **four classes** — 8 groups total, 60 instances per group (**480 instances total**).

### Two Families (distance type)

| Family | Transition cost | Context |
|--------|----------------|---------|
| **H** | Horizontal distance only: \|uᵢ − uⱼ\| | Mimics the real telescope (bars slide in one direction) |
| **E** | Euclidean distance: √((uᵢ−uⱼ)²+(vᵢ−vⱼ)²) | Generalizes to vehicle routing on a plane |

### Four Classes (synchronization intensity)

| Class | Bands per object | Synchronization | Difficulty |
|-------|-----------------|-----------------|------------|
| **1** | 1 | None — 55 independent orienteering problems | Easy |
| **2** | 2 | Moderate | Medium |
| **3** | 3 | Strong | Hard |
| **4** | 4 | Maximum | Hardest |

### Sizes by Class

| Class | n (targets per instance) | Why smaller as class grows |
|-------|--------------------------|---------------------------|
| 1 | 425, 450, 475, 500 | No sync — many targets can fit in budget |
| 2 | 100, 110, 120, 130 | Sync overhead reduces feasible set |
| 3 | 55, 60, 65, 70 | Even small instances become hard |
| 4 | 45, 50, 55, 60 | 4-way sync makes even 60 targets very hard |

### Time Tightness Parameter α ∈ {0.25, 0.50, 0.75}

L is set as a fraction α of the maximum TSP tour length across all processors.
- Small α (0.25) = tight budget = fewer objects selected = **harder**
- Large α (0.75) = generous budget = more objects = **easier**

Each group: 5 catalogues × 4 sizes × 3 α values = **60 instances per group**, **480 total**.

---

## Published Algorithm and Results (What We Are Competing Against)

The authors solve with **Branch-and-Price-and-Cut (BPC)** — an exact algorithm. Two variants:
- **BPC-SRPS-E:** elementary routes (stronger bounds, slower pricing)
- **BPC-SRPS-N:** non-elementary routes (faster pricing, weaker bounds)

**Time limit: 1 CPU hour per instance** on Intel Core i5 6600 3.3 GHz, 8 GB RAM.

### Results Summary

| Group | n | Solved to optimality? | Remaining gap at timeout |
|-------|---|-----------------------|--------------------------|
| H1/E1 (no sync) | 425–500 | All 5/5 | 0.00% |
| H2 (2 bands, line) | 100–130 | Most; some timeout | Up to 0.76% |
| E2 (2 bands, plane) | 100–130 | Most; some timeout | Up to 0.70% |
| H3 (3 bands, line) | 55–70 | Most; some timeout | Up to 0.08% |
| E3 (3 bands, plane) | 55–70 | Mix; harder | Up to 0.17% |
| H4 (4 bands, line) | 45–60 | Mix; several timeout | Up to 0.12% |
| **E4 (4 bands, plane)** | **45–60** | **Several timeout** | **Up to 1.64%** |

**The hardest unsolved instances:** E4 and H4 with α = 0.50 and 0.75, n = 55–60. The exact algorithm hits the 1-hour limit and leaves meaningful gaps. These are our primary competitive territory.

---

## Critical Implication for Our Framework

The authors use an **exact** algorithm. Most published solutions are **optimal**. This means:

- For **solved instances**: our heuristic target is to **match the optimal value** in much less time. This is already a strong result — matching optimal fast.
- For **unsolved instances** (H4/E4, tight α, large n): the exact algorithm timed out with a non-zero gap. Here we can genuinely **beat the incumbent** the exact solver found before timeout. If our lower bound is also tighter, we can claim a smaller certified optimality gap.

The paper explicitly notes that Class 4 instances with tight α remain hard even for their exact method. **That is our primary competitive territory.**

---

## Instance File Format (from downloaded benchmark)

Instances are JSON files organized in families (A, B, C, D = Family H; EA, EB, EC, ED = Family E):

- **`targets/`** — target coordinates `(u, v)`, processing time `p`, profit `b`
- **`instances/`** — scenario configuration including `Jk` (list of targets per processor band) and time limit `L`

Coordinates are in arcseconds within a 240 × 398.4 arcsecond field of view.
Transition costs in Family H = |uᵢ − uⱼ|; in Family E = Euclidean distance.

---

*Document created: June 2026*
*Source: Full paper read from OPS_Riera-Ledesma_Salazar-Gonzalez_2021_COR.pdf*
