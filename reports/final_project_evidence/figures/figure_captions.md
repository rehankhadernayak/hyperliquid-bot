# Figure Captions — Final Project Report

Generated 2026-08-05 from existing artifacts only. No models rerun.

---

## Figure 1 — Candidate F Cumulative Net Return

**File:** `candidate_f_cumulative_net_bps.png`

**Caption:** Cumulative net return (bps) across 15 accepted Candidate F paper trades after estimated costs, ordered exactly as stored in the accepted-trades CSV. Config `ETH_UP_H20_C0.65_E2_M1.25_S8`. Final cumulative net bps: +45.16.

**Interpretation cautions:**
- n=15 is a **sparse sample**; the curve is descriptive, not predictive.
- Trade ordering follows CSV row order, not chronological re-sorting; intermediate path shape is order-dependent.
- Final cumulative total is order-invariant but intermediate values are not.
- **Do not infer statistical significance** from this curve.
- Most gains concentrate on 2023-09-23 (13 of 15 trades per robustness diagnostics).

---

## Figure 2 — Candidate F Return Distribution

**File:** `candidate_f_return_distribution.png`

**Caption:** Distribution of net returns (bps) for 15 Candidate F accepted paper trades (config `ETH_UP_H20_C0.65_E2_M1.25_S8`). Histogram (left) and individual trade values (right). One trade (−2.68 bps) is the sole loser; mean net return is +3.01 bps.

**Interpretation cautions:**
- n=15; distribution shape is **not statistically meaningful**.
- Several adjacent trades share identical net returns (same walk-forward window on 2023-09-23).
- Win rate (14/15) should not be extrapolated to future periods.
- **Do not imply statistical significance** or generalization from this sample.

---

## Figure 3 — Candidate Development Timeline

**File:** `candidate_development_timeline.png`

**Caption:** Chronological development of Candidates B through F on paper-only walk-forward and grid evaluation. Color coding: red = rejected, orange = diagnostic-positive (sparse), blue = frozen. Only Candidate F was frozen; none achieved robust market-ready status.

**Interpretation cautions:**
- Dates are approximate where inferred from report filename stamps (`YYYYMMDD`) and research notes.
- Not all intermediate runs are plotted; key decision points only.
- Candidate C metrics reflect best-ranked grid config (still negative).
- Candidate D shown as rejected standalone; label infrastructure adopted by E/F.
- Missing evidence: dedicated robustness reports for Candidates C and E.

---

## Figure 4 — Data Pipeline Workflow

**File:** `data_pipeline_workflow.png`

**Caption:** Paper-only bid/ask research pipeline: S3 backfill → 1s CSV → resample → validate → features → walk-forward → candidate gates. Dark boxes = research logic; green boxes = engineering safeguards. Execution paths disabled.

**Interpretation cautions:**
- Diagram reflects design intent from research notes and logs.
- Live scheduler/orchestrator wiring may include additional audit steps not shown.
- Engineering fixes (2026-07-29 stale-tail, atomic writes, in-memory cleaning) were prerequisites for trusting depth/regime features in Candidates E and F.
- Dataset coverage ~348 h as of 2026-07-29 milestone; 30-day target not yet reached.

---

*All figures generated from existing local artifacts. Paper-only research; execution disabled.*
