# Project Timeline — Model Research and Pipeline Engineering

Chronological summary from local artifacts. Dates are approximate where inferred from report filename stamps (`YYYYMMDD`) or research notes. **No models were rerun for this document.**

---

## Model Research Timeline

### Candidate B — `BTC_DOWN_MICRO_H20_CANDIDATE_B`

| When | Event |
|------|-------|
| Early Jul 2026 | Initial strict config: ~5 paper trades, ~+3.8 cumulative net bps on BTC DOWN microstructure features |
| Early Jul 2026 | Relaxed filters (25–62 trades) strongly negative after costs |
| **2026-07-06** | Robustness report: 7 slices, 0 met min trades; `unstable_sparse` |
| **2026-07-09** | Threshold/regime tuning grid (12 runs) — all negative post-cost; tuning abandoned |
| **2026-07-30** | Latest validation + walk-forward (`20260730T074926`) — family **rejected** |
| Outcome | 63 accepted trades, +11.77 cum. net bps, +0.19 avg — marginal positivity but not robust |

**Key artifacts:** `candidate_validation_BTC_DOWN_MICRO_H20_CANDIDATE_B_20260730T034523.json`, `robustness_BTC_DOWN_MICRO_H20_CANDIDATE_B_20260706T115055.json`, `walk_forward_BTC_20260730T074926.json`

---

### Candidate C — `candidate_c_core`

| When | Event |
|------|-------|
| **2026-07-09** | Feature-engineering pivot from B's post-hoc filters |
| **2026-07-09** | BTC/ETH DOWN grid (`20260709T124717`) — no positive post-cost configs |
| **2026-07-09** | ETH UP/DOWN grid (`20260709T125740`) — all configs negative |
| **2026-07-09** | Walk-forward runs on BTC/ETH — confirm grid negatives |
| Outcome | Best-ranked config: 157 trades, −437.98 cum. net bps — **rejected** |

**Key artifacts:** `candidate_grid_20260709T124717.json`, `candidate_grid_20260709T125740.json`, `candidate_c_feature_engineering_plan.md`

**Missing evidence:** No dedicated rolling-slice robustness report.

---

### Candidate D — cost-aware labels

| When | Event |
|------|-------|
| **2026-07-09** | Label methodology designed after C failure |
| **2026-07-09** | Walk-forward diagnostics (`20260709T150853`) — ~66.7% NO_TRADE fraction |
| Ongoing | `cost_aware_3class` diagnostics-only; binary tradeable modes adopted by E/F |
| Outcome | 0 accepted trades as standalone candidate — **rejected** as model family; kept as infrastructure |

**Key artifacts:** `candidate_d_cost_aware_labeling_plan.md`, `walk_forward_BTC_ETH_20260709T150853.json`

---

### Candidate E — `candidate_e_depth_aware`

| When | Event |
|------|-------|
| **2026-07-14** | Depth-aware features (bid/ask size, microprice) on ETH UP H20 |
| **2026-07-14** | Walk-forward (`20260714T070309`) — 14 trades, +31.28 cum. net bps |
| **2026-07-14** | 80-config grid (`20260714T071013`) — best config only 3 trades |
| Outcome | Sparse positive signal; failed min-trade robustness (threshold 10) — **not continued** |

**Key artifacts:** `walk_forward_ETH_20260714T070309.json`, `candidate_grid_20260714T071013.json`

**Missing evidence:** No `candidate_e_robustness` report analogous to F.

---

### Candidate F — `ETH_UP_H20_C0.65_E2_M1.25_S8` (frozen)

| When | Event |
|------|-------|
| **2026-07-17** | Candidate F grid (`20260717T080653`) — 1 positive config in grid (17 trades, +28.75 cum. net bps in grid metrics) |
| **2026-07-23 – 2026-07-29** | Repeated walk-forward reruns on same dataset bounds (4 superseded by latest) |
| **2026-07-29** | Final frozen walk-forward (`20260729T101645`) — 15 trades, +45.16 cum. net bps |
| **2026-07-29** | Robustness analyzer — `diagnostic_positive_sparse`, `robust_candidate: false` |
| **2026-07-29** | Milestone comparison — 1 valid milestone (duplicate reruns collapsed) |
| Outcome | **Frozen** as final diagnostic reference; no further tuning |

**Key artifacts:** `walk_forward_ETH_20260729T101645.json`, `candidate_f_robustness_latest.json`, `candidate_f_milestone_comparison_latest.json`, `candidate_f_final_summary.json`

---

## Pipeline Engineering Timeline

Separate from model research — data reliability and operational fixes.

| When | Category | Event |
|------|----------|-------|
| Early Jul 2026 | Data quality | Mock/bad rows in early 1s CSV; repair pass (dedupe, recompute ISO timestamps) |
| Early Jul 2026 | Resampler | `bid_size` / `ask_size` were dropped during 15s/1m resample — fixed by carrying last-valid 1s tick |
| Early Jul 2026 | Backfill | Import stall (`no_import_progress`) documented; phase3 sprint toward 14-day target |
| Early Jul 2026 | Validation | BTC/ETH 1s/15s/1m validators pass with gap warnings (~80k+ missing seconds) |
| **2026-07-15** | Operations | Scheduler 90-minute tick timeout added for long backfill runs |
| Mid Jul 2026 | Coverage | Data grows toward ~183 h (briefing note) then ~348 h by 2026-07-29 |
| **2026-07-29** | Resampler | Stale 15s/1m tails diagnosed after 1s canonical file changes |
| **2026-07-29** | Resampler | Atomic resampled-file writes, source-bound trimming, alignment assertions |
| **2026-07-29** | Importer | In-memory cleaning of malformed numeric rows; source files preserved |
| **2026-07-29** | Validation | Soft gap warnings vs hard blocking failures clarified |
| **2026-07-29** | Testing | Stale-tail regression coverage; resampler test suite passed |
| **2026-07-29** | Milestone | BTC + ETH **14-day** historical milestone completed (~348 h, 48.3% toward 30-day) |

**Key artifacts:** `hyperliquid_bid_ask_research_notes.md`, `whatsapp_call_briefing_20260715.md`, `research_log/2026-07-29.md`, `backfill_milestones_latest.json`

---

## Narrative Arc (One Paragraph Each)

**Model research:** The project progressed from Candidate B's fragile BTC DOWN filter hypothesis through C's richer features (uniformly negative), D's cost-aware abstention framing (diagnostic only), E's first sparse ETH UP positive (14 trades), to Candidate F's regime-depth extension (15 trades, +45 bps, frozen). Each step increased methodological discipline; none produced a robust, baseline-beating candidate.

**Pipeline engineering:** Parallel work hardened the data layer — repairing corrupt 1s rows, restoring depth columns in resampled bars, fixing stale-tail propagation, and reaching the 14-day coverage milestone with passing validators. These fixes were prerequisites for trusting E and F depth/regime features.

---

## Report-Preparation Phase

| When | Event |
|------|-------|
| **2026-07-29** | Research log: organise metrics, evidence, tables/figures for final report |
| **2026-08-05** | Report-preparation artifacts created (`report_outline.md`, `table_figure_plan.md`, metrics tables, timeline); no tuning or execution |

---

*Generated 2026-08-05 from existing artifacts only.*
