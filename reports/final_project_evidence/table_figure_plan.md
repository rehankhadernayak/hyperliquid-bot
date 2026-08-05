# Table and Figure Plan — Final Project Report

Paper-only report artifacts. Values below are taken from existing local files; unsupported items are marked explicitly.

---

## Required Tables

### Table 1 — Candidate B–F Comparison

| Field | Source |
|-------|--------|
| Candidate ID (B–F) | `reports/final_project_evidence/candidate_history.json` → `comparison_table` |
| Status | same |
| Feature / method | same → `feature_or_method` |
| Coin / interval | same → `coin_interval` |
| Accepted trades | same → `accepted_trades` |
| Cumulative net bps | same → `cumulative_net_bps` |
| Average net bps | same → `average_net_bps` |
| Robust (yes/no/n/a) | same → `robust` |
| Outcome summary | same → `outcome` |

**Notes:**
- Candidate C metrics are from the **best-ranked grid config** (`ETH_DOWN_H20_C0.55_E1_M1.25_S8`); all grid configs negative. Mark footnote in report.
- Candidate D `robust: null` → display as **n/a**.

**Can produce now:** Yes — copy from `candidate_history.md` compact comparison or regenerate from JSON.

**Caption draft:** *Comparison of Candidates B through F on paper-only walk-forward or grid evaluation. Positive cumulative net bps after estimated costs do not imply robustness or market readiness.*

---

### Table 2 — Candidate F Frozen Configuration

| Field | Value (from artifacts) |
|-------|------------------------|
| config_id | `ETH_UP_H20_C0.65_E2_M1.25_S8` |
| coin | ETH |
| interval | 15s |
| horizon | 20 |
| trade_class | UP |
| feature_set | `candidate_f_regime_depth` |
| label_mode | `binary_tradeable_up` |
| min_confidence | 0.65 |
| min_edge_bps | 2.0 |
| cost_multiplier | 1.25 |
| max_spread_bps | 8.0 |

**Source:** `reports/candidate_f_milestone_comparison/candidate_f_milestone_comparison_latest.json` → `frozen_config`

**Can produce now:** Yes.

**Caption draft:** *Frozen Candidate F configuration used for final walk-forward evaluation (20260729T101645). No further tuning applied after freeze.*

---

### Table 3 — Candidate F Final Walk-Forward Metrics

| Field | Value |
|-------|-------|
| Report stamp | 20260729T101645 |
| total_windows | 705 |
| completed_windows | 695 |
| skipped_windows | 10 |
| accepted_trades | 15 |
| cumulative_net_bps | 45.164969776189494 |
| average_net_bps | 3.0109979850792996 |
| win_rate | 0.9333333333333333 |
| beats_baseline_accuracy | false |
| baseline_accuracy_mean | 0.8482014388489209 |
| accuracy_edge_vs_baseline | −0.010618705035971177 |
| average_cost_bps | 6.42368215429615 |
| blocked_by_confidence | 1376 |
| blocked_by_cost | 529 |
| blocked_by_spread | 8 |
| dataset coverage | 2023-09-16T00:00:00Z → 2023-09-30T14:59:45Z (351 h) |

**Sources:**
- `reports/final_project_evidence/candidate_f_final_summary.json`
- `reports/bid_ask_walk_forward/walk_forward_ETH_20260729T101645.json` (summary fields)
- `reports/candidate_f_milestone_comparison/candidate_f_milestone_comparison_latest.json` → `milestones[0].dataset_summary`

**Can produce now:** Yes.

**Caption draft:** *Candidate F walk-forward paper evaluation on ETH 15s historical data. Fifteen accepted UP trades passed confidence, edge, spread, and cost filters.*

---

### Table 4 — Candidate F Robustness Summary

| Field | Value |
|-------|-------|
| overall_verdict | `diagnostic_positive_sparse` |
| robust_candidate | false |
| trade_count | 15 |
| cumulative_net_bps | 45.164969776189494 |
| win_rate | 0.9333333333333333 |
| dominant_day | 2023-09-23 (86.7% of trades) |
| dominant_utc_hour | 14 (86.7% of trades) |
| dominant_volatility_regime | low_volatility_regime (86.7%) |
| dominant_spread_regime | tight_spread_regime (100%) |
| excluding_dominant_day trades | 2 |
| excluding_dominant_day cumulative_net_bps | 4.958076196600178 |
| top_1_trade_contribution_pct | 15.97% |
| top_3_trades_contribution_pct | 31.02% |
| grid vs walk-forward trade count | grid 17 vs walk-forward 15 *(both from robustness artifact)* |
| next_recommended_action | Continue paper-only data collection until ≥30 accepted trades |

**Sources:**
- `reports/candidate_f_robustness/candidate_f_robustness_latest.json`
- `reports/candidate_f_milestone_comparison/candidate_f_milestone_comparison_latest.json` → `comparison_table[0]`

**Can produce now:** Yes.

**Caption draft:** *Robustness diagnostics for frozen Candidate F. Verdict reflects sparse trade count and temporal/regime concentration, not statistical significance.*

---

### Table 5 — Dataset Coverage and Validation Summary

| Coin | Interval | Coverage (h) | First TS | Last TS | Rows | Validator passed | Missing/incomplete warnings |
|------|----------|-------------:|----------|---------|-----:|:----------------:|----------------------------:|
| BTC | 1s | 347.998 | 2023-09-16T00:00:05.332Z | 2023-09-30T11:59:59.480Z | 1,495,036 | yes | 284,130 |
| BTC | 15s | 348.0 | 2023-09-16T00:00:00Z | 2023-09-30T11:59:45Z | 66,274 | yes | 17,321 |
| BTC | 1m | 348.0 | 2023-09-16T00:00:00Z | 2023-09-30T11:59:00Z | 16,591 | yes | 4,332 |
| ETH | 1s | 347.998 | 2023-09-16T00:00:05.332Z | 2023-09-30T11:59:59.480Z | 1,583,836 | yes | 225,332 |
| ETH | 15s | 348.0 | 2023-09-16T00:00:00Z | 2023-09-30T11:59:45Z | 70,358 | yes | 13,242 |
| ETH | 1m | 348.0 | 2023-09-16T00:00:00Z | 2023-09-30T11:59:00Z | 17,610 | yes | 3,311 |

**Milestone progress (all coin/interval pairs):** 3-day ✓, 7-day ✓, 14-day ✓, 30-day ✗ (~48.33% complete).

**Sources:**
- `reports/backfill_milestones/backfill_milestones_latest.json` → `coverage`
- `data/bid_ask_csv/reports/*_validation_20260729T*.json`

**Can produce now:** Yes.

**Caption draft:** *Historical bid/ask dataset coverage and validation status as of 2026-07-29. Validators pass with expected gap warnings for sparse L2 snapshots.*

---

### Table 6 — Major Pipeline Failures and Fixes

| Date (approx.) | Problem | Fix / outcome | Evidence |
|----------------|---------|---------------|----------|
| Early Jul 2026 | Mock/bad rows in 1s CSV | Repair pass: dedupe timestamps, remove mock rows | `hyperliquid_bid_ask_research_notes.md` §6–7 |
| Early Jul 2026 | `bid_size`/`ask_size` dropped in resample | Last-valid 1s tick carried into 15s/1m bars | `whatsapp_call_briefing_20260715.md` |
| Early Jul 2026 | Backfill import stall (`no_import_progress`) | Documented; phase3 audit tracking | `hyperliquid_bid_ask_research_notes.md` §6 |
| Mid Jul 2026 | Scheduler ticks timing out | 90-minute tick timeout | `whatsapp_call_briefing_20260715.md` |
| 2026-07-29 | Stale 15s/1m tails after 1s updates | Atomic resampled writes, source-bound trimming, alignment assertions | `research_log/2026-07-29.md` |
| 2026-07-29 | Malformed numeric rows caused import failures | In-memory cleaning; preserve source files | `research_log/2026-07-29.md` |
| 2026-07-29 | Soft vs hard validation behavior | Gap warnings non-blocking; hard failures still block pipeline | `research_log/2026-07-29.md` |

**Can produce now:** Yes (narrative table from research notes and logs).

**Unsupported:** Formal incident ticket IDs or automated failure counts beyond validator warning totals — not in artifact set.

**Caption draft:** *Selected data-pipeline failures and fixes during the research period. Fixes enabled trustworthy depth-aware feature evaluation.*

---

## Recommended Figures

### Figure 1 — Candidate Development Timeline

**Purpose:** Visualize B→F research arc and key decision points (reject, supersede, freeze).

**Source files:**
- `reports/final_project_evidence/candidate_history.json`
- `reports/final_project_evidence/project_timeline.md`
- Report stamp dates from walk-forward filenames (e.g., `20260709`, `20260714`, `20260729`)

**Fields needed:** candidate ID, date, status transition, one-line outcome, optional cumulative net bps.

**Draft caption:** *Chronological development of Candidates B through F. Only Candidate F was frozen; none achieved robust market-ready status.*

**Can produce now:** Yes — timeline diagram from dated artifacts.

**Interpretation caution:** Dates inferred from report filenames and research notes; not all intermediate runs are plotted.

---

### Figure 2 — Accepted-Trade Net-Return Distribution (Candidate F)

**Purpose:** Show dispersion and outliers among 15 accepted trades.

**Source file:** `reports/bid_ask_walk_forward/accepted_trades_ETH_20260729T101645.csv`

**Columns needed:** `net_return_bps` (primary); optional `timestamp_iso`, `confidence`.

**Alternative source:** `walk_forward_ETH_20260729T101645.json` → `accepted_trade_net_returns_bps` (15 values).

**Draft caption:** *Distribution of net returns (bps) for 15 Candidate F accepted paper trades after estimated costs. One trade (−2.68 bps) is the sole loser.*

**Can produce now:** Yes — histogram or strip plot from CSV/JSON.

**Interpretation caution:** n=15; distribution shape is not statistically meaningful. Several adjacent bars share identical net returns (same window).

---

### Figure 3 — Cumulative Net Bps Across Accepted Trades

**Purpose:** Show running cumulative performance and dependence on trade ordering.

**Source file:** `reports/bid_ask_walk_forward/accepted_trades_ETH_20260729T101645.csv`

**Columns needed:** `timestamp_iso`, `net_return_bps` (compute cumulative sum in sort order).

**Draft caption:** *Cumulative net bps across Candidate F accepted trades ordered by timestamp. Most gains concentrate on 2023-09-23.*

**Can produce now:** Yes.

**Interpretation caution:** Path-dependent; reordering would change intermediate curve. Final cumulative (+45.16 bps) is order-invariant.

---

### Figure 4 — Accepted Trades by Day or Regime

**Purpose:** Expose temporal and regime concentration driving robustness failure.

**Source files:**
- `reports/candidate_f_robustness/candidate_f_robustness_latest.json` → `config_analyses[0].trade_analysis.by_day`, `by_utc_hour`, `by_volatility_regime`, `by_spread_regime`
- `reports/bid_ask_walk_forward/accepted_trades_ETH_20260729T101645.csv` → `low_volatility_regime`, `tight_spread_regime`, `timestamp_iso`

**Fields needed:**
- By day: date, count, cumulative_net_bps
- By regime: regime label, count, avg_net_bps

**Draft caption:** *Accepted Candidate F trades by calendar day (left) and volatility/spread regime (right). 13 of 15 trades occurred on 2023-09-23 in low-volatility, tight-spread conditions.*

**Can produce now:** Yes — robustness JSON has pre-aggregated buckets; CSV supports raw scatter.

**Interpretation caution:** High concentration does not prove regime-specific alpha; may reflect filter interaction with a short sample. High-volatility and wide-spread regimes have **zero** trades — absence is not evidence of negative edge there.

---

### Figure 5 — Data-Pipeline Workflow Diagram

**Purpose:** Architecture figure for report methods section.

**Source files:**
- `reports/research_notes/hyperliquid_bid_ask_research_notes.md` (§4 Architecture ASCII)
- `reports/backfill_milestones/backfill_milestones_latest.json` (data root paths)

**Fields needed:** Stage names, script names, output paths (`historical_1s/`, `historical_15s/`, validators, walk-forward).

**Draft caption:** *Paper-only bid/ask research pipeline: S3 backfill → 1s CSV → resample → validate → features → walk-forward → candidate gates. Execution paths disabled.*

**Can produce now:** Yes — diagram from research notes architecture block.

**Interpretation caution:** Diagram reflects design intent; live scheduler/orchestrator wiring may include additional audit steps not shown.

---

## Production Checklist

| Item | Status |
|------|--------|
| Tables 1–6 drafted from JSON/CSV | Ready |
| Figure 2–4 data available | Ready |
| Figure 1 dates | Ready (approximate from stamps) |
| Figure 5 architecture | Ready |
| Rendered PNG/SVG outputs | **Not yet created** — next step |
| Statistical confidence bands | **Unsupported** — not in artifacts |

---

*Plan generated 2026-08-05. No models rerun. No figure files written in this step.*
