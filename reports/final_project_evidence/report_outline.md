# Final Project Report — Outline

Paper-only research report for Hyperliquid BTC/ETH bid/ask microstructure ML. All sections map to local artifacts; unsupported claims are flagged where evidence is missing.

**Research-only safeguards:** `paper_only: true`, `execution_enabled: false`, `real_execution_ready: false` across final evidence artifacts.

---

## 1. Executive Summary

**Purpose:** One-page synthesis of the B→F candidate arc, frozen Candidate F outcome, pipeline maturity, and honest negative/constrained-positive findings.

**Key points to cover:**
- Candidate F (`ETH_UP_H20_C0.65_E2_M1.25_S8`) is **frozen** with `diagnostic_positive_sparse` verdict: +45.16 cumulative net bps on 15 accepted trades, `robust_candidate: false`.
- No candidate B–F demonstrated robust, baseline-beating, market-ready paper performance.
- Data pipeline reached the **14-day milestone** (~348 h) but not the **30-day milestone** (~48% complete).
- Real execution remains disabled throughout.

**Supporting artifacts:**
- `reports/final_project_evidence/candidate_f_final_summary.json`
- `reports/final_project_evidence/candidate_history.json`
- `reports/final_project_evidence/candidate_history.md`
- `reports/final_project_evidence/evidence_inventory.json`
- `reports/research_status/research_status_latest.json`

---

## 2. Research Question

**Purpose:** State the primary and secondary questions the project attempted to answer.

**Questions:**
- Can microstructure features on Hyperliquid BTC/ETH L2 bid/ask data produce **paper-only** trade signals that remain positive after estimated costs (fees + slippage + spread)?
- Which labeling, feature-engineering, and filter strategies survive walk-forward evaluation and robustness checks?
- What pipeline and validation infrastructure is required to test such claims honestly?

**Supporting artifacts:**
- `reports/research_notes/hyperliquid_bid_ask_research_notes.md`
- `reports/research_notes/candidate_c_feature_engineering_plan.md`
- `reports/research_notes/candidate_d_cost_aware_labeling_plan.md`
- `reports/final_project_evidence/candidate_history.json`

---

## 3. Project Scope and Paper-Only Safeguards

**Purpose:** Document boundaries, flags, and what the project explicitly does **not** do.

**Points:**
- No wallets, signing, orders, leverage, or exchange execution endpoints.
- Simulated returns are for research validation design, not profitability proof.
- Candidate F frozen; no further tuning in final-report phase.

**Supporting artifacts:**
- `reports/research_notes/hyperliquid_bid_ask_research_notes.md` (§2 Safety Boundary)
- `reports/candidate_f_robustness/candidate_f_robustness_latest.json` (`safety_note`, `paper_only`, `execution_enabled`)
- `reports/candidate_f_milestone_comparison/candidate_f_milestone_comparison_latest.json` (`safety_note`, `paper_only`)
- `reports/backfill_milestones/backfill_milestones_latest.json` (`safety_statement`, `research_only_note`)
- `reports/bid_ask_walk_forward/walk_forward_ETH_20260729T101645.json` (`paper_only`, `execution_enabled`)

---

## 4. Data Collection and Validation

**Purpose:** Describe L2 backfill, CSV storage, coverage milestones, validation gates, and known gaps.

**Points:**
- BTC/ETH historical 1s → 15s/1m resample pipeline; Requester Pays S3 backfill.
- Coverage as of 2026-07-29: ~348 h per coin/interval (2023-09-16 → 2023-09-30); 14-day milestone met, 30-day at ~48.3%.
- Validators pass with missing-period warnings (e.g., ETH 15s: 13,242 incomplete periods).
- Depth columns (`bid_size`, `ask_size`) present; microprice features expected.

**Supporting artifacts:**
- `reports/backfill_milestones/backfill_milestones_latest.json`
- `data/bid_ask_csv/reports/BTC_15s_validation_20260729T090909Z.json`
- `data/bid_ask_csv/reports/BTC_1m_validation_20260729T090909Z.json`
- `data/bid_ask_csv/reports/BTC_1s_validation_20260729T090841Z.json`
- `data/bid_ask_csv/reports/ETH_15s_validation_20260729T090943Z.json`
- `data/bid_ask_csv/reports/ETH_1m_validation_20260729T090943Z.json`
- `data/bid_ask_csv/reports/ETH_1s_validation_20260729T090913Z.json`
- `reports/research_log/2026-07-29.md`
- `reports/research_notes/hyperliquid_bid_ask_research_notes.md` (§5–7)

---

## 5. Mathematical and Statistical Methods

**Purpose:** Walk-forward protocol, classifier choice, cost model, filter gates, and robustness criteria.

**Points:**
- Rolling walk-forward windows on 15s bars; Random Forest classifier (per research notes architecture).
- Cost formula: `estimated_cost_bps = fee_bps + slippage_bps + spread_bps`; cost multiplier 1.25 on filter thresholds.
- Post-prediction filters: min confidence, min edge, max spread, trade class.
- Robustness: trade-count thresholds, regime/day concentration, milestone comparison, rolling-slice checks (Candidate B).

**Supporting artifacts:**
- `reports/bid_ask_walk_forward/walk_forward_ETH_20260729T101645.json`
- `reports/candidate_f_robustness/candidate_f_robustness_latest.json`
- `reports/bid_ask_candidate_validation/robustness_BTC_DOWN_MICRO_H20_CANDIDATE_B_20260706T115055.json`
- `reports/research_notes/hyperliquid_bid_ask_research_notes.md` (§4 Architecture)

**Unsupported / partial evidence:**
- Formal statistical significance tests not located in artifact set.
- Candidate C lacks dedicated rolling-slice robustness report (`candidate_history.json` → `missing_evidence_summary`).

---

## 6. Feature Engineering

**Purpose:** Trace feature-set evolution from microstructure (B) through regime-depth (F).

**Evolution:**
| Candidate | Feature set | Notes |
|-----------|-------------|-------|
| B | `microstructure` | Post-hoc filters on direction labels |
| C | `candidate_c_core` | Spread regime, imbalance dynamics, returns, time |
| E | `candidate_e_depth_aware` | Adds bid/ask size, microprice, size imbalance |
| F | `candidate_f_regime_depth` | 34 features: E + realized vol, spread/vol regimes, interactions |

**Supporting artifacts:**
- `reports/final_project_evidence/candidate_history.json`
- `reports/research_notes/candidate_c_feature_engineering_plan.md`
- `reports/candidate_grid/candidate_grid_20260709T124717.json`
- `reports/candidate_grid/candidate_grid_20260714T071013.json`
- `reports/candidate_grid/candidate_grid_20260717T080653.json`
- `reports/bid_ask_walk_forward/accepted_trades_ETH_20260729T101645.csv` (regime feature columns)

---

## 7. Labeling and Cost Assumptions

**Purpose:** Document direction vs cost-aware vs binary tradeable label modes and abstention rates.

**Points:**
- **Direction labels** (B, C): high trade counts, uniformly negative after costs.
- **Cost-aware 3-class** (D): ~66.7% NO_TRADE; diagnostics-only for walk-forward training.
- **Binary tradeable_up** (E, F): ~85% NO_TRADE in Candidate F walk-forward; labels bars with sufficient edge after costs.
- Candidate F: `threshold_bps: 1.0`, `horizon_bars: 20`, `label_mode: binary_tradeable_up`.

**Supporting artifacts:**
- `reports/research_notes/candidate_d_cost_aware_labeling_plan.md`
- `reports/bid_ask_walk_forward/walk_forward_BTC_ETH_20260709T150853.json`
- `reports/bid_ask_walk_forward/walk_forward_ETH_20260729T101645.json` (`candidate_d_label_diagnostics`)
- `reports/final_project_evidence/candidate_history.json` (Candidate D section)

---

## 8. Walk-Forward Evaluation

**Purpose:** Methodology and Candidate F final run details.

**Candidate F final run (20260729T101645):**
- 705 total windows, 695 completed, 10 skipped
- 15 accepted UP trades; cumulative +45.16 net bps; average +3.01 net bps
- `beats_baseline_accuracy: false`; baseline accuracy mean ~0.848
- Block breakdown: 67,572 FLAT class blocks, 1,913 UP class blocks, 1,376 confidence, 529 cost, 8 spread

**Supporting artifacts:**
- `reports/bid_ask_walk_forward/walk_forward_ETH_20260729T101645.json`
- `reports/bid_ask_walk_forward/walk_forward_ETH_20260729T101645.md` *(if present)*
- `reports/bid_ask_walk_forward/accepted_trades_ETH_20260729T101645.csv`
- `reports/final_project_evidence/candidate_f_final_summary.json`

**Prior candidate walk-forward runs (comparison context):**
- `reports/bid_ask_walk_forward/walk_forward_BTC_20260730T074926.json` (B)
- `reports/bid_ask_walk_forward/walk_forward_BTC_20260709T153038.json` (C)
- `reports/bid_ask_walk_forward/walk_forward_ETH_20260709T152926.json` (C)
- `reports/bid_ask_walk_forward/walk_forward_ETH_20260714T070309.json` (E)

---

## 9. Candidate B–F Development

**Purpose:** Chronological narrative of hypothesis → test → rejection/freeze for each candidate family.

**Summary table source:** `reports/final_project_evidence/candidate_history.md` (compact comparison).

| Candidate | Status | Trades | Cum. net bps | Outcome |
|-----------|--------|-------:|-------------:|---------|
| B | rejected | 63 | +11.77 | Unstable/sparse; did not scale |
| C | rejected | 157* | −437.98* | All grid configs negative |
| D | rejected | 0 | 0.00 | Diagnostic labels; adopted by E/F |
| E | diagnostic_positive_sparse | 14 | +31.28 | Failed min-trade robustness |
| F | frozen | 15 | +45.16 | Sparse positive; not robust |

\*Representative best-ranked grid config; all configs negative.

**Supporting artifacts:**
- `reports/final_project_evidence/candidate_history.json`
- `reports/final_project_evidence/candidate_history.md`
- `reports/candidate_ledger/candidate_ledger.json`
- `reports/research_status/research_status_latest.json` (`candidate_timeline`)
- Per-candidate artifacts listed in `candidate_history.json` → `candidates.*.source_artifacts`

**Missing evidence (do not overclaim):**
- Candidate C: no dedicated rolling-slice robustness report.
- Candidate D: no positive standalone trading run without E/F features.
- Candidate E: no `candidate_e_robustness` report analogous to F.

---

## 10. Candidate F Final Results

**Purpose:** Frozen configuration, final metrics, trade concentration, baseline comparison.

**Frozen config:** `ETH_UP_H20_C0.65_E2_M1.25_S8`
- Coin ETH, interval 15s, horizon 20, trade class UP
- min_confidence 0.65, min_edge_bps 2.0, cost_multiplier 1.25, max_spread_bps 8.0
- feature_set: `candidate_f_regime_depth`, label_mode: `binary_tradeable_up`

**Final metrics:** see `reports/final_project_evidence/final_metrics_table.md`.

**Concentration flags:**
- 13/15 trades (86.7%) on 2023-09-23; dominant UTC hour 14 (86.7%)
- All trades in tight_spread_regime and low_volatility_regime
- Excluding dominant day: 2 trades, +4.96 cumulative net bps

**Supporting artifacts:**
- `reports/final_project_evidence/candidate_f_final_summary.json`
- `reports/bid_ask_walk_forward/walk_forward_ETH_20260729T101645.json`
- `reports/bid_ask_walk_forward/accepted_trades_ETH_20260729T101645.csv`
- `reports/candidate_f_milestone_comparison/candidate_f_milestone_comparison_latest.json`

---

## 11. Robustness Analysis

**Purpose:** Candidate F robustness verdict, concentration diagnostics, milestone stability.

**Findings:**
- Overall verdict: `diagnostic_positive_sparse`
- `robust_candidate: false`
- Grid source (20260717): 17 accepted trades at +1.69 avg net bps in grid vs 15 trades at +3.01 in latest walk-forward — **count mismatch between grid and final walk-forward** (both positive but sparse).
- Next recommended action: continue paper-only data collection until ≥30 accepted trades.

**Supporting artifacts:**
- `reports/candidate_f_robustness/candidate_f_robustness_latest.json`
- `reports/candidate_f_robustness/candidate_f_robustness_latest.md` *(if present)*
- `reports/candidate_f_milestone_comparison/candidate_f_milestone_comparison_latest.json`
- `reports/candidate_f_milestone_comparison/candidate_f_milestone_comparison_latest.md` *(if present)*
- `reports/candidate_grid/candidate_grid_20260717T080653.json`

**Candidate B robustness (historical comparison):**
- `reports/bid_ask_candidate_validation/robustness_BTC_DOWN_MICRO_H20_CANDIDATE_B_20260706T115055.json`

---

## 12. Pipeline Engineering Findings

**Purpose:** Data pipeline bugs, fixes, and operational lessons separate from model research.

**Major themes:**
- 1s CSV repair (dedupe, mock-row removal, monotonic timestamps)
- Resampler fix: carry last-valid 1s tick for `bid_size`/`ask_size` into 15s/1m bars
- Stale 15s/1m tails after 1s updates → atomic writes, source-bound trimming, alignment assertions (2026-07-29)
- Importer/resampler in-memory cleaning of malformed numeric rows
- Scheduler 90-minute tick timeout for long backfill runs
- Soft vs hard validation: gap warnings vs blocking failures

**Supporting artifacts:**
- `reports/research_log/2026-07-29.md`
- `reports/research_notes/hyperliquid_bid_ask_research_notes.md` (§6–7)
- `reports/research_notes/whatsapp_call_briefing_20260715.md`
- `reports/backfill_milestones/backfill_milestones_latest.json`
- `reports/final_project_evidence/project_timeline.md`

---

## 13. Limitations

**Purpose:** Explicit constraints on interpretability and generalizability.

**Points:**
- **Sample size:** 15 accepted trades for Candidate F; below 30-trade robustness target.
- **Concentration:** 86.7% of trades on a single calendar day (2023-09-23).
- **Dataset span:** ~14.5 days of historical data; 30-day milestone not met.
- **Single coin/class:** Final frozen config is ETH UP H20 only.
- **Cost model:** Estimated fees/slippage/spread; not live fill simulation.
- **Baseline:** Model does not beat baseline accuracy (`beats_baseline_accuracy: false`).
- **Missing robustness artifacts** for Candidates C and E.
- **No out-of-sample held-out date block** formally documented as complete (milestone 6 at 0% in early notes).

**Supporting artifacts:**
- `reports/final_project_evidence/candidate_f_final_summary.json`
- `reports/candidate_f_robustness/candidate_f_robustness_latest.json`
- `reports/candidate_f_milestone_comparison/candidate_f_milestone_comparison_latest.json`
- `reports/final_project_evidence/candidate_history.json` (`missing_evidence_summary`)
- `reports/research_notes/hyperliquid_bid_ask_research_notes.md` (§3 Milestone Journey)

---

## 14. Conclusion

**Purpose:** Final honest assessment aligned with frozen evidence.

**Draft conclusion points:**
1. A reproducible paper-only research pipeline was built and validated for Hyperliquid BTC/ETH microstructure data.
2. Candidate F shows a **diagnostic** post-cost positive signal (+45.16 bps, 15 trades) but fails robustness and baseline-improvement criteria.
3. Candidates B–E were rejected or superseded; cost-aware labeling and depth/regime features improved abstention and sparsity but did not yield a market-ready candidate.
4. Real execution should remain disabled; further work requires more data and out-of-sample validation, not parameter tuning on the frozen config.

**Supporting artifacts:**
- `reports/final_project_evidence/candidate_f_final_summary.json`
- `reports/final_project_evidence/candidate_history.json`
- `reports/research_log/2026-07-29.md`

---

## 15. Appendices

### A. Evidence inventory and file index
- `reports/final_project_evidence/evidence_inventory.json`
- `reports/final_project_evidence/candidate_history.json` → `source_artifacts_index`

### B. Final metrics table
- `reports/final_project_evidence/final_metrics_table.csv`
- `reports/final_project_evidence/final_metrics_table.md`

### C. Table and figure plan
- `reports/final_project_evidence/table_figure_plan.md`

### D. Project timeline
- `reports/final_project_evidence/project_timeline.md`

### E. Accepted trades detail (Candidate F)
- `reports/bid_ask_walk_forward/accepted_trades_ETH_20260729T101645.csv`

### F. Research logs
- `reports/research_log/2026-07-29.md`
- `reports/research_log/2026-08-05.md`

### G. Candidate grid archives
- `reports/candidate_grid/candidate_grid_20260709T124717.json` (C — BTC/ETH DOWN)
- `reports/candidate_grid/candidate_grid_20260709T125740.json` (C — ETH UP/DOWN)
- `reports/candidate_grid/candidate_grid_20260714T071013.json` (E)
- `reports/candidate_grid/candidate_grid_20260717T080653.json` (F)

### H. Validation reports (latest 2026-07-29)
- `data/bid_ask_csv/reports/*_validation_20260729T*.json`

---

*Outline generated 2026-08-05. No models rerun. No tuning. Execution disabled.*
