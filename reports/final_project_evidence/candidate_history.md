# Candidate B–F Research History

Research-only summary compiled from existing local artifacts. No models were rerun. Real execution remains disabled.

**Status legend**

| Category | Meaning |
|----------|---------|
| **rejected** | Candidate family abandoned |
| **diagnostic_positive_sparse** | Positive after estimated costs on paper, but too few trades and/or unstable; not robust |
| **frozen** | Configuration locked; retained as final diagnostic reference |

---

## Compact Comparison (Final Report Table)

| Candidate | Status | Feature / method | Coin / interval | Trades | Cum. net bps | Avg net bps | Robust | Outcome |
|-----------|--------|------------------|-----------------|-------:|-------------:|------------:|:------:|---------|
| **B** | rejected | microstructure + direction filters | BTC / 15s | 63 | +11.77 | +0.19 | no | Early signal did not scale; unstable/sparse |
| **C** | rejected | `candidate_c_core` + direction labels | BTC, ETH / 15s | 157* | −437.98* | −2.79* | no | All grid configs negative after costs |
| **D** | rejected | cost-aware / binary tradeable labels | BTC, ETH / 15s | 0 | 0.00 | 0.00 | n/a | Diagnostic labels only; adopted by E/F |
| **E** | diagnostic_positive_sparse | `candidate_e_depth_aware` + binary_tradeable_up | ETH / 15s H20 | 14 | +31.28 | +2.23 | no | Positive but failed min-trade robustness |
| **F** | frozen | `candidate_f_regime_depth` + binary_tradeable_up | ETH / 15s H20 | 15 | +45.16 | +3.01 | no | `diagnostic_positive_sparse`; no baseline improvement |

\*Representative best-ranked grid config (`ETH_DOWN_H20_C0.55_E1_M1.25_S8`); all configs negative.

---

## Candidate B — `BTC_DOWN_MICRO_H20_CANDIDATE_B`

**Status: rejected**

| Field | Value |
|-------|-------|
| Objective | Tradeable BTC DOWN moves via microstructure features with strict filters |
| Main change | `microstructure` features; direction labels; confidence ≥ 0.65, edge ≥ 2 bps, spread ≤ 8 bps |
| Dataset | BTC, 15s, historical |
| Evaluation | Walk-forward + candidate validation + data-slice robustness |
| Accepted trades | 63 |
| Cumulative net bps | +11.77 |
| Average net bps | +0.19 |
| Robustness | `robustness_pass: false`; 0/7 slices met min trades; unstable_sparse |
| Decision | Rejected |
| Reason | Early strict-config positivity (~5 trades) collapsed under data expansion; relaxed configs strongly negative; threshold tuning abandoned 2026-07-09 |

**Source artifacts**

- `reports/bid_ask_candidate_validation/candidate_validation_BTC_DOWN_MICRO_H20_CANDIDATE_B_20260730T034523.json`
- `reports/bid_ask_candidate_validation/robustness_BTC_DOWN_MICRO_H20_CANDIDATE_B_20260706T115055.json`
- `reports/bid_ask_walk_forward/walk_forward_BTC_20260730T074926.json`
- `reports/research_notes/hyperliquid_bid_ask_research_notes.md`

---

## Candidate C — `candidate_c_core`

**Status: rejected**

| Field | Value |
|-------|-------|
| Objective | Fix post-cost losses via richer engineered features instead of post-hoc filters |
| Main change | `candidate_c_core` feature set (spread regime, imbalance dynamics, returns, time) |
| Dataset | BTC + ETH, 15s, historical |
| Evaluation | Walk-forward candidate grids (16 configs each for DOWN-only and UP/DOWN) |
| Accepted trades | 157 (best-ranked config) |
| Cumulative net bps | −437.98 (best-ranked config) |
| Average net bps | −2.79 (best-ranked); BTC DOWN H20 −6.99; ETH UP H20 −5.15 |
| Robustness | No positive post-cost config in either grid |
| Decision | Rejected |
| Reason | Feature engineering alone did not overcome costs under direction labels |

**Source artifacts**

- `reports/candidate_grid/candidate_grid_20260709T124717.json`
- `reports/candidate_grid/candidate_grid_20260709T125740.json`
- `reports/research_notes/candidate_c_feature_engineering_plan.md`
- `reports/research_notes/hyperliquid_bid_ask_research_notes.md`

**Missing evidence:** dedicated rolling-slice robustness report not located.

---

## Candidate D — cost-aware labels

**Status: rejected** (continued as label infrastructure)

| Field | Value |
|-------|-------|
| Objective | Label bars by tradeable edge after costs, not direction alone |
| Main change | `cost_aware_3class` (UP_EDGE / DOWN_EDGE / NO_TRADE); `binary_tradeable_up/down` |
| Dataset | BTC + ETH, 15s, historical |
| Evaluation | Label diagnostics in walk-forward reports; 3-class mode diagnostics-only |
| Accepted trades | 0 |
| Cumulative / avg net bps | 0.00 / 0.00 |
| Label diagnostics | ~66.7% NO_TRADE fraction; 3-class walk-forward training not supported |
| Robustness | Not applicable as standalone trading candidate |
| Decision | Rejected standalone; binary modes adopted by E and F |
| Reason | Clarified cost/abstention problem but did not yield predictive edge alone |

**Source artifacts**

- `reports/bid_ask_walk_forward/walk_forward_BTC_ETH_20260709T150853.json`
- `reports/research_notes/candidate_d_cost_aware_labeling_plan.md`
- `reports/research_status/research_status_latest.json`

**Missing evidence:** no positive post-cost run attributed solely to D without later feature sets.

---

## Candidate E — `candidate_e_depth_aware`

**Status: diagnostic_positive_sparse**

| Field | Value |
|-------|-------|
| Objective | Restore sparse positive ETH UP H20 using depth/microprice features |
| Main change | Extends `candidate_c_core` with bid/ask size, microprice, size imbalance; `binary_tradeable_up` labels |
| Dataset | ETH, 15s, H20, historical |
| Evaluation | Walk-forward + 80-config grid (`min_trades_after_regime_filter=5`) |
| Accepted trades | 14 (walk-forward); grid best 3 |
| Cumulative net bps | +31.28 |
| Average net bps | +2.23 (walk-forward); grid best +6.96 on 3 trades |
| Robustness | Failed minimum-trade check (threshold 10) |
| Decision | Not continued — superseded by Candidate F |
| Reason | Interesting sparse positive signal but insufficient trade count and concentration |

**Source artifacts**

- `reports/bid_ask_walk_forward/walk_forward_ETH_20260714T070309.json`
- `reports/candidate_grid/candidate_grid_20260714T071013.json`
- `reports/research_status/research_status_latest.json`

**Missing evidence:** no `candidate_e_robustness` report analogous to Candidate F.

---

## Candidate F — `ETH_UP_H20_C0.65_E2_M1.25_S8` (frozen)

**Status: frozen — diagnostic_positive_sparse**

| Field | Value |
|-------|-------|
| Objective | Tradeable upward ETH moves using volatility/spread/depth regime features |
| Main change | `candidate_f_regime_depth` (34 features); frozen thresholds C0.65/E2/M1.25/S8 |
| Dataset | ETH, 15s, H20, historical (2023-09-16 → 2023-09-30) |
| Evaluation | Frozen walk-forward; robustness analyzer; milestone comparison |
| Total / completed / skipped windows | 705 / 695 / 10 |
| Accepted trades | 15 |
| Cumulative net bps | +45.16 |
| Average net bps | +3.01 |
| Verdict | `diagnostic_positive_sparse` |
| Robust candidate | `false` |
| Decision | Frozen |
| Reason | Positive after estimated costs, but sparse, non-robust, and no demonstrated improvement over baseline |

**Source artifacts**

- `reports/final_project_evidence/candidate_f_final_summary.json`
- `reports/bid_ask_walk_forward/walk_forward_ETH_20260729T101645.json`
- `reports/candidate_f_robustness/candidate_f_robustness_latest.json`
- `reports/candidate_f_milestone_comparison/candidate_f_milestone_comparison_latest.json`
- `reports/research_log/2026-07-29.md`

---

## Narrative (B → F)

**Candidate B** tested whether strict post-hoc filters on microstructure features could isolate BTC DOWN edges. An early positive snippet on ~5 strict trades did not survive backfill; relaxed configs turned strongly negative, and rolling-slice robustness never reached minimum trade counts. The family was rejected as a scalable paper candidate but retained value as a diagnostic baseline.

**Candidate C** shifted from filtering to feature engineering (`candidate_c_core`). Walk-forward grids on expanded BTC/ETH data produced ample trades but uniformly negative average net bps, confirming that richer inputs under direction labels still could not cover fees, slippage, and spread.

**Candidate D** reframed the labeling problem: roughly two-thirds of bars are correctly labeled NO_TRADE under cost-aware rules. The 3-class diagnostic mode validated abstention mechanics but did not train profitably; binary tradeable modes became infrastructure for later candidates rather than a standalone winner.

**Candidate E** added depth-aware features and produced the first sparse positive ETH UP H20 walk-forward (+31 bps on 14 trades). Grid searches showed similar positivity only at 3-trade configs. The signal was too sparse to pass the minimum-trade robustness bar and was not frozen.

**Candidate F** extended E with regime-depth interactions under a frozen config. The latest walk-forward reports +45 bps on 15 trades with `diagnostic_positive_sparse` verdict, `robust_candidate: false`, and no demonstrated improvement over baseline. F is frozen as the final reference configuration for the project report.

---

## Files Created

- `reports/final_project_evidence/candidate_history.json`
- `reports/final_project_evidence/candidate_history.md`

## Missing Evidence

1. Candidate C — no dedicated robustness slicing report (grid/walk-forward negatives used).
2. Candidate D — no positive standalone trading run without E/F feature sets.
3. Candidate E — no dedicated robustness analyzer report (research status + walk-forward used).
