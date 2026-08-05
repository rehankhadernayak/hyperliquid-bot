# Hyperliquid BTC/ETH Bid/Ask Microstructure ML — Final Project Report (Draft v1)

**Document status:** Paper-only research report. No models rerun. No Candidate F tuning. Real execution disabled.

**Generated:** 2026-08-05  
**Primary evidence base:** Local artifacts listed in Appendix A.

---

## Abstract

This report documents a paper-only research project that asked whether machine-learning signals derived from Hyperliquid Level-2 (L2) bid/ask microstructure data for BTC and ETH could produce trade ideas that remain positive after estimated transaction costs. The project built a reproducible data pipeline, iterated through six candidate model families (B through F), and froze one final configuration—**Candidate F** (`ETH_UP_H20_C0.65_E2_M1.25_S8`)—for diagnostic reference.

In plain terms: the project collected historical order-book snapshots, engineered features from bid/ask prices and sizes, trained classifiers in a walk-forward (time-respecting) protocol, and simulated whether predicted trades would have beaten costs on paper. Candidate F produced **15 accepted trades** with **+45.16 cumulative net basis points (bps)** and **+3.01 average net bps**, but the evidence is **sparse**, **concentrated in time and regime**, and **does not demonstrate improvement over a simple baseline**. The robustness verdict is `diagnostic_positive_sparse`; `robust_candidate: false`. This is **not** evidence of profitability or market readiness.

All work remained within explicit safety boundaries: `paper_only: true`, `execution_enabled: false`, `real_execution_ready: false`.

---

## 1. Executive Summary

### 1.1 What this project is

Hyperliquid is a decentralised exchange. Like any exchange, its order book shows the best prices at which participants are willing to buy (bid) and sell (ask). Small, rapid changes in that book—spread width, order imbalance, depth at the top of book—sometimes precede short-term price moves. This project tested whether those microstructure patterns, combined with machine learning, could identify moments when a simulated **upward** trade on ETH would still be worthwhile **after fees, slippage, and spread**.

The work was deliberately limited to **research simulation**. No wallets, signing, orders, or live exchange endpoints were used.

### 1.2 What was built

Two parallel efforts ran throughout the project:

1. **Model research (Candidates B–F):** A sequence of hypotheses about features, labels, and filters, each evaluated with walk-forward testing and cost-aware gates.
2. **Pipeline engineering:** Infrastructure to backfill historical L2 data from S3, resample 1-second bars to 15-second and 1-minute intervals, validate CSV integrity, and support depth-aware feature computation.

By 2026-07-29, the pipeline reached the **14-day data milestone** (~348 hours of BTC and ETH history from 2023-09-16 through 2023-09-30) but remained at **~48.3% progress toward the 30-day milestone**.

### 1.3 Final outcome

| Outcome dimension | Result |
|-------------------|--------|
| Frozen candidate | F — `ETH_UP_H20_C0.65_E2_M1.25_S8` |
| Accepted trades | 15 |
| Cumulative net bps | +45.16 |
| Average net bps | +3.01 |
| Robustness verdict | `diagnostic_positive_sparse` |
| Robust candidate | `false` |
| Baseline improvement | No demonstrated improvement (`beats_baseline_accuracy: false`) |
| Market readiness | Not claimed; execution disabled |

No candidate B through F achieved robust, baseline-beating, market-ready paper performance. Candidate F is retained as a **frozen diagnostic reference**, not as a production trading system.

**See also:** Table 1 (candidate comparison), Table 3 (final metrics), Figure 3 (`candidate_development_timeline.png`).

---

## 2. Research Questions

### 2.1 Primary question

Can microstructure features derived from Hyperliquid BTC/ETH L2 bid/ask data produce **paper-only** trade signals that remain positive after estimated costs (fees + slippage + spread)?

### 2.2 Secondary questions

1. Which combinations of labeling strategy, feature engineering, and post-prediction filters survive walk-forward evaluation and robustness checks?
2. What pipeline and validation infrastructure is required to test such claims honestly without live execution?

These questions guided the B→F candidate arc documented in Section 9 and Table 1.

---

## 3. Project Scope and Paper-Only Safeguards

### 3.1 What the project does

- Collects and validates historical bid/ask CSV data for BTC and ETH at 1s, 15s, and 1m intervals.
- Engineers microstructure, depth, and regime features from validated bars.
- Trains classifiers (Random Forest, per research architecture notes) in rolling walk-forward windows.
- Applies post-prediction filters (confidence, edge, spread, trade class) before counting an **accepted trade**.
- Computes **net returns in basis points** after an estimated cost model.
- Runs robustness diagnostics (trade count, temporal concentration, regime concentration, milestone comparison).

### 3.2 What the project explicitly does not do

- Connect wallets, sign transactions, or submit orders.
- Enable leverage, margin, deposits, withdrawals, or exchange execution endpoints.
- Claim statistical significance, profitability, or live-trading readiness.
- Tune Candidate F after the 2026-07-29 freeze.

Every cited final artifact carries consistent safety flags:

| Flag | Value |
|------|-------|
| `paper_only` | `true` |
| `execution_enabled` | `false` |
| `real_execution_ready` | `false` |

**Sources:** `candidate_f_robustness_latest.json`, `candidate_f_milestone_comparison_latest.json`, `backfill_milestones_latest.json`, `walk_forward_ETH_20260729T101645.json`.

Simulated returns exist to **design and validate research methodology**, not to prove that a strategy would profit in live markets.

---

## 4. Data Collection and Validation

### 4.1 Overview

Historical Hyperliquid L2 bid/ask data was backfilled via Requester Pays S3 into local CSV archives under `data/bid_ask_csv/`. The canonical 1-second files were resampled to 15-second and 1-minute bars for model training. Validators checked schema, timestamps, and gap warnings before features or walk-forward runs consumed the data.

### 4.2 Coverage as of 2026-07-29

**Table 5** summarises dataset coverage and validation status. Highlights:

| Coin | Interval | Coverage (h) | First timestamp | Last timestamp | Validator passed |
|------|----------|-------------:|-----------------|----------------|:----------------:|
| BTC | 1s | 347.998 | 2023-09-16T00:00:05.332Z | 2023-09-30T11:59:59.480Z | yes |
| BTC | 15s | 348.0 | 2023-09-16T00:00:00Z | 2023-09-30T11:59:45Z | yes |
| ETH | 15s | 348.0 | 2023-09-16T00:00:00Z | 2023-09-30T11:59:45Z | yes |
| ETH | 1m | 348.0 | 2023-09-16T00:00:00Z | 2023-09-30T11:59:00Z | yes |

**Milestone progress (all coin/interval pairs):** 3-day ✓, 7-day ✓, 14-day ✓, **30-day ✗ (~48.33% complete)**.

Candidate F's final walk-forward used ETH 15s data spanning **351 hours** (`2023-09-16T00:00:00Z` → `2023-09-30T14:59:45Z`), slightly wider than the backfill milestone snapshot due to report-specific CSV bounds.

### 4.3 Validation warnings

Validators **passed** for all six primary files (BTC/ETH × 1s/15s/1m) but emitted expected gap warnings reflecting sparse L2 snapshots—for example, ETH 15s reported **13,242** incomplete periods. These are **soft warnings**: they do not block the pipeline but document missing intervals that reduce effective sample size.

Depth columns (`bid_size`, `ask_size`) are present in resampled files; microprice-related features are expected and were prerequisites for Candidates E and F.

**See also:** Figure 4 (`data_pipeline_workflow.png`), Table 6 (pipeline fixes).

---

## 5. Mathematical and Statistical Methods

This section explains how mathematics enters the project, first in plain language, then with the formulas and thresholds used in artifacts.

### 5.1 Returns and basis points

A **return** measures how much a price changed over a holding period. If ETH mid-price rises from 1,600 to 1,601.60 over 20 fifteen-second bars (five minutes), the gross return is roughly 0.1%. In this project, returns are expressed in **basis points (bps)**, where **1 bps = 0.01%**. Thus +10 bps means +0.10%.

For each accepted paper trade, the walk-forward engine records a **gross return** (price change before costs) and a **net return** (after subtracting estimated costs). Candidate F's **cumulative net bps** (+45.16) is the **sum** of net returns across all 15 accepted trades. **Average net bps** (+3.01) is that sum divided by 15.

### 5.2 Spread

The **spread** is the gap between the best ask and best bid. A tight spread means the market is liquid at the top of book; a wide spread means crossing the spread is expensive. Candidate F blocks trades when spread exceeds **8 bps** (`max_spread_bps: 8.0`). In the final run, **8 predictions** were blocked by spread filters; all 15 accepted trades fell in the **tight_spread_regime** (100%).

### 5.3 Transaction costs

Real trading incurs **fees** (exchange charges), **slippage** (price movement while executing), and **spread crossing** (buying at ask vs. mid). The project uses an estimated cost formula documented in walk-forward artifacts:

```
estimated_cost_bps = fee_bps + slippage_bps + spread_bps
```

**Net return** for a simulated trade is approximately:

```
net_return_bps ≈ gross_return_bps − estimated_cost_bps
```

Candidate F's walk-forward reports **average cost bps ≈ 6.42** across the evaluation windows and **≈ 3.75** on accepted UP trades specifically. Filters also apply a **cost multiplier** of **1.25** on edge thresholds—requiring predicted edge to exceed costs by a safety margin before acceptance.

### 5.4 Volatility

**Volatility** measures how much prices fluctuate. Candidate F's feature set includes **realised volatility** and classifies bars into **low** vs. **high volatility regimes**. Robustness analysis shows **13 of 15** accepted trades (86.7%) occurred in **low_volatility_regime**; **zero** occurred in high-volatility regime. Absence of trades in a regime is **not** evidence of negative edge there—it may reflect filter interaction with a short sample.

### 5.5 Probabilities and confidence

The Random Forest classifier outputs **class probabilities** for each bar—for example, the probability that an upward tradeable move will occur. Candidate F requires **minimum confidence 0.65** (`min_confidence: 0.65`): the model must assign at least 65% probability to the UP class before a prediction is eligible. In the final run, **1,376** UP-class predictions were blocked for insufficient confidence.

Confidence is a **model-estimated probability**, not a guarantee of future correctness. Accepted trades averaged **≈ 0.923** confidence, but with only 15 trades this average is not generalisable.

### 5.6 Walk-forward windows

**Walk-forward evaluation** respects time: the model trains on past bars and predicts on subsequent bars, rolling forward through history. This mimics how a strategy would have been deployed day by day without peeking at future data.

Candidate F's final run processed **705 total windows**, of which **695 completed** and **10 were skipped** (typically due to insufficient training history at window start). Each window produces predictions; post-prediction filters then decide which predictions become **accepted trades**.

### 5.7 Cumulative return vs. average return

- **Cumulative net bps** sums all accepted-trade net returns. Candidate F: **+45.16 bps** total across 15 trades.
- **Average net bps** divides cumulative by trade count: **+3.01 bps** per trade.

Cumulative return answers "how much would paper P&L have added up?"; average return answers "how much per trade on average?". With sparse trades, a few large winners can lift cumulative return while average return remains modest.

**See Figure 1** (`candidate_f_cumulative_net_bps.png`) for the running cumulative curve. Intermediate path shape depends on trade ordering; the final total is order-invariant.

### 5.8 Robustness

**Robustness** here means: would modest changes in time period, regime mix, or evaluation rerun still support the same conclusion? The project applies **qualitative and count-based** robustness gates—not formal statistical hypothesis tests (which are **unavailable** in the artifact set).

Criteria include:

- Minimum accepted trade count (project target: **≥ 30** for robustness; Candidate F has **15**).
- Temporal concentration (dominant calendar day, UTC hour).
- Regime concentration (volatility, spread buckets).
- Comparison between grid search metrics and final walk-forward metrics.
- Baseline accuracy comparison (`beats_baseline_accuracy`).

Candidate F fails these gates: verdict **`diagnostic_positive_sparse`**, **`robust_candidate: false`**.

### 5.9 Classifier and filter pipeline (summary)

| Stage | Candidate F setting |
|-------|---------------------|
| Classifier | Random Forest (per architecture notes) |
| Feature set | `candidate_f_regime_depth` (34 features) |
| Label mode | `binary_tradeable_up` |
| Horizon | 20 bars (15s each ≈ 5 minutes) |
| min_confidence | 0.65 |
| min_edge_bps | 2.0 |
| cost_multiplier | 1.25 |
| max_spread_bps | 8.0 |
| trade_class | UP only |

**Block breakdown (final run):** 67,572 FLAT-class blocks, 1,913 UP-class blocks, 1,376 blocked by confidence, 529 by cost, 8 by spread.

---

## 6. Feature Engineering

Features evolved from simple microstructure inputs (Candidate B) through engineered spread/imbalance dynamics (C), depth-aware microprice signals (E), to regime-depth interactions (F).

| Candidate | Feature set | Description |
|-----------|-------------|-------------|
| B | `microstructure` | Basic bid/ask microstructure; post-hoc direction filters |
| C | `candidate_c_core` | Spread regime, imbalance dynamics, returns, time features |
| D | *(label infrastructure)* | No standalone feature set; reframed labels |
| E | `candidate_e_depth_aware` | Extends C with bid/ask size, microprice, size imbalance |
| F | `candidate_f_regime_depth` | 34 features: E + realised vol, spread/vol regimes, interactions |

Candidate F adds **regime-aware** features so the model can distinguish, for example, tight-spread low-volatility conditions from wide-spread high-volatility conditions. This was the main methodological advance over Candidate E.

---

## 7. Labeling and Cost Assumptions

### 7.1 Direction labels (B, C)

Early candidates labelled bars by **price direction** (up/down) regardless of whether the move exceeded costs. This produced many trades but uniformly **negative average net bps** after costs—Candidate C's best grid config averaged **−2.79 bps** across 157 trades.

### 7.2 Cost-aware labels (D)

Candidate D introduced labels based on **tradeable edge after costs**:

- **`cost_aware_3class`:** UP_EDGE / DOWN_EDGE / NO_TRADE (~66.7% NO_TRADE).
- **`binary_tradeable_up` / `binary_tradeable_down`:** Binary edge labels adopted by E and F.

The 3-class mode was **diagnostics-only** for walk-forward training (not supported as a profitable standalone candidate). Candidate D produced **zero accepted trades** as a trading family but clarified that most bars correctly should not be traded.

### 7.3 Binary tradeable labels (E, F)

Candidates E and F use **`binary_tradeable_up`**: a bar is labelled positive only if a hypothetical UP trade at that bar would exceed a **threshold** after costs. Candidate F uses **`threshold_bps: 1.0`** and **`horizon_bars: 20`**.

Label diagnostics in the final walk-forward show **≈ 85.0% NO_TRADE fraction** (`no_trade_fraction: 0.8496`), meaning the labeler correctly abstains on most bars. The model must therefore operate in a **high-abstention** regime—consistent with only 15 accepted trades from 705 windows.

---

## 8. Walk-Forward Evaluation

### 8.1 Methodology

Walk-forward evaluation rolls a training window through historical 15-second ETH bars, fits the classifier, predicts the next segment, applies filters, and records accepted trades with net returns. Safety flags remain `paper_only: true`, `execution_enabled: false`.

### 8.2 Candidate F final run (20260729T101645)

**Table 2** lists the frozen configuration. **Table 3** lists final metrics.

| Metric | Value |
|--------|------:|
| Report stamp | 20260729T101645 |
| total_windows | 705 |
| completed_windows | 695 |
| skipped_windows | 10 |
| accepted_trades | 15 |
| cumulative_net_bps | +45.16 |
| average_net_bps | +3.01 |
| win_rate | 0.933 (14 wins, 1 loss) |
| beats_baseline_accuracy | false |
| baseline_accuracy_mean | 0.848 |
| accuracy_edge_vs_baseline | −0.0106 |
| average_cost_bps | 6.42 |

The model's classification accuracy **does not beat** a simple baseline that always predicts the majority class (~84.8% accuracy). Positive net bps on accepted trades therefore reflects **filtering and trade selection**, not overall predictive superiority.

**See Figure 2** (`candidate_f_return_distribution.png`) for per-trade net return dispersion. One trade (−2.68 bps on 2023-09-21) is the sole loser.

---

## 9. Candidate B–F Development

### 9.1 Summary comparison

**Table 1 — Candidate B–F Comparison**

| Candidate | Status | Feature / method | Coin / interval | Trades | Cum. net bps | Avg net bps | Robust | Outcome |
|-----------|--------|------------------|-----------------|-------:|-------------:|------------:|:------:|---------|
| **B** | rejected | microstructure + direction filters | BTC / 15s | 63 | +11.77 | +0.19 | no | Early signal did not scale; unstable/sparse |
| **C** | rejected | `candidate_c_core` + direction labels | BTC, ETH / 15s | 157* | −437.98* | −2.79* | no | All grid configs negative after costs |
| **D** | rejected | cost-aware / binary tradeable labels | BTC, ETH / 15s | 0 | 0.00 | 0.00 | n/a | Diagnostic labels only; adopted by E/F |
| **E** | diagnostic_positive_sparse | `candidate_e_depth_aware` + binary_tradeable_up | ETH / 15s H20 | 14 | +31.28 | +2.23 | no | Positive but failed min-trade robustness |
| **F** | frozen | `candidate_f_regime_depth` + binary_tradeable_up | ETH / 15s H20 | 15 | +45.16 | +3.01 | no | `diagnostic_positive_sparse`; no baseline improvement |

\*Candidate C metrics are from the best-ranked grid config (`ETH_DOWN_H20_C0.55_E1_M1.25_S8`); **all grid configs were negative**.

*Caption:* Comparison of Candidates B through F on paper-only walk-forward or grid evaluation. Positive cumulative net bps after estimated costs do not imply robustness or market readiness.

### 9.2 Narrative progression

**Candidate B** tested strict post-hoc filters on BTC DOWN microstructure features. An early positive snippet (~5 trades) did not survive data expansion; relaxed configs turned strongly negative. Rolling-slice robustness (7 slices, 0 meeting minimum trades) yielded verdict **`unstable_sparse`**. Rejected 2026-07-30.

**Candidate C** pivoted to richer engineered features under direction labels. Grid searches on BTC and ETH produced ample trades but **uniformly negative** post-cost performance. Rejected 2026-07-09.

**Candidate D** reframed the problem around cost-aware abstention. ~66.7% of bars are NO_TRADE under 3-class diagnostics. No standalone trading edge; binary modes became infrastructure for E/F. Rejected as model family; retained as label methodology.

**Candidate E** added depth-aware features on ETH UP H20. Walk-forward produced **14 trades, +31.28 cumulative net bps**—the first sparse positive signal—but failed the **minimum 10-trade** robustness threshold. Superseded by F; not frozen.

**Candidate F** extended E with regime-depth interactions under frozen thresholds. Final walk-forward: **15 trades, +45.16 cumulative net bps**, verdict **`diagnostic_positive_sparse`**, **`robust_candidate: false`**. Frozen 2026-07-29 as the final diagnostic reference.

**See Figure 3** (`candidate_development_timeline.png`) for the chronological arc.

---

## 10. Candidate F Final Results

### 10.1 Frozen configuration

**Table 2 — Candidate F Frozen Configuration**

| Field | Value |
|-------|-------|
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

*Caption:* Frozen Candidate F configuration used for final walk-forward evaluation (20260729T101645). No further tuning applied after freeze.

The config_id encodes: **ETH**, **UP** trade class, **H20** horizon, **C0.65** confidence, **E2** min edge (2 bps), **M1.25** cost multiplier, **S8** max spread (8 bps).

### 10.2 Final metrics

**Table 3 — Candidate F Final Walk-Forward Metrics**

| Field | Value |
|-------|-------|
| accepted_trades | 15 |
| cumulative_net_bps | +45.16 |
| average_net_bps | +3.01 |
| total_windows | 705 |
| completed_windows | 695 |
| skipped_windows | 10 |
| win_rate | 0.933 |
| beats_baseline_accuracy | false |
| baseline_accuracy_mean | 0.848 |
| robustness_verdict | `diagnostic_positive_sparse` |
| robust_candidate | false |
| research_conclusion | Positive after estimated costs, but sparse, non-robust, no baseline improvement |

*Caption:* Candidate F walk-forward paper evaluation on ETH 15s historical data. Fifteen accepted UP trades passed confidence, edge, spread, and cost filters.

### 10.3 Sample size caveat

**The 15-trade sample is too small for strong statistical claims.** With n = 15, win rate (93.3%), average return (+3.01 bps), and cumulative return (+45.16 bps) are **descriptive summaries only**. They must not be interpreted as statistically significant, nor extrapolated to future periods or live trading. Formal hypothesis tests and confidence intervals are **unavailable** in the current artifact set.

### 10.4 Trade concentration

Accepted trades cluster heavily:

| Dimension | Dominant value | Fraction |
|-----------|----------------|----------:|
| Calendar day | 2023-09-23 | 86.7% (13/15) |
| UTC hour | 14 | 86.7% (13/15) |
| Volatility regime | low_volatility_regime | 86.7% (13/15) |
| Spread regime | tight_spread_regime | 100% (15/15) |

Excluding the dominant day (2023-09-23) leaves **2 trades** totalling **+4.96 cumulative net bps**—far too few to support generalisation.

Day-level breakdown from robustness artifacts:

| Date | Trades | Cumulative net bps |
|------|-------:|-------------------:|
| 2023-09-21 | 1 | −2.68 |
| 2023-09-23 | 13 | +40.21 |
| 2023-09-27 | 1 | +7.64 |

---

## 11. Robustness Analysis

### 11.1 Overall verdict

**Table 4 — Candidate F Robustness Summary**

| Field | Value |
|-------|-------|
| overall_verdict | `diagnostic_positive_sparse` |
| robust_candidate | false |
| trade_count | 15 |
| cumulative_net_bps | +45.16 |
| win_rate | 0.933 |
| dominant_day | 2023-09-23 (86.7%) |
| dominant_utc_hour | 14 (86.7%) |
| excluding_dominant_day trades | 2 |
| excluding_dominant_day cumulative_net_bps | +4.96 |
| top_1_trade_contribution_pct | 15.97% |
| top_3_trades_contribution_pct | 31.02% |
| next_recommended_action | Continue paper-only data collection until ≥30 accepted trades |

*Caption:* Robustness diagnostics for frozen Candidate F. Verdict reflects sparse trade count and temporal/regime concentration, not statistical significance.

### 11.2 Grid vs. walk-forward mismatch

The Candidate F grid search (20260717T080653) reported **17 accepted trades** at **+1.69 average net bps** and **+28.75 cumulative net bps** for the same config_id. The final walk-forward (20260729T101645) reports **15 trades** at **+3.01 average net bps** and **+45.16 cumulative net bps**. Both runs are positive but sparse; the **trade-count mismatch** (17 vs. 15) reflects evaluation pipeline differences between grid and walk-forward modes, not parameter tuning (config was frozen).

### 11.3 Milestone comparison

The milestone comparator collapsed **four duplicate walk-forward reruns** (20260723–20260729) with identical dataset bounds into **one valid milestone**. This confirms reproducibility of the frozen config on the same data slice but does not expand temporal coverage.

### 11.4 Candidate B robustness (historical context)

Candidate B's rolling-slice analysis (7 slices, 0 meeting minimum trades, verdict **`unstable_sparse`**) established early that sparse positivity without slice stability is insufficient—a pattern repeated with Candidates E and F.

---

## 12. Pipeline Engineering Findings

Model research and pipeline engineering were **distinct but interdependent** efforts. Depth-aware and regime features (E, F) would have been untrustworthy without reliable `bid_size`, `ask_size`, and resampled bar alignment.

**Table 6 — Major Pipeline Failures and Fixes**

| Date (approx.) | Problem | Fix / outcome |
|----------------|---------|---------------|
| Early Jul 2026 | Mock/bad rows in 1s CSV | Repair pass: dedupe timestamps, remove mock rows |
| Early Jul 2026 | `bid_size`/`ask_size` dropped in resample | Last-valid 1s tick carried into 15s/1m bars |
| Early Jul 2026 | Backfill import stall | Documented; phase3 audit tracking |
| 2026-07-15 | Scheduler ticks timing out | 90-minute tick timeout added |
| 2026-07-29 | Stale 15s/1m tails after 1s updates | Atomic resampled writes, source-bound trimming, alignment assertions |
| 2026-07-29 | Malformed numeric rows caused import failures | In-memory cleaning; preserve source files |
| 2026-07-29 | Soft vs hard validation behaviour | Gap warnings non-blocking; hard failures still block pipeline |

*Caption:* Selected data-pipeline failures and fixes during the research period. Fixes enabled trustworthy depth-aware feature evaluation.

On 2026-07-29, stale-tail regression tests passed and frozen Candidate F was rerun **without configuration changes**, producing the final evidence set for this report.

**See Figure 4** (`data_pipeline_workflow.png`) for the end-to-end paper-only pipeline diagram.

Formal incident ticket IDs and automated failure counts beyond validator warning totals are **unavailable** in the artifact set.

---

## 13. Limitations

1. **Sample size:** 15 accepted trades for Candidate F; below the 30-trade robustness target. Too small for strong statistical claims.
2. **Temporal concentration:** 86.7% of trades on a single calendar day (2023-09-23).
3. **Dataset span:** ~14.5 days of historical data; 30-day milestone not met (~48.3% complete).
4. **Single coin/class:** Final frozen config is ETH UP H20 only; BTC and DOWN classes not validated at freeze.
5. **Cost model:** Estimated fees/slippage/spread; not live fill simulation or order-book impact modelling.
6. **Baseline:** Model does not beat baseline accuracy (`beats_baseline_accuracy: false`).
7. **Missing robustness artifacts:** No dedicated rolling-slice report for Candidate C; no `candidate_e_robustness` report analogous to F.
8. **No formal significance testing:** Hypothesis tests and confidence bands are **unavailable** in artifacts.
9. **No out-of-sample held-out date block** formally documented as complete.
10. **Regime coverage gaps:** Zero accepted trades in high-volatility or wide-spread regimes; absence is not evidence of edge either way.

---

## 14. Conclusion

This project set out to determine whether Hyperliquid L2 bid/ask microstructure could support cost-aware, paper-only trade signals for BTC and ETH. The honest answer from existing evidence is **mixed and constrained**:

1. **Pipeline success:** A reproducible paper-only research pipeline was built and validated. Historical data reached the 14-day milestone with passing validators, depth columns restored, and stale-tail bugs fixed.

2. **Model research outcome:** Six candidate families (B–F) were tested with increasing methodological discipline—direction labels → engineered features → cost-aware abstention → depth features → regime-depth features. **None** achieved robust, baseline-beating, market-ready paper performance.

3. **Candidate F (frozen):** Configuration `ETH_UP_H20_C0.65_E2_M1.25_S8` shows a **diagnostic** post-cost positive signal (+45.16 cumulative net bps on 15 trades, +3.01 average net bps) with verdict **`diagnostic_positive_sparse`** and **`robust_candidate: false`**. There is **no demonstrated improvement over baseline**. This is **not** profitability proof and **not** market readiness.

4. **Recommended next step (from artifacts):** Continue **paper-only data collection** until at least **30 accepted trades** exist for the leading config—**not** further parameter tuning on the frozen configuration.

Real execution should remain **disabled**. Further work requires more data and out-of-sample validation under the same safety boundaries, not deployment.

---

## 15. Future Work

Based strictly on artifact recommendations and identified gaps:

1. **Extend historical coverage** toward the 30-day milestone (~720 hours; currently ~48.3% complete).
2. **Accumulate ≥ 30 accepted trades** before revisiting robustness verdicts for Candidate F.
3. **Generate remaining planned figures** (e.g., day/regime concentration panel from `table_figure_plan.md` Figure 4 proposal)—**not yet rendered** as PNG.
4. **Formal out-of-sample held-out date block** evaluation when sufficient calendar span exists.
5. **Dedicated robustness reports** for Candidates C and E if historical comparison is needed—currently **unavailable**.
6. **Statistical inference framework** (hypothesis tests, confidence intervals)—currently **unsupported** by artifacts.

No item above authorises enabling execution or tuning the frozen Candidate F config.

---

## Appendix A — Source Artifacts

| Artifact | Role |
|----------|------|
| `reports/final_project_evidence/report_outline.md` | Section structure |
| `reports/final_project_evidence/final_metrics_table.md` | Final metrics |
| `reports/final_project_evidence/candidate_history.md` | B–F comparison |
| `reports/final_project_evidence/project_timeline.md` | Chronology |
| `reports/final_project_evidence/table_figure_plan.md` | Table/figure plan |
| `reports/final_project_evidence/candidate_f_final_summary.json` | Frozen summary |
| `reports/final_project_evidence/figures/figure_captions.md` | Figure captions |
| `reports/final_project_evidence/figures/figure_manifest.json` | Figure manifest |
| `reports/bid_ask_walk_forward/walk_forward_ETH_20260729T101645.json` | Final walk-forward |
| `reports/bid_ask_walk_forward/accepted_trades_ETH_20260729T101645.csv` | Accepted trades |
| `reports/candidate_f_robustness/candidate_f_robustness_latest.json` | Robustness |
| `reports/candidate_f_milestone_comparison/candidate_f_milestone_comparison_latest.json` | Milestones |
| `reports/backfill_milestones/backfill_milestones_latest.json` | Data coverage |
| `reports/research_log/2026-07-29.md` | Pipeline + freeze log |
| `reports/research_log/2026-08-05.md` | Report prep log |

---

## Appendix B — Exact Commands (Evidence Generation)

Commands below are **documented from research logs and artifact metadata** for reproducibility. They were **not re-executed** for this draft.

### B.1 Data validation and milestones

```bash
# Backfill milestone tracker (read-only coverage report)
python bid_ask_backfill_milestone_tracker.py
# Output: reports/backfill_milestones/backfill_milestones_latest.json
```

### B.2 Candidate F final walk-forward (2026-07-29)

```bash
# Frozen config rerun — no tuning
python bid_ask_walk_forward.py --coin ETH --interval 15s \
  --config ETH_UP_H20_C0.65_E2_M1.25_S8
# Output: reports/bid_ask_walk_forward/walk_forward_ETH_20260729T101645.json
#         reports/bid_ask_walk_forward/accepted_trades_ETH_20260729T101645.csv
```

*Note: Exact CLI flags should be verified against the script's `--help` output; command form inferred from artifact paths and research log.*

### B.3 Candidate F robustness and milestone comparison (2026-07-29)

```bash
python candidate_f_robustness_analyzer.py
# Output: reports/candidate_f_robustness/candidate_f_robustness_latest.json

python candidate_f_milestone_comparison.py
# Output: reports/candidate_f_milestone_comparison/candidate_f_milestone_comparison_latest.json
```

### B.4 Report preparation (2026-08-05)

No model reruns, tuning, data download, or execution enablement. Artifacts assembled manually from existing JSON/CSV/MD files per `reports/research_log/2026-08-05.md`.

---

## Appendix C — Figure and Table Index

### Tables

| Table | Title | Primary source |
|-------|-------|----------------|
| Table 1 | Candidate B–F Comparison | `candidate_history.md` |
| Table 2 | Candidate F Frozen Configuration | `candidate_f_milestone_comparison_latest.json` |
| Table 3 | Candidate F Final Walk-Forward Metrics | `candidate_f_final_summary.json`, walk-forward JSON |
| Table 4 | Candidate F Robustness Summary | `candidate_f_robustness_latest.json` |
| Table 5 | Dataset Coverage and Validation Summary | `backfill_milestones_latest.json` |
| Table 6 | Major Pipeline Failures and Fixes | `project_timeline.md`, research logs |

### Figures (rendered per `figure_manifest.json`)

| Figure | Filename | Title |
|--------|----------|-------|
| Figure 1 | `candidate_f_cumulative_net_bps.png` | Candidate F Cumulative Net Return (bps) |
| Figure 2 | `candidate_f_return_distribution.png` | Candidate F Accepted-Trade Net Return Distribution |
| Figure 3 | `candidate_development_timeline.png` | Candidate Development Timeline (B → F) |
| Figure 4 | `data_pipeline_workflow.png` | Bid/Ask Research Pipeline — Paper-Only Workflow |

### Planned but not rendered

| Planned figure (`table_figure_plan.md`) | Status |
|----------------------------------------|--------|
| Accepted Trades by Day or Regime (plan Figure 4) | **Not yet created** as PNG; data available in robustness JSON |

---

## Document Metadata

| Field | Value |
|-------|-------|
| Version | draft_v1 |
| Word count | 4,623 |
| Models rerun | No |
| Candidate F tuning | No |
| Execution enabled | No |
| Data download | No |

---

*End of draft v1. Generated 2026-08-05 from existing local evidence only.*
