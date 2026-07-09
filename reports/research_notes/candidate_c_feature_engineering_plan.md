# Candidate C Feature Engineering Plan

Last updated: 2026-07-09. Paper-only research. No model training changes in this phase.

---

## 1. Context and Motivation

Candidate B tuning (threshold/regime filters, hour exclusions) is **rejected for now**:

- More historical data turned previously positive BTC H10/H20 DOWN-only configs **negative after costs**.
- Hour filters reduced losses on older samples but **did not survive** when new data was added.
- Post-hoc trade filters increase trade count at the cost of generalization; relaxed Candidate B runs lost heavily (−102 bps at 25 trades, −268 bps at 62 trades).

**Candidate C** shifts focus from *filtering bad trades* to *giving the model better inputs* so signal is learned in-feature rather than bolted on after prediction.

| Approach | Candidate B (rejected path) | Candidate C (this plan) |
|----------|----------------------------|-------------------------|
| Primary lever | Confidence/edge/spread/hour filters | Feature families + label-aware inputs |
| Risk | Overfit to sparse regimes | Feature redundancy / leakage if mis-specified |
| Success metric | Positive net bps with ≥30 trades, stable buckets | Features with stable correlation to cost-adjusted future returns across walk-forward windows |

---

## 2. Scope

**In scope**

- Define seven feature families for audit and eventual training integration.
- Run read-only feature audit on existing resampled CSVs (`historical_15s`, `historical_1m`).
- Compare feature distributions and correlations with future returns before retraining.
- Document which features require 1s size columns vs resampled-only proxies.

**Out of scope (this phase)**

- Changing `bid_ask_train.py` or walk-forward training defaults.
- New post-hoc trade filters (hour, spread gate, regime gate).
- Any execution, wallet, or network paths.

---

## 3. Data Inputs

| Source | Path pattern | Required columns |
|--------|--------------|------------------|
| Resampled 15s/1m | `data/bid_ask_csv/historical_{interval}/{COIN}_{interval}.csv` | `RESAMPLED_COLUMNS` from `bid_ask_resampler.py` |
| Optional 1s sizes | Not in current resampled bars | `bid_size`, `ask_size` on 1s CSV for microprice / liquidity |

Current resampled bars aggregate spread and imbalance but **drop top-of-book sizes**. Microprice and size-imbalance features will be marked `skipped` in audit until resampler exports `avg_bid_size` / `avg_ask_size` (future enhancement).

---

## 4. Feature Families

### 4.1 Spread / Cost Regime

**Hypothesis:** Model should see execution friction directly, not rely on post-hoc spread filters.

| Feature | Definition | Source column(s) |
|---------|------------|------------------|
| `spread_bps` | Bar-average spread in bps | `avg_spread_bps` |
| `rolling_spread_mean_20` | 20-bar rolling mean of spread | derived |
| `rolling_spread_std_20` | 20-bar rolling std | derived |
| `spread_zscore_20` | (spread − mean) / std over 20 bars | derived |
| `estimated_cost_bps` | `fee_bps + slippage_bps + spread_bps` | defaults: 2 + 1 + spread |

**Audit checks:** missingness, tail quantiles (p95/p99), correlation with `future_return_bps` at H1/H5/H10/H20, class balance in tight vs wide spread terciles.

### 4.2 Order-Book Imbalance Regime

**Hypothesis:** Persistent imbalance predicts short-horizon pressure; level alone may be weak without regime context.

| Feature | Definition |
|---------|------------|
| `bbo_imbalance` | `(bid_size − ask_size) / (bid_size + ask_size)` or `avg_bbo_imbalance` |
| `imbalance_mean_20` | 20-bar rolling mean |
| `imbalance_std_20` | 20-bar rolling std |
| `imbalance_zscore_20` | rolling z-score |
| `imbalance_change_1` | first difference |

**Audit checks:** sign persistence, correlation with signed future returns, UP/DOWN class balance by imbalance tercile.

### 4.3 Microprice / Pressure

**Hypothesis:** Size-weighted mid captures short-term buy/sell pressure better than simple mid.

| Feature | Definition | Availability |
|---------|------------|--------------|
| `microprice` | `(ask × bid_size + bid × ask_size) / (bid_size + ask_size)` | Requires sizes |
| `microprice_mid_deviation_bps` | `(microprice − mid) / mid × 10_000` | Requires sizes |
| `rolling_pressure_change` | Δ microprice deviation over 5 bars | Requires sizes |

**Fallback until resampler change:** skip in audit; note in report. Do **not** proxy with unweighted mid.

### 4.4 Volatility / Momentum

**Hypothesis:** Return and vol at multiple lookbacks help disambiguate mean-reversion vs continuation.

| Feature | 15s bar lag | 1m bar lag |
|---------|-------------|------------|
| `return_15s_bps` | 1 | — |
| `return_30s_bps` | 2 | — |
| `return_60s_bps` | 4 | — |
| `realized_volatility_20` | std of 1-bar returns, 20 windows | same |
| `short_term_drawdown_bps` | rolling min return over 10 bars | same |
| `short_term_upmove_bps` | rolling max return over 10 bars | same |

For 1m audits, scale lags as `max(1, round(seconds / 60))`.

**Audit checks:** correlation stability across walk-forward halves; avoid features with >30% missing at start of series.

### 4.5 Time / Session

**Hypothesis:** Session effects exist but should be **features**, not hard hour filters.

| Feature | Definition |
|---------|------------|
| `utc_hour` | hour from `start_timestamp_ms` |
| `hour_sin`, `hour_cos` | cyclical encoding |
| `is_weekend` | Saturday/Sunday UTC |

**Audit checks:** future return mean/std by hour; class balance by hour; **do not** auto-exclude hours based on audit alone.

### 4.6 Liquidity / Size

**Hypothesis:** Thin books widen effective cost and reduce signal reliability.

| Feature | Definition | Availability |
|---------|------------|--------------|
| `bid_size`, `ask_size` | top-of-book sizes | 1s only today |
| `total_top_size` | bid + ask | 1s only |
| `size_imbalance` | same formula as BBO imbalance on sizes | 1s only |

**Planned resampler addition:** `avg_bid_size`, `avg_ask_size` on 15s/1m bars.

### 4.7 Label / Target Diagnostics

Not model inputs — audit outputs to validate whether features align with tradable edge.

| Diagnostic | Description |
|------------|-------------|
| Future return distribution | mean/std/skew by hour, spread tercile, imbalance tercile |
| Class balance | UP / DOWN / FLAT fractions by regime |
| Net paper return by regime | If `accepted_trades_{COIN}_*.csv` exists under `reports/bid_ask_walk_forward/` |

Label definition for audit (matches training baseline):

```text
future_return_bps = (close_mid[t+h] − close_mid[t]) / close_mid[t] × 10_000
label: UP if > threshold_bps, DOWN if < −threshold_bps, else FLAT
```

Horizons: H1, H5, H10, H20 bars (configurable).

---

## 5. Feature Audit Workflow

```bash
# BTC 15s historical (primary research interval)
python3 bid_ask_feature_audit.py audit --coin BTC --source historical --interval 15s

# ETH 15s historical
python3 bid_ask_feature_audit.py audit --coin ETH --source historical --interval 15s
```

Outputs:

- `reports/feature_audit/feature_audit_{COIN}_{interval}_{stamp}.json`
- `reports/feature_audit/feature_audit_{COIN}_{interval}_{stamp}.md`

Audit report sections:

1. Safety flags (`paper_only`, `execution_enabled`, `real_execution_ready`)
2. Input CSV metadata (path, row count, time range)
3. Per-feature missingness, min/median/max, quantiles
4. Pearson correlation with `future_return_bps` per horizon
5. Skipped features (missing optional columns)
6. Label diagnostics by regime
7. Accepted-trades regime summary (if CSV found)

---

## 6. Selection Criteria (Pre-Training Gate)

A feature graduates from audit to Candidate C training set if:

1. **Coverage:** missingness < 5% after warmup period (first 20 bars excluded).
2. **Variance:** not constant (std > 1e-6).
3. **Stability:** correlation sign with H20 future returns consistent in first vs second chronological half of sample (same sign, |Δr| < 0.15).
4. **No obvious leakage:** feature uses only data at or before bar `t`.
5. **Regime sanity:** not entirely driven by a single hour bucket (>70% of absolute correlation mass).

Features failing gate remain documented but excluded from initial Candidate C feature set.

---

## 7. Candidate C Training Integration (Future Phase)

After audit sign-off:

1. Add `create_candidate_c_features()` mirroring audit definitions in a shared module.
2. Extend walk-forward evaluator with `feature_set="candidate_c"`.
3. Retrain with **same** cost model (fee 2 bps, slippage 1 bps, spread from bar).
4. Compare against Candidate B baseline on:
   - cumulative net bps
   - trade count
   - stability buckets (hour, spread, imbalance)
   - out-of-sample second half only

**Do not** combine new features with relaxed hour/spread filters in the first experiment — isolate feature effect.

---

## 8. Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Resampled bars lose size information | Extend resampler; until then, skip microprice/size family |
| Feature collinearity (spread vs cost) | Audit correlation matrix; drop redundant pair |
| Small sample (~48h historical) | Treat audit as directional; re-run after backfill grows |
| Overfitting to H20 DOWN class | Train multiclass; evaluate per-class calibration |

---

## 9. Success Criteria

Candidate C feature phase succeeds when:

- Audit runs cleanly for BTC and ETH on `historical_15s`.
- At least **two feature families** show stable H10/H20 correlation with future returns.
- Accepted-trades regime diagnostic explains **why** Candidate B concentrated in one bucket (if CSV available).
- Feature set documented and ready for training PR without post-hoc hour filters.

---

## 10. Safety

All audit tooling is read-only:

- `paper_only: true`
- `execution_enabled: false`
- `real_execution_ready: false`
- No network, AWS, subprocess, wallets, or order paths.

Simulated correlations and regime tables do **not** prove profitability.
