# Hyperliquid Bid/Ask ML Research Notes

Rough log — last updated 2026-07-06. Paper-only research. Not a trading bot demo.

---

## 1. Project Goal

Build a **paper-only** research pipeline on Hyperliquid BTC/ETH bid/ask microstructure data:

- extract / backfill historical L2
- repair and validate CSVs
- resample to 15s / 1m
- train classifiers, walk-forward test
- filter candidates with realistic costs
- log **paper decisions** (no real orders)

Goal for the presentation: show uncle + friend a **serious research system**, not a fake profitable bot.

---

## 2. Safety Boundary

Hard rules (non-negotiable):

| Flag | Value |
|------|-------|
| `paper_only` | `true` |
| `execution_enabled` | `false` |
| `real_execution_ready` | `false` |

No wallets, private keys, signing, deposits, withdrawals, leverage, margin, or order endpoints. Dashboard and all research scripts are read-only / offline from execution.

**Simulated returns do not prove profitability.** Positive backtest snippets can be noise, overfitting, or too few trades.

---

## 3. Milestone Journey

| # | Milestone | Progress | Notes |
|---|-----------|----------|-------|
| 1 | Data Pipeline Works | 80% | 1s import, repair, validate, 15s/1m resample working for BTC/ETH |
| 2 | Paper Backtest Engine Works | 85% | Walk-forward + cost model + paper trade CSV export |
| 3 | First Candidate Found | 40% | Candidate B exists but only 5 trades |
| 4 | Candidate Scaling | 20% | Relaxing filters kills performance |
| 5 | Robustness | 15% | Stability checks flag concentration / sparse buckets |
| 6 | Out-of-Sample Paper Test | 0% | Not started |
| 7 | Paper Operations | 70% | Scheduler, orchestrator, daily summaries running |
| 8 | Market Ready | 0% | Explicitly not ready — real execution disabled |

---

## 4. Architecture

```
Hyperliquid S3 / WebSocket
        │
        ▼
   Raw L2 data
        │
        ▼
Bid/Ask importer / extractor  (AWS Requester Pays backfill)
        │
        ▼
     1s CSV  (historical_1s/)
        │
        ▼
  15s / 1m resampler
        │
        ▼
    Validator  (+ repair pass for bad rows)
        │
        ▼
  Feature builder  (microstructure set, ~44 features)
        │
        ▼
  Model training  (RandomForestClassifier)
        │
        ▼
Walk-forward evaluator
        │
        ▼
Candidate validator  (costs, stability, min trades)
        │
        ▼
Paper decision dashboard  (read-only Streamlit)
```

Key scripts (not modified for dashboard work):

- `bid_ask_research_scheduler.py` — autopilot / backfill cycles
- `bid_ask_data_pipeline.py` — import / resample / validate orchestration
- `bid_ask_walk_forward_evaluator.py` — rolling train/test + paper trades
- `bid_ask_candidate_validator.py` — Candidate B gates

---

## 5. Progress From Last Few Days

**Data**

- Phase 3 backfill sprint: `phase3_btc_eth_recent`, target **14 days**, focused **BTC + ETH**
- ~105 `.lz4` files downloaded via Requester Pays (BTC 53, ETH 52)
- Imported **1s historical CSVs** cover ~**48 hours** (2023-09-16 → 2023-09-17)
  - BTC: 145,137 rows
  - ETH: 139,528 rows
- **15s** resample: BTC 6,242 rows, ETH 6,002 rows
- **1m** resample: BTC 1,562 rows, ETH 1,502 rows

**Models / evaluation**

- Walk-forward on 15s bars, horizon 20, microstructure features
- Model accuracy ~**42.3%** vs baseline ~**38.2%** (small edge — not enough alone)
- Completed **57 windows** in latest BTC run

**Candidate B**

- Name: `BTC_DOWN_MICRO_H20_CANDIDATE_B`
- Strict filters: confidence ≥ 0.65, edge ≥ 2 bps, **DOWN class only**
- **5 paper trades**, cumulative net **+3.84 bps**, avg net **+0.77 bps**, win rate **60%**
- Candidate validator still marks: `insufficient_trades`, `not_market_ready`, unstable

---

## 6. Bugs / Problems Found

1. **Backfill import stall** — scheduler downloads chunks but `historical_1s` timestamps sometimes don't advance (`no_import_progress`). Phase3 audit recommends: *investigate date/hour planning or importer path*.

2. **1s data gaps** — validation passes but warns ~**80k missing seconds** in coverage range (expected for sparse L2 snapshots, not a full continuous feed).

3. **Mock / bad rows in early 1s CSV** — repair logs removed **24 mock rows** from BTC (and similar for ETH).

4. **Candidate B trade count vs performance cliff**
   - Strict (5 trades): slightly positive after costs
   - Relaxed to 25 trades (lower confidence/edge, DOWN only): **cumulative −101.8 bps**
   - Relaxed to 62 trades (DOWN + UP): **cumulative −268.0 bps**

5. **Stability** — one time bucket contributes >70% of positive net bps; 4 buckets have <3 trades.

6. **Repeated unchanged candidate reports** — validation output identical across last 3 runs (data/candidate frozen until backfill grows).

---

## 7. Fixes Made

- **1s CSV repair** — dedupe timestamps, remove mock rows, recompute ISO timestamps, confirm `monotonic_after: true`, zero duplicate timestamps post-repair.
- **Validation pipeline** — latest BTC/ETH 1s/15s/1m validation reports **pass** (with gap warnings).
- **Resampling** — 15s and 1m bars regenerated from repaired 1s source.
- **Research dashboard** — new read-only Streamlit view (`bid_ask_research_dashboard.py`) for presentation; no execution paths.

---

## 8. Current Candidate Results

### Strict Candidate B (5 trades)

| Metric | Value |
|--------|-------|
| Trades | 5 |
| Cumulative net | +3.84 bps |
| Average net | +0.77 bps |
| Win rate | 60% |
| Positive after costs | yes (simulated) |
| Beats baseline accuracy | yes |
| Market ready | **no** |

### Relaxed — 25 trades (conf 0.60, edge 1 bps, DOWN only)

| Metric | Value |
|--------|-------|
| Trades | 25 |
| Cumulative net | **−101.82 bps** |
| Average net | −4.07 bps |
| Win rate | 16% |
| Positive after costs | **no** |

### Relaxed — 62 trades (conf 0.60, edge 1 bps, DOWN + UP)

| Metric | Value |
|--------|-------|
| Trades | 62 |
| Cumulative net | **−267.95 bps** |
| Average net | −4.32 bps |
| Win rate | ~19% blended |
| Positive after costs | **no** |

**Important:** the 5-trade “positive” result is **not** evidence the strategy works. Sample size is tiny; stability checks fail; relaxed versions lose money after costs in simulation.

---

## 9. Current Blocker

> Find a candidate that **increases trade count** while staying **positive after realistic costs** (fees + slippage + spread).

Right now it's one or the other:

- **Strict filters** → few trades, slightly positive (maybe noise)
- **Looser filters** → more trades, clearly negative

Also blocked on **more historical data** (~48h imported vs 14-day sprint target) and **import progress stalls** during backfill.

---

## 10. Next Plan

1. Finish **BTC/ETH 14-day data sprint** — fix import planning stall
2. **Inspect losing trades** from relaxed Candidate B (which windows/regimes hurt?)
3. Test horizons **10 / 15 / 20 / 30 / 40**
4. **Compare BTC vs ETH** on same feature set
5. Build a **candidate grid** (confidence × edge × class × horizon)
6. **Freeze** one promising config if anything survives scaling
7. **Out-of-sample paper test** on held-out dates (not started)

---

## 11. Questions I Want Feedback On

1. Is **+3.8 bps on 5 trades** worth anything, or should I treat it as zero evidence until N≥20–30?

2. When relaxing filters kills performance, is the usual answer **better features / regime filters**, or **accept low frequency**?

3. For presentation: is showing the **negative relaxed results** enough to prove I'm not cherry-picking, or should I add explicit “null hypothesis” baselines?

4. **48 hours of data** — too early for any candidate claim? Should I delay all candidate talk until 14-day backfill lands?

5. Cost model uses **fee 2 bps + slippage 1 bps × 1.25 multiplier** — does that sound conservative or optimistic for Hyperliquid BTC perps?

6. Anyone seen microstructure **DOWN-only** edges before, or is class imbalance a red flag?

---

*Reminder: this is a student research log. Real execution is disabled. Nothing here is investment advice.*
