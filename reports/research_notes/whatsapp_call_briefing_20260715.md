# WhatsApp Call Briefing — Hyperliquid Paper Research

**Date:** 15 July 2026  
**Audience:** Uncle's friend (research mentor check-in)  
**Purpose:** Update on progress, ask for guidance on research direction

---

## Safety (say this first)

This is **educational research only**. The project is **paper-only** — no real trading, no order execution, no wallets, and no investment claims. Simulated backtest numbers are for learning and validation design, not profitability proof.

---

## 1. 30-Second Summary

> I built a local research pipeline on Hyperliquid BTC/ETH bid/ask microstructure data. It backfills historical L2 snapshots, validates and resamples them, engineers features, trains classifiers, and runs walk-forward paper tests with costs — all offline, no execution. Data now covers ~183 hours through **2023-09-23**. Infrastructure is stable: scheduler timeouts fixed, depth-aware features restored (27 active features). Several strategy candidates (B–E) were tested; none is robust yet. Candidate E showed a sparse positive ETH signal but failed trade-count robustness. Next focus: expand data and write a detailed research report by early August.

---

## 2. Two-Minute Explanation

**What the project is**

A structured ML research lab on short-horizon crypto order-book microstructure — not a trading bot.

**Pipeline (end to end)**

1. Download historical L2 data from Hyperliquid (Requester Pays S3 backfill)
2. Import to 1-second CSVs → validate → resample to 15s and 1m bars
3. Build microstructure features (spread, imbalance, microprice, depth/size, volatility, session time)
4. Train Random Forest classifiers on direction labels
5. Walk-forward evaluation with realistic costs (fees + slippage + spread)
6. Candidate validation gates (min trades, stability, positive-after-costs)
7. Log everything to a candidate ledger and research dashboard (read-only)

**What I tested**

Four research candidates exploring different hypotheses:

| Candidate | Hypothesis |
|-----------|------------|
| **B** | Strict microstructure filter on BTC DOWN, horizon 20 bars |
| **C** | Broader feature-engineered microstructure set |
| **D** | Cost-aware labeling (trade only when edge exceeds costs) |
| **E** | Depth-aware features using bid/ask size at top of book |

**Current honest result**

The pipeline works reliably. Promising-looking snippets appear (e.g., small positive ETH UP configs), but they are **sparse, unstable, or negative after costs** when sample size grows. That is useful research output — it tells us what does *not* survive scrutiny.

---

## 3. What Changed Since Last Update (~6 July)

| Area | Before | Now |
|------|--------|-----|
| **Data coverage** | ~48 hours (2023-09-16 → 09-17) | **~183 hours** through **2023-09-23T14:59:59Z** |
| **BTC 1s rows** | ~145k | **808,308** |
| **ETH 1s rows** | ~140k | **813,431** |
| **Scheduler** | Ticks timing out on long backfill runs | **90-minute tick timeout** — autopilot completes reliably |
| **Depth features** | `bid_size` / `ask_size` dropped during resample; size features skipped | **Resampler fixed** — last-valid 1s tick carried into 15s/1m bars |
| **Feature audit** | Size/microprice families skipped | **27 active features**, `skipped_features=[]` |
| **Candidate B** | 5 strict trades, small positive snippet | **Rejected** — 28 strict trades, **−127 bps** after costs with more data |
| **Candidate E** | Not yet run with depth features | Sparse positive ETH UP H20 signal; **failed min-trade robustness** |
| **Research artifacts** | Early notes only | Daily progress digests, candidate ledger (330+ rows), final report skeleton started |

---

## 4. Current Technical Status

**Safety flags (all correct)**

- `paper_only: true`
- `execution_enabled: false`
- `real_execution_ready: false`

**Data (BTC / ETH, as of today)**

| Interval | BTC rows | ETH rows | Last timestamp |
|----------|----------|----------|----------------|
| 1s | 808,308 | 813,431 | 2023-09-23 14:59 UTC |
| 15s | 35,902 | 36,141 | 2023-09-23 14:59 UTC |
| 1m | 8,989 | 9,048 | 2023-09-23 14:59 UTC |

- `bid_size` and `ask_size` present on all intervals ✓
- Validation passes with known gap warnings (~130k missing 1s seconds — expected for sparse L2 snapshots)

**Infrastructure**

- Scheduler autopilot: running, backfill phase `phase3_btc_eth_recent`
- Orchestrator: resample → train → walk-forward → validate cycle working
- Feature audit: 27 features across spread/cost, imbalance, microprice, volatility, session, liquidity/size families
- Candidate ledger: 333 logged experiment rows

**Known data-quality warnings (not blockers, but documented)**

- Missing seconds in 1s coverage (~130k per coin)
- Missing/incomplete resampled bars (15s ~8k, 1m ~2k per coin)
- ~100 rows exceed extreme spread threshold (>100 bps)

---

## 5. Candidate B / C / D / E Status

### Candidate B — `BTC_DOWN_MICRO_H20_CANDIDATE_B`

**Status: Rejected**

- Strict config: 28 paper trades, **−126.9 bps** cumulative net (research-only)
- Relaxed config: 113 trades, **−609.3 bps**
- Early positive signal on tiny sample did **not** hold with expanded data
- Stability flags: concentration in one time bucket, sparse trade buckets

### Candidate C — `candidate_c_core` (feature-engineered microstructure)

**Status: Rejected / diagnostic**

- Full feature set (27 features) now active and auditable
- Walk-forward runs remain **negative after costs**
- Useful for feature correlation analysis, not a viable signal on current sample

### Candidate D — cost-aware labels

**Status: Diagnostic / useful but negative**

- Labeling that requires predicted edge > estimated cost is a good diagnostic filter
- Alone, not predictive enough to pass validation gates
- Will re-run after more out-of-sample data lands

### Candidate E — `candidate_e_depth_aware` (bid/ask size features)

**Status: Interesting but too sparse — failed robustness**

- Depth-aware features restored after resampler fix
- **ETH UP, horizon 20:** 14 paper trades, +31.3 bps cumulative net (research-only — **not** a profitability claim)
- Only 3 trades in top grid configs → flagged **SPARSE (<10 trades)**
- Failed minimum-trade robustness check; loss concentration in worst 5 trades
- BTC configs negative after costs

**Bottom line across all candidates:** No candidate is market-ready. The research value is in the disciplined pipeline and honest negative results.

---

## 6. Main Lesson Learned

> **Small positive backtest snippets on tiny samples are misleading. Expanding data and applying strict robustness gates (min trades, stability, costs) is essential — and it correctly kills most candidates.**

Specifically:

1. Candidate B looked slightly positive on 5 trades; with 28+ trades it turned clearly negative.
2. Relaxing filters to get more trades made performance worse, not better.
3. Fixing the resampler bug (dropped `bid_size`/`ask_size`) was necessary before trusting any depth-aware result.
4. A research pipeline that produces honest failures is more valuable than one that cherry-picks winners.

---

## 7. What I Am NOT Claiming

- ❌ This is **not** a trading bot or investment system
- ❌ I am **not** claiming profitability or alpha
- ❌ Positive paper-trade snippets (e.g., ETH +31 bps on 14 trades) are **not** evidence of a working strategy
- ❌ Model accuracy ~42% vs baseline ~38% is a small diagnostic edge, not tradable edge
- ❌ I am **not** ready for live execution — and have no intention to enable it
- ❌ Backtest metrics do **not** predict future results

**What I am claiming:**

- ✅ A working, reproducible, paper-only research pipeline
- ✅ Systematic candidate testing with cost-aware evaluation
- ✅ Honest documentation of what failed and why
- ✅ A learning project in ML validation for financial microstructure data

---

## 8. Next Steps (Now → Early August)

| Priority | Action |
|----------|--------|
| 1 | **Continue BTC/ETH historical backfill** — extend beyond 2023-09-23 |
| 2 | **Resolve missing-bar warnings** before trusting resampled features on new chunks |
| 3 | **Re-run feature audit** after each major data increment |
| 4 | **Re-run Candidates D and E** only after sufficient new out-of-sample data (avoid threshold tuning on current sample) |
| 5 | **Write final research report** — methodology, data quality, candidate timeline, negative results, lessons learned |
| 6 | **Prepare presentation narrative** framed as research system demo, not trading pitch |

**Target deliverable:** Detailed research report + dashboard walkthrough by **early August 2026**.

---

## 9. Questions to Ask Him

1. **Focus trade-off:** Should I prioritize **data quality**, **feature engineering**, or **validation design** in the remaining weeks?

2. **Reporting failures:** How should I present negative/failed candidates in the final report — as a dedicated "what we learned" section, or integrated into each candidate's timeline?

3. **Research framing:** What would make this project look **stronger as a research project** (methodology, rigor, reproducibility) rather than a trading system pitch?

4. **ML validation methods:** Are there specific validation approaches he recommends for **small, imbalanced financial datasets** (e.g., purged k-fold, combinatorial purged CV, bootstrap confidence intervals, deflated Sharpe)?

---

## Quick Reference Numbers (if asked)

- Data: ~183 hours, 808k BTC / 813k ETH 1s rows
- Features: 27 active, 0 skipped
- Candidate ledger: 333 experiment rows
- Best ETH snippet (Candidate E): 14 trades, +31.3 bps — **sparse, not robust**
- Candidate B (strict): 28 trades, −126.9 bps — **rejected**
- Scheduler: 90-min tick timeout, autopilot stable
- Report target: early August 2026

---

*Generated for personal use before WhatsApp call. Research-only. Not investment advice.*
