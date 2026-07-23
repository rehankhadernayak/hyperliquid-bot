# Candidate F Milestone Comparison

Generated at: 2026-07-23T13:53:49.720486Z
Frozen config: ETH_UP_H20_C0.65_E2_M1.25_S8
Feature set: candidate_f_regime_depth

## Safety statement

- paper_only: true
- execution_enabled: false
- real_execution_ready: false

Candidate F milestone comparison is read-only and paper-only. Reads local walk-forward and robustness report artifacts only. Dataset bounds are derived read-only from CSV paths declared in walk-forward reports. No orders, execution, wallets, AWS, network, or shell commands.

## Discovery summary

- discovered_walk_forward_count: 4
- discovered_robustness_count: 4
- milestone_count: 1
- duplicate_runs_superseded: 2

## Chronological comparison table

| Milestone | Dataset first | Dataset latest | Coverage h | Trades | Avg bps | Cum bps | Win rate | Completed | Skipped | Dom hour (frac) | Dom day (frac) | Ex-dom avg bps | Vol regimes | Spread regimes | Verdict |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | --- | --- | --- |
| 1 | 2023-09-16T00:00:00Z | 2023-09-28T23:59:45Z | 312.0 | 17 | 1.69 | 28.75 | 76.5% | 646 | 9 | 14 (94.1%) | 2023-09-23 (94.1%) | -2.68 | high_volatility_regime:2@-3.63bps; low_volatility_regime:11@3.34bps | tight_spread_regime:17@1.69bps; wide_spread_regime:0@0.00bps | diagnostic_positive_sparse |

## Superseded duplicate runs

- 20260723T085108 superseded by 20260723T092114 (reports/bid_ask_walk_forward/walk_forward_ETH_20260723T085108.json)
- 20260723T085350 superseded by 20260723T092114 (reports/bid_ask_walk_forward/walk_forward_ETH_20260723T085350.json)

Bid/ask schema helpers are read-only, paper-trading only, and research-only. No wallets, private keys, signing, leverage, margin, exchange actions, real orders, deposits, withdrawals, or execution endpoints are used.

