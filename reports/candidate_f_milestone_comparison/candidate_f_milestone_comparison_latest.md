# Candidate F Milestone Comparison

Generated at: 2026-07-29T10:05:36.292829Z
Frozen config: ETH_UP_H20_C0.65_E2_M1.25_S8
Feature set: candidate_f_regime_depth

## Safety statement

- paper_only: true
- execution_enabled: false
- real_execution_ready: false

Candidate F milestone comparison is read-only and paper-only. Reads local walk-forward and robustness report artifacts only. Dataset bounds are derived read-only from CSV paths declared in walk-forward reports. No orders, execution, wallets, AWS, network, or shell commands.

## Discovery summary

- discovered_walk_forward_count: 5
- discovered_robustness_count: 5
- milestone_count: 1
- duplicate_runs_superseded: 3

## Chronological comparison table

| Milestone | Dataset first | Dataset latest | Coverage h | Trades | Avg bps | Cum bps | Win rate | Completed | Skipped | Dom hour (frac) | Dom day (frac) | Ex-dom avg bps | Vol regimes | Spread regimes | Verdict |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | --- | --- | --- |
| 1 | 2023-09-16T00:00:00Z | 2023-09-30T14:59:45Z | 351.0 | 15 | 3.01 | 45.16 | 93.3% | 695 | 10 | 14 (86.7%) | 2023-09-23 (86.7%) | 2.48 | high_volatility_regime:0@0.00bps; low_volatility_regime:13@3.60bps | tight_spread_regime:15@3.01bps; wide_spread_regime:0@0.00bps | diagnostic_positive_sparse |

## Superseded duplicate runs

- 20260723T085108 superseded by 20260729T100529 (reports/bid_ask_walk_forward/walk_forward_ETH_20260723T085108.json)
- 20260723T085350 superseded by 20260729T100529 (reports/bid_ask_walk_forward/walk_forward_ETH_20260723T085350.json)
- 20260723T092114 superseded by 20260729T100529 (reports/bid_ask_walk_forward/walk_forward_ETH_20260723T092114.json)

Bid/ask schema helpers are read-only, paper-trading only, and research-only. No wallets, private keys, signing, leverage, margin, exchange actions, real orders, deposits, withdrawals, or execution endpoints are used.

