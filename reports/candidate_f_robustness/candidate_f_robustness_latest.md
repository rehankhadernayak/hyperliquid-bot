# Candidate F Robustness Report

Generated at: 2026-07-29T10:05:34.308702Z
Candidate grid source: reports/candidate_grid/candidate_grid_20260717T080653.json

## Safety statement

- paper_only: true
- execution_enabled: false
- real_execution_ready: false

Candidate F robustness analysis is read-only and paper-only. Reads local report artifacts only. No orders, execution, wallets, AWS, network, or shell commands.

## Candidate F summary

- feature_set: candidate_f_regime_depth
- grid_row_count: 1
- positive_config_count: 1
- best_config_id: ETH_UP_H20_C0.65_E2_M1.25_S8
- best_average_net_return_bps: 1.69 bps
- best_accepted_trades_count: 17

## Positive configs

1. **ETH_UP_H20_C0.65_E2_M1.25_S8** — avg=1.69 bps, trades=17

## Accepted-trades robustness

### ETH_UP_H20_C0.65_E2_M1.25_S8

- matching_status: matched
- accepted_trades_csv_path: reports/bid_ask_walk_forward/accepted_trades_ETH_20260729T100529.csv
- trade_count: 15
- cumulative_net_bps: 45.16 bps
- average_net_bps: 3.01 bps
- win_rate: 93.3%
- top_1_trade_contribution_pct: 16.0%
- top_3_trades_contribution_pct: 31.0%
- worst_3_trades_loss_share_pct: 100.0%
- verdict: diagnostic_positive_sparse

## Regime/hour/day concentration

- dominant UTC hour: 14 (86.7% of trades)
- high_volatility_regime: count=0, cumulative=0.00 bps, avg=0.00 bps
- low_volatility_regime: count=13, cumulative=46.76 bps, avg=3.60 bps
- tight_spread_regime: count=15, cumulative=45.16 bps, avg=3.01 bps
- wide_spread_regime: count=0, cumulative=0.00 bps, avg=0.00 bps
- by volatility regime:
  - high_volatility_regime: count=0, cumulative=0.00 bps, avg=0.00 bps
  - low_volatility_regime: count=13, cumulative=46.76 bps, avg=3.60 bps
- dominant volatility regime: low_volatility_regime (86.7% of trades)
- by spread regime:
  - tight_spread_regime: count=15, cumulative=45.16 bps, avg=3.01 bps
  - wide_spread_regime: count=0, cumulative=0.00 bps, avg=0.00 bps
- dominant spread regime: tight_spread_regime (100.0% of trades)
- dominant regime (overall): tight_spread_regime (100.0% of trades)
- by UTC hour:
  - 14: count=13, sum=40.21 bps, avg=3.09 bps
  - 15: count=1, sum=-2.68 bps, avg=-2.68 bps
  - 20: count=1, sum=7.64 bps, avg=7.64 bps
- by day:
  - 2023-09-21: count=1, sum=-2.68 bps, avg=-2.68 bps
  - 2023-09-23: count=13, sum=40.21 bps, avg=3.09 bps
  - 2023-09-27: count=1, sum=7.64 bps, avg=7.64 bps

## Verdict

- overall_verdict: diagnostic_positive_sparse
- robust_candidate: false

## Next recommended action

Continue paper-only data collection until at least 30 accepted trades exist for the leading Candidate F config.

Bid/ask schema helpers are read-only, paper-trading only, and research-only. No wallets, private keys, signing, leverage, margin, exchange actions, real orders, deposits, withdrawals, or execution endpoints are used.

