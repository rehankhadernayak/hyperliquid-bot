# Walk-Forward Paper Evaluation

Generated at: 2026-07-29T10:05:29.246814+00:00
Coins: ETH
Interval: 15s | Source: historical
Horizon: 20 bars | Threshold: 1.0 bps

## Safety

- paper_only: true
- execution_enabled: false
- real_execution_ready: false

## Labels

- label_mode: binary_tradeable_up
- label_cost_multiplier: 1.0
- label_min_edge_bps: 0.0
- average_label_required_edge_bps: 7.3211
- min_label_required_edge_bps: 4.5947
- max_label_required_edge_bps: 129.8152
- label 0: count=60392 fraction=0.8497
- label 1: count=10686 fraction=0.1503

## Candidate D Cost-Aware Label Diagnostics

- label_mode: binary_tradeable_up
- horizon_bars: 20
- threshold_bps: 1.0
- estimated_cost_formula: fee_bps + slippage_bps + spread_bps
- no_trade_fraction: 0.8496
- edge_fraction: 0.1504
- accepted_edge_predictions: 15
- false_edge_rate: n/a (no predictions)
- walk_forward_training_supported: true

### Class counts

- 0: count=60372 fraction=0.8496
- 1: count=10686 fraction=0.1504

### False edge rate

- false_edge_rate_if_predictions_available: n/a (no predictions)


## Features

- feature_set: candidate_f_regime_depth
- feature_count: 34
- feature_columns: bbo_imbalance, imbalance_mean_20, imbalance_std_20, imbalance_zscore_20, imbalance_change_1, spread_bps, rolling_spread_std_20, spread_zscore_20, return_15s_bps, return_30s_bps, return_60s_bps, short_term_drawdown_bps, short_term_upmove_bps, utc_hour, hour_sin, hour_cos, is_weekend, bid_size, ask_size, total_top_size, size_imbalance, microprice, microprice_mid_deviation_bps, rolling_pressure_change, realized_volatility_5m_bps, realized_volatility_15m_bps, volatility_ratio_5m_15m, spread_volatility_interaction, microprice_volatility_interaction, imbalance_volatility_interaction, low_volatility_regime, high_volatility_regime, tight_spread_regime, wide_spread_regime

## Aggregate metrics

- total_windows: 705
- completed_windows: 695
- skipped_windows: 10
- skipped_windows_count: 10
- one_class_training_windows_count: 10
- skipped_window_reasons: {'one_class_training_window': 10}
- status: completed
- rejection_reason: n/a
- model_accuracy_mean: 0.8376
- baseline_accuracy_mean: 0.8482
- accuracy_edge_vs_baseline: -0.0106
- total_paper_signals: 1928
- total_paper_trades: 15
- raw_predictions: 1928
- filtered_paper_trades: 15
- blocked_by_confidence: 1376
- blocked_by_cost: 529
- blocked_by_spread: 8
- blocked_by_volatility: 0
- blocked_by_trade_class: 0
- average_cost_bps: 6.4237
- average_required_edge_bps: 10.0296
- average_gross_return_bps: 0.0015
- filtered_paper_trades (accepted): 15
- cumulative_net_return_bps: 45.1650
- average_net_return_bps: 3.0110
- expected_average_from_cumulative: 3.0110
- average_net_return_difference_abs: 0.000000
- net_return_consistency_check: true
- max_drawdown_bps: 2.6839
- win_rate_after_costs: 0.9333333333333333
- longest_losing_streak: 1

## Regime Filters

- regime_filters_enabled: false
- exploratory_overfit_warning: false
- exclude_hours: none
- include_hours: none
- max_cost_bps: none
- min_cost_bps: none
- trades_removed_by_regime_filters: 0

### Before regime filters

- trades_before_regime_filters: 15
- cumulative_net_return_before_regime_filters: 45.1650
- average_net_return_before_regime_filters: 3.0110
- win_rate_before_regime_filters: 0.9333

### After regime filters

- trades_after_regime_filters: 15
- cumulative_net_return_after_regime_filters: 45.1650
- average_net_return_after_regime_filters: 3.0110
- win_rate_after_regime_filters: 0.9333

## Readiness

- beats_baseline_accuracy: False
- positive_after_costs: True
- enough_windows: True
- enough_trades: False
- ready_for_more_research: False
- real_execution_ready: False

## Model vs baseline after costs

Model does not clearly beat baseline on accuracy and/or net returns after simulated costs.

## Per-coin summary

| Coin | Windows | Model acc | Baseline acc | Edge | Trades | Net bps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ETH | 695 | 0.8376 | 0.8482 | -0.0106 | 15 | 45.1650 |
## Trade Diagnostics

### UP

- trades: 15
- cumulative_net_return_bps: 45.1650
- average_net_return_bps: 3.0110
- win_rate_after_costs: 0.9333
- average_cost_bps: 3.7543
- average_confidence: 0.9227

### DOWN

- trades: 0
- cumulative_net_return_bps: 0.0000
- average_net_return_bps: 0.0000
- win_rate_after_costs: 0.0000
- average_cost_bps: 0.0000
- average_confidence: 0.0000

### FLAT

- trades: 0
- cumulative_net_return_bps: 0.0000
- average_net_return_bps: 0.0000
- win_rate_after_costs: 0.0000
- average_cost_bps: 0.0000
- average_confidence: 0.0000

### Accepted vs blocked by class

- accepted UP: 15
- accepted DOWN: 0
- accepted FLAT: 0
- blocked UP: 1913
- blocked DOWN: 0
- blocked FLAT: 67572

### Coin-class trade diagnostics

#### ETH

### UP

- trades: 15
- cumulative_net_return_bps: 45.1650
- average_net_return_bps: 3.0110
- win_rate_after_costs: 0.9333
- average_cost_bps: 3.7543
- average_confidence: 0.9227

### DOWN

- trades: 0
- cumulative_net_return_bps: 0.0000
- average_net_return_bps: 0.0000
- win_rate_after_costs: 0.0000
- average_cost_bps: 0.0000
- average_confidence: 0.0000

### FLAT

- trades: 0
- cumulative_net_return_bps: 0.0000
- average_net_return_bps: 0.0000
- win_rate_after_costs: 0.0000
- average_cost_bps: 0.0000
- average_confidence: 0.0000


## Stability Validation

- stability_report_enabled: true
- stability_min_trades_per_bucket: 3
- profitable_bucket_fraction: 0.7692
- unprofitable_bucket_fraction: 0.2308
- buckets_with_too_few_trades: 7
- largest_positive_bucket_net_bps: 45.1650
- largest_negative_bucket_net_bps: -2.6839
- concentration_warning: true
- sparse_stability_warning: true

### Stability warnings

- stability_concentration: one time/window bucket contributes more than 70% of total positive net bps
- stability_sparse_buckets: 7 bucket(s) have fewer than 3 trades
### by_coin

### ETH

- trades: 15
- cumulative_net_return_bps: 45.1650
- average_net_return_bps: 3.0110
- win_rate_after_costs: 0.9333
- average_cost_bps: 3.7543
- average_confidence: 0.9227
- positive_after_costs: true

### by_predicted_class

### DOWN

- trades: 0
- cumulative_net_return_bps: 0.0000
- average_net_return_bps: 0.0000
- win_rate_after_costs: 0.0000
- average_cost_bps: 0.0000
- average_confidence: 0.0000
- positive_after_costs: false

### FLAT

- trades: 0
- cumulative_net_return_bps: 0.0000
- average_net_return_bps: 0.0000
- win_rate_after_costs: 0.0000
- average_cost_bps: 0.0000
- average_confidence: 0.0000
- positive_after_costs: false

### UP

- trades: 15
- cumulative_net_return_bps: 45.1650
- average_net_return_bps: 3.0110
- win_rate_after_costs: 0.9333
- average_cost_bps: 3.7543
- average_confidence: 0.9227
- positive_after_costs: true

### by_coin_predicted_class

### ETH:UP

- trades: 15
- cumulative_net_return_bps: 45.1650
- average_net_return_bps: 3.0110
- win_rate_after_costs: 0.9333
- average_cost_bps: 3.7543
- average_confidence: 0.9227
- positive_after_costs: true

### by_hour

### 14

- trades: 13
- cumulative_net_return_bps: 40.2069
- average_net_return_bps: 3.0928
- win_rate_after_costs: 1.0000
- average_cost_bps: 3.6286
- average_confidence: 0.9246
- positive_after_costs: true

### 15

- trades: 1
- cumulative_net_return_bps: -2.6839
- average_net_return_bps: -2.6839
- win_rate_after_costs: 0.0000
- average_cost_bps: 4.2645
- average_confidence: 0.9000
- positive_after_costs: false

### 20

- trades: 1
- cumulative_net_return_bps: 7.6420
- average_net_return_bps: 7.6420
- win_rate_after_costs: 1.0000
- average_cost_bps: 4.8780
- average_confidence: 0.9200
- positive_after_costs: true

### by_date

### 2023-09-21

- trades: 1
- cumulative_net_return_bps: -2.6839
- average_net_return_bps: -2.6839
- win_rate_after_costs: 0.0000
- average_cost_bps: 4.2645
- average_confidence: 0.9000
- positive_after_costs: false

### 2023-09-23

- trades: 13
- cumulative_net_return_bps: 40.2069
- average_net_return_bps: 3.0928
- win_rate_after_costs: 1.0000
- average_cost_bps: 3.6286
- average_confidence: 0.9246
- positive_after_costs: true

### 2023-09-27

- trades: 1
- cumulative_net_return_bps: 7.6420
- average_net_return_bps: 7.6420
- win_rate_after_costs: 1.0000
- average_cost_bps: 4.8780
- average_confidence: 0.9200
- positive_after_costs: true

### by_window_index

### 263

- trades: 1
- cumulative_net_return_bps: -2.6839
- average_net_return_bps: -2.6839
- win_rate_after_costs: 0.0000
- average_cost_bps: 4.2645
- average_confidence: 0.9000
- positive_after_costs: false

### 354

- trades: 1
- cumulative_net_return_bps: 1.0851
- average_net_return_bps: 1.0851
- win_rate_after_costs: 1.0000
- average_cost_bps: 3.6285
- average_confidence: 0.8600
- positive_after_costs: true

### 355

- trades: 12
- cumulative_net_return_bps: 39.1218
- average_net_return_bps: 3.2601
- win_rate_after_costs: 1.0000
- average_cost_bps: 3.6286
- average_confidence: 0.9300
- positive_after_costs: true

### 590

- trades: 1
- cumulative_net_return_bps: 7.6420
- average_net_return_bps: 7.6420
- win_rate_after_costs: 1.0000
- average_cost_bps: 4.8780
- average_confidence: 0.9200
- positive_after_costs: true


## Warnings

- Walk-forward evaluation is research-only and paper-only. Simulated returns do not prove profitability. No real orders are placed.
- one_class_training_windows: 10 aggregate walk-forward window(s) skipped because training labels contained only one class
- one_class_training_windows: 10 walk-forward window(s) skipped because training labels contained only one class for ETH
- model_not_better_than_baseline
- binary_tradeable_positive_class_sparse: test class-1 is 5.0% (under 10%)
- binary_tradeable_no_positive_class: test has zero class-1 labels
- binary_tradeable_positive_class_sparse: test class-1 is 3.0% (under 10%)
- binary_tradeable_positive_class_sparse: test class-1 is 2.0% (under 10%)
- binary_tradeable_positive_class_sparse: test class-1 is 1.0% (under 10%)
- binary_tradeable_positive_class_sparse: test class-1 is 9.0% (under 10%)
- binary_tradeable_positive_class_sparse: test class-1 is 7.0% (under 10%)
- binary_tradeable_positive_class_sparse: test class-1 is 8.0% (under 10%)
- binary_tradeable_positive_class_sparse: test class-1 is 6.0% (under 10%)
- binary_tradeable_positive_class_sparse: test class-1 is 4.0% (under 10%)
- stability_concentration: one time/window bucket contributes more than 70% of total positive net bps
- stability_sparse_buckets: 7 bucket(s) have fewer than 3 trades

## Research-only disclaimer

Walk-forward evaluation is research-only and paper-only. Simulated returns do not prove profitability. No real orders are placed.

Bid/ask schema helpers are read-only, paper-trading only, and research-only. No wallets, private keys, signing, leverage, margin, exchange actions, real orders, deposits, withdrawals, or execution endpoints are used.
