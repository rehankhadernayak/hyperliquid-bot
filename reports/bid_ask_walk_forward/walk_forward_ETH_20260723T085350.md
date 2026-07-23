# Walk-Forward Paper Evaluation

Generated at: 2026-07-23T08:53:50.309572+00:00
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
- average_label_required_edge_bps: 7.3200
- min_label_required_edge_bps: 4.6037
- max_label_required_edge_bps: 129.8152
- label 0: count=56264 fraction=0.8513
- label 1: count=9829 fraction=0.1487

## Candidate D Cost-Aware Label Diagnostics

- label_mode: binary_tradeable_up
- horizon_bars: 20
- threshold_bps: 1.0
- estimated_cost_formula: fee_bps + slippage_bps + spread_bps
- no_trade_fraction: 0.8512
- edge_fraction: 0.1488
- accepted_edge_predictions: 17
- false_edge_rate: n/a (no predictions)
- walk_forward_training_supported: true

### Class counts

- 0: count=56242 fraction=0.8512
- 1: count=9829 fraction=0.1488

### False edge rate

- false_edge_rate_if_predictions_available: n/a (no predictions)


## Features

- feature_set: candidate_f_regime_depth
- feature_count: 34
- feature_columns: bbo_imbalance, imbalance_mean_20, imbalance_std_20, imbalance_zscore_20, imbalance_change_1, spread_bps, rolling_spread_std_20, spread_zscore_20, return_15s_bps, return_30s_bps, return_60s_bps, short_term_drawdown_bps, short_term_upmove_bps, utc_hour, hour_sin, hour_cos, is_weekend, bid_size, ask_size, total_top_size, size_imbalance, microprice, microprice_mid_deviation_bps, rolling_pressure_change, realized_volatility_5m_bps, realized_volatility_15m_bps, volatility_ratio_5m_15m, spread_volatility_interaction, microprice_volatility_interaction, imbalance_volatility_interaction, low_volatility_regime, high_volatility_regime, tight_spread_regime, wide_spread_regime

## Aggregate metrics

- total_windows: 655
- completed_windows: 646
- skipped_windows: 9
- skipped_windows_count: 9
- one_class_training_windows_count: 9
- skipped_window_reasons: {'one_class_training_window': 9}
- status: completed
- rejection_reason: n/a
- model_accuracy_mean: 0.8392
- baseline_accuracy_mean: 0.8496
- accuracy_edge_vs_baseline: -0.0105
- total_paper_signals: 2102
- total_paper_trades: 17
- raw_predictions: 2102
- filtered_paper_trades: 17
- blocked_by_confidence: 1464
- blocked_by_cost: 614
- blocked_by_spread: 7
- blocked_by_volatility: 0
- blocked_by_trade_class: 0
- average_cost_bps: 6.2463
- average_required_edge_bps: 9.8078
- average_gross_return_bps: 0.0014
- filtered_paper_trades (accepted): 17
- cumulative_net_return_bps: 28.7497
- average_net_return_bps: 1.6912
- expected_average_from_cumulative: 1.6912
- average_net_return_difference_abs: 0.000000
- net_return_consistency_check: true
- max_drawdown_bps: 8.0287
- win_rate_after_costs: 0.7647058823529411
- longest_losing_streak: 3

## Regime Filters

- regime_filters_enabled: false
- exploratory_overfit_warning: false
- exclude_hours: none
- include_hours: none
- max_cost_bps: none
- min_cost_bps: none
- trades_removed_by_regime_filters: 0

### Before regime filters

- trades_before_regime_filters: 17
- cumulative_net_return_before_regime_filters: 28.7497
- average_net_return_before_regime_filters: 1.6912
- win_rate_before_regime_filters: 0.7647

### After regime filters

- trades_after_regime_filters: 17
- cumulative_net_return_after_regime_filters: 28.7497
- average_net_return_after_regime_filters: 1.6912
- win_rate_after_regime_filters: 0.7647

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
| ETH | 646 | 0.8392 | 0.8496 | -0.0105 | 17 | 28.7497 |
## Trade Diagnostics

### UP

- trades: 17
- cumulative_net_return_bps: 28.7497
- average_net_return_bps: 1.6912
- win_rate_after_costs: 0.7647
- average_cost_bps: 3.7265
- average_confidence: 0.8835

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

- accepted UP: 17
- accepted DOWN: 0
- accepted FLAT: 0
- blocked UP: 2085
- blocked DOWN: 0
- blocked FLAT: 62498

### Coin-class trade diagnostics

#### ETH

### UP

- trades: 17
- cumulative_net_return_bps: 28.7497
- average_net_return_bps: 1.6912
- win_rate_after_costs: 0.7647
- average_cost_bps: 3.7265
- average_confidence: 0.8835

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
- profitable_bucket_fraction: 0.6000
- unprofitable_bucket_fraction: 0.4000
- buckets_with_too_few_trades: 3
- largest_positive_bucket_net_bps: 36.7784
- largest_negative_bucket_net_bps: -5.3448
- concentration_warning: true
- sparse_stability_warning: true

### Stability warnings

- stability_concentration: one time/window bucket contributes more than 70% of total positive net bps
- stability_sparse_buckets: 3 bucket(s) have fewer than 3 trades
### by_coin

### ETH

- trades: 17
- cumulative_net_return_bps: 28.7497
- average_net_return_bps: 1.6912
- win_rate_after_costs: 0.7647
- average_cost_bps: 3.7265
- average_confidence: 0.8835
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

- trades: 17
- cumulative_net_return_bps: 28.7497
- average_net_return_bps: 1.6912
- win_rate_after_costs: 0.7647
- average_cost_bps: 3.7265
- average_confidence: 0.8835
- positive_after_costs: true

### by_coin_predicted_class

### ETH:UP

- trades: 17
- cumulative_net_return_bps: 28.7497
- average_net_return_bps: 1.6912
- win_rate_after_costs: 0.7647
- average_cost_bps: 3.7265
- average_confidence: 0.8835
- positive_after_costs: true

### by_hour

### 14

- trades: 16
- cumulative_net_return_bps: 31.4336
- average_net_return_bps: 1.9646
- win_rate_after_costs: 0.8125
- average_cost_bps: 3.6929
- average_confidence: 0.8812
- positive_after_costs: true

### 15

- trades: 1
- cumulative_net_return_bps: -2.6839
- average_net_return_bps: -2.6839
- win_rate_after_costs: 0.0000
- average_cost_bps: 4.2645
- average_confidence: 0.9200
- positive_after_costs: false

### by_date

### 2023-09-21

- trades: 1
- cumulative_net_return_bps: -2.6839
- average_net_return_bps: -2.6839
- win_rate_after_costs: 0.0000
- average_cost_bps: 4.2645
- average_confidence: 0.9200
- positive_after_costs: false

### 2023-09-23

- trades: 16
- cumulative_net_return_bps: 31.4336
- average_net_return_bps: 1.9646
- win_rate_after_costs: 0.8125
- average_cost_bps: 3.6929
- average_confidence: 0.8812
- positive_after_costs: true

### by_window_index

### 263

- trades: 1
- cumulative_net_return_bps: -2.6839
- average_net_return_bps: -2.6839
- win_rate_after_costs: 0.0000
- average_cost_bps: 4.2645
- average_confidence: 0.9200
- positive_after_costs: false

### 354

- trades: 5
- cumulative_net_return_bps: -5.3448
- average_net_return_bps: -1.0690
- win_rate_after_costs: 0.4000
- average_cost_bps: 3.8342
- average_confidence: 0.8840
- positive_after_costs: false

### 355

- trades: 11
- cumulative_net_return_bps: 36.7784
- average_net_return_bps: 3.3435
- win_rate_after_costs: 1.0000
- average_cost_bps: 3.6286
- average_confidence: 0.8800
- positive_after_costs: true


## Warnings

- Walk-forward evaluation is research-only and paper-only. Simulated returns do not prove profitability. No real orders are placed.
- one_class_training_windows: 9 aggregate walk-forward window(s) skipped because training labels contained only one class
- one_class_training_windows: 9 walk-forward window(s) skipped because training labels contained only one class for ETH
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
- stability_sparse_buckets: 3 bucket(s) have fewer than 3 trades

## Research-only disclaimer

Walk-forward evaluation is research-only and paper-only. Simulated returns do not prove profitability. No real orders are placed.

Bid/ask schema helpers are read-only, paper-trading only, and research-only. No wallets, private keys, signing, leverage, margin, exchange actions, real orders, deposits, withdrawals, or execution endpoints are used.
