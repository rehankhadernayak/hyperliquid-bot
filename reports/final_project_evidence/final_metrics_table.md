# Final Metrics Table — Candidate F (Frozen)

Source: `reports/final_project_evidence/candidate_f_final_summary.json`, `reports/bid_ask_walk_forward/walk_forward_ETH_20260729T101645.json`, `reports/candidate_f_milestone_comparison/candidate_f_milestone_comparison_latest.json`.

Paper-only / execution disabled across all cited runs.

| Field | Value |
|-------|-------|
| **config_id** | `ETH_UP_H20_C0.65_E2_M1.25_S8` |
| **feature_set** | `candidate_f_regime_depth` |
| **total_windows** | 705 |
| **completed_windows** | 695 |
| **skipped_windows** | 10 |
| **accepted_trades** | 15 |
| **cumulative_net_bps** | 45.164969776189494 |
| **average_net_bps** | 3.0109979850792996 |
| **robustness_verdict** | `diagnostic_positive_sparse` |
| **robust_candidate** | `false` |
| **milestone_count** | 1 |
| **baseline_improvement_conclusion** | No demonstrated improvement over baseline (`beats_baseline_accuracy: false`; `accuracy_edge_vs_baseline`: −0.010618705035971177) |
| **research_conclusion** | Candidate F was positive after estimated costs, but trade evidence was sparse, non-robust, and did not demonstrate improvement over the baseline. |
| **paper_only** | `true` |
| **execution_enabled** | `false` |
| **real_execution_ready** | `false` |

## Supplementary metrics (walk-forward report)

| Field | Value |
|-------|-------|
| win_rate | 0.9333333333333333 |
| baseline_accuracy_mean | 0.8482014388489209 |
| average_cost_bps | 6.42368215429615 |
| label_mode | `binary_tradeable_up` |
| coin / interval / horizon | ETH / 15s / H20 |
| dataset coverage | 2023-09-16T00:00:00Z → 2023-09-30T14:59:45Z (351 h) |
| walk-forward report | `reports/bid_ask_walk_forward/walk_forward_ETH_20260729T101645.json` |
| accepted trades CSV | `reports/bid_ask_walk_forward/accepted_trades_ETH_20260729T101645.csv` |

## CSV export

Machine-readable copy: `reports/final_project_evidence/final_metrics_table.csv`

---

*Generated 2026-08-05. Values not invented; see source JSON for full precision.*
