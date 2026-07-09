# Candidate D Cost-Aware Labeling Plan

Last updated: 2026-07-09. Paper-only research. **No model training changes in this phase.**

---

## 1. Problem Statement

Direction labels are not enough for post-cost paper evaluation.

Candidate B (threshold/regime filters) and Candidate C (feature engineering) both failed to produce positive post-cost paper results after expanded historical data:

| Candidate | Lever | Outcome |
|-----------|-------|---------|
| B | Post-hoc confidence/edge/hour filters | Positive on 5 strict trades; all relaxed configs negative after backfill |
| C | `candidate_c_core` features + direction labels | BTC/ETH DOWN grids negative; ETH UP worse than DOWN; both negative after costs |

The shared failure mode: **correct direction is not the same as tradeable edge.** A model can predict UP or DOWN accurately while the realized move is smaller than fees + slippage + spread. Those trades look "right" on direction accuracy but lose net bps on paper.

Examples from recent grids:

- ETH DOWN H20 (Candidate C): 157 trades, ~−2.79 bps average net — directionally active but cost-negative.
- ETH UP H20 (Candidate C): 96 trades, ~−5.15 bps average net — worse than DOWN.
- BTC DOWN H20 (Candidate C): 103 trades, ~−6.99 bps average net.

Further threshold grid search on direction labels risks **overfitting** sparse regimes. Candidate D shifts the research question from "which direction?" to "is there enough edge after costs to trade at all?"

**Hypothesis:** Labels that encode cost-aware abstention (NO_TRADE / FLAT) will teach the evaluator and eventual model to skip low-edge bars, reducing trade count intelligently and improving average net bps versus direction-only baselines.

---

## 2. Proposed Label Modes

All modes use row-level cost estimation consistent with the existing walk-forward evaluator:

```
estimated_cost_bps = fee_bps + slippage_bps + spread_bps(row)
```

Defaults: `fee_bps = 2`, `slippage_bps = 1`, spread from bar `avg_spread_bps` (fallback to default spread).

`threshold_bps` is a configurable minimum edge above estimated cost (analogous to `label_min_edge_bps` today). Optional `label_cost_multiplier` may scale estimated cost before adding threshold (existing parameter).

### 2.1 `cost_aware_3class` (primary mode)

Multiclass labels aligned with abstention:

| Label | Condition |
|-------|-----------|
| `UP_EDGE` | `future_return_bps > estimated_cost_bps + threshold_bps` |
| `DOWN_EDGE` | `future_return_bps < -(estimated_cost_bps + threshold_bps)` |
| `NO_TRADE` | otherwise |

Maps conceptually to existing `cost_aware` UP/DOWN/FLAT but uses explicit EDGE/NO_TRADE naming for paper reporting and abstention metrics.

**Use when:** evaluating whether a multiclass model can learn when *not* to trade while preserving directional edge classes.

### 2.2 `binary_tradeable_down`

Binary label for DOWN-only research path (Candidate B/C heritage):

| Label | Condition |
|-------|-----------|
| `1` (tradeable) | `future_return_bps < -(estimated_cost_bps + threshold_bps)` |
| `0` (not tradeable) | otherwise |

**Use when:** isolating DOWN microstructure hypothesis without UP class noise. Paper accepts only predictions of class `1` mapped to DOWN_EDGE.

### 2.3 `binary_tradeable_up`

Binary label for UP-side experiments:

| Label | Condition |
|-------|-----------|
| `1` (tradeable) | `future_return_bps > estimated_cost_bps + threshold_bps` |
| `0` (not tradeable) | otherwise |

**Use when:** comparing UP vs DOWN tradeability under identical cost assumptions (ETH UP/DOWN grid follow-up).

### 2.4 Label mode summary

| `label_mode` | Classes | Abstention class | Primary research question |
|--------------|---------|-------------------|---------------------------|
| `direction` (baseline) | UP / DOWN / FLAT | FLAT | Which direction? (existing) |
| `cost_aware` (existing) | UP / DOWN / FLAT | FLAT | Same as 3-class, legacy names |
| `cost_aware_3class` | UP_EDGE / DOWN_EDGE / NO_TRADE | NO_TRADE | Direction + explicit abstention |
| `binary_tradeable_down` | 1 / 0 | 0 | Is DOWN move big enough to trade? |
| `binary_tradeable_up` | 1 / 0 | 0 | Is UP move big enough to trade? |

**Phase scope:** implement label creation and evaluator diagnostics only. Do **not** change `bid_ask_train.py` defaults or retrain models until label diagnostics pass acceptance gates below.

---

## 3. Required Evaluator Changes

Extend `bid_ask_walk_forward_evaluator.py` (and grid runner label_mode passthrough) — **labeling and reporting only**:

### 3.1 Label creation

1. Add `cost_aware_3class`, `binary_tradeable_down`, `binary_tradeable_up` to `SUPPORTED_LABEL_MODES`.
2. Implement `create_cost_aware_3class_labels()` and binary variants sharing one cost-edge helper.
3. Preserve walk-forward boundary: labels computed per fold from training slice statistics only; no future rows in feature matrix.
4. Emit `label_required_edge_bps` per row for diagnostics (already exists for `cost_aware`).

### 3.2 Reporting — class balance

For each walk-forward fold and aggregate report:

- Count and fraction per label class.
- UP_EDGE / DOWN_EDGE / NO_TRADE (or 1/0) percentages.
- Compare to `direction` mode class balance on same fold windows.

### 3.3 Reporting — abstention rate

- **`no_trade_abstention_rate`:** fraction of bars labeled NO_TRADE (or class 0 in binary modes).
- **`predicted_abstention_rate`:** fraction of paper decisions where model predicts NO_TRADE / 0 (when inference path exists).
- **`accepted_edge_rate`:** fraction of bars where paper trade is accepted (non-abstaining prediction passing existing gates).

Distinguish **label abstention** (ground truth) from **model abstention** (prediction) in markdown/JSON output.

### 3.4 Reporting — post-cost paper return (EDGE only)

- Compute cumulative and average net bps **only for accepted EDGE predictions** (UP_EDGE, DOWN_EDGE, or binary class 1 mapped to a direction).
- Do not include NO_TRADE / class 0 bars in trade PnL numerators.
- Side-by-side table: direction-mode accepted trades vs cost-aware EDGE accepted trades on identical folds.

### 3.5 CLI / grid runner

- `--label-mode` accepts new modes.
- Grid runner `label_modes` tuple includes new modes for diagnostic sweeps (threshold_bps grid only — no confidence/ hour filter expansion in first Candidate D pass).

---

## 4. Required Diagnostics

Run read-only label diagnostics before any training integration.

### 4.1 Label class balance by coin / horizon

For BTC and ETH, horizons H10 and H20 (primary), optionally H1/H5:

| Slice | Metrics |
|-------|---------|
| Per coin | UP_EDGE %, DOWN_EDGE %, NO_TRADE % |
| Per horizon | Class counts, min class fraction |
| Per fold | Fold-level balance (detect collapse to all NO_TRADE) |

Warning thresholds (reuse existing patterns):

- All NO_TRADE → `cost_aware_labels_all_flat`
- UP_EDGE + DOWN_EDGE < 5% of bars → `cost_aware_labels_sparse`

### 4.2 Future edge distribution versus estimated cost

Per coin/horizon:

- Histogram / quantiles of `future_return_bps`
- Overlay `estimated_cost_bps + threshold_bps` threshold line
- Report fraction of bars where `|future_return_bps|` exceeds cost hurdle (raw tradeable universe)
- Compare p50/p90/p99 of edge conditional on UP_EDGE vs DOWN_EDGE labels

### 4.3 Accepted trades by predicted class

When model inference is available (future phase):

- Count paper trades by predicted class (UP_EDGE, DOWN_EDGE)
- Mean net bps per predicted class
- Confusion matrix: predicted class vs realized label

**This phase (labels only):** simulate "oracle" acceptance — accept all bars labeled UP_EDGE or DOWN_EDGE — to establish upper bound on tradeable universe size.

### 4.4 False edge rate

Define **false edge** for a labeled UP_EDGE bar: `future_return_bps <= 0` (move did not go up) or net bps ≤ 0 after costs.

Define **false edge** for DOWN_EDGE symmetrically.

Report:

- `false_edge_rate_up`: P(net ≤ 0 | label = UP_EDGE)
- `false_edge_rate_down`: P(net ≤ 0 | label = DOWN_EDGE)
- Compare false edge rate under `direction` labels vs cost-aware EDGE labels at same horizon

Goal: confirm cost-aware labels reduce false edges even if total EDGE count drops.

### 4.5 Diagnostic outputs

```
reports/label_diagnostics/label_diag_{COIN}_{horizon}_{label_mode}_{stamp}.json
reports/label_diagnostics/label_diag_{COIN}_{horizon}_{label_mode}_{stamp}.md
```

---

## 5. Validation Rules

Same rigor as Candidates B and C. No shortcuts.

| Rule | Requirement |
|------|-------------|
| Walk-forward only | Expanding or rolling folds; no random train/test split |
| No future leakage | Labels at time t use mid at t and mid at t+H only; features ≤ t |
| Holdout integrity | Final chronological holdout (most recent 20% or last N days) **never** used for threshold_bps tuning |
| Threshold tuning | `threshold_bps` grid searched on inner walk-forward folds only |
| Minimum trades per fold | ≥ 5 accepted EDGE paper trades per fold for fold-level metrics; flag `insufficient_trades` otherwise |
| Baseline comparison | Every Candidate D run reports alongside Candidate B (direction + filters) and Candidate C (direction + `candidate_c_core`) on identical date ranges |
| Expanded data | Primary evaluation on full backfill through 2023-09-19T20:59:59Z and growing; no claims from pre-backfill snippets |
| No filter stacking | First Candidate D experiments use cost-aware labels only — no new hour/spread/regime post-hoc filters |

### 5.1 Suggested threshold grid (labels only)

Initial diagnostic sweep:

```
threshold_bps ∈ {0, 1, 2, 5}
label_cost_multiplier ∈ {1.0, 1.25}  # optional, match existing default
horizons: H10, H20
coins: BTC, ETH
label_modes: cost_aware_3class, binary_tradeable_down
```

Do **not** expand confidence or exclude_hours grids until label diagnostics show EDGE classes are non-sparse and false edge rate improves vs direction baseline.

---

## 6. Safety

Non-negotiable for all Candidate D work:

| Flag | Value |
|------|-------|
| `paper_only` | `true` |
| `execution_enabled` | `false` |
| `real_execution_ready` | `false` |

- No wallets, private keys, signing, deposits, withdrawals, leverage, margin, or order endpoints.
- No network calls for execution.
- Label diagnostics and walk-forward runs are offline research on local CSVs.
- Simulated oracle-acceptance PnL does **not** prove profitability.

---

## 7. Acceptance Criteria

Candidate D label phase succeeds when **all** of the following hold:

1. **Intelligent trade reduction:** Cost-aware EDGE labels yield fewer oracle-acceptance trades than direction-mode accepted trades at the same horizon, with documented abstention rate (target: ≥ 30% NO_TRADE on H20 unless spread regime explains otherwise).

2. **Average net bps improvement vs Candidate C:** On identical BTC/ETH windows, oracle-acceptance average net bps per EDGE trade ≥ Candidate C best direction config average net bps (even if both remain negative — improvement is relative until holdout passes).

3. **False edge rate reduction:** `false_edge_rate` for EDGE labels < direction-label false edge rate at same horizon (cost-aware labeling filters moves too small to cover friction).

4. **Non-degenerate classes:** No config where UP_EDGE + DOWN_EDGE < 2% of bars unless explicitly documented as spread-dominated regime.

5. **No positive claim without holdout:** Any statement of "positive after costs" requires:
   - Survival on expanded BTC **and** ETH data
   - Positive average net bps on chronological out-of-sample holdout
   - ≥ 30 accepted EDGE trades on holdout
   - Stability warnings ≤ 1

Until then, Candidate D remains **diagnostic research**, not a paper candidate promotion.

---

## 8. Relationship to Prior Candidates

| Candidate | Status | Role in Candidate D comparison |
|-----------|--------|----------------------------------|
| B | Rejected for scaling; useful baseline | Direction labels + strict/relaxed filters; threshold tuning stopped |
| C | Implemented and tested; negative post-cost | `candidate_c_core` features + direction labels; primary baseline for net bps |
| D | This plan | Cost-aware labels + abstention; no new features, no new post-hoc filters in phase 1 |

Candidate D does not discard Candidate C features. Phase 2 (future) may combine `candidate_c_core` features with `cost_aware_3class` labels **only after** label-only diagnostics pass.

---

## 9. Implementation Sequence

1. **Label functions + unit tests** — new modes, parity with existing `cost_aware` for equivalent thresholds.
2. **Evaluator reporting** — class balance, abstention rate, EDGE-only PnL sections in walk-forward markdown/JSON.
3. **Label diagnostic script** — batch run over BTC/ETH × H10/H20 × threshold grid.
4. **Baseline comparison report** — Candidate D diagnostics vs B/C ledger entries on same date range.
5. **Research note update** — document findings; decide whether to proceed to training integration.

**Explicitly deferred:** `bid_ask_train.py` changes, hyperparameter tuning, new feature families, regime/hour filters.

---

## 10. Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| All NO_TRADE — nothing to learn | Threshold grid includes 0 bps; report cost hurdle distribution |
| EDGE classes too sparse for ML | Binary modes; per-coin diagnostics; do not train until ≥ 5% combined EDGE |
| Oracle acceptance overstates model PnL | Label phase uses oracle for universe sizing only; model metrics in phase 2 |
| threshold_bps overfit on inner folds | Fixed holdout; limit grid to 4 values; compare B/C baselines on holdout untouched |
| DOWN-only bias persists | Run binary_tradeable_up in parallel; report asymmetry explicitly |

---

*Reminder: Candidate D is a research plan. Simulated label diagnostics and paper returns do not prove profitability. Real execution remains disabled.*
