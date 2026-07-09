# Top-of-Book Size Recovery Audit

Generated: 2026-07-09

## Executive Summary

**Root cause:** `bid_size` and `ask_size` were present in historical 1s CSVs but **dropped during resampling** into 15s/1m bars. The feature audit correctly skipped size/microprice families because resampled columns lacked these fields.

**Resolution:** Updated `bid_ask_resampler.py` to carry top-of-book sizes into resampled bars using **last-valid 1s tick within each bar** (close-of-bar snapshot). Feature audit microprice/liquidity features activate automatically once resampled CSVs are regenerated.

**Importer status:** No importer changes required. L2 sizes are **not** lost during historical import.

---

## Task 1: Historical 1s CSV Inspection (BTC, ETH)

### Columns Present

Both `data/bid_ask_csv/historical_1s/BTC_1s.csv` and `ETH_1s.csv` contain:

```
timestamp_ms, timestamp_iso, coin, best_bid, best_ask, mid_price, spread, spread_bps,
bid_size, ask_size, bbo_imbalance, source_session_id, source_channel
```

This matches `BID_ASK_CSV_COLUMNS` in `bid_ask_schema.py`.

### Missingness (Full Files)

| Coin | Rows   | bid_size null | ask_size null | bbo_imbalance null |
|------|--------|---------------|---------------|--------------------|
| BTC  | 467,288 | 0.0000%      | 0.0000%       | 0.0011%            |
| ETH  | 467,154 | 0.0000%      | 0.0000%       | 0.0011%            |

### Sample Values (First Rows)

**BTC:**
- bid_size: 0.56287, 0.59928, 0.58195
- ask_size: 0.5774, 0.55713, 0.56967
- bbo_imbalance: -0.0127, 0.0364, 0.0107

**ETH:**
- bid_size: 10.258, 10.258, 9.7265
- ask_size: 9.602, 9.602, 8.7337
- bbo_imbalance: 0.0330, 0.0330, 0.0538

### Distribution Summary

| Coin | bid_size median | ask_size median | bbo_imbalance median |
|------|-----------------|-----------------|----------------------|
| BTC  | 0.596           | 0.535           | 0.047                |
| ETH  | 9.864           | 9.071           | 0.042                |

---

## Task 2: Resampler Fix

### Where Sizes Were Lost

`bid_ask_resampler.py` `RESAMPLED_COLUMNS` previously included only OHLC mid, averaged bid/ask/spread, and `avg_bbo_imbalance`. The `resample_bid_ask()` loop never read `BID_SIZE` or `ASK_SIZE` from the 1s input despite schema requiring them on 1s rows.

### Aggregation Method (Documented)

| Field | Method |
|-------|--------|
| `bid_size` | Last non-null 1s `bid_size` within the bar interval |
| `ask_size` | Last non-null 1s `ask_size` within the bar interval |
| `avg_bbo_imbalance` | **Unchanged** — mean of 1s `bbo_imbalance` values in the bucket |

Last-valid sizing is causal: each bar uses only ticks whose `timestamp_ms` falls inside `[start_timestamp_ms, end_timestamp_ms]`. No future leakage.

### Schema Split

- `REQUIRED_RESAMPLED_COLUMNS` — original 17 columns (backward compatible load/validate)
- `OPTIONAL_RESAMPLED_COLUMNS` — `bid_size`, `ask_size`
- `RESAMPLED_COLUMNS` — full output schema for newly resampled files

Legacy 15s/1m CSVs without size columns still load and validate. Feature audit skips size/microprice families until data is re-resampled.

---

## Task 3: Importer (Not Modified)

Historical 1s files include `source_channel=historical_l2Book`, confirming L2 book data was imported with sizes intact. **No importer rewrite needed.**

If sizes were missing at 1s level, the loss point would be the historical archive → 1s CSV writer. Inspection shows sizes are present at import time; loss occurred exclusively in the resampler.

---

## Task 4: Microprice Feature Support

Once resampled bars include `bid_size` and `ask_size`, `bid_ask_feature_audit.py` computes:

| Feature | Formula |
|---------|---------|
| `microprice` | `(best_ask × bid_size + best_bid × ask_size) / (bid_size + ask_size)` |
| `microprice_mid_deviation_bps` | `10000 × (microprice − mid_price) / mid_price` |
| `total_top_size` | `bid_size + ask_size` |
| `size_imbalance` | `(bid_size − ask_size) / (bid_size + ask_size)` |

Uses `avg_bid` / `avg_ask` from resampled bars when computing microprice. Logic was already implemented; it was blocked only by missing size columns.

---

## Task 5: Safety Confirmation

| Flag | Value |
|------|-------|
| paper_only | True |
| execution_enabled | False |
| real_execution_ready | False |
| read_only | True |
| no_network_calls | True |
| no_subprocess_calls | True |
| research_only | True |

Resampler does not modify 1s source files. Validator and feature audit remain read-only.

---

## Regeneration Commands

Re-resample historical 1s → 15s/1m with restored sizes:

```bash
python3 bid_ask_resampler.py --coins BTC ETH --from-interval 1s --to-interval all --source historical
```

Validate resampled output:

```bash
python3 bid_ask_validator.py --coins BTC ETH --interval 15s --source historical
python3 bid_ask_validator.py --coins BTC ETH --interval 1m --source historical
```

Run feature audits (BTC and ETH):

```bash
python3 -m bid_ask_feature_audit audit --coin BTC --source historical --interval 15s
python3 -m bid_ask_feature_audit audit --coin ETH --source historical --interval 15s
python3 -m bid_ask_feature_audit audit --coin BTC --source historical --interval 1m
python3 -m bid_ask_feature_audit audit --coin ETH --source historical --interval 1m
```

After re-resampling, feature audit reports should show **empty `skipped_features`** for microprice and liquidity families (assuming sizes present in all bars).

---

## Files Modified

- `bid_ask_resampler.py` — size carry-forward, schema split
- `bid_ask_validator.py` — import required columns from resampler; optional negative-size warnings
- `bid_ask_feature_audit.py` — load validates required columns only
- `tests/test_bid_ask_resampler.py` — size preservation, no-leakage tests
- `tests/test_bid_ask_validator.py` — legacy CSV compat, negative size warning
- `tests/test_bid_ask_feature_audit.py` — microprice formula, full family enablement

`bid_ask_schema.py` unchanged — `BID_SIZE` / `ASK_SIZE` constants already defined.
