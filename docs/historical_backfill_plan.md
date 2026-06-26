# Hyperliquid Historical L2 Backfill Plan

Read-only research and importer skeleton for converting Hyperliquid historical L2
archive snapshots into bid/ask CSV rows. This phase does **not** download archive data
by default.

## Safety

- Read-only, research-only, paper-only
- No wallets, keys, signing, orders, or execution endpoints
- No live trading, private APIs, or user account data
- No automatic AWS/S3 downloads in this phase
- Do not start `main.py` or `websocket_recorder.py` for backfill work

## Historical Facts

### Live data

Hyperliquid live WebSocket **BBO** provides best bid/ask updates. The project's
existing recorder and `bid_ask_extractor` pipeline can turn locally captured BBO or
L2 messages into 1-second bid/ask CSV rows.

### Historical archive

Hyperliquid publishes historical **L2 book snapshots** in the public requester-pays
archive bucket:

```text
s3://hyperliquid-archive/market_data/[YYYYMMDD]/[hour]/l2Book/[COIN].lz4
```

Example:

```text
s3://hyperliquid-archive/market_data/20230916/9/l2Book/SOL.lz4
```

Each `.lz4` file holds JSON lines for one coin and one hour. After decompression,
each line is an L2 snapshot with `levels` (bids and asks). Snapshots are emitted
about **twice per second**, not necessarily every second.

### Archive cadence and reliability

- Historical archive is uploaded **about once a month**
- Data **may be missing** and is **not guaranteed timely**
- There is no SLA for complete hourly coverage across all coins

### API limitations

Hyperliquid `candleSnapshot` only supports **1m+ intervals** and the **most recent
5000 candles**. It cannot reconstruct sub-minute bid/ask history for all time.

### What we do not claim

**All-time 1-second bid/ask is not directly available** from a single public source.
Any 1-second series derived from archive L2 must be validated for gaps, duplicates,
and missing hours before use in research or training.

## Best Path Forward

1. **Inspect paths** — print expected S3 locations without downloading
2. **Download manually** (future phase) — requester-pays S3 copy + local `unlz4`
3. **Import locally** — parse decompressed files with `historical_l2_importer.py`
4. **Extract top of book** — best bid/ask and sizes from L2 `levels`
5. **Validate coverage** — row counts, missing seconds, duplicates, invalid quotes
6. **Resample** — use existing `bid_ask_resampler` for 15s/1m intervals

```mermaid
flowchart LR
  A[Archive S3 .lz4] --> B[Manual download + decompress]
  B --> C[historical_l2_importer]
  C --> D[data/bid_ask_csv/historical_1s]
  D --> E[coverage report JSON]
  D --> F[bid_ask_resampler 15s/1m]
```

## Required Local Tools

Manual historical backfill (Phase 2 download/decompress) depends on external CLI
tools installed on your machine. The aws CLI is needed for manual S3 copy; lz4 is
needed for decompression. The repo does **not** install or download anything
automatically — no automatic installs and no automatic downloads.

| Tool | Required? | Purpose |
| --- | --- | --- |
| `aws` CLI | **Yes** (manual download) | AWS CLI for requester-pays S3 copy from `hyperliquid-archive` |
| `lz4` | **Yes** (manual download) | Decompress `.lz4` archive files to JSONL |
| `brew` | Optional (macOS) | One possible install method for `aws` and `lz4`; not required |

Check what is available locally:

```bash
python3 historical_backfill_tool_check.py
```

The diagnostic uses `shutil.which` only — no subprocess execution, no network calls,
and no automatic installs. If `aws` or `lz4` is missing, `ready_for_manual_download`
is false; you can still capture bid/ask CSV rows via the local BBO recorder workflow.

## Manual Phase 2 Workflow

Phase 2 adds a manual-first guide (`historical_backfill_manual_guide.py`) that prints
commands for the user to run locally. **Nothing is downloaded, decompressed, or
executed automatically.**

### Why manual first

- Archive files are large and requester-pays; bulk downloads are easy to misconfigure
- One coin + one hour is enough to validate parsing, schema compatibility, and coverage
- Keeps the workflow read-only and safe until coverage metrics look acceptable
- Avoids accidental network/subprocess automation in research tooling

### Inspect path

Confirm the expected object before downloading:

```bash
python3 historical_l2_importer.py inspect-path --date 20230916 --hour 9 --coin SOL
```

Or print the full manual plan (commands only, no execution):

```bash
python3 historical_backfill_manual_guide.py --date 20230916 --hour 9 --coin SOL
```

### Download one hour manually

Run the printed `aws s3 cp --no-sign-request ...` command in your shell. Example:

```bash
aws s3 cp --no-sign-request \
  s3://hyperliquid-archive/market_data/20230916/9/l2Book/SOL.lz4 \
  ./downloads/hyperliquid/20230916/9/SOL.lz4
```

Download **one hour for one coin** first. Do not bulk-fetch months of data until a
sample import validates cleanly.

### Decompress one hour manually

```bash
lz4 -d ./downloads/hyperliquid/20230916/9/SOL.lz4 \
  ./downloads/hyperliquid/20230916/9/SOL.jsonl
```

Use local `lz4` or `unlz4`. The importer reads the decompressed JSONL file only.

### Import local decompressed file

```bash
python3 historical_l2_importer.py import-local \
  --input ./downloads/hyperliquid/20230916/9/SOL.jsonl \
  --coin SOL
```

Output CSV lands under `data/bid_ask_csv/historical_1s/` with:

- `source_session_id = historical_archive`
- `source_channel = historical_l2Book`

### Validate

Historical L2-derived bid/ask is stored separately from live WebSocket BBO. Use the
historical summarize CLI (not `bid_ask_data_pipeline.py validate`, which targets live
recorder output):

```bash
python3 historical_l2_importer.py summarize \
  --csv data/bid_ask_csv/historical_1s/SOL_1s.csv
```

Review the JSON report under `data/bid_ask_csv/reports/` for row counts, missing
seconds, duplicates, and invalid quotes.

### Avoid huge downloads

- Start with a single `(date, hour, coin)` tuple
- Check archive availability with `inspect-path` before paying for S3 transfer
- Expand hour-by-hour only after coverage reports look acceptable
- Do not claim all-time 1-second BBO without validating each imported segment

### Label historical vs live BBO

| Source | CSV location | Session / channel labels |
| --- | --- | --- |
| Live WebSocket BBO | `data/bid_ask_csv/1s/` (via recorder) | recorder session IDs, live channels |
| Historical archive L2 | `data/bid_ask_csv/historical_1s/` | `historical_archive`, `historical_l2Book` |

Keep historical and live CSVs in separate directories. Do not merge them without an
explicit deduplication and provenance policy.

## Module Overview

| File | Purpose |
| --- | --- |
| `historical_backfill_tool_check.py` | PATH diagnostic for `brew`/`aws`/`lz4` (shutil.which only) |
| `historical_backfill_manual_guide.py` | Manual download/decompress/import command strings and step plan |
| `historical_backfill_research.py` | Research constants, facts, and summary helpers |
| `historical_l2_importer.py` | Path builder, parser, local importer, coverage summary, CLI |
| `tests/test_historical_backfill_manual_guide.py` | Offline tests for manual guide commands and safety |
| `tests/test_historical_l2_importer.py` | Offline tests with fake L2 lines |
| `docs/historical_backfill_plan.md` | This document |

## Importer Functions

### `build_archive_s3_path(date, hour, coin, datatype="l2Book")`

Returns the expected archive path, e.g.:

```text
s3://hyperliquid-archive/market_data/20230916/9/l2Book/SOL.lz4
```

### `parse_l2_snapshot_line(line)`

Parses one decompressed JSON line. Extracts:

- `timestamp_ms`, `coin`, `best_bid`, `best_ask`, `bid_size`, `ask_size`
- `source_channel = historical_l2Book`

Malformed rows return `None` and are skipped safely.

### `l2_snapshot_to_bid_ask_row(snapshot)`

Converts parsed top-of-book into the standard bid/ask CSV schema used by
`bid_ask_schema.py`, with:

- `source_session_id = historical_archive`
- `source_channel = historical_l2Book`

### `import_decompressed_l2_file(input_path, coin, output_path)`

Reads a **local decompressed file only**. Writes CSV under:

```text
data/bid_ask_csv/historical_1s/
```

### `summarize_historical_coverage(csv_path)`

Returns coverage metrics:

- row count, start/end timestamps
- missing seconds, duplicate timestamps, invalid bid/ask rows
- `source = historical_archive_l2Book`

### `write_historical_import_report(...)`

Saves JSON under:

```text
data/bid_ask_csv/reports/
```

## CLI Commands

Print expected S3 path (no download):

```bash
python3 historical_l2_importer.py inspect-path --date 20230916 --hour 9 --coin SOL
```

Import a local decompressed file:

```bash
python3 historical_l2_importer.py import-local --input /path/to/decompressed/SOL --coin SOL
```

Summarize a local historical CSV (coverage only, no download):

```bash
python3 historical_l2_importer.py summarize --csv data/bid_ask_csv/historical_1s/SOL_1s.csv
```

Or resolve the default historical CSV path by coin:

```bash
python3 historical_l2_importer.py summarize --coin SOL
```

Optional output path:

```bash
python3 historical_l2_importer.py import-local \
  --input /tmp/SOL \
  --coin SOL \
  --output data/bid_ask_csv/historical_1s/SOL_1s.csv
```

## Output Layout

```text
data/bid_ask_csv/
  historical_1s/
    SOL_1s.csv
  reports/
    historical_SOL_20230916T090000Z.json
```

Historical CSV columns match `BID_ASK_CSV_COLUMNS` in `bid_ask_schema.py`.

## Future Work (Not in This Phase)

- Requester-pays S3 downloader with explicit opt-in
- Hour-range batch import and deduplication across files
- Integration hook in `bid_ask_data_pipeline.py` for `--source historical`
- Gap-fill policy documentation (forward-fill vs leave missing)

## References

- [Hyperliquid historical data docs](https://hyperliquid.gitbook.io/hyperliquid-docs/historical-data)
- Archive format: `market_data/[date]/[hour]/l2Book/[coin].lz4`
- Decompression: `unlz4` or system `lz4` package
