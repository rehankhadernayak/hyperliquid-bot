"""Markdown document generators for the review package."""

from __future__ import annotations

import json
from typing import Any

from config import (
    EVIDENCE,
    MARKET,
    PAPER_TRADING,
    SIGNAL,
    TEST_MODE,
    VERSION,
)
from strategy_profiles import STRATEGY_PROFILES, list_profile_ids
from review_data import TestSummary


def generate_executive_summary(
    *,
    test_summary: TestSummary,
    experiments: list[dict[str, Any]],
    latest: dict[str, Any] | None,
    readiness_label: str,
    readiness_summary: str,
) -> str:
    profiles = ", ".join(list_profile_ids())
    coins = ", ".join(MARKET.watchlist)
    mode = "TEST_MODE" if TEST_MODE else "production"

    exp_section = _format_latest_experiment(latest)
    evidence_section = _format_evidence_labels(latest)

    return f"""# Executive Summary

## Project

| Field | Value |
|-------|-------|
| **Name** | Hyperliquid Paper Trading Bot |
| **Version** | `{VERSION}` |
| **Purpose** | Supervised technical and risk review of a read-only paper-trading system |

## What This System Does

- Polls the Hyperliquid **public Info API** for mid prices (BTC, ETH, SOL).
- Computes rolling percentage-change signals and contrarian trade logic.
- Applies shared risk rules before any simulated action.
- Produces read-only recommendations (`PAPER BUY`, `PAPER SELL`, `HOLD`, `BLOCKED`).
- Simulates fills against a fake USDC balance with fees and slippage.
- Logs prices, trades, equity, sessions, backtests, and offline experiments locally.
- Provides a Streamlit dashboard and this review package for adviser inspection.

## What This System Explicitly Cannot Do

- Place real trades or connect to exchange execution endpoints.
- Store or use wallets, private keys, seed phrases, or signing credentials.
- Accept deposits, withdrawals, or leverage.
- Autonomously deploy capital or approve live trading.
- Claim any strategy is proven or profitable.

**This release cannot place real trades and is intended only for supervised technical and risk review.**

## Development Stage

This is a **review-ready paper-trading release** (`9.0.0-review`). The system collects
observational and simulated evidence. No capital deployment decision is implied.

## Monitored Coins

{coins}

## Strategy Profiles Available

{profiles}

Live analyzer mode: **{mode}** (thresholds differ from experiment profiles).

## Paper Balance and Risk Assumptions

| Parameter | Value |
|-----------|-------|
| Starting balance | ${PAPER_TRADING.starting_balance_usdc:,.2f} USDC (simulated) |
| Trade size | ${PAPER_TRADING.trade_size_usdc:,.2f} |
| Max allocation per coin | ${PAPER_TRADING.max_allocation_per_coin_usdc:,.2f} |
| Max entries per coin | {PAPER_TRADING.max_entries_per_coin} |
| Buy cooldown | {PAPER_TRADING.min_seconds_between_buys} seconds |
| Max total open exposure | ${PAPER_TRADING.max_total_open_exposure_usdc:,.2f} |
| Session trade limit (live) | {PAPER_TRADING.max_successful_trades_per_session} |
| Equity pause threshold | ${PAPER_TRADING.min_equity_before_pause_usdc:,.2f} |
| Trading fee | {PAPER_TRADING.trading_fee_pct}% per fill |
| Slippage | {PAPER_TRADING.slippage_pct}% per fill |

## Test Status

| Metric | Value |
|--------|-------|
| Tests passing | {"Yes" if test_summary.passed else "No"} |
| Total tests | {test_summary.total} |
| Failed | {test_summary.failed} |

## Latest Experiment Results

{exp_section}

## Current Evidence Labels

{evidence_section}

## Capital Readiness (Descriptive Only)

**Assessment:** `{readiness_label}`

{readiness_summary}

This assessment is descriptive only. It does **not** authorize real trading.

## Major Warnings

- Historical and paper results **do not guarantee future performance**.
- Live analyzer uses `signal_from_1m_change()` directly; strategy registry is for offline experiments.
- Short data history may produce misleading evidence labels.
- TEST_MODE thresholds generate more frequent paper trades than production thresholds.
- One week of data alone is insufficient for capital deployment decisions.
- Positive return, high win rate, or low drawdown alone do not qualify a strategy.

## Disclaimer

All results shown in this package are simulated or observational. Past paper-trading
performance does not predict future results. Any decision to use real capital requires
independent review by a qualified adult adviser and appropriate legal supervision.
"""


def _format_latest_experiment(latest: dict[str, Any] | None) -> str:
    if latest is None:
        return "**No experiment results available.** Run `python3 experiment_runner.py` first."

    lines = [
        f"- Experiment ID: `{latest.get('experiment_id', 'unknown')}`",
        f"- Created: {latest.get('created_at', 'unknown')}",
        f"- Date range: {latest.get('date_range', {}).get('start', '?')} → "
        f"{latest.get('date_range', {}).get('end', '?')}",
        f"- Timestamps: {latest.get('timestamp_count', 0):,}",
        f"- Rows: {latest.get('row_count', 0):,}",
        "",
        "| Profile | OOS Return | OOS Trades | Evidence |",
        "|---------|------------|------------|----------|",
    ]
    for profile_id, payload in latest.get("profile_results", {}).items():
        oos = payload.get("out_of_sample", {})
        lines.append(
            f"| {profile_id} | {oos.get('net_return_pct', 0):+.2f}% | "
            f"{oos.get('completed_trades', 0)} | {payload.get('evidence_assessment', '—')} |"
        )
    return "\n".join(lines)


def _format_evidence_labels(latest: dict[str, Any] | None) -> str:
    if latest is None:
        return "No evidence labels — insufficient experiment data."

    lines = []
    for profile_id, payload in latest.get("profile_results", {}).items():
        label = payload.get("evidence_assessment", "UNKNOWN")
        warnings = payload.get("overfitting_warnings") or []
        warn_text = f" ({len(warnings)} warning(s))" if warnings else ""
        lines.append(f"- **{profile_id}:** `{label}`{warn_text}")
    return "\n".join(lines) if lines else "No profiles evaluated."


def generate_system_architecture() -> str:
    return """# System Architecture

## Pipeline Overview

```mermaid
flowchart TD
    A[Hyperliquid public Info API] --> B[Market data collection]
    B --> C[Strategy / signal engine]
    C --> D[Signal result]
    D --> E[Shared risk rules]
    E --> F[Decision engine]
    F --> G[Paper trader]
    G --> H[Equity / session logging]
    H --> I[Dashboard]
    B --> J[Backtester]
    J --> K[Experiment framework]
    K --> L[Evidence assessment]
    L --> M[Review package]
    I --> M
```

## Module Responsibilities

| Module | Responsibility |
|--------|----------------|
| `hyperliquid_client.py` | Read-only HTTP calls to public Info API |
| `market_analyzer.py` | Price history, rolling percent changes |
| `signals.py` | Threshold application, UP/DOWN/FLAT labels |
| `strategies/` | Pluggable strategy interface (offline experiments) |
| `strategy_profiles.py` | Immutable named experiment configurations |
| `risk_rules.py` | Live paper-trading risk blocks |
| `profile_risk_rules.py` | Experiment-only risk blocks (same logic, profile limits) |
| `decision_engine.py` | Read-only recommendations with reasons |
| `paper_trader.py` | Simulated fills, fees, slippage |
| `equity_logger.py` | Append-only equity snapshots |
| `data_logger.py` | Append-only price log |
| `session.py` / `session_report.py` | Session IDs and per-session analytics |
| `market_snapshot.py` | Terminal + `recommendations.json` snapshot |
| `backtester.py` | Offline replay of `price_log.csv` |
| `experiment_runner.py` | Multi-profile offline experiments |
| `experiment_evidence.py` | Descriptive evidence labels |
| `experiment_overfitting.py` | Overfitting and concentration warnings |
| `dashboard.py` | Read-only Streamlit visualization |
| `review_package.py` | Adviser review package generator |
| `capital_readiness.py` | Descriptive capital-readiness assessment |

## Important Data Files

| File | Purpose |
|------|---------|
| `data/price_log.csv` | Historical mid prices (live + experiment input) |
| `data/paper_trades.csv` | Live paper trade log |
| `data/equity_log.csv` | Per-cycle equity snapshots |
| `data/recommendations.json` | Latest analyzer snapshot |
| `data/backtest_results.json` | Test vs production backtest comparison |
| `data/session_configs/` | Per-run configuration snapshots |
| `data/session_reports/` | Per-session analytics JSON |
| `data/experiments/` | Offline experiment results |

## Fees and Slippage

Applied on **every simulated fill** in:

- `paper_trader.py` — live paper trading
- `backtester.py` — offline backtests
- `experiment_trader.py` — offline experiments

Default: **0.05% fee** + **0.02% slippage** per fill (configurable in `config.py`).

## Isolation Between Subsystems

| Subsystem | Network | Writes live CSVs | Uses strategy registry |
|-----------|---------|------------------|------------------------|
| Live analyzer (`main.py`) | Yes (read-only API) | Yes | No (uses `signals.py` directly) |
| Backtester | No | No | No |
| Experiment runner | No | No | Yes |
| Review package | No | No | N/A (reads only) |
| Dashboard | No | No | N/A (reads only) |

**Experiments do not use network access.** They replay local `price_log.csv` only.

## Safety Confirmation

- No wallet, signing, leverage, or exchange execution modules exist.
- Review package export is CLI-only (`python3 review_package.py`).
- Dashboard Review Mode is read-only with no export or execution actions.
"""


def generate_strategy_documentation() -> str:
    sections = [
        "# Strategy Documentation",
        "",
        "This document describes every registered strategy and profile.",
        "None of these strategies use trained AI/ML models — they are rule-based.",
        "",
        "## Live Analyzer Note",
        "",
        "The live analyzer (`main.py`) uses `signals.signal_from_1m_change()` directly,",
        "not the strategy registry. Experiment profiles mirror this logic for offline testing.",
        "",
    ]

    for profile_id in list_profile_ids():
        profile = STRATEGY_PROFILES[profile_id]
        thresholds = profile.thresholds
        sections.extend(
            [
                f"## Profile: `{profile_id}`",
                "",
                "| Field | Value |",
                "|-------|-------|",
                f"| Strategy | `{profile.strategy_name}` v{profile.strategy_version} |",
                f"| Lookback | {profile.lookback_seconds} seconds (1-minute rolling change) |",
                f"| Cooldown | {profile.cooldown_seconds} seconds between buys |",
                f"| Trade size | ${profile.trade_size_usdc:,.2f} |",
                f"| Fee assumption | {profile.trading_fee_pct}% |",
                f"| Slippage assumption | {profile.slippage_pct}% |",
                f"| Starting balance | ${profile.starting_balance_usdc:,.2f} |",
                f"| Max allocation/coin | ${profile.max_allocation_per_coin_usdc:,.2f} |",
                f"| Max entries/coin | {profile.max_entries_per_coin} |",
                f"| Max total exposure | ${profile.max_total_open_exposure_usdc:,.2f} |",
                f"| Session trade limit | {profile.max_successful_trades_per_session} |",
                f"| Equity pause | ${profile.min_equity_before_pause_usdc:,.2f} |",
                "",
                "### Signal Thresholds",
                "",
                "| Label | Threshold |",
                "|-------|-----------|",
                f"| STRONG_UP | ≥ +{thresholds.strong_up_pct}% |",
                f"| UP | ≥ +{thresholds.up_pct}% |",
                f"| STRONG_DOWN | ≤ {thresholds.strong_down_pct}% |",
                f"| DOWN | ≤ {thresholds.down_pct}% |",
                f"| FLAT | between DOWN and UP bands |",
                "",
            ]
        )

        if profile.strategy_name == "contrarian_v1":
            sections.extend(
                [
                    "### Logic",
                    "",
                    "1. Compute rolling 1-minute percentage price change.",
                    "2. Map change to signal label using thresholds above.",
                    "3. **BUY** on DOWN or STRONG_DOWN (contrarian — buy the dip).",
                    "4. **SELL** on UP or STRONG_UP when a position is open (take profit).",
                    "5. FLAT / N/A → no trade.",
                    "",
                    "### Required Inputs",
                    "",
                    "- Current mid price for the coin.",
                    "- Price history covering at least the lookback window.",
                    "- Profile thresholds.",
                    "",
                    "### Trade Direction",
                    "",
                    "Contrarian: buy after short-term decline, sell after short-term rise.",
                    "",
                    "### Expected Market Behavior",
                    "",
                    "Assumes short-term price moves partially mean-revert within minutes.",
                    "Works best in range-bound or mildly trending markets.",
                    "",
                    "### Known Weaknesses",
                    "",
                    "- Strong trends can produce repeated losing buys (falling knife).",
                    "- 1-minute lookback is noisy and may react to microstructure spikes.",
                    "- No volume, order book, or funding rate inputs.",
                    "- Three-coin watchlist may not represent broader market conditions.",
                    "",
                    "### Conditions Where It May Fail",
                    "",
                    "- Sustained directional moves (crash or rally).",
                    "- Low liquidity periods with stale or gapped prices.",
                    "- High fee environments relative to expected move size.",
                    "",
                ]
            )
        elif profile.strategy_name == "hold_baseline":
            sections.extend(
                [
                    "### Logic",
                    "",
                    "Always returns FLAT. Never generates tradable signals.",
                    "Used as a zero-trade baseline for experiment comparison.",
                    "",
                    "### Expected Market Behavior",
                    "",
                    "N/A — no trades are placed.",
                    "",
                    "### Known Weaknesses",
                    "",
                    "None (baseline only). Does not test any alpha hypothesis.",
                    "",
                    "### Conditions Where It May Fail",
                    "",
                    "N/A — hold baseline cannot lose from trading but also cannot profit.",
                    "",
                ]
            )

        sections.append(f"### Full Configuration\n\n```json\n{json.dumps(profile.to_dict(), indent=2)}\n```\n")

    sections.extend(
        [
            "## TEST_MODE vs Production Thresholds",
            "",
            "| Mode | UP | STRONG_UP | DOWN | STRONG_DOWN |",
            "|------|-----|-----------|------|-------------|",
            f"| TEST_MODE | {SIGNAL.test_mode.up_pct}% | {SIGNAL.test_mode.strong_up_pct}% | "
            f"{SIGNAL.test_mode.down_pct}% | {SIGNAL.test_mode.strong_down_pct}% |",
            f"| Production | {SIGNAL.production.up_pct}% | {SIGNAL.production.strong_up_pct}% | "
            f"{SIGNAL.production.down_pct}% | {SIGNAL.production.strong_down_pct}% |",
            "",
            "Live analyzer currently uses "
            f"**{'TEST_MODE' if TEST_MODE else 'production'}** thresholds.",
        ]
    )
    return "\n".join(sections)


def generate_risk_management() -> str:
    pt = PAPER_TRADING
    scenarios = _hypothetical_loss_scenarios()
    return f"""# Risk Management

## Currently Implemented Protections

| Safeguard | Value / Behavior |
|-----------|------------------|
| Starting paper balance | ${pt.starting_balance_usdc:,.2f} USDC (simulated) |
| Trade size | ${pt.trade_size_usdc:,.2f} per fill |
| Per-coin allocation cap | ${pt.max_allocation_per_coin_usdc:,.2f} |
| Maximum entries per coin | {pt.max_entries_per_coin} |
| Buy cooldown | {pt.min_seconds_between_buys} seconds |
| Total exposure cap | ${pt.max_total_open_exposure_usdc:,.2f} |
| Session trade limit (live) | {pt.max_successful_trades_per_session} successful trades |
| Equity pause threshold | Trading pauses if equity < ${pt.min_equity_before_pause_usdc:,.2f} |
| Insufficient cash check | BUY blocked if cash < trade size |
| Shared risk rules | `risk_rules.py` used by both decision engine and paper trader |
| Fees on every fill | {pt.trading_fee_pct}% |
| Slippage on every fill | {pt.slippage_pct}% |
| TEST_MODE vs production | Different signal thresholds (`config.TEST_MODE`) |
| No real execution | No wallet, signing, or exchange order endpoints |

## Recommended Future Protections (Not Implemented)

- Maximum daily loss limit across sessions.
- Automatic halt on repeated API failures.
- Position-level stop-loss and take-profit beyond signal-based exits.
- Multi-session drawdown circuit breaker.
- Configuration change audit log with approval workflow.
- Independent reconciliation of paper vs logged trades.

## Protections Requiring Adviser Review

- Whether ${pt.trade_size_usdc:,.2f} trade size is appropriate for any future pilot.
- Whether {pt.max_successful_trades_per_session} session trades is too high or low.
- Whether ${pt.min_equity_before_pause_usdc:,.2f} equity pause is sufficient.
- Whether fee/slippage assumptions ({pt.trading_fee_pct}% / {pt.slippage_pct}%) are realistic.
- Whether leverage should remain permanently prohibited.

## Hypothetical Loss Scenarios (Paper Simulation Only)

These scenarios are calculated analytically — no trades are placed.

{scenarios}

## Risk Rule Alignment

The decision engine (`decision_engine.py`) and paper trader (`paper_trader.py`) share
identical risk checks via `risk_rules.py`. Experiments use the same logic through
`profile_risk_rules.py` with profile-specific limits.
"""


def _hypothetical_loss_scenarios() -> str:
    pt = PAPER_TRADING
    trade = pt.trade_size_usdc
    fee_pct = pt.trading_fee_pct / 100
    slip_pct = pt.slippage_pct / 100
    round_trip_cost = trade * (fee_pct + slip_pct) * 2

    lines = [
        "### One Losing Trade",
        "",
        f"A single round trip with ${trade:,.2f} size and full round-trip costs "
        f"(fee + slippage both sides ≈ ${round_trip_cost:.2f}) plus a 1% adverse move "
        f"could reduce equity by approximately ${trade * 0.01 + round_trip_cost:.2f}.",
        "",
        "### Three Consecutive Losing Trades",
        "",
        f"Three losing round trips at ${trade:,.2f} each with ~1% adverse move each: "
        f"approximately ${3 * (trade * 0.01 + round_trip_cost):.2f} paper loss "
        f"(before exposure overlap effects).",
        "",
        "### Maximum Allocated Exposure Decline",
        "",
        f"| Market Move | Approx. Paper Impact (full ${pt.max_total_open_exposure_usdc:,.0f} exposure) |",
        "|-------------|----------------------------------------------------------------|",
    ]
    for move in (1, 3, 5, 10):
        impact = pt.max_total_open_exposure_usdc * (move / 100)
        lines.append(f"| −{move}% | −${impact:,.2f} unrealized |")

    lines.extend(
        [
            "",
            "### Fees During High Trading Frequency",
            "",
            f"At {pt.max_successful_trades_per_session} round trips × ${round_trip_cost:.2f} "
            f"cost per round trip ≈ ${pt.max_successful_trades_per_session * round_trip_cost:.2f} "
            "in fees/slippage alone (ignoring PnL).",
            "",
            "### Gap or API Data Failure",
            "",
            "If API is unavailable, the analyzer skips the cycle. No new trades are placed.",
            "Open paper positions retain last known prices until next successful poll.",
            "",
            "### Stale Market Data",
            "",
            "Stale prices may produce incorrect signals. No staleness detection is currently implemented.",
            "",
            "### Process Crash During Open Paper Position",
            "",
            "Open positions persist in `paper_trader` state only during runtime.",
            "On restart, positions are reloaded from `paper_trades.csv` history.",
            "Unrealized PnL may differ until prices are refreshed.",
        ]
    )
    return "\n".join(lines)


def generate_evidence_summary(latest: dict[str, Any] | None) -> str:
    if latest is None:
        return """# Evidence Summary

## ⚠️ Insufficient Data

**No experiment results are available.** Evidence cannot be assessed.

Run:
```bash
python3 experiment_runner.py --profiles contrarian_v1_test contrarian_v1_production hold_baseline
```

Historical and paper results do not guarantee future performance.
"""

    date_range = latest.get("date_range", {})
    split = latest.get("split", {})
    lines = [
        "# Evidence Summary",
        "",
        "## Input Data",
        "",
        "| Field | Value |",
        "|-------|-------|",
        f"| Experiment ID | `{latest.get('experiment_id')}` |",
        f"| Date range | {date_range.get('start', '?')} → {date_range.get('end', '?')} |",
        f"| Timestamps | {latest.get('timestamp_count', 0):,} |",
        f"| Rows | {latest.get('row_count', 0):,} |",
        f"| Development timestamps | {split.get('development_timestamps', 0)} |",
        f"| Out-of-sample timestamps | {split.get('out_of_sample_timestamps', 0)} |",
        f"| Input hash | `{latest.get('input_data_hash', '')[:16]}…` |",
        "",
        "Historical and paper results do not guarantee future performance.",
        "",
        "## Profile Results (Hold Baseline Shown Alongside Active Profiles)",
        "",
    ]

    for profile_id, payload in latest.get("profile_results", {}).items():
        dev = payload.get("development", {})
        oos = payload.get("out_of_sample", {})
        combined = payload.get("combined", {})
        label = payload.get("evidence_assessment", "UNKNOWN")
        warnings = payload.get("overfitting_warnings") or []

        lines.extend(
            [
                f"### {profile_id}",
                "",
                f"**Evidence label:** `{label}`",
                "",
                "| Period | Return | Trades | Max DD | Profit Factor | Fees | Slippage |",
                "|--------|--------|--------|--------|---------------|------|----------|",
                f"| Development | {dev.get('net_return_pct', 0):+.2f}% | "
                f"{dev.get('completed_trades', 0)} | "
                f"{dev.get('maximum_drawdown_pct', 0):.2f}% | "
                f"{dev.get('profit_factor', '—')} | "
                f"${dev.get('fees_paid', 0):.2f} | ${dev.get('slippage_cost', 0):.2f} |",
                f"| Out-of-sample | {oos.get('net_return_pct', 0):+.2f}% | "
                f"{oos.get('completed_trades', 0)} | "
                f"{oos.get('maximum_drawdown_pct', 0):.2f}% | "
                f"{oos.get('profit_factor', '—')} | "
                f"${oos.get('fees_paid', 0):.2f} | ${oos.get('slippage_cost', 0):.2f} |",
                f"| Combined | {combined.get('net_return_pct', 0):+.2f}% | "
                f"{combined.get('completed_trades', 0)} | "
                f"{combined.get('maximum_drawdown_pct', 0):.2f}% | "
                f"{combined.get('profit_factor', '—')} | "
                f"${combined.get('fees_paid', 0):.2f} | ${combined.get('slippage_cost', 0):.2f} |",
                "",
            ]
        )

        by_coin = oos.get("results_by_coin") or {}
        if by_coin:
            lines.append("**OOS per-coin realized PnL:**")
            for coin, stats in by_coin.items():
                lines.append(
                    f"- {coin}: ${stats.get('realized_pnl', 0):.2f} "
                    f"({stats.get('completed_trades', 0)} trades)"
                )
            lines.append("")

        if warnings:
            lines.append("**Overfitting / concentration warnings:**")
            for w in warnings:
                lines.append(f"- {w}")
            lines.append("")

        gap = dev.get("net_return_pct", 0) - oos.get("net_return_pct", 0)
        if abs(gap) >= EVIDENCE.max_dev_oos_return_gap_pct:
            lines.append(
                f"⚠️ Dev/OOS return gap ({gap:+.2f}%) exceeds "
                f"{EVIDENCE.max_dev_oos_return_gap_pct}% threshold."
            )
            lines.append("")

    global_warnings = latest.get("warnings") or []
    if global_warnings:
        lines.append("## Global Warnings")
        for w in global_warnings:
            lines.append(f"- {w}")
        lines.append("")

    if split.get("out_of_sample_timestamps", 0) < 50:
        lines.append(
            "⚠️ **Short out-of-sample period** — evidence labels may not be reliable."
        )

    return "\n".join(lines)


def generate_known_limitations() -> str:
    return """# Known Limitations

1. **Paper trading only** — no real market impact, partial fills, or liquidity constraints.
2. **Public mid prices only** — no order book depth, funding rates, or trade flow.
3. **Single lookback window** — 1-minute rolling change only; no multi-timeframe fusion.
4. **Three-coin watchlist** — BTC, ETH, SOL may not represent broader market behavior.
5. **Short data history** — limited timestamps reduce statistical confidence.
6. **No staleness detection** — stale API data is not explicitly flagged.
7. **No stop-loss** — exits only on contrarian sell signals or forced close at experiment end.
8. **Strategy registry is experiment-only** — live path uses `signals.py` directly.
9. **Evidence labels are descriptive** — they do not certify profitability or readiness.
10. **Capital readiness is advisory** — highest tier means "present to adviser," not "approved."
11. **No walk-forward by default** — single 70/30 split unless explicitly enabled.
12. **Configuration changes are not audited** — manual edits to `config.py` take effect immediately.
13. **Dashboard is read-only** — no export or execution from the UI.
14. **Review package excludes raw trading history** — unless explicitly enabled in future versions.
"""


def generate_failure_scenarios() -> str:
    scenarios = [
        _failure_row(
            "API unavailable",
            "Network outage, Hyperliquid downtime, rate limiting",
            "Analyzer skips cycle; last snapshot retained; no new trades",
            "None during outage; open positions carry stale unrealized PnL",
            "Missing new price_log rows; repeated error in terminal",
            "Read-only API; no retry budget exhaustion on execution",
            "Automatic retry with backoff; circuit breaker; alert on sustained failure",
            "How long can the system safely operate without fresh data?",
        ),
        _failure_row(
            "Malformed API response",
            "API schema change, partial JSON, unexpected field types",
            "Exception caught; cycle skipped; no trade placed",
            "None for that cycle",
            "Error log in terminal; missing price row for timestamp",
            "Request timeout; basic response validation",
            "Schema validation layer; versioned API adapter",
            "What happens if price field is missing or zero?",
        ),
        _failure_row(
            "Stale price",
            "Slow polling, cached response, clock skew",
            "Signal computed on stale price; may produce incorrect recommendation",
            "Paper PnL may diverge from actual market",
            "Compare timestamp gaps in price_log.csv",
            "Fixed polling interval",
            "Staleness threshold; reject prices older than N seconds",
            "What maximum price age is acceptable?",
        ),
        _failure_row(
            "Duplicate timestamp",
            "Clock reset, rapid restarts, CSV append race",
            "Duplicate rows in price_log; signals may re-evaluate",
            "Minimal in paper sim; possible double-counting in analytics",
            "Duplicate timestamp rows in CSV",
            "Append-only logging",
            "Deduplication on read; unique constraint",
            "Could duplicate timestamps cause look-ahead bias?",
        ),
        _failure_row(
            "Duplicate trade",
            "Logic bug, missing cooldown enforcement",
            "Would produce two paper trades for same signal",
            "Increased exposure and fees in simulation",
            "Duplicate entries in paper_trades.csv for same timestamp/coin",
            "Cooldown, max entries, session limit",
            "Idempotent trade keys; dedup on write",
            "Are trade identifiers unique and auditable?",
        ),
        _failure_row(
            "Corrupted CSV",
            "Disk error, partial write, manual edit",
            "Load failure or partial data in dashboard/backtester",
            "Analytics may be wrong or incomplete",
            "Parse errors; pytest live-data protection",
            "conftest protects live CSVs during tests",
            "CSV checksum validation; backup/restore",
            "How would you detect and recover from corruption?",
        ),
        _failure_row(
            "Disk write failure",
            "Full disk, permission error",
            "Trade may execute in memory but not persist",
            "State divergence between memory and disk",
            "Missing rows; exception on write",
            "None",
            "Write-ahead log; fsync confirmation",
            "Is in-memory state reconciled with disk on restart?",
        ),
        _failure_row(
            "Dashboard unavailable",
            "Streamlit crash, port conflict",
            "Analyzer continues independently; no dashboard impact",
            "None — dashboard is read-only",
            "Streamlit error page",
            "Dashboard is optional",
            "Health check for dashboard process",
            "Is dashboard required for safe operation?",
        ),
        _failure_row(
            "Restart with open positions",
            "Process kill, machine reboot",
            "Positions rebuilt from paper_trades.csv on restart",
            "Unrealized PnL may be stale until next price poll",
            "Compare position state before/after restart",
            "CSV persistence of trades",
            "Explicit position snapshot file",
            "Are open positions correctly restored?",
        ),
        _failure_row(
            "Extreme price movement",
            "Flash crash, liquidation cascade",
            "Contrarian buys during crash; large unrealized losses",
            "Paper drawdown may exceed pause threshold",
            "Equity log drawdown; pause flag in recommendations.json",
            "Equity pause at configured threshold",
            "Hard stop-loss; volatility filter",
            "What maximum drawdown is acceptable before halting?",
        ),
        _failure_row(
            "Excessive fees",
            "High trade frequency in TEST_MODE",
            "Fees accumulate; net return reduced",
            "Paper equity erosion from costs alone",
            "Fee totals in session reports and experiments",
            "Session trade limit; fee modeling on every fill",
            "Fee budget per session",
            "Are fee assumptions conservative enough?",
        ),
        _failure_row(
            "Strategy overfitting",
            "Too many parameters, short data, repeated tuning",
            "Good dev results, poor OOS results",
            "Misleading paper performance estimates",
            "Dev/OOS gap in experiment results; evidence UNSTABLE/REJECT",
            "Chronological split; overfitting warnings",
            "Walk-forward; holdout registry; parameter freeze",
            "How many tuning iterations were performed?",
        ),
        _failure_row(
            "Too few trades",
            "Production thresholds, short data window",
            "INSUFFICIENT_DATA evidence label",
            "No statistical confidence in results",
            "completed_trades < min_completed_trades",
            "Evidence rules block premature conclusions",
            "Longer data collection; lower thresholds for testing only",
            "What minimum trade count is required?",
        ),
        _failure_row(
            "Inconsistent OOS results",
            "Regime change, small holdout set",
            "Variable OOS returns across folds",
            "Unreliable performance estimates",
            "Walk-forward variance; UNSTABLE label",
            "Single chronological split",
            "Multiple OOS windows; regime tagging",
            "Are OOS periods representative?",
        ),
        _failure_row(
            "One-coin profit concentration",
            "Strategy works on one coin only",
            "Most profit from single coin in OOS",
            "Diversification illusion",
            "Single-coin dominance warning in experiments",
            "Per-coin analytics in dashboard",
            "Minimum per-coin trade count; concentration limits",
            "Should all coins meet minimum performance?",
        ),
        _failure_row(
            "Accidental configuration change",
            "Manual edit to config.py, wrong TEST_MODE",
            "Different thresholds or limits take effect immediately",
            "Inconsistent behavior vs prior sessions",
            "session_configs/ snapshot comparison",
            "Session config saved at start",
            "Config versioning; change approval workflow",
            "Who can approve configuration changes?",
        ),
        _failure_row(
            "Future real-execution module accidentally introduced",
            "Developer adds wallet/signing code",
            "health_check.py and review_check.py should fail",
            "Potential real capital risk if checks bypassed",
            "Forbidden pattern scan in health_check.py",
            "Forbidden module name/pattern blocklist",
            "CI gate; code review; separate execution repo",
            "Should execution code live in a separate repository?",
        ),
    ]

    header = (
        "# Failure Scenarios\n\n"
        "| Scenario | Possible Cause | Expected Behavior | "
        "Paper Financial Risk | Detection | Existing Mitigation | "
        "Missing Mitigation | Adviser Question |\n"
        "|----------|----------------|-------------------|"
        "---------------------|-----------|---------------------|"
        "--------------------|------------------|\n"
    )
    return header + "\n".join(scenarios)


def _failure_row(
    scenario: str,
    cause: str,
    behavior: str,
    risk: str,
    detection: str,
    mitigation: str,
    missing: str,
    question: str,
) -> str:
    return (
        f"| {scenario} | {cause} | {behavior} | {risk} | "
        f"{detection} | {mitigation} | {missing} | {question} |"
    )


def generate_advisor_questions() -> str:
    return """# Adviser Questions

These are questions for a qualified adult adviser and appropriate professionals.
This document does not provide legal, financial, or tax conclusions.

## Technical

1. Are the percentage-change calculations and signal thresholds mathematically correct?
2. Are the fee (0.05%) and slippage (0.02%) assumptions realistic for Hyperliquid perps?
3. Is the 70/30 chronological development/out-of-sample split a fair testing methodology?
4. Is there any data leakage between development and out-of-sample periods?
5. Are the failure controls (equity pause, session limits, exposure caps) adequate?
6. Is the lack of staleness detection on API prices acceptable?
7. Should walk-forward validation be required before any capital consideration?
8. Are the pytest tests and health checks sufficient for ongoing verification?

## Strategy

1. Does the contrarian premise (buy dips, sell rips on 1-minute moves) make economic sense?
2. Is a one-minute lookback window sufficient, or is it too noisy?
3. What market regimes (trending, ranging, high-volatility) should be tested?
4. What additional baseline strategies are necessary beyond hold_baseline?
5. What minimum sample size (trades, days, sessions) should be acceptable for evaluation?
6. Should the strategy be tested on additional coins or timeframes?
7. Is the difference between TEST_MODE and production thresholds well understood?

## Risk

1. What maximum drawdown is acceptable before halting any future pilot?
2. What capital amount could be completely lost in a worst-case scenario?
3. Should leverage remain permanently prohibited?
4. What manual approval process should be required before any real capital use?
5. What emergency stop rules are needed (daily loss limit, API failure halt)?
6. Is the $100 trade size appropriate relative to a potential pilot allocation?
7. Should per-coin concentration limits be tighter?

## Legal / Account

1. Who legally owns and supervises any trading account that might be considered in the future?
2. What age, identity, and platform rules apply to the account holder?
3. What permissions are required before using another person's funds?
4. What reporting should a funder receive (frequency, metrics, access)?
5. What regulatory requirements apply to automated trading systems in the relevant jurisdiction?
6. Should a formal written agreement exist before any supervised pilot?

---

*All questions above require independent professional review. This system does not provide legal advice.*
"""


def generate_review_checklist() -> str:
    return """# Review Checklist

Use this checklist during an adviser review meeting.

## Safety Verification

- [ ] Confirm VERSION is `9.0.0-review`
- [ ] Confirm no wallet, signing, or execution modules exist
- [ ] Confirm dashboard has no "go live" button or export action
- [ ] Confirm review package was generated via CLI (`python3 review_package.py`)
- [ ] Verify checksums.sha256 matches all package files
- [ ] Confirm no secrets, .env files, or absolute local paths in package

## Technical Review

- [ ] Run `python3 review_check.py` — all checks pass
- [ ] Run `python3 -m pytest` — all tests pass
- [ ] Inspect `system_architecture.md` pipeline diagram
- [ ] Verify fee/slippage assumptions in `risk_management.md`
- [ ] Review `failure_scenarios.md` for unmitigated risks
- [ ] Confirm experiments do not use network access

## Evidence Review

- [ ] Review `evidence_summary.md` for all profiles including hold_baseline
- [ ] Check evidence labels — no strategy labeled as "proven"
- [ ] Review overfitting warnings
- [ ] Confirm out-of-sample trade count meets minimum thresholds
- [ ] Compare dev vs OOS returns for consistency
- [ ] Check per-coin concentration

## Capital Readiness

- [ ] Run `python3 capital_readiness.py` and review all passed/failed criteria
- [ ] Confirm assessment is descriptive only (not authorization)
- [ ] Verify unexplained trades and risk violations would block progression
- [ ] Confirm one week of data alone does not qualify

## Strategy Review

- [ ] Read `strategy_documentation.md` for each profile
- [ ] Understand contrarian logic and known weaknesses
- [ ] Review TEST_MODE vs production threshold differences
- [ ] Discuss `advisor_questions.md` with qualified adviser

## Decision

- [ ] Document adviser feedback and open questions
- [ ] Record decision: continue paper testing / gather more evidence / schedule follow-up
- [ ] **Do not proceed to real trading based on this checklist alone**
"""
