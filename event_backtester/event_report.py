"""JSON and markdown report generation for event backtests."""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from event_backtester.event_metrics import EVIDENCE_INSUFFICIENT
from event_backtester.event_safety import EXECUTION_ENABLED, PAPER_ONLY, SAFETY_STATEMENT

EVENT_BACKTEST_SCHEMA_VERSION = "1.0.0"


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def profile_config_hash(profile_dict: dict[str, Any]) -> str:
    return hashlib.sha256(_stable_json(profile_dict).encode()).hexdigest()


def compute_deterministic_backtest_hash(
    *,
    stream_hash: str,
    profile_dict: dict[str, Any],
    strategy_version: str,
    cost_assumptions: dict[str, Any],
    fill_model: str,
    coins: list[str],
    channels: list[str],
    trade_log: list[dict[str, Any]],
    blocked_signals: list[dict[str, Any]],
    final_equity: float,
) -> str:
    payload = {
        "stream_hash": stream_hash,
        "profile_config_hash": profile_config_hash(profile_dict),
        "strategy_version": strategy_version,
        "cost_assumptions": cost_assumptions,
        "fill_model": fill_model,
        "coins": sorted(coins),
        "channels": sorted(channels),
        "trade_log": trade_log,
        "blocked_signals": blocked_signals,
        "final_equity": round(final_equity, 8),
    }
    return hashlib.sha256(_stable_json(payload).encode()).hexdigest()


def build_event_backtest_report(
    *,
    event_backtest_id: str,
    replay_session_id: str,
    source_recorder_session_id: str,
    quality_report_path: str | None,
    quality_label: str | None,
    strategy_name: str,
    strategy_version: str,
    profile_name: str,
    profile_configuration: dict[str, Any],
    coins: list[str],
    channels: list[str],
    starting_equity: float,
    ending_equity: float,
    realized_pnl: float,
    unrealized_pnl: float,
    total_fees: float,
    total_slippage_cost: float,
    gross_pnl_before_costs: float,
    completed_trades: int,
    buys: int,
    sells: int,
    blocked_signals: int,
    blocked_reasons: dict[str, int],
    win_rate: float,
    average_win: float,
    average_loss: float,
    average_net_return_per_trade: float,
    profit_factor: float | None,
    maximum_drawdown: float,
    exposure_time_pct: float,
    average_holding_time_seconds: float,
    results_by_coin: dict[str, Any],
    events_processed: int,
    bbo_availability_pct: float,
    stale_bbo_count: int,
    missing_bbo_count: int,
    stream_hash: str,
    deterministic_backtest_hash: str,
    warnings: list[str],
    limitations: list[str],
    evidence_label: str,
    dry_run: bool,
    adapter_notes: list[str],
    comparison_section: dict[str, Any],
    generated_at: str | None = None,
) -> dict[str, Any]:
    net_return = ((ending_equity - starting_equity) / starting_equity * 100.0) if starting_equity else 0.0
    return {
        "schema_version": EVENT_BACKTEST_SCHEMA_VERSION,
        "event_backtest_id": event_backtest_id,
        "replay_session_id": replay_session_id,
        "source_recorder_session_id": source_recorder_session_id,
        "quality_report_path": quality_report_path,
        "quality_label": quality_label,
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(),
        "strategy_name": strategy_name,
        "strategy_version": strategy_version,
        "profile_name": profile_name,
        "profile_configuration": profile_configuration,
        "coins": coins,
        "channels": channels,
        "starting_equity": starting_equity,
        "ending_equity": ending_equity,
        "net_return_pct": net_return,
        "realized_pnl": realized_pnl,
        "unrealized_pnl": unrealized_pnl,
        "total_fees": total_fees,
        "total_slippage_cost": total_slippage_cost,
        "gross_pnl_before_costs": gross_pnl_before_costs,
        "completed_trades": completed_trades,
        "buys": buys,
        "sells": sells,
        "blocked_signals": blocked_signals,
        "blocked_reasons": blocked_reasons,
        "win_rate": win_rate,
        "average_win": average_win,
        "average_loss": average_loss,
        "average_net_return_per_trade": average_net_return_per_trade,
        "profit_factor": profit_factor,
        "maximum_drawdown_pct": maximum_drawdown,
        "exposure_time_pct": exposure_time_pct,
        "average_holding_time_seconds": average_holding_time_seconds,
        "results_by_coin": results_by_coin,
        "events_processed": events_processed,
        "bbo_availability_pct": bbo_availability_pct,
        "stale_bbo_count": stale_bbo_count,
        "missing_bbo_count": missing_bbo_count,
        "stream_hash": stream_hash,
        "deterministic_backtest_hash": deterministic_backtest_hash,
        "warnings": warnings,
        "limitations": limitations,
        "evidence_label": evidence_label,
        "dry_run": dry_run,
        "adapter_notes": adapter_notes,
        "comparison_section": comparison_section,
        "paper_only": PAPER_ONLY,
        "execution_enabled": EXECUTION_ENABLED,
    }


def render_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Event-Driven Backtest Report",
        "",
        f"**Event backtest ID:** `{report.get('event_backtest_id', '—')}`",
        f"**Replay session:** `{report.get('replay_session_id', '—')}`",
        f"**Source recorder session:** `{report.get('source_recorder_session_id', '—')}`",
        f"**Generated at:** {report.get('generated_at', '—')}",
        "",
        "## Strategy / Profile",
        "",
        f"- **Strategy:** {report.get('strategy_name', '—')} v{report.get('strategy_version', '—')}",
        f"- **Profile:** `{report.get('profile_name', '—')}`",
        f"- **Coins:** {', '.join(report.get('coins') or [])}",
        f"- **Channels:** {', '.join(report.get('channels') or [])}",
        "",
        "## Safety",
        "",
        SAFETY_STATEMENT,
        "",
        "## Data Quality",
        "",
        f"- **Quality label:** {report.get('quality_label', '—')}",
        f"- **Quality report:** `{report.get('quality_report_path', '—')}`",
        f"- **Stream hash:** `{report.get('stream_hash', '—')}`",
        f"- **BBO availability:** {report.get('bbo_availability_pct', 0):.1f}%",
        "",
        "## Key Metrics",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Starting equity | ${report.get('starting_equity', 0):,.2f} |",
        f"| Ending equity | ${report.get('ending_equity', 0):,.2f} |",
        f"| Net return | {report.get('net_return_pct', 0):.2f}% |",
        f"| Realized PnL | ${report.get('realized_pnl', 0):,.2f} |",
        f"| Unrealized PnL | ${report.get('unrealized_pnl', 0):,.2f} |",
        f"| Total fees | ${report.get('total_fees', 0):,.2f} |",
        f"| Total slippage | ${report.get('total_slippage_cost', 0):,.2f} |",
        f"| Completed trades | {report.get('completed_trades', 0)} |",
        f"| Max drawdown | {report.get('maximum_drawdown_pct', 0):.2f}% |",
        f"| Profit factor | {report.get('profit_factor') or '—'} |",
        f"| Evidence label | **{report.get('evidence_label', EVIDENCE_INSUFFICIENT)}** |",
        "",
        "## Trade Summary",
        "",
        f"- **Buys:** {report.get('buys', 0)}",
        f"- **Sells:** {report.get('sells', 0)}",
        f"- **Win rate:** {report.get('win_rate', 0):.1f}%",
        f"- **Avg win:** {report.get('average_win', 0):.2f}%",
        f"- **Avg loss:** {report.get('average_loss', 0):.2f}%",
        "",
        "## Blocked Signals",
        "",
        f"- **Total blocked:** {report.get('blocked_signals', 0)}",
    ]
    for reason, count in sorted((report.get("blocked_reasons") or {}).items(), key=lambda x: -x[1]):
        lines.append(f"- {reason}: {count}")

    lines.extend(["", "## Results by Coin", ""])
    for coin, data in sorted((report.get("results_by_coin") or {}).items()):
        lines.append(
            f"- **{coin}:** buys={data.get('buys', 0)}, sells={data.get('sells', 0)}, "
            f"realized_pnl=${data.get('realized_pnl', 0):,.2f}"
        )

    lines.extend(["", "## Warnings", ""])
    warnings = report.get("warnings") or []
    if warnings:
        for w in warnings:
            lines.append(f"- {w}")
    else:
        lines.append("_No warnings._")

    lines.extend(
        [
            "",
            "## Evidence Label",
            "",
            f"**{report.get('evidence_label', EVIDENCE_INSUFFICIENT)}**",
            "",
            "_This label is conservative and does not claim the strategy is proven or profitable._",
            "",
            "## Limitations",
            "",
        ]
    )
    for lim in report.get("limitations") or []:
        lines.append(f"- {lim}")

    comparison = report.get("comparison_section") or {}
    if comparison:
        lines.extend(["", "## Comparison vs Row-Based Experiments", ""])
        for key, value in comparison.items():
            lines.append(f"- **{key}:** {value}")

    lines.extend(
        [
            "",
            "## Next Recommended Test",
            "",
            "Run a longer recorder session (hours, not seconds) with GOOD_FOR_RESEARCH "
            "quality label, then replay and backtest with the same profile to accumulate "
            "more completed trades and stable BBO coverage.",
            "",
            f"**Deterministic backtest hash:** `{report.get('deterministic_backtest_hash', '—')}`",
            "",
        ]
    )
    return "\n".join(lines)


def write_event_backtest_report(
    report: dict[str, Any],
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    backtest_id = report["event_backtest_id"]
    json_path = output_dir / f"{backtest_id}.json"
    md_path = output_dir / f"{backtest_id}.md"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)

    md_path.write_text(render_markdown_report(report), encoding="utf-8")
    return {"json": json_path, "markdown": md_path}


def list_event_backtest_reports(output_dir: Path) -> list[dict[str, Any]]:
    if not output_dir.exists():
        return []

    reports: list[dict[str, Any]] = []
    for path in sorted(output_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            with path.open(encoding="utf-8") as handle:
                data = json.load(handle)
            reports.append(
                {
                    "event_backtest_id": data.get("event_backtest_id", path.stem),
                    "replay_session_id": data.get("replay_session_id"),
                    "source_recorder_session_id": data.get("source_recorder_session_id"),
                    "generated_at": data.get("generated_at"),
                    "strategy_name": data.get("strategy_name"),
                    "profile_name": data.get("profile_name"),
                    "starting_equity": data.get("starting_equity"),
                    "ending_equity": data.get("ending_equity"),
                    "net_return_pct": data.get("net_return_pct"),
                    "completed_trades": data.get("completed_trades", 0),
                    "total_fees": data.get("total_fees", 0),
                    "total_slippage_cost": data.get("total_slippage_cost", 0),
                    "maximum_drawdown_pct": data.get("maximum_drawdown_pct"),
                    "profit_factor": data.get("profit_factor"),
                    "bbo_availability_pct": data.get("bbo_availability_pct"),
                    "evidence_label": data.get("evidence_label"),
                    "warnings": data.get("warnings") or [],
                    "dry_run": data.get("dry_run", False),
                    "json_path": str(path),
                    "markdown_path": str(path.with_suffix(".md")),
                }
            )
        except (json.JSONDecodeError, OSError):
            continue
    return reports


def new_event_backtest_id() -> str:
    return uuid.uuid4().hex
