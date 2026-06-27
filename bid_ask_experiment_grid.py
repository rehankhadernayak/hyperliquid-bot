"""Paper-only bid/ask model experiment grid (research-only, read-only inputs)."""

from __future__ import annotations

import argparse
import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from bid_ask_schema import EXECUTION_ENABLED, PAPER_ONLY
from bid_ask_train import (
    DEFAULT_DATA_ROOT,
    DEFAULT_MODELS_DIR,
    DEFAULT_REPORTS_DIR,
    SAFETY_STATEMENT,
    run_training_pipeline,
)

DEFAULT_GRID_REPORTS_DIR = Path("reports/bid_ask_experiments")

TrainingRunner = Callable[..., dict[str, Any]]


@dataclass(frozen=True)
class ExperimentConfig:
    coin: str
    interval: str
    horizon: int
    threshold_bps: float
    train_fraction: float = 0.70
    max_rows: int | None = None
    notes: str = ""


def default_experiment_grid() -> list[ExperimentConfig]:
    """Return the default bid/ask model experiment grid."""
    return [
        ExperimentConfig(
            coin="BTC",
            interval="15s",
            horizon=1,
            threshold_bps=1.0,
            notes="BTC 15s short-horizon tight threshold",
        ),
        ExperimentConfig(
            coin="BTC",
            interval="1m",
            horizon=1,
            threshold_bps=3.0,
            notes="BTC 1m short-horizon moderate threshold",
        ),
        ExperimentConfig(
            coin="BTC",
            interval="1m",
            horizon=5,
            threshold_bps=5.0,
            notes="BTC 1m medium-horizon wider threshold",
        ),
        ExperimentConfig(
            coin="ETH",
            interval="1m",
            horizon=5,
            threshold_bps=5.0,
            notes="ETH 1m medium-horizon wider threshold",
        ),
        ExperimentConfig(
            coin="SOL",
            interval="1m",
            horizon=5,
            threshold_bps=5.0,
            notes="SOL 1m medium-horizon wider threshold",
        ),
    ]


def _normalize_coin(coin: str) -> str:
    return coin.strip().upper()


def run_single_experiment(
    config: ExperimentConfig,
    *,
    data_root: Path = DEFAULT_DATA_ROOT,
    reports_dir: Path = DEFAULT_REPORTS_DIR,
    models_dir: Path = DEFAULT_MODELS_DIR,
    save_report: bool = True,
    training_runner: TrainingRunner | None = None,
) -> dict[str, Any]:
    """Run one experiment by calling the training pipeline directly."""
    runner = training_runner or run_training_pipeline
    experiment_id = uuid.uuid4().hex
    pipeline_result = runner(
        coin=_normalize_coin(config.coin),
        interval=config.interval,
        horizon=config.horizon,
        threshold_bps=config.threshold_bps,
        train_fraction=config.train_fraction,
        max_rows=config.max_rows,
        data_root=data_root,
        reports_dir=reports_dir,
        models_dir=models_dir,
        save_report=save_report,
    )
    report = pipeline_result["report"]
    evaluation = pipeline_result["evaluation"]
    warnings = list(report.get("warnings", []))
    if config.notes:
        warnings.append(f"Config note: {config.notes}")

    return {
        "experiment_id": experiment_id,
        "coin": report["coin"],
        "interval": report["interval"],
        "horizon": report["horizon"],
        "threshold_bps": report["threshold_bps"],
        "train_rows": evaluation["train_rows"],
        "test_rows": evaluation["test_rows"],
        "model_used": report["model_used"],
        "model_accuracy": float(report["model_accuracy"]),
        "baseline_accuracy": float(report["baseline_accuracy"]),
        "model_edge_vs_baseline": float(report["model_edge_vs_baseline"]),
        "model_artifact_path": report.get("model_artifact_path"),
        "warnings": warnings,
        "notes": config.notes,
        "train_fraction": config.train_fraction,
        "max_rows": config.max_rows,
        "paper_only": PAPER_ONLY,
        "execution_enabled": EXECUTION_ENABLED,
    }


def run_experiment_grid(
    configs: list[ExperimentConfig],
    *,
    data_root: Path = DEFAULT_DATA_ROOT,
    reports_dir: Path = DEFAULT_REPORTS_DIR,
    models_dir: Path = DEFAULT_MODELS_DIR,
    save_report: bool = True,
    training_runner: TrainingRunner | None = None,
) -> dict[str, Any]:
    """Run experiments sequentially; continue after individual failures."""
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for config in configs:
        try:
            results.append(
                run_single_experiment(
                    config,
                    data_root=data_root,
                    reports_dir=reports_dir,
                    models_dir=models_dir,
                    save_report=save_report,
                    training_runner=training_runner,
                )
            )
        except Exception as exc:
            failures.append(
                {
                    "config": asdict(config),
                    "coin": _normalize_coin(config.coin),
                    "interval": config.interval,
                    "horizon": config.horizon,
                    "threshold_bps": config.threshold_bps,
                    "error": str(exc),
                }
            )

    return {
        "results": results,
        "failures": failures,
        "ranked": rank_experiments(results),
    }


def rank_experiments(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank by model edge, then accuracy, then test row count."""
    return sorted(
        results,
        key=lambda row: (
            float(row.get("model_edge_vs_baseline", 0.0)),
            float(row.get("model_accuracy", 0.0)),
            int(row.get("test_rows", 0)),
        ),
        reverse=True,
    )


def _report_stamp(report: dict[str, Any]) -> str:
    if report.get("generated_at"):
        raw = str(report["generated_at"])
        return raw.replace(":", "").replace("+", "").replace("-", "")[:15]
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _aggregate_warnings(
    results: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> list[str]:
    warnings: list[str] = []
    seen: set[str] = set()
    for result in results:
        for warning in result.get("warnings", []):
            if warning not in seen:
                seen.add(warning)
                warnings.append(warning)
    for failure in failures:
        message = f"Experiment failed for {failure.get('coin')} {failure.get('interval')}: {failure.get('error')}"
        if message not in seen:
            seen.add(message)
            warnings.append(message)
    return warnings


def build_grid_report(
    *,
    configs: list[ExperimentConfig],
    results: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    ranked: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a grid summary report payload."""
    ranked_results = ranked if ranked is not None else rank_experiments(results)
    best = ranked_results[0] if ranked_results else None
    return {
        "grid_id": uuid.uuid4().hex,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_count": len(configs),
        "completed_count": len(results),
        "failed_count": len(failures),
        "configs": [asdict(config) for config in configs],
        "results": results,
        "failures": failures,
        "ranked": ranked_results,
        "best_result": best,
        "warnings": _aggregate_warnings(results, failures),
        "paper_only": PAPER_ONLY,
        "execution_enabled": EXECUTION_ENABLED,
        "safety_statement": SAFETY_STATEMENT,
    }


def write_grid_report_json(
    report: dict[str, Any],
    output_dir: Path = DEFAULT_GRID_REPORTS_DIR,
) -> Path:
    """Save grid report JSON under reports/bid_ask_experiments/."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = _report_stamp(report)
    output_path = output_dir / f"bid_ask_experiment_grid_{stamp}.json"
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    report["json_report_path"] = str(output_path)
    return output_path


def _format_accuracy(value: float) -> str:
    return f"{value:.4f}"


def _format_config_label(row: dict[str, Any]) -> str:
    return (
        f"{row.get('coin')} {row.get('interval')} "
        f"h={row.get('horizon')} thr={row.get('threshold_bps')}bps"
    )


def write_grid_report_markdown(
    report: dict[str, Any],
    output_dir: Path = DEFAULT_GRID_REPORTS_DIR,
) -> Path:
    """Save grid report Markdown under reports/bid_ask_experiments/."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = _report_stamp(report)
    output_path = output_dir / f"bid_ask_experiment_grid_{stamp}.md"

    ranked = report.get("ranked", [])
    failures = report.get("failures", [])
    warnings = report.get("warnings", [])
    best = report.get("best_result")

    lines = [
        "# Bid/Ask Experiment Grid Report",
        "",
        f"Generated at: {report.get('generated_at', 'n/a')}",
        f"Experiments: {report.get('experiment_count', 0)} "
        f"(completed={report.get('completed_count', 0)}, "
        f"failed={report.get('failed_count', 0)})",
        "",
        "## Experiment table",
        "",
        "| Rank | Coin | Interval | Horizon | Threshold (bps) | Model | Edge | Accuracy | Test rows |",
        "| ---: | --- | --- | ---: | ---: | --- | ---: | ---: | ---: |",
    ]

    for index, row in enumerate(ranked, start=1):
        lines.append(
            f"| {index} | {row.get('coin')} | {row.get('interval')} | "
            f"{row.get('horizon')} | {row.get('threshold_bps')} | "
            f"{row.get('model_used')} | "
            f"{_format_accuracy(float(row.get('model_edge_vs_baseline', 0.0)))} | "
            f"{_format_accuracy(float(row.get('model_accuracy', 0.0)))} | "
            f"{row.get('test_rows', 0)} |"
        )

    if not ranked:
        lines.append("| - | - | - | - | - | - | - | - | - |")

    lines.extend(["", "## Best result", ""])
    if best is None:
        lines.append("No successful experiments.")
    else:
        lines.extend(
            [
                f"- config: {_format_config_label(best)}",
                f"- model_used: {best.get('model_used')}",
                f"- model_edge_vs_baseline: "
                f"{_format_accuracy(float(best.get('model_edge_vs_baseline', 0.0)))}",
                f"- model_accuracy: "
                f"{_format_accuracy(float(best.get('model_accuracy', 0.0)))}",
                f"- baseline_accuracy: "
                f"{_format_accuracy(float(best.get('baseline_accuracy', 0.0)))}",
                f"- test_rows: {best.get('test_rows', 0)}",
            ]
        )
        if best.get("model_artifact_path"):
            lines.append(f"- model_artifact_path: {best['model_artifact_path']}")

    lines.extend(["", "## Failed experiments", ""])
    if not failures:
        lines.append("None.")
    else:
        lines.extend(["| Coin | Interval | Horizon | Threshold (bps) | Error |", "| --- | --- | ---: | ---: | --- |"])
        for failure in failures:
            lines.append(
                f"| {failure.get('coin')} | {failure.get('interval')} | "
                f"{failure.get('horizon')} | {failure.get('threshold_bps')} | "
                f"{failure.get('error')} |"
            )

    lines.extend(["", "## Warnings", ""])
    if not warnings:
        lines.append("None.")
    else:
        for warning in warnings:
            lines.append(f"- {warning}")

    lines.extend(
        [
            "",
            "## Research interpretation",
            "",
            "Results compare offline bid/ask movement models on local CSV data only. "
            "Higher model edge versus baseline suggests the selected model may capture "
            "signal beyond a simple majority-class baseline, but this does not prove "
            "live profitability. Use the ranked table to choose promising configurations "
            "for further paper-only validation.",
            "",
            "## Paper-only disclaimer",
            "",
            SAFETY_STATEMENT,
            "",
            "Paper-only. No trades placed.",
        ]
    )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    report["markdown_report_path"] = str(output_path)
    return output_path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a paper-only bid/ask model experiment grid on local CSV data "
            "(research-only, read-only inputs)."
        )
    )
    parser.add_argument(
        "--default",
        action="store_true",
        help="Run the default experiment grid",
    )
    parser.add_argument("--coin", help="Single experiment coin symbol, e.g. BTC")
    parser.add_argument(
        "--interval",
        choices=["15s", "1m"],
        help="Single experiment resampled interval",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=1,
        help="Single experiment label horizon in bars (default: 1)",
    )
    parser.add_argument(
        "--threshold-bps",
        type=float,
        default=1.0,
        help="Single experiment UP/DOWN threshold in basis points (default: 1.0)",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.70,
        help="Chronological train split fraction (default: 0.70)",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional row cap for each experiment",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Root directory containing bid/ask CSV files",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=DEFAULT_REPORTS_DIR,
        help="Directory for per-experiment training reports",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help="Directory for saved model artifacts",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_GRID_REPORTS_DIR,
        help="Directory for grid summary JSON/Markdown reports",
    )
    return parser


def _resolve_configs(args: argparse.Namespace) -> list[ExperimentConfig]:
    if args.default:
        return default_experiment_grid()
    if args.coin and args.interval:
        return [
            ExperimentConfig(
                coin=args.coin,
                interval=args.interval,
                horizon=args.horizon,
                threshold_bps=args.threshold_bps,
                train_fraction=args.train_fraction,
                max_rows=args.max_rows,
            )
        ]
    raise SystemExit("Provide --default or both --coin and --interval.")


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    configs = _resolve_configs(args)
    grid_output = run_experiment_grid(
        configs,
        data_root=args.data_root,
        reports_dir=args.reports_dir,
        models_dir=args.models_dir,
    )
    report = build_grid_report(
        configs=configs,
        results=grid_output["results"],
        failures=grid_output["failures"],
        ranked=grid_output["ranked"],
    )
    json_path = write_grid_report_json(report, output_dir=args.output_dir)
    markdown_path = write_grid_report_markdown(report, output_dir=args.output_dir)

    print(f"experiments={len(configs)} completed={len(grid_output['results'])} failed={len(grid_output['failures'])}")
    best = report.get("best_result")
    if best is not None:
        print(f"best_config={_format_config_label(best)}")
        print(f"best_edge={float(best.get('model_edge_vs_baseline', 0.0)):.4f}")
    else:
        print("best_config=none")
    print(f"json_report_path={json_path}")
    print(f"markdown_report_path={markdown_path}")
    for warning in report.get("warnings", []):
        print(f"warning: {warning}")
    print("Paper-only. No trades placed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
