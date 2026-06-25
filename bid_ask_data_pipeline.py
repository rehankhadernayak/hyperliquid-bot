"""Command-line wrapper for the local bid/ask data pipeline (research-only, paper-only)."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import bid_ask_extractor
import bid_ask_resampler
import bid_ask_train
import bid_ask_validator

DATA_COVERAGE_STATEMENT = (
    "Using locally collected data only. This is not full exchange all-time history."
)
HISTORICAL_WARNING = (
    "Historical exchange backfill is not implemented yet. Using locally collected data only."
)

DEFAULT_DATA_ROOT = bid_ask_extractor.DEFAULT_OUTPUT_ROOT
DEFAULT_MARKET_STREAM_DIR = bid_ask_extractor.DEFAULT_MARKET_STREAM_DIR
DEFAULT_TRAINING_REPORTS_DIR = bid_ask_train.DEFAULT_REPORTS_DIR
DEFAULT_MODELS_DIR = bid_ask_train.DEFAULT_MODELS_DIR

SUPPORTED_EXTRACT_INTERVALS = frozenset({"1s"})
SUPPORTED_VALIDATE_INTERVALS = frozenset({"1s", "15s", "1m"})
SUPPORTED_RESAMPLE_FROM = frozenset({"1s"})
SUPPORTED_RESAMPLE_TO = frozenset({"15s", "1m", "all"})
SUPPORTED_TRAIN_INTERVALS = frozenset({"15s", "1m"})
SUPPORTED_SOURCES = frozenset({"local"})
HISTORICAL_SOURCE_ALIASES = frozenset({"historical", "exchange", "all-time", "alltime"})
RESAMPLE_ALL_INTERVALS = ("15s", "1m")


@dataclass
class StepResult:
    action: str
    coin: str
    intervals: list[str] = field(default_factory=list)
    output_paths: list[str] = field(default_factory=list)
    report_paths: list[str] = field(default_factory=list)
    row_counts: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    passed: bool | None = None
    skipped: bool = False
    error: str | None = None


def _normalize_coins(coins: list[str]) -> list[str]:
    return [coin.strip().upper() for coin in coins if coin.strip()]


def _print_data_coverage() -> None:
    print(DATA_COVERAGE_STATEMENT)


def _print_step_summary(result: StepResult, *, next_command: str | None = None) -> None:
    print(f"action={result.action}")
    print(f"coin={result.coin}")
    if result.intervals:
        print(f"intervals={','.join(result.intervals)}")
    if result.output_paths:
        print(f"output_paths={','.join(result.output_paths)}")
    if result.report_paths:
        print(f"report_paths={','.join(result.report_paths)}")
    for interval, count in result.row_counts.items():
        print(f"row_count[{interval}]={count}")
    if result.passed is not None:
        print(f"passed={result.passed}")
    if result.skipped:
        print("skipped=true")
    if result.error:
        print(f"error={result.error}")
    for warning in result.warnings:
        print(f"warning: {warning}")
    if next_command:
        print(f"next_recommended_command={next_command}")
    print()


def _maybe_warn_historical_source(source: str | None) -> None:
    if source is None:
        return
    normalized = source.strip().lower()
    if normalized in HISTORICAL_SOURCE_ALIASES or normalized not in SUPPORTED_SOURCES:
        print(HISTORICAL_WARNING)


def _strict_blocks_continue(report_passed: bool | None, *, strict: bool) -> bool:
    return strict and report_passed is False


def cmd_extract(
    coins: list[str],
    *,
    source: str = "local",
    interval: str = "1s",
    dry_run: bool = False,
    data_root: Path = DEFAULT_DATA_ROOT,
    market_stream_dir: Path = DEFAULT_MARKET_STREAM_DIR,
) -> list[StepResult]:
    _maybe_warn_historical_source(source)

    if interval not in SUPPORTED_EXTRACT_INTERVALS:
        raise ValueError(f"Extract supports interval 1s only; got {interval!r}")

    _print_data_coverage()
    results: list[StepResult] = []
    normalized = _normalize_coins(coins)
    for index, coin in enumerate(normalized):
        result = StepResult(action="extract", coin=coin, intervals=[interval])
        if dry_run:
            output_path = str(data_root / "1s" / f"{coin}_1s.csv")
            result.output_paths = [output_path]
            result.report_paths = [str(data_root / "reports" / f"{coin}_<timestamp>.json")]
            result.warnings.append("dry-run: no files written")
            next_cmd = None
            if index == len(normalized) - 1:
                next_cmd = (
                    f"python3 bid_ask_data_pipeline.py validate --coins {' '.join(normalized)} "
                    f"--interval {interval}"
                )
            _print_step_summary(result, next_command=next_cmd)
            results.append(result)
            continue

        try:
            report = bid_ask_extractor.extract_coin_bid_ask_1s(
                coin,
                market_stream_dir=market_stream_dir,
                output_root=data_root,
            )
            result.output_paths = [report.output_path]
            result.report_paths = [
                str(data_root / "reports" / f"{report.coin}_*.json")
            ]
            result.row_counts[interval] = report.row_count
            if report.missing_seconds:
                result.warnings.append(f"{report.missing_seconds} missing second(s) in coverage")
            if report.duplicate_timestamps:
                result.warnings.append(
                    f"{report.duplicate_timestamps} duplicate timestamp(s) collapsed during sampling"
                )
            next_cmd = (
                f"python3 bid_ask_data_pipeline.py validate --coins {coin} --interval {interval}"
            )
            _print_step_summary(result, next_command=next_cmd)
        except Exception as exc:  # noqa: BLE001 - CLI boundary
            result.error = str(exc)
            result.passed = False
            _print_step_summary(result)
        results.append(result)
    return results


def cmd_validate(
    coins: list[str],
    *,
    interval: str,
    dry_run: bool = False,
    data_root: Path = DEFAULT_DATA_ROOT,
) -> list[StepResult]:
    if interval not in SUPPORTED_VALIDATE_INTERVALS:
        raise ValueError(f"Unsupported interval {interval!r}")

    results: list[StepResult] = []
    normalized = _normalize_coins(coins)
    for index, coin in enumerate(normalized):
        result = StepResult(action="validate", coin=coin, intervals=[interval])
        input_path = str(bid_ask_validator.bid_ask_csv_path(coin, interval, data_root))
        result.output_paths = [input_path]

        if dry_run:
            result.report_paths = [
                str(data_root / "reports" / f"{coin}_{interval}_validation_<timestamp>.json")
            ]
            result.warnings.append("dry-run: no files written")
            next_cmd = None
            if index == len(normalized) - 1 and interval == "1s":
                next_cmd = (
                    f"python3 bid_ask_data_pipeline.py resample --coins {' '.join(normalized)} "
                    "--from-interval 1s --to-interval all"
                )
            _print_step_summary(result, next_command=next_cmd)
            results.append(result)
            continue

        try:
            report = bid_ask_validator.validate_coin_interval(
                coin,
                interval,
                data_root=data_root,
            )
            result.report_paths = [report.report_path]
            result.row_counts[interval] = report.row_count
            result.warnings.extend(report.warnings)
            result.passed = report.passed
            next_cmd = None
            if interval == "1s":
                next_cmd = (
                    f"python3 bid_ask_data_pipeline.py resample --coins {coin} "
                    "--from-interval 1s --to-interval all"
                )
            elif interval == "15s":
                next_cmd = (
                    f"python3 bid_ask_data_pipeline.py train --coin {coin} "
                    "--interval 15s --horizon 1"
                )
            _print_step_summary(result, next_command=next_cmd)
        except Exception as exc:  # noqa: BLE001 - CLI boundary
            result.error = str(exc)
            result.passed = False
            _print_step_summary(result)
        results.append(result)
    return results


def cmd_resample(
    coins: list[str],
    *,
    from_interval: str = "1s",
    to_interval: str,
    dry_run: bool = False,
    data_root: Path = DEFAULT_DATA_ROOT,
) -> list[StepResult]:
    if from_interval not in SUPPORTED_RESAMPLE_FROM:
        raise ValueError(f"Resample supports from-interval 1s only; got {from_interval!r}")
    if to_interval not in SUPPORTED_RESAMPLE_TO:
        raise ValueError(f"Unsupported to-interval {to_interval!r}")

    intervals = list(RESAMPLE_ALL_INTERVALS) if to_interval == "all" else [to_interval]
    results: list[StepResult] = []
    normalized = _normalize_coins(coins)

    for coin in normalized:
        result = StepResult(action="resample", coin=coin, intervals=intervals)
        if dry_run:
            result.output_paths = [
                str(bid_ask_resampler.resampled_csv_path(coin, interval, data_root))
                for interval in intervals
            ]
            result.report_paths = [
                str(data_root / "reports" / f"{coin}_{interval}_resample_<timestamp>.json")
                for interval in intervals
            ]
            result.warnings.append("dry-run: no files written")
            _print_step_summary(
                result,
                next_command=(
                    f"python3 bid_ask_data_pipeline.py validate --coins {coin} "
                    f"--interval {intervals[0]}"
                ),
            )
            results.append(result)
            continue

        try:
            resample_results = bid_ask_resampler.resample_coin(
                coin,
                intervals=tuple(intervals),
                data_root=data_root,
            )
            for interval, paths in resample_results.items():
                result.output_paths.append(str(paths["output_path"]))
                result.report_paths.append(str(paths["report_path"]))
            for interval in intervals:
                report_path = resample_results[interval]["report_path"]
                if report_path.is_file():
                    payload = json.loads(report_path.read_text(encoding="utf-8"))
                    result.row_counts[interval] = int(payload.get("output_rows", 0))
            _print_step_summary(
                result,
                next_command=(
                    f"python3 bid_ask_data_pipeline.py validate --coins {coin} "
                    f"--interval {intervals[0]}"
                ),
            )
        except Exception as exc:  # noqa: BLE001 - CLI boundary
            result.error = str(exc)
            result.passed = False
            _print_step_summary(result)
        results.append(result)
    return results


def cmd_train(
    coin: str,
    *,
    interval: str,
    horizon: int = 1,
    dry_run: bool = False,
    data_root: Path = DEFAULT_DATA_ROOT,
    reports_dir: Path = DEFAULT_TRAINING_REPORTS_DIR,
    models_dir: Path = DEFAULT_MODELS_DIR,
) -> StepResult:
    if interval not in SUPPORTED_TRAIN_INTERVALS:
        raise ValueError(f"Training supports intervals 15s and 1m only; got {interval!r}")

    normalized_coin = coin.strip().upper()
    result = StepResult(action="train", coin=normalized_coin, intervals=[interval])
    _print_data_coverage()

    source_path = str(bid_ask_resampler.resampled_csv_path(normalized_coin, interval, data_root))
    result.output_paths = [source_path]

    if dry_run:
        result.report_paths = [str(reports_dir / "<report_id>.json")]
        result.warnings.append("dry-run: no files written")
        _print_step_summary(result)
        return result

    try:
        pipeline_result = bid_ask_train.run_training_pipeline(
            coin=normalized_coin,
            interval=interval,
            horizon=horizon,
            data_root=data_root,
            reports_dir=reports_dir,
            models_dir=models_dir,
        )
        evaluation = pipeline_result["evaluation"]
        report = pipeline_result["report"]
        result.row_counts["train"] = evaluation["train_rows"]
        result.row_counts["test"] = evaluation["test_rows"]
        result.warnings.extend(evaluation.get("warnings", []))
        if pipeline_result.get("report_path") is not None:
            result.report_paths = [str(pipeline_result["report_path"])]
        if report.get("model_path"):
            result.output_paths.append(str(report["model_path"]))
        result.passed = True
        _print_step_summary(result)
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        result.error = str(exc)
        result.passed = False
        _print_step_summary(result)
    return result


def run_one_coin_workflow(
    coin: str,
    *,
    train_interval: str = "15s",
    horizon: int = 1,
    dry_run: bool = False,
    strict: bool = False,
    data_root: Path = DEFAULT_DATA_ROOT,
    market_stream_dir: Path = DEFAULT_MARKET_STREAM_DIR,
    reports_dir: Path = DEFAULT_TRAINING_REPORTS_DIR,
    models_dir: Path = DEFAULT_MODELS_DIR,
) -> tuple[list[StepResult], bool]:
    """Run extract -> validate 1s -> resample -> validate 15s/1m -> train."""
    all_results: list[StepResult] = []
    success = True

    extract_results = cmd_extract(
        [coin],
        interval="1s",
        dry_run=dry_run,
        data_root=data_root,
        market_stream_dir=market_stream_dir,
    )
    all_results.extend(extract_results)
    if any(r.error for r in extract_results):
        return all_results, False

    validate_1s_results = cmd_validate(
        [coin],
        interval="1s",
        dry_run=dry_run,
        data_root=data_root,
    )
    all_results.extend(validate_1s_results)
    validate_1s = validate_1s_results[0] if validate_1s_results else None
    if validate_1s and validate_1s.error:
        return all_results, False
    if validate_1s and _strict_blocks_continue(bool(validate_1s.passed), strict=strict):
        validate_1s.warnings.append("strict: stopping before resample/training")
        success = False
        return all_results, success

    resample_results = cmd_resample(
        [coin],
        from_interval="1s",
        to_interval="all",
        dry_run=dry_run,
        data_root=data_root,
    )
    all_results.extend(resample_results)
    if any(r.error for r in resample_results):
        return all_results, False

    for interval in ("15s", "1m"):
        validate_results = cmd_validate(
            [coin],
            interval=interval,
            dry_run=dry_run,
            data_root=data_root,
        )
        all_results.extend(validate_results)
        step = validate_results[0] if validate_results else None
        if step and step.error:
            success = False
            if interval == train_interval and strict:
                step.warnings.append("strict: stopping before training")
                return all_results, success
            continue
        if (
            interval == train_interval
            and step
            and _strict_blocks_continue(bool(step.passed), strict=strict)
        ):
            step.warnings.append("strict: stopping before training")
            success = False
            return all_results, success

    train_result = cmd_train(
        coin,
        interval=train_interval,
        horizon=horizon,
        dry_run=dry_run,
        data_root=data_root,
        reports_dir=reports_dir,
        models_dir=models_dir,
    )
    all_results.append(train_result)
    if train_result.error:
        success = False
    return all_results, success


def cmd_run_one(
    coin: str,
    *,
    train_interval: str = "15s",
    horizon: int = 1,
    dry_run: bool = False,
    strict: bool = False,
    data_root: Path = DEFAULT_DATA_ROOT,
    market_stream_dir: Path = DEFAULT_MARKET_STREAM_DIR,
    reports_dir: Path = DEFAULT_TRAINING_REPORTS_DIR,
    models_dir: Path = DEFAULT_MODELS_DIR,
) -> bool:
    normalized = coin.strip().upper()
    print(f"action=run-one coin={normalized} train_interval={train_interval} horizon={horizon}")
    _print_data_coverage()
    _, success = run_one_coin_workflow(
        normalized,
        train_interval=train_interval,
        horizon=horizon,
        dry_run=dry_run,
        strict=strict,
        data_root=data_root,
        market_stream_dir=market_stream_dir,
        reports_dir=reports_dir,
        models_dir=models_dir,
    )
    print(f"run-one_complete coin={normalized} success={success}")
    return success


def cmd_run_all(
    coins: list[str],
    *,
    train_interval: str = "15s",
    horizon: int = 1,
    dry_run: bool = False,
    strict: bool = False,
    data_root: Path = DEFAULT_DATA_ROOT,
    market_stream_dir: Path = DEFAULT_MARKET_STREAM_DIR,
    reports_dir: Path = DEFAULT_TRAINING_REPORTS_DIR,
    models_dir: Path = DEFAULT_MODELS_DIR,
) -> dict[str, bool]:
    normalized = _normalize_coins(coins)
    total = len(normalized)
    outcomes: dict[str, bool] = {}

    print(
        f"action=run-all coins={','.join(normalized)} "
        f"train_interval={train_interval} horizon={horizon}"
    )
    _print_data_coverage()

    for index, coin in enumerate(normalized, start=1):
        print(f"[{index}/{total}] {coin}")
        try:
            _, success = run_one_coin_workflow(
                coin,
                train_interval=train_interval,
                horizon=horizon,
                dry_run=dry_run,
                strict=strict,
                data_root=data_root,
                market_stream_dir=market_stream_dir,
                reports_dir=reports_dir,
                models_dir=models_dir,
            )
            outcomes[coin] = success
        except Exception as exc:  # noqa: BLE001 - CLI boundary
            outcomes[coin] = False
            print(f"error={exc}")

    succeeded = [coin for coin, ok in outcomes.items() if ok]
    failed = [coin for coin, ok in outcomes.items() if not ok]
    print("run-all_summary")
    print(f"succeeded={','.join(succeeded) if succeeded else '(none)'}")
    print(f"failed={','.join(failed) if failed else '(none)'}")
    return outcomes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Local bid/ask data pipeline: extract, validate, resample, and train "
            "(read-only, research-only, paper-only)."
        ),
    )

    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned steps without writing output files",
    )
    parent.add_argument(
        "--strict",
        action="store_true",
        help="Stop resample/training for a coin when validation fails",
    )
    parent.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Root directory for bid/ask CSV output (default: data/bid_ask_csv)",
    )
    parent.add_argument(
        "--market-stream-dir",
        type=Path,
        default=DEFAULT_MARKET_STREAM_DIR,
        help="Local market stream directory for extraction (default: data/market_stream)",
    )
    parent.add_argument(
        "--reports-dir",
        type=Path,
        default=DEFAULT_TRAINING_REPORTS_DIR,
        help="Training reports directory (default: reports/bid_ask_training)",
    )
    parent.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help="Sklearn model output directory (default: models/bid_ask)",
    )
    parent.add_argument(
        "--historical",
        action="store_true",
        help="Request historical exchange backfill (not implemented; uses local data)",
    )
    parent.add_argument(
        "--all-time",
        action="store_true",
        help="Request all-time exchange history (not implemented; uses local data)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    extract_parser = subparsers.add_parser(
        "extract",
        parents=[parent],
        help="Extract 1s bid/ask CSV from local data",
    )
    extract_parser.add_argument("--coins", nargs="+", required=True)
    extract_parser.add_argument("--source", default="local")
    extract_parser.add_argument("--interval", default="1s")

    validate_parser = subparsers.add_parser(
        "validate",
        parents=[parent],
        help="Validate bid/ask CSV files",
    )
    validate_parser.add_argument("--coins", nargs="+", required=True)
    validate_parser.add_argument("--interval", required=True)

    resample_parser = subparsers.add_parser(
        "resample",
        parents=[parent],
        help="Resample 1s CSV to 15s or 1m",
    )
    resample_parser.add_argument("--coins", nargs="+", required=True)
    resample_parser.add_argument("--from-interval", default="1s")
    resample_parser.add_argument("--to-interval", required=True)

    train_parser = subparsers.add_parser(
        "train",
        parents=[parent],
        help="Train bid/ask movement model",
    )
    train_parser.add_argument("--coin", required=True)
    train_parser.add_argument("--interval", required=True)
    train_parser.add_argument("--horizon", type=int, default=1)

    run_one_parser = subparsers.add_parser(
        "run-one",
        parents=[parent],
        help="Full local workflow for one coin",
    )
    run_one_parser.add_argument("--coin", required=True)
    run_one_parser.add_argument("--train-interval", default="15s")
    run_one_parser.add_argument("--horizon", type=int, default=1)

    run_all_parser = subparsers.add_parser(
        "run-all",
        parents=[parent],
        help="Full local workflow for multiple coins",
    )
    run_all_parser.add_argument("--coins", nargs="+", required=True)
    run_all_parser.add_argument("--train-interval", default="15s")
    run_all_parser.add_argument("--horizon", type=int, default=1)

    return parser


def _pipeline_options(args: argparse.Namespace) -> dict[str, object]:
    return {
        "dry_run": args.dry_run,
        "data_root": args.data_root,
        "market_stream_dir": args.market_stream_dir,
        "reports_dir": args.reports_dir,
        "models_dir": args.models_dir,
    }


def dispatch(args: argparse.Namespace) -> int:
    if args.historical or args.all_time:
        print(HISTORICAL_WARNING)

    options = _pipeline_options(args)

    if args.command == "extract":
        cmd_extract(
            args.coins,
            source=args.source,
            interval=args.interval,
            dry_run=bool(options["dry_run"]),
            data_root=options["data_root"],  # type: ignore[arg-type]
            market_stream_dir=options["market_stream_dir"],  # type: ignore[arg-type]
        )
        return 0

    if args.command == "validate":
        cmd_validate(
            args.coins,
            interval=args.interval,
            dry_run=bool(options["dry_run"]),
            data_root=options["data_root"],  # type: ignore[arg-type]
        )
        return 0

    if args.command == "resample":
        cmd_resample(
            args.coins,
            from_interval=args.from_interval,
            to_interval=args.to_interval,
            dry_run=bool(options["dry_run"]),
            data_root=options["data_root"],  # type: ignore[arg-type]
        )
        return 0

    if args.command == "train":
        result = cmd_train(
            args.coin,
            interval=args.interval,
            horizon=args.horizon,
            dry_run=bool(options["dry_run"]),
            data_root=options["data_root"],  # type: ignore[arg-type]
            reports_dir=options["reports_dir"],  # type: ignore[arg-type]
            models_dir=options["models_dir"],  # type: ignore[arg-type]
        )
        return 0 if result.error is None else 1

    if args.command == "run-one":
        success = cmd_run_one(
            args.coin,
            train_interval=args.train_interval,
            horizon=args.horizon,
            dry_run=bool(options["dry_run"]),
            strict=args.strict,
            data_root=options["data_root"],  # type: ignore[arg-type]
            market_stream_dir=options["market_stream_dir"],  # type: ignore[arg-type]
            reports_dir=options["reports_dir"],  # type: ignore[arg-type]
            models_dir=options["models_dir"],  # type: ignore[arg-type]
        )
        return 0 if success else 1

    if args.command == "run-all":
        outcomes = cmd_run_all(
            args.coins,
            train_interval=args.train_interval,
            horizon=args.horizon,
            dry_run=bool(options["dry_run"]),
            strict=args.strict,
            data_root=options["data_root"],  # type: ignore[arg-type]
            market_stream_dir=options["market_stream_dir"],  # type: ignore[arg-type]
            reports_dir=options["reports_dir"],  # type: ignore[arg-type]
            models_dir=options["models_dir"],  # type: ignore[arg-type]
        )
        return 0 if all(outcomes.values()) else 1

    raise ValueError(f"Unknown command: {args.command}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return dispatch(args)
    except ValueError as exc:
        print(f"error={exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
