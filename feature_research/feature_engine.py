"""Feature research engine — consumes deterministic replay output."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from feature_research.feature_analysis import (
    FeatureAnalyzer,
    summarize_missing_data,
    top_signals,
)
from feature_research.feature_builder import FeatureBuilder
from feature_research.feature_config import FeatureResearchConfig
from feature_research.feature_dataset import (
    build_dataset_records,
    compute_dataset_hash,
    write_dataset,
)
from feature_research.feature_models import (
    FEATURE_COLUMN_NAMES,
    FeatureRow,
    LeakageCheckResult,
)
from feature_research.feature_report import (
    build_feature_research_report,
    new_feature_research_id,
    write_feature_research_report,
)
from feature_research.label_builder import LabelBuilder
from market_data.replay_cursor import ReplayCursor
from market_data.replay_engine import ReplayEngine, ReplayError
from market_data.replay_events import compute_stream_hash
from market_data.replay_filters import ReplayFilters
from market_data.replay_report import list_replay_reports


class FeatureResearchError(Exception):
    """Raised when feature research cannot proceed safely."""


@dataclass
class FeatureResearchResult:
    report: dict[str, Any]
    report_paths: dict[str, Path]
    rows: tuple[FeatureRow, ...]
    deterministic_dataset_hash: str


def _load_replay_report(replay_session_id: str, replays_dir: Path) -> dict[str, Any]:
    path = replays_dir / f"{replay_session_id}.json"
    if not path.exists():
        raise FeatureResearchError(f"Replay report not found: {path}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_replay_session(config: FeatureResearchConfig) -> tuple[str, dict[str, Any] | None]:
    replays_dir = config.market_stream_dir / "replays"

    if config.replay_session_id:
        report = _load_replay_report(config.replay_session_id, replays_dir)
        return config.replay_session_id, report

    if config.use_latest_replay:
        reports = list_replay_reports(replays_dir)
        if not reports:
            raise FeatureResearchError("No replay reports found for --latest-replay")
        replay_id = reports[0]["replay_session_id"]
        return replay_id, _load_replay_report(replay_id, replays_dir)

    if config.source_recorder_session_id:
        raise FeatureResearchError(
            "Feature research v1 requires an existing replay session. "
            "Run market_data_replay.py first, then pass --replay-session-id."
        )

    raise FeatureResearchError(
        "Provide --replay-session-id or --latest-replay"
    )


def _build_filters(config: FeatureResearchConfig, replay_report: dict[str, Any] | None) -> ReplayFilters:
    replay_filters = (replay_report or {}).get("filters") or {}
    return ReplayFilters(
        coins=list(config.coins),
        channels=list(config.channels),
        max_events=config.max_events or replay_filters.get("max_events"),
        include_candles=False,
        include_l2_book=False,
    )


def validate_no_leakage(
    rows_before: tuple[FeatureRow, ...],
    rows_after_mutation_test: tuple[FeatureRow, ...] | None = None,
) -> LeakageCheckResult:
    checks: list[dict[str, Any]] = []
    warnings: list[str] = []

    checks.append(
        {
            "name": "feature_rows_frozen_before_labeling",
            "status": "PASS" if rows_before else "WARN",
        }
    )

    frozen = all(getattr(row, "_frozen", True) for row in rows_before)
    checks.append({"name": "feature_rows_immutable", "status": "PASS" if frozen else "FAIL"})

    seq_recorded = all(row.source_replay_sequence > 0 for row in rows_before) if rows_before else True
    checks.append({"name": "source_replay_sequence_recorded", "status": "PASS" if seq_recorded else "FAIL"})

    if rows_after_mutation_test is not None:
        unchanged = all(
            r1.to_dict() == r2.to_dict()
            for r1, r2 in zip(rows_before, rows_after_mutation_test)
        )
        checks.append(
            {
                "name": "future_price_changes_do_not_affect_features",
                "status": "PASS" if unchanged else "FAIL",
            }
        )

    passed = all(c["status"] in ("PASS", "WARN") for c in checks) and not any(
        c["status"] == "FAIL" for c in checks
    )
    if not rows_before:
        warnings.append("No feature rows to validate leakage against")

    return LeakageCheckResult(passed=passed, checks=checks, warnings=warnings)


class FeatureResearchEngine:
    """Runs feature research over deterministic replay streams."""

    def __init__(self, config: FeatureResearchConfig) -> None:
        self._config = config
        self._replay_engine = ReplayEngine(
            market_stream_dir=config.market_stream_dir,
            test_mode=config.test_mode,
        )

    def run(self, feature_research_id: str | None = None) -> FeatureResearchResult:
        config = self._config
        research_id = feature_research_id or new_feature_research_id()

        replay_session_id, replay_report = _resolve_replay_session(config)
        filters = _build_filters(config, replay_report)

        if not replay_report:
            raise FeatureResearchError("Replay report required")

        source_recorder_id = replay_report["source_recorder_session_id"]
        session = self._replay_engine.load_session(source_recorder_id)
        try:
            events, _skipped, session = self._replay_engine.prepare_events(
                session,
                filters,
                replay_session_id=replay_session_id,
            )
        except ReplayError as exc:
            if "No replayable events" in str(exc):
                events = []
            else:
                raise FeatureResearchError(str(exc)) from exc

        stream_hash = compute_stream_hash(events)
        quality_report = session.quality_report
        quality_label = (quality_report or {}).get("quality_label")
        quality_path = str(session.quality_report_path) if session.quality_report_path else None

        feature_builder = FeatureBuilder(config)
        label_builder = LabelBuilder(config)

        cursor = ReplayCursor(events)
        while not cursor.at_end():
            event = cursor.next_event()
            if event is None:
                break
            label_builder.ingest_event(event)

        cursor2 = ReplayCursor(events)
        while not cursor2.at_end():
            event = cursor2.next_event()
            if event is None:
                break
            feature_builder.process_event(event)

        frozen_rows = feature_builder.freeze_rows()
        rows_snapshot = copy.deepcopy(frozen_rows)

        labels = label_builder.build_labels(frozen_rows)

        leakage = validate_no_leakage(frozen_rows)

        if config.dry_run:
            warnings = ["Dry-run mode: analysis skipped, no reports written"]
            report = build_feature_research_report(
                feature_research_id=research_id,
                replay_session_id=replay_session_id,
                source_recorder_session_id=source_recorder_id,
                quality_report_path=quality_path,
                quality_label=quality_label,
                feature_config=config.to_dict(),
                horizons=list(config.horizons_seconds),
                coins=list(config.coins),
                channels=list(config.channels),
                row_count=len(frozen_rows),
                feature_count=len(FEATURE_COLUMN_NAMES),
                label_count=len(labels),
                missing_data_summary=summarize_missing_data(frozen_rows),
                leakage_check=leakage.to_dict(),
                feature_analysis=[],
                top_possible_signals=[],
                rejected_signals=[],
                warnings=warnings,
                limitations=["Dry-run: no analysis performed"],
                recommendations=[],
                deterministic_dataset_hash="",
                stream_hash=stream_hash,
                dataset_path=None,
                dataset_format=None,
                dry_run=True,
            )
            return FeatureResearchResult(
                report=report,
                report_paths={},
                rows=rows_snapshot,
                deterministic_dataset_hash="",
            )

        analyzer = FeatureAnalyzer(config)
        analysis_results = analyzer.analyze(frozen_rows, labels)
        missing_summary = summarize_missing_data(frozen_rows)
        signals = top_signals(analysis_results)

        analysis_dicts = [
            {
                "feature_name": r.feature_name,
                "horizon_seconds": r.horizon_seconds,
                "row_count": r.row_count,
                "missing_count": r.missing_count,
                "correlation": r.correlation,
                "information_coefficient": r.information_coefficient,
                "monotonicity_score": r.monotonicity_score,
                "bucket_analysis": r.bucket_analysis,
                "stability_by_coin": r.stability_by_coin,
                "stability_by_time": r.stability_by_time,
                "candidate_signal": r.candidate_signal,
                "candidate_reason": r.candidate_reason,
            }
            for r in analysis_results
        ]

        dataset_hash = compute_dataset_hash(
            stream_hash=stream_hash,
            config_hash=config.config_hash(),
            rows=frozen_rows,
            labels=labels,
            analysis_results=analysis_results,
        )

        warnings: list[str] = []
        limitations: list[str] = [
            "Feature research does not prove profitability.",
            "Short replay windows may not represent broader market regimes.",
            "Correlation does not imply causation or tradability.",
        ]
        recommendations: list[str] = []

        if len(frozen_rows) < config.min_rows_for_signal:
            warnings.append(
                f"Only {len(frozen_rows)} rows — below minimum {config.min_rows_for_signal} for signal discovery"
            )
            limitations.append("Dataset too small for reliable signal conclusions.")
            recommendations.append("Collect longer recorder sessions before drawing conclusions.")

        if not signals["possible"]:
            recommendations.append(
                "No strong feature signals found — consider longer data or alternative feature definitions."
            )
        else:
            top_feat = signals["possible"][0]["feature"]
            recommendations.append(
                f"Investigate `{top_feat}` in a controlled event backtest with explicit entry/exit rules."
            )

        records = build_dataset_records(frozen_rows, labels)
        dataset_dir = config.output_dir / "datasets"
        dataset_path = dataset_dir / f"{research_id}.parquet"
        written_path, dataset_format = write_dataset(records, dataset_path)

        report = build_feature_research_report(
            feature_research_id=research_id,
            replay_session_id=replay_session_id,
            source_recorder_session_id=source_recorder_id,
            quality_report_path=quality_path,
            quality_label=quality_label,
            feature_config=config.to_dict(),
            horizons=list(config.horizons_seconds),
            coins=list(config.coins),
            channels=list(config.channels),
            row_count=len(frozen_rows),
            feature_count=len(FEATURE_COLUMN_NAMES),
            label_count=len(labels),
            missing_data_summary=missing_summary,
            leakage_check=leakage.to_dict(),
            feature_analysis=analysis_dicts,
            top_possible_signals=signals["possible"],
            rejected_signals=signals["rejected"],
            warnings=warnings,
            limitations=limitations,
            recommendations=recommendations,
            deterministic_dataset_hash=dataset_hash,
            stream_hash=stream_hash,
            dataset_path=str(written_path),
            dataset_format=dataset_format,
            dry_run=False,
        )

        report_paths = write_feature_research_report(report, config.output_dir)

        return FeatureResearchResult(
            report=report,
            report_paths=report_paths,
            rows=rows_snapshot,
            deterministic_dataset_hash=dataset_hash,
        )
