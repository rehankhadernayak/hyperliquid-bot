"""Chronological train/test splits for offline experiments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChronologicalSplit:
    """Development and out-of-sample timestamp partitions."""

    development: tuple[str, ...]
    out_of_sample: tuple[str, ...]
    development_pct: float
    label: str = "holdout"

    @property
    def combined(self) -> tuple[str, ...]:
        return self.development + self.out_of_sample


def split_timestamps_chronologically(
    timestamps: list[str],
    *,
    development_pct: float = 0.7,
) -> ChronologicalSplit:
    """
    Split timestamps in file order — never shuffle.

    Default: first 70% development, final 30% out-of-sample.
    """
    if not timestamps:
        return ChronologicalSplit((), (), development_pct)

    if len(timestamps) == 1:
        return ChronologicalSplit((timestamps[0],), (), development_pct)

    split_index = max(1, int(len(timestamps) * development_pct))
    if split_index >= len(timestamps):
        split_index = len(timestamps) - 1

    development = tuple(timestamps[:split_index])
    out_of_sample = tuple(timestamps[split_index:])
    return ChronologicalSplit(development, out_of_sample, development_pct)


@dataclass(frozen=True)
class WalkForwardFold:
    """One walk-forward evaluation fold."""

    fold_index: int
    development: tuple[str, ...]
    evaluation: tuple[str, ...]


def walk_forward_folds(
    timestamps: list[str],
    *,
    min_development_timestamps: int = 3,
    evaluation_window: int = 2,
) -> list[WalkForwardFold]:
    """
    Build expanding-window walk-forward folds.

    Each fold uses all prior timestamps as development and the next fixed-size
    window as evaluation — no future data in development windows.
    """
    if len(timestamps) <= min_development_timestamps:
        return []

    folds: list[WalkForwardFold] = []
    fold_index = 0
    start_eval = min_development_timestamps

    while start_eval + evaluation_window <= len(timestamps):
        development = tuple(timestamps[:start_eval])
        evaluation = tuple(timestamps[start_eval : start_eval + evaluation_window])
        folds.append(
            WalkForwardFold(
                fold_index=fold_index,
                development=development,
                evaluation=evaluation,
            )
        )
        fold_index += 1
        start_eval += evaluation_window

    return folds
