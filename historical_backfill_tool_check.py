"""Diagnostic for local tools needed for manual Hyperliquid historical backfill.

Read-only check using shutil.which only. Does not install tools, download data,
or execute shell commands.
"""

from __future__ import annotations

import shutil
from typing import Any

BREW_MISSING_WARNING = (
    "brew is not installed. On macOS, brew is one possible install method, "
    "but installation instructions may need another method."
)

BULK_DOWNLOAD_WARNING = (
    "Do not bulk download archive data. Start with one coin and one hour."
)

ALL_TIME_BBO_NOT_CLAIMED = (
    "All-time 1-second bid/ask is not directly available and is not claimed."
)


def check_tool_available(tool_name: str) -> bool:
    """Return True when tool_name is found on PATH via shutil.which."""
    return shutil.which(tool_name) is not None


def check_backfill_tools() -> dict[str, Any]:
    """Check brew/aws/lz4 availability and whether manual download is possible."""
    brew_available = check_tool_available("brew")
    aws_available = check_tool_available("aws")
    lz4_available = check_tool_available("lz4")

    missing_tools: list[str] = []
    if not aws_available:
        missing_tools.append("aws")
    if not lz4_available:
        missing_tools.append("lz4")

    warnings: list[str] = []
    if not brew_available:
        warnings.append(BREW_MISSING_WARNING)
    warnings.append(BULK_DOWNLOAD_WARNING)
    warnings.append(ALL_TIME_BBO_NOT_CLAIMED)

    return {
        "brew_available": brew_available,
        "aws_available": aws_available,
        "lz4_available": lz4_available,
        "ready_for_manual_download": aws_available and lz4_available,
        "missing_tools": missing_tools,
        "warnings": warnings,
    }


def _status_label(available: bool) -> str:
    return "available" if available else "missing"


def _next_action(ready: bool) -> str:
    if ready:
        return (
            "Run historical_backfill_manual_guide.py for one-hour manual download "
            "commands, or continue with the local BBO CSV recorder workflow."
        )
    return (
        "Install required tools manually or continue with the local BBO CSV recorder workflow."
    )


def print_tool_check() -> None:
    """Print tool status, missing tools, next action, and safety warnings."""
    result = check_backfill_tools()
    print("Historical Backfill Tool Check")
    print(f"brew: {_status_label(result['brew_available'])}")
    print(f"aws: {_status_label(result['aws_available'])}")
    print(f"lz4: {_status_label(result['lz4_available'])}")
    print(f"ready_for_manual_download: {str(result['ready_for_manual_download']).lower()}")

    if result["missing_tools"]:
        print(f"missing_tools: {', '.join(result['missing_tools'])}")

    print(f"Next: {_next_action(result['ready_for_manual_download'])}")

    for warning in result["warnings"]:
        print(f"Warning: {warning}")


def main() -> int:
    print_tool_check()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
