#!/usr/bin/env python3
"""
Generate a read-only quality report for a WebSocket recorder session.

Paper-only, research-only. Reads manifest and data files; writes separate
quality report files only. Does not modify raw, normalized, or session data.

Usage:
    python3 market_data_quality_report.py --session-id <RECORDER_SESSION_ID>
    python3 market_data_quality_report.py --latest
    python3 market_data_quality_report.py --session-id <ID> --format json
    python3 market_data_quality_report.py --session-id <ID> --format markdown
"""

from __future__ import annotations

import sys

from market_data.quality_cli import main

if __name__ == "__main__":
    sys.exit(main())
