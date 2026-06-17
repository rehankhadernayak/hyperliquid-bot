"""Tests for central configuration defaults and data paths."""

from config import DATA, DEFAULT_BACKTEST, PAPER_TRADING, VERSION


def test_version_is_paper_release():
    assert VERSION == "9.0.0-review"


def test_live_session_trade_limit_defaults_to_10():
    assert PAPER_TRADING.max_successful_trades_per_session == 10


def test_backtest_session_trade_limit_defaults_to_1000():
    assert DEFAULT_BACKTEST.max_session_trades == 1000


def test_required_data_paths_are_defined():
    assert DATA.data_dir.name == "data"
    assert DATA.price_log_csv.name == "price_log.csv"
    assert DATA.paper_trades_csv.name == "paper_trades.csv"
    assert DATA.backtest_results_json.name == "backtest_results.json"
    assert DATA.equity_log_csv.name == "equity_log.csv"
    assert DATA.price_log_csv.parent == DATA.data_dir
    assert DATA.paper_trades_csv.parent == DATA.data_dir
    assert DATA.backtest_results_json.parent == DATA.data_dir
    assert DATA.equity_log_csv.parent == DATA.data_dir
