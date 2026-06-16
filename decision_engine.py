"""
Read-only decision and recommendation layer for paper trading.

Converts market signals, positions, and risk state into per-coin recommendations.
Does not execute trades, sign transactions, or call exchange order endpoints.

Confidence labels are rule-based guidance only — not a probability of profit.
"""

from dataclasses import dataclass

from config import MARKET, PAPER_TRADING
from paper_trader import BUY_SIGNALS, SELL_SIGNALS, PaperTrader, Position
from risk_rules import buy_block_reason, sell_block_reason

RECOMMENDATION_BUY = "PAPER BUY"
RECOMMENDATION_SELL = "PAPER SELL"
RECOMMENDATION_HOLD = "HOLD"
RECOMMENDATION_BLOCKED = "BLOCKED"

CONFIDENCE_HIGH = "HIGH"
CONFIDENCE_MEDIUM = "MEDIUM"
CONFIDENCE_LOW = "LOW"

BUY_SIGNALS_SET = BUY_SIGNALS
SELL_SIGNALS_SET = SELL_SIGNALS


@dataclass
class CoinDecision:
    """One coin's read-only trading recommendation."""

    coin: str
    recommendation: str
    signal: str
    signal_strength: str
    current_price: float | None
    change_1m: float | None
    change_5m: float | None
    current_position_value: float
    per_coin_allocation_limit: float
    risk_status: str
    primary_reason: str
    supporting_reasons: list[str]
    confidence: str


@dataclass
class DecisionSessionSummary:
    """Session-level counts for recommendations and paper trading state."""

    active_recommendations: int
    blocked_recommendations: int
    open_positions: int
    session_trades: int
    total_equity: float | None
    current_drawdown_pct: float | None


def signal_strength_label(signal: str) -> str:
    """Map a signal name to a simple strength label."""
    if signal in ("STRONG_UP", "STRONG_DOWN"):
        return "STRONG"
    if signal in ("UP", "DOWN"):
        return "MODERATE"
    if signal == "FLAT":
        return "WEAK"
    return "NONE"


def confidence_label(
    signal: str,
    recommendation: str,
) -> str:
    """
    Rule-based confidence (not a profit probability).

    HIGH   — STRONG_DOWN / STRONG_UP with PAPER BUY or PAPER SELL and risk OK
    MEDIUM — DOWN / UP with PAPER BUY or PAPER SELL and risk OK;
             or STRONG signal blocked by risk
    LOW    — HOLD, missing data, FLAT / N/A, or weaker blocked cases
    """
    if recommendation in (RECOMMENDATION_BUY, RECOMMENDATION_SELL):
        if signal in ("STRONG_DOWN", "STRONG_UP"):
            return CONFIDENCE_HIGH
        return CONFIDENCE_MEDIUM

    if recommendation == RECOMMENDATION_BLOCKED and signal in (
        "STRONG_DOWN",
        "STRONG_UP",
    ):
        return CONFIDENCE_MEDIUM

    return CONFIDENCE_LOW


def _get_position(trader: PaperTrader, coin: str) -> Position:
    return trader.positions.get(coin, Position())


def position_market_value(trader: PaperTrader, coin: str, price: float | None) -> float:
    """Current market value of an open position, or zero."""
    if price is None:
        return 0.0

    position = _get_position(trader, coin)
    if not position.is_open:
        return 0.0

    return position.quantity * price


def proposed_action(signal: str, has_open_position: bool) -> str | None:
    """
    Map a signal to a proposed recommendation before risk checks.

    Returns RECOMMENDATION_BUY, RECOMMENDATION_SELL, RECOMMENDATION_HOLD, or None
    when price data is unavailable.
    """
    if signal in BUY_SIGNALS_SET:
        return RECOMMENDATION_BUY

    if signal in SELL_SIGNALS_SET:
        if has_open_position:
            return RECOMMENDATION_SELL
        return RECOMMENDATION_HOLD

    if signal in ("FLAT", "N/A"):
        return RECOMMENDATION_HOLD

    return RECOMMENDATION_HOLD


def build_supporting_reasons(
    signal: str,
    recommendation: str,
    block_reason: str | None,
    trader: PaperTrader,
    coin: str,
) -> list[str]:
    """Build short supporting context lines for a decision."""
    reasons: list[str] = []

    if signal in BUY_SIGNALS_SET:
        reasons.append("Contrarian rule: buy after short-term price decline")
    elif signal in SELL_SIGNALS_SET:
        reasons.append("Take-profit rule: sell after short-term price rise")

    if recommendation == RECOMMENDATION_HOLD and signal in SELL_SIGNALS_SET:
        reasons.append("No open position to sell")

    if recommendation == RECOMMENDATION_HOLD and signal in ("FLAT", "N/A"):
        if signal == "FLAT":
            reasons.append("1-minute change is within the flat band")
        else:
            reasons.append("Waiting for enough price history")

    if recommendation == RECOMMENDATION_BLOCKED and block_reason:
        reasons.append(f"Risk rule: {block_reason}")

    position = _get_position(trader, coin)
    if recommendation == RECOMMENDATION_BUY and block_reason is None:
        reasons.append(
            f"Cash available: ${trader.cash_balance:,.2f} "
            f"(trade size ${PAPER_TRADING.trade_size_usdc:,.2f})"
        )
        reasons.append(
            f"Entries used: {position.entry_count}/{PAPER_TRADING.max_entries_per_coin}"
        )

    if recommendation == RECOMMENDATION_SELL and block_reason is None:
        reasons.append(f"Open position value on {coin}")

    return reasons


def primary_reason_for_decision(
    signal: str,
    recommendation: str,
    block_reason: str | None,
) -> str:
    """Return the main explanation shown for a coin decision."""
    if recommendation == RECOMMENDATION_BLOCKED and block_reason:
        return block_reason

    if recommendation == RECOMMENDATION_BUY:
        return f"{signal} signal — contrarian paper buy"

    if recommendation == RECOMMENDATION_SELL:
        return f"{signal} signal — paper sell with open position"

    if signal in SELL_SIGNALS_SET:
        return "no open position to sell"

    if signal == "FLAT":
        return "price change is flat — no action"

    if signal == "N/A":
        return "not enough history — hold"

    return "no actionable signal — hold"


def risk_status_label(recommendation: str, block_reason: str | None) -> str:
    """Short risk status for display."""
    if recommendation == RECOMMENDATION_BLOCKED:
        return f"BLOCKED — {block_reason or 'risk rule active'}"

    if recommendation in (RECOMMENDATION_BUY, RECOMMENDATION_SELL):
        return "OK — all risk checks passing"

    return "OK — no trade proposed"


def decide_for_coin(
    *,
    coin: str,
    signal: str,
    current_price: float | None,
    change_since_start: float | None,
    change_1m: float | None,
    change_5m: float | None,
    trader: PaperTrader,
    now: float,
) -> CoinDecision:
    """Build one coin decision from market data and paper trader state."""
    position = _get_position(trader, coin)
    has_open_position = position.is_open and current_price is not None
    position_value = position_market_value(trader, coin, current_price)

    if current_price is None or change_since_start is None:
        return CoinDecision(
            coin=coin,
            recommendation=RECOMMENDATION_HOLD,
            signal=signal,
            signal_strength=signal_strength_label(signal),
            current_price=current_price,
            change_1m=change_1m,
            change_5m=change_5m,
            current_position_value=position_value,
            per_coin_allocation_limit=PAPER_TRADING.max_allocation_per_coin_usdc,
            risk_status="OK — price data unavailable",
            primary_reason="price data unavailable — hold",
            supporting_reasons=["Waiting for a valid market price"],
            confidence=CONFIDENCE_LOW,
        )

    action = proposed_action(signal, has_open_position)
    block_reason: str | None = None

    if action == RECOMMENDATION_BUY:
        block_reason = buy_block_reason(trader, coin, now)
    elif action == RECOMMENDATION_SELL:
        block_reason = sell_block_reason(trader, coin)

    if action in (RECOMMENDATION_BUY, RECOMMENDATION_SELL) and block_reason:
        recommendation = RECOMMENDATION_BLOCKED
    else:
        recommendation = action or RECOMMENDATION_HOLD

    primary_reason = primary_reason_for_decision(signal, recommendation, block_reason)
    supporting = build_supporting_reasons(
        signal,
        recommendation,
        block_reason,
        trader,
        coin,
    )
    confidence = confidence_label(signal, recommendation)

    return CoinDecision(
        coin=coin,
        recommendation=recommendation,
        signal=signal,
        signal_strength=signal_strength_label(signal),
        current_price=current_price,
        change_1m=change_1m,
        change_5m=change_5m,
        current_position_value=position_value,
        per_coin_allocation_limit=PAPER_TRADING.max_allocation_per_coin_usdc,
        risk_status=risk_status_label(recommendation, block_reason),
        primary_reason=primary_reason,
        supporting_reasons=supporting,
        confidence=confidence,
    )


def current_drawdown_pct(trader: PaperTrader, current_prices: dict[str, float]) -> float | None:
    """
    Drawdown from session starting equity to current total equity.

    Returns None when equity cannot be calculated.
    """
    if not current_prices:
        return None

    equity = trader.total_equity(current_prices)
    starting = PAPER_TRADING.starting_balance_usdc

    if starting <= 0:
        return None

    if equity >= starting:
        return 0.0

    return ((starting - equity) / starting) * 100


def build_decisions(
    market_rows: list,
    trader: PaperTrader,
    current_prices: dict[str, float],
    now: float,
) -> tuple[list[CoinDecision], DecisionSessionSummary]:
    """Build all coin decisions and a session summary."""
    decisions = [
        decide_for_coin(
            coin=row.coin,
            signal=row.signal,
            current_price=row.current_price,
            change_since_start=row.change_since_start,
            change_1m=row.change_1m,
            change_5m=row.change_5m,
            trader=trader,
            now=now,
        )
        for row in market_rows
    ]

    active = sum(
        1
        for decision in decisions
        if decision.recommendation in (RECOMMENDATION_BUY, RECOMMENDATION_SELL)
    )
    blocked = sum(
        1 for decision in decisions if decision.recommendation == RECOMMENDATION_BLOCKED
    )
    open_positions = sum(
        1
        for coin in MARKET.watchlist
        if _get_position(trader, coin).is_open and current_prices.get(coin) is not None
    )

    total_equity = trader.total_equity(current_prices) if current_prices else None
    drawdown = current_drawdown_pct(trader, current_prices) if current_prices else None

    summary = DecisionSessionSummary(
        active_recommendations=active,
        blocked_recommendations=blocked,
        open_positions=open_positions,
        session_trades=trader.successful_trades_this_session,
        total_equity=total_equity,
        current_drawdown_pct=drawdown,
    )

    return decisions, summary
