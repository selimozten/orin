"""Market data connector using yfinance."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import pandas as pd


def get_returns(
    ticker: str,
    event_date: str,
    timeframes: list[str] | None = None,
) -> dict[str, float]:
    """Fetch actual returns for a ticker after an event date.

    Args:
        ticker: Stock ticker symbol (e.g. "AAPL").
        event_date: Date string in YYYY-MM-DD format.
        timeframes: List of timeframes to compute returns for.
            Supported: "1d", "5d", "20d". Defaults to ["1d"].

    Returns:
        Dict mapping timeframe to return (e.g. {"1d": 0.028}).
        Returns are signed floats (positive = up, negative = down).

    Requires: pip install orin[all]
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance not installed. Run: pip install orin[all]")

    if timeframes is None:
        timeframes = ["1d"]

    timeframe_days = {"1d": 1, "5d": 5, "20d": 20}
    max_days = max(timeframe_days.get(tf, 1) for tf in timeframes)

    dt = datetime.strptime(event_date, "%Y-%m-%d")
    start = dt - timedelta(days=5)
    end = dt + timedelta(days=max_days + 10)

    data = yf.download(
        ticker, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), progress=False
    )

    if data.empty:
        return {tf: 0.0 for tf in timeframes}

    # Handle multi-level columns from yfinance
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data.index = pd.to_datetime(data.index).tz_localize(None)

    # Find the close price on or just before the event date
    event_dt = pd.Timestamp(event_date)
    pre_event = data.loc[data.index <= event_dt]
    if pre_event.empty:
        return {tf: 0.0 for tf in timeframes}

    base_price = float(pre_event["Close"].iloc[-1])
    post_event = data.loc[data.index > event_dt]

    returns = {}
    for tf in timeframes:
        n_days = timeframe_days.get(tf, 1)
        if len(post_event) >= n_days:
            future_price = float(post_event["Close"].iloc[n_days - 1])
            returns[tf] = (future_price - base_price) / base_price
        elif len(post_event) > 0:
            future_price = float(post_event["Close"].iloc[-1])
            returns[tf] = (future_price - base_price) / base_price
        else:
            returns[tf] = 0.0

    return returns


def compute_outcome(
    ticker: str,
    event_date: str,
    timeframe: str = "1d",
) -> dict[str, Any]:
    """Compute a full outcome dict from market data.

    Args:
        ticker: Stock ticker symbol.
        event_date: Date of the event (YYYY-MM-DD).
        timeframe: Lookahead window ("1d", "5d", "20d").

    Returns:
        Outcome dict: {"direction": "up"/"down"/"flat",
                       "magnitude": float, "timeframe": str}
    """
    returns = get_returns(ticker, event_date, timeframes=[timeframe])
    ret = returns.get(timeframe, 0.0)

    if ret > 0.005:
        direction = "up"
    elif ret < -0.005:
        direction = "down"
    else:
        direction = "flat"

    return {
        "direction": direction,
        "magnitude": abs(ret),
        "timeframe": timeframe,
    }


def bulk_returns(
    records: list[dict[str, Any]],
    timeframe: str = "1d",
) -> list[dict[str, Any]]:
    """Enrich a list of records with real market outcomes.

    For each record with "ticker" and "date" fields, fetches the actual
    return and replaces the "outcome" field.

    Args:
        records: List of data records.
        timeframe: Lookahead window for returns.

    Returns:
        Records with updated "outcome" fields.
    """
    enriched = []
    for record in records:
        ticker = record.get("ticker", "")
        date = record.get("date", "")
        if ticker and date:
            try:
                outcome = compute_outcome(ticker, date, timeframe)
                record = {**record, "outcome": outcome}
            except Exception:
                pass
        enriched.append(record)
    return enriched
