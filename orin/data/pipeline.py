"""End-to-end data pipeline: fetch text + enrich with real market outcomes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_filing_dataset(
    tickers: list[str],
    form_types: list[str] | None = None,
    max_per_ticker: int = 5,
    timeframe: str = "5d",
    enrich: bool = True,
    output_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Build a dataset of SEC filings with real market outcomes.

    Fetches filing text from EDGAR and computes actual post-filing
    returns using yfinance.

    Args:
        tickers: List of ticker symbols.
        form_types: Filing types (default: 10-K, 10-Q, 8-K).
        max_per_ticker: Max filings per ticker.
        timeframe: Lookahead window for returns.
        enrich: Whether to enrich with real market data.
        output_path: If set, save records to this JSONL file.

    Returns:
        List of orin-format records.
    """
    from orin.data.edgar import fetch_filings_as_records

    all_records: list[dict[str, Any]] = []
    for ticker in tickers:
        records = fetch_filings_as_records(
            ticker, form_types, max_per_ticker, timeframe=timeframe,
        )
        all_records.extend(records)

    if enrich and all_records:
        from orin.data.market import bulk_returns

        all_records = bulk_returns(all_records, timeframe=timeframe)

    if output_path is not None:
        _save_jsonl(all_records, output_path)

    return all_records


def build_earnings_dataset(
    tickers: list[str],
    transcripts: list[dict[str, Any]],
    timeframe: str = "1d",
    enrich: bool = True,
    output_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Build an earnings dataset from transcripts with real market outcomes.

    Transcripts should already have "text", "ticker", and "date" fields.
    This function enriches them with actual post-earnings returns.

    Args:
        tickers: Filter transcripts to these tickers (empty = all).
        transcripts: Pre-loaded transcript records.
        timeframe: Lookahead window for returns.
        enrich: Whether to enrich with real market data.
        output_path: If set, save records to this JSONL file.

    Returns:
        List of orin-format records.
    """
    if tickers:
        ticker_set = {t.upper() for t in tickers}
        records = [r for r in transcripts if r.get("ticker", "").upper() in ticker_set]
    else:
        records = list(transcripts)

    for r in records:
        r.setdefault("source", "earnings_call")
        r.setdefault("outcome", {"direction": "flat", "magnitude": 0.0, "timeframe": timeframe})

    if enrich and records:
        from orin.data.market import bulk_returns

        records = bulk_returns(records, timeframe=timeframe)

    if output_path is not None:
        _save_jsonl(records, output_path)

    return records


def build_news_dataset(
    headlines: list[dict[str, Any]],
    timeframe: str = "1d",
    enrich: bool = True,
    output_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Build a news dataset from headlines with real market outcomes.

    Headlines should have "text", "ticker", and "date" fields.

    Args:
        headlines: Pre-loaded headline/article records.
        timeframe: Lookahead window for returns.
        enrich: Whether to enrich with real market data.
        output_path: If set, save records to this JSONL file.

    Returns:
        List of orin-format records.
    """
    records = list(headlines)
    for r in records:
        r.setdefault("source", "news")
        r.setdefault("outcome", {"direction": "flat", "magnitude": 0.0, "timeframe": timeframe})

    if enrich and records:
        from orin.data.market import bulk_returns

        records = bulk_returns(records, timeframe=timeframe)

    if output_path is not None:
        _save_jsonl(records, output_path)

    return records


def enrich_dataset(
    input_path: str | Path,
    output_path: str | Path | None = None,
    timeframe: str = "1d",
) -> list[dict[str, Any]]:
    """Load an existing JSONL dataset and enrich with real market outcomes.

    Useful for adding outcomes to any dataset that has ticker + date fields.

    Args:
        input_path: Path to input JSONL file.
        output_path: If set, save enriched records here.
        timeframe: Lookahead window for returns.

    Returns:
        Enriched records.
    """
    from orin.data.loaders import load_jsonl
    from orin.data.market import bulk_returns

    records = load_jsonl(input_path)
    records = bulk_returns(records, timeframe=timeframe)

    if output_path is not None:
        _save_jsonl(records, output_path)

    return records


def _save_jsonl(records: list[dict[str, Any]], path: str | Path) -> None:
    """Save records to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
