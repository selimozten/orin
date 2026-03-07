"""Tests for the data pipeline (unit tests with mocked network calls)."""

from __future__ import annotations

from pathlib import Path

from orin.data.pipeline import _save_jsonl, build_earnings_dataset, build_news_dataset


def test_save_jsonl(tmp_path: Path):
    records = [{"text": "hello", "ticker": "X"}]
    out = tmp_path / "test.jsonl"
    _save_jsonl(records, out)
    assert out.exists()
    import json

    with open(out) as f:
        loaded = json.loads(f.readline())
    assert loaded["text"] == "hello"


def test_build_earnings_dataset_no_enrich():
    transcripts = [
        {"text": "AAPL earnings call...", "ticker": "AAPL", "date": "2024-01-01"},
        {"text": "MSFT earnings call...", "ticker": "MSFT", "date": "2024-01-01"},
        {"text": "GOOGL earnings call...", "ticker": "GOOGL", "date": "2024-01-01"},
    ]
    records = build_earnings_dataset(
        tickers=["AAPL", "MSFT"],
        transcripts=transcripts,
        enrich=False,
    )
    assert len(records) == 2
    assert all(r["source"] == "earnings_call" for r in records)
    assert all("outcome" in r for r in records)


def test_build_earnings_dataset_all_tickers():
    transcripts = [
        {"text": "call 1", "ticker": "A", "date": "2024-01-01"},
        {"text": "call 2", "ticker": "B", "date": "2024-01-01"},
    ]
    records = build_earnings_dataset(tickers=[], transcripts=transcripts, enrich=False)
    assert len(records) == 2


def test_build_news_dataset_no_enrich():
    headlines = [
        {"text": "Market rallies on CPI", "ticker": "SPY", "date": "2024-08-14"},
    ]
    records = build_news_dataset(headlines, enrich=False)
    assert len(records) == 1
    assert records[0]["source"] == "news"


def test_build_earnings_saves_output(tmp_path: Path):
    transcripts = [
        {"text": "test", "ticker": "X", "date": "2024-01-01"},
    ]
    out = tmp_path / "earnings.jsonl"
    build_earnings_dataset(tickers=[], transcripts=transcripts, enrich=False, output_path=out)
    assert out.exists()
