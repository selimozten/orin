"""Tests for data loading."""

from __future__ import annotations

from pathlib import Path

from orin.data.sources import (
    generate_sample_earnings,
    generate_sample_filing,
    generate_sample_macro,
    generate_sample_news,
    write_sample_data,
)


def test_sample_earnings():
    data = generate_sample_earnings()
    assert len(data) > 0
    for record in data:
        assert "text" in record
        assert "ticker" in record
        assert "date" in record
        assert "outcome" in record
        assert "direction" in record["outcome"]
        assert "magnitude" in record["outcome"]


def test_sample_news():
    data = generate_sample_news()
    assert len(data) > 0


def test_sample_filing():
    data = generate_sample_filing()
    assert len(data) > 0


def test_sample_macro():
    data = generate_sample_macro()
    assert len(data) > 0


def test_write_sample_data(tmp_path: Path):
    write_sample_data(tmp_path)
    for name in ["earnings", "news", "filing", "macro"]:
        assert (tmp_path / f"{name}.jsonl").exists()


def test_load_jsonl(tmp_path: Path):
    write_sample_data(tmp_path)
    from orin.data.loaders import load_jsonl

    records = load_jsonl(tmp_path / "earnings.jsonl")
    assert len(records) > 0
    assert "text" in records[0]


def test_load_dataframe(tmp_path: Path):
    write_sample_data(tmp_path)
    from orin.data.loaders import load_dataframe, load_jsonl

    records = load_jsonl(tmp_path / "earnings.jsonl")
    df = load_dataframe(records)
    assert len(df) > 0
    assert "text" in df.columns
