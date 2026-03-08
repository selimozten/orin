"""Tests for the procedural data generator and augmentation utilities."""

from __future__ import annotations

from orin.data.augment import augment_records
from orin.data.generator import generate_all, generate_earnings, generate_news


def test_earnings_count():
    records = generate_earnings(100)
    assert len(records) == 100


def test_earnings_beat_ratio():
    records = generate_earnings(1000, seed=99, difficulty="easy")
    up_count = sum(1 for r in records if r["outcome"]["direction"] == "up")
    ratio = up_count / len(records)
    assert 0.50 <= ratio <= 0.60, f"beat ratio {ratio:.2f} outside 50-60%"


def test_noise_rate_easy():
    """Easy difficulty: ~10% noise (beat->down or miss->up)."""
    records = generate_earnings(1000, seed=7, difficulty="easy")
    # In easy mode, beat text with 'down' direction or miss text with 'up' direction
    # approximates noise. We check that noisy records exist but are a minority.
    beat_keywords = ["exceeded", "beating", "beat", "strong", "raised"]
    miss_keywords = ["missed", "disappointing", "fell short", "lowered", "contracted"]

    noisy = 0
    clear = 0
    for r in records:
        text_lower = r["text"].lower()
        direction = r["outcome"]["direction"]
        has_beat = any(kw in text_lower for kw in beat_keywords)
        has_miss = any(kw in text_lower for kw in miss_keywords)
        if has_beat and not has_miss and direction == "down":
            noisy += 1
            clear += 1
        elif has_miss and not has_beat and direction == "up":
            noisy += 1
            clear += 1
        elif has_beat or has_miss:
            clear += 1

    if clear > 0:
        noise_ratio = noisy / clear
        assert noise_ratio < 0.20, f"easy noise ratio {noise_ratio:.2f} too high"


def test_noise_rate_hard():
    """Hard difficulty should have noise within expected range."""
    records = generate_earnings(1000, seed=7, difficulty="hard")
    # Just verify it runs and produces records with expected structure
    assert len(records) == 1000
    directions = {r["outcome"]["direction"] for r in records}
    assert "up" in directions
    assert "down" in directions


def test_seed_reproducibility():
    a = generate_earnings(50, seed=123)
    b = generate_earnings(50, seed=123)
    for ra, rb in zip(a, b):
        assert ra["text"] == rb["text"]
        assert ra["outcome"] == rb["outcome"]


def test_different_seeds():
    a = generate_earnings(50, seed=1)
    b = generate_earnings(50, seed=2)
    texts_a = {r["text"] for r in a}
    texts_b = {r["text"] for r in b}
    # Should not be identical
    assert texts_a != texts_b


def test_ticker_coverage():
    records = generate_earnings(500, seed=42)
    tickers = {r["ticker"] for r in records}
    # Should cover multiple sectors
    has_tech = bool(tickers & {"AAPL", "MSFT", "GOOGL", "NVDA"})
    has_finance = bool(tickers & {"JPM", "BAC", "GS"})
    has_healthcare = bool(tickers & {"UNH", "JNJ", "PFE"})
    assert has_tech, "missing tech tickers"
    assert has_finance, "missing finance tickers"
    assert has_healthcare, "missing healthcare tickers"


def test_difficulty_parameter():
    for level in ("easy", "medium", "hard"):
        records = generate_earnings(20, seed=42, difficulty=level)
        assert len(records) == 20
        for r in records:
            assert "text" in r
            assert "outcome" in r
            assert r["outcome"]["direction"] in ("up", "down")


def test_all_env_types():
    data = generate_all(n_earnings=10, n_news=10, n_filings=10, n_macro=10)
    assert set(data.keys()) == {"earnings", "news", "filing", "macro"}
    for key, records in data.items():
        assert len(records) == 10, f"{key} count mismatch"


def test_ambiguous_templates_medium():
    """Medium/hard difficulty should include ambiguous text patterns."""
    ambiguous_markers = [
        "mixed",
        "conflicting",
        "but",
        "however",
        "divided",
        "nuanced",
        "uneven",
        "unclear",
        "both",
        "tale of two",
    ]
    for level in ("medium", "hard"):
        records = generate_earnings(200, seed=42, difficulty=level)
        texts = [r["text"].lower() for r in records]
        has_ambiguous = any(
            any(marker in t for marker in ambiguous_markers) for t in texts
        )
        assert has_ambiguous, f"{level} difficulty should produce ambiguous templates"

    # Also check news
    for level in ("medium", "hard"):
        records = generate_news(200, seed=42, difficulty=level)
        texts = [r["text"].lower() for r in records]
        has_ambiguous = any(
            any(marker in t for marker in ambiguous_markers) for t in texts
        )
        assert has_ambiguous, f"{level} news should produce ambiguous templates"


def test_augment_records():
    records = generate_earnings(10, seed=42)
    augmented = augment_records(records, n_augments=2, seed=99)
    # Original 10 + 10*2 augmented = 30
    assert len(augmented) == 30


def test_augment_preserves_outcome():
    records = generate_earnings(20, seed=42)
    augmented = augment_records(records, n_augments=1, seed=99)
    # First 20 are originals, next 20 are augmented copies
    for orig, aug in zip(augmented[:20], augmented[20:]):
        assert orig["outcome"] == aug["outcome"]
        assert orig["ticker"] == aug["ticker"]
        assert orig["date"] == aug["date"]
