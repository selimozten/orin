"""Tests for market data module (unit tests that don't call yfinance)."""

from __future__ import annotations

from orin.data.market import compute_outcome


def test_compute_outcome_structure():
    """Test that compute_outcome returns the right structure.

    Note: this test uses a mock -- real yfinance tests require network.
    """
    # We test the internal logic by monkeypatching get_returns
    import orin.data.market as market_mod

    original = market_mod.get_returns

    def mock_returns(ticker, event_date, timeframes=None):
        return {"1d": 0.035, "5d": 0.02}

    market_mod.get_returns = mock_returns
    try:
        outcome = compute_outcome("AAPL", "2024-01-01", "1d")
        assert outcome["direction"] == "up"
        assert outcome["magnitude"] == 0.035
        assert outcome["timeframe"] == "1d"
    finally:
        market_mod.get_returns = original


def test_compute_outcome_down():
    import orin.data.market as market_mod

    original = market_mod.get_returns

    def mock_returns(ticker, event_date, timeframes=None):
        return {"1d": -0.05}

    market_mod.get_returns = mock_returns
    try:
        outcome = compute_outcome("TSLA", "2024-01-01", "1d")
        assert outcome["direction"] == "down"
        assert outcome["magnitude"] == 0.05
    finally:
        market_mod.get_returns = original


def test_compute_outcome_flat():
    import orin.data.market as market_mod

    original = market_mod.get_returns

    def mock_returns(ticker, event_date, timeframes=None):
        return {"1d": 0.002}

    market_mod.get_returns = mock_returns
    try:
        outcome = compute_outcome("SPY", "2024-01-01", "1d")
        assert outcome["direction"] == "flat"
    finally:
        market_mod.get_returns = original


def test_bulk_returns_enriches():
    import orin.data.market as market_mod
    from orin.data.market import bulk_returns

    original = market_mod.get_returns

    def mock_returns(ticker, event_date, timeframes=None):
        return {"1d": 0.02}

    market_mod.get_returns = mock_returns
    try:
        records = [
            {
                "text": "test",
                "ticker": "AAPL",
                "date": "2024-01-01",
                "outcome": {"direction": "flat", "magnitude": 0.0, "timeframe": "1d"},
            },
        ]
        enriched = bulk_returns(records, timeframe="1d")
        assert enriched[0]["outcome"]["direction"] == "up"
        assert enriched[0]["outcome"]["magnitude"] == 0.02
    finally:
        market_mod.get_returns = original
