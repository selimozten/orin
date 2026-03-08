"""Metadata encoder for environment observations."""
from __future__ import annotations

import numpy as np

SECTORS = ["Technology", "Financials", "Healthcare", "Consumer", "Industrials"]
SOURCES = ["earnings_call", "news", "10-K", "10-Q", "8-K"]


class MetadataEncoder:
    """Encode metadata into numeric features.

    Produces a fixed-size vector:
    - sector one-hot (5 dims)
    - date cyclical encoding (4 dims: sin/cos for month and day-of-week)
    - source one-hot (5 dims)
    Total: 14 dimensions
    """

    n_features: int = 14

    # Map tickers to sectors
    _SECTOR_MAP = {
        # Tech
        **{t: 0 for t in [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "CRM",
            "ORCL", "ADBE", "INTC", "AMD", "AVGO", "CSCO", "NOW", "SHOP",
            "SQ", "SNOW", "PLTR", "NET", "DDOG", "MDB", "ZS", "CRWD", "PANW",
        ]},
        # Finance
        **{t: 1 for t in [
            "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP",
            "V", "MA", "COF", "USB", "PNC", "TFC",
        ]},
        # Healthcare
        **{t: 2 for t in [
            "UNH", "JNJ", "LLY", "PFE", "ABBV", "MRK", "TMO", "ABT",
            "DHR", "BMY", "AMGN", "GILD", "VRTX", "REGN", "ISRG",
        ]},
        # Consumer
        **{t: 3 for t in [
            "WMT", "COST", "HD", "MCD", "NKE", "SBUX", "TGT", "LOW",
            "TJX", "ROST", "DG", "DLTR", "YUM", "CMG", "DPZ",
        ]},
        # Industrial
        **{t: 4 for t in [
            "CAT", "DE", "BA", "HON", "UPS", "RTX", "LMT", "GE", "MMM", "EMR",
        ]},
    }

    def encode(self, metadata: dict) -> np.ndarray:
        """Encode metadata dict into a fixed-size numeric feature vector."""
        features = np.zeros(self.n_features, dtype=np.float32)

        # Sector one-hot (dims 0-4)
        ticker = metadata.get("ticker", "")
        sector_idx = self._SECTOR_MAP.get(ticker, -1)
        if 0 <= sector_idx < 5:
            features[sector_idx] = 1.0

        # Date cyclical (dims 5-8)
        date_str = metadata.get("date", "")
        if date_str and len(date_str) >= 10:
            try:
                month = int(date_str[5:7])
                day = int(date_str[8:10])
                features[5] = np.sin(2 * np.pi * month / 12)
                features[6] = np.cos(2 * np.pi * month / 12)
                # Approximate day-of-week from day-of-month (rough)
                features[7] = np.sin(2 * np.pi * day / 31)
                features[8] = np.cos(2 * np.pi * day / 31)
            except (ValueError, IndexError):
                pass

        # Source one-hot (dims 9-13)
        source = metadata.get("source", "")
        if source in SOURCES:
            features[9 + SOURCES.index(source)] = 1.0

        return features
