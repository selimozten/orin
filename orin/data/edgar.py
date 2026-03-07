"""SEC EDGAR filing fetcher.

Uses the public EDGAR full-text search API (EFTS) -- no API key required.
Rate limit: 10 requests/second with a User-Agent header.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

_USER_AGENT = "orin-fintext-gym/0.1 (https://github.com/selimozten/orin)"
_EFTS_BASE = "https://efts.sec.gov/LATEST/search-index"
_SUBMISSIONS_BASE = "https://data.sec.gov/submissions"
_FILING_BASE = "https://www.sec.gov/Archives/edgar/data"
_LAST_REQUEST_TIME: float = 0.0


def _rate_limit() -> None:
    """Enforce 100ms between requests to respect SEC rate limits."""
    global _LAST_REQUEST_TIME
    elapsed = time.time() - _LAST_REQUEST_TIME
    if elapsed < 0.1:
        time.sleep(0.1 - elapsed)
    _LAST_REQUEST_TIME = time.time()


def _fetch_json(url: str) -> dict[str, Any]:
    """Fetch JSON from a URL with proper User-Agent."""
    _rate_limit()
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _fetch_text(url: str) -> str:
    """Fetch raw text from a URL."""
    _rate_limit()
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="replace")


def get_cik(ticker: str) -> str | None:
    """Look up the CIK number for a ticker symbol.

    Args:
        ticker: Stock ticker (e.g. "AAPL").

    Returns:
        CIK as zero-padded string, or None if not found.
    """
    url = "https://www.sec.gov/files/company_tickers.json"
    data = _fetch_json(url)
    ticker_upper = ticker.upper()
    for entry in data.values():
        if entry.get("ticker", "").upper() == ticker_upper:
            return str(entry["cik_str"]).zfill(10)
    return None


def get_recent_filings(
    ticker: str,
    form_types: list[str] | None = None,
    max_results: int = 10,
) -> list[dict[str, Any]]:
    """Get recent filing metadata for a company.

    Args:
        ticker: Stock ticker symbol.
        form_types: Filter by form type (e.g. ["10-K", "10-Q", "8-K"]).
            Defaults to ["10-K", "10-Q", "8-K"].
        max_results: Maximum number of filings to return.

    Returns:
        List of filing metadata dicts with keys:
        accession, form, date, description, url.
    """
    if form_types is None:
        form_types = ["10-K", "10-Q", "8-K"]

    cik = get_cik(ticker)
    if cik is None:
        return []

    url = f"{_SUBMISSIONS_BASE}/CIK{cik}.json"
    data = _fetch_json(url)

    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])
    descriptions = recent.get("primaryDocDescription", [])

    filings = []
    for i in range(len(forms)):
        if forms[i] in form_types:
            accession_clean = accessions[i].replace("-", "")
            filing_url = f"{_FILING_BASE}/{cik.lstrip('0')}/{accession_clean}/{primary_docs[i]}"
            filings.append(
                {
                    "accession": accessions[i],
                    "form": forms[i],
                    "date": dates[i],
                    "description": descriptions[i] if i < len(descriptions) else "",
                    "url": filing_url,
                    "ticker": ticker.upper(),
                }
            )
            if len(filings) >= max_results:
                break

    return filings


def fetch_filing_text(
    filing: dict[str, Any],
    max_chars: int = 4000,
) -> str:
    """Fetch the text content of a filing.

    Downloads the primary document and extracts readable text,
    truncated to max_chars.

    Args:
        filing: Filing metadata dict (from get_recent_filings).
        max_chars: Maximum characters to return.

    Returns:
        Plain text excerpt from the filing.
    """
    url = filing["url"]
    try:
        raw = _fetch_text(url)
    except urllib.error.HTTPError:
        return ""

    # Strip HTML tags for a rough plain-text extraction
    import re

    text = re.sub(r"<[^>]+>", " ", raw)
    text = re.sub(r"&[a-zA-Z]+;", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text[:max_chars]


def fetch_filings_as_records(
    ticker: str,
    form_types: list[str] | None = None,
    max_results: int = 5,
    max_chars: int = 4000,
    timeframe: str = "5d",
) -> list[dict[str, Any]]:
    """Fetch filings and format them as orin data records.

    Each record has text, ticker, date, source, and an empty outcome
    that can be enriched with market.bulk_returns().

    Args:
        ticker: Stock ticker.
        form_types: Filing types to fetch.
        max_results: Max filings to return.
        max_chars: Max text length per filing.
        timeframe: Timeframe for outcome (used as placeholder).

    Returns:
        List of orin-format records.
    """
    filings = get_recent_filings(ticker, form_types, max_results)
    records = []
    for filing in filings:
        text = fetch_filing_text(filing, max_chars)
        if not text:
            continue
        records.append(
            {
                "text": text,
                "ticker": filing["ticker"],
                "date": filing["date"],
                "source": filing["form"],
                "outcome": {"direction": "flat", "magnitude": 0.0, "timeframe": timeframe},
            }
        )
    return records
