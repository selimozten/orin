"""Build a real dataset from SEC EDGAR filings + yfinance market data.

Usage:
    pip install orin[all]
    python examples/build_dataset.py
    python examples/build_dataset.py --tickers AAPL MSFT NVDA --output data/filings.jsonl

This fetches real filing text from SEC EDGAR and enriches each record
with actual post-filing stock returns from yfinance.
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Build orin dataset from real data")
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"],
    )
    parser.add_argument("--forms", nargs="+", default=["10-K", "10-Q", "8-K"])
    parser.add_argument("--max-per-ticker", type=int, default=5)
    parser.add_argument("--timeframe", default="5d")
    parser.add_argument("--output", default="data/real/filings.jsonl")
    args = parser.parse_args()

    from orin.data.pipeline import build_filing_dataset

    print(f"Fetching filings for {args.tickers}...")
    print(f"Form types: {args.forms}")
    print(f"Timeframe: {args.timeframe}")

    records = build_filing_dataset(
        tickers=args.tickers,
        form_types=args.forms,
        max_per_ticker=args.max_per_ticker,
        timeframe=args.timeframe,
        enrich=True,
        output_path=args.output,
    )

    print(f"\nBuilt {len(records)} records")
    print(f"Saved to {args.output}")

    # Summary
    up = sum(1 for r in records if r.get("outcome", {}).get("direction") == "up")
    down = sum(1 for r in records if r.get("outcome", {}).get("direction") == "down")
    flat = sum(1 for r in records if r.get("outcome", {}).get("direction") == "flat")
    print(f"Distribution: up={up} down={down} flat={flat}")

    # Show a sample
    if records:
        r = records[0]
        print("\nSample record:")
        print(f"  Ticker: {r['ticker']}")
        print(f"  Date:   {r['date']}")
        print(f"  Source: {r['source']}")
        print(f"  Text:   {r['text'][:200]}...")
        print(f"  Outcome: {r['outcome']}")


if __name__ == "__main__":
    main()
