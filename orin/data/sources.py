"""Data source connectors and sample data generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def generate_sample_earnings() -> list[dict[str, Any]]:
    """Generate sample earnings call data for testing."""
    samples = [
        {
            "text": (
                "Good afternoon. This is the AAPL Q3 2024 earnings call. "
                "Revenue came in at $81.8 billion, up 5% year over year. "
                "iPhone revenue was $39.3 billion. Services revenue hit a new "
                "all-time record of $21.2 billion. We returned over $24 billion "
                "to shareholders. Gross margin was 46.3%, up 100 basis points."
            ),
            "ticker": "AAPL",
            "date": "2024-08-01",
            "source": "earnings_call",
            "outcome": {"direction": "up", "magnitude": 0.028, "timeframe": "1d"},
        },
        {
            "text": (
                "Welcome to MSFT fiscal Q4 2024 earnings conference call. "
                "Revenue was $64.7 billion, increasing 15% year over year. "
                "Azure and other cloud services revenue grew 29%. "
                "Operating income increased 15% to $27.9 billion. "
                "We are seeing strong demand for AI services across our cloud platform."
            ),
            "ticker": "MSFT",
            "date": "2024-07-30",
            "source": "earnings_call",
            "outcome": {"direction": "down", "magnitude": 0.032, "timeframe": "1d"},
        },
        {
            "text": (
                "Good evening. AMZN Q2 2024 results. Net sales increased 10% "
                "to $148.0 billion. AWS segment sales of $26.3 billion, up 19% "
                "year over year. Operating income was $14.7 billion, up from "
                "$7.7 billion. North America segment operating margin improved "
                "to 5.6%. We continue to invest heavily in generative AI."
            ),
            "ticker": "AMZN",
            "date": "2024-08-01",
            "source": "earnings_call",
            "outcome": {"direction": "down", "magnitude": 0.045, "timeframe": "1d"},
        },
        {
            "text": (
                "TSLA Q2 2024 earnings call. Total revenue was $25.5 billion, "
                "up 2% year over year. Automotive revenue declined 7% to "
                "$19.9 billion. Energy generation and storage revenue more than "
                "doubled to $3.0 billion. GAAP operating margin was 6.3%. "
                "We delivered 443,956 vehicles in Q2."
            ),
            "ticker": "TSLA",
            "date": "2024-07-23",
            "source": "earnings_call",
            "outcome": {"direction": "down", "magnitude": 0.082, "timeframe": "1d"},
        },
        {
            "text": (
                "NVDA fiscal Q1 2025 earnings. Revenue was $26.0 billion, up "
                "262% year over year. Data Center revenue was $22.6 billion, "
                "up 427%. GAAP earnings per diluted share was $5.98, up 629%. "
                "We announced a ten-for-one stock split. Demand for Hopper "
                "architecture remains very strong."
            ),
            "ticker": "NVDA",
            "date": "2024-05-22",
            "source": "earnings_call",
            "outcome": {"direction": "up", "magnitude": 0.095, "timeframe": "1d"},
        },
        {
            "text": (
                "Good afternoon. GOOGL Q2 2024 earnings. Revenue of $84.7 "
                "billion, up 14% year over year. Google Cloud revenue was "
                "$10.3 billion, up 29%. YouTube ads revenue was $8.7 billion. "
                "Operating margin expanded to 32%. We are seeing strong "
                "momentum in AI-driven search improvements."
            ),
            "ticker": "GOOGL",
            "date": "2024-07-23",
            "source": "earnings_call",
            "outcome": {"direction": "down", "magnitude": 0.051, "timeframe": "1d"},
        },
        {
            "text": (
                "META Q2 2024 earnings conference call. Total revenue was "
                "$39.1 billion, up 22%. Family daily active people was 3.27 "
                "billion. Ad impressions delivered across Family of Apps "
                "increased 10%. Average price per ad increased 10%. "
                "Reality Labs revenue was $353 million with operating loss "
                "of $4.5 billion."
            ),
            "ticker": "META",
            "date": "2024-07-31",
            "source": "earnings_call",
            "outcome": {"direction": "down", "magnitude": 0.041, "timeframe": "1d"},
        },
        {
            "text": (
                "JPM Q3 2024 earnings. Net revenue of $43.3 billion, up 6%. "
                "Net income was $12.9 billion. CIB revenue was $17.0 billion. "
                "Investment banking fees up 31%. Credit costs were $3.1 billion. "
                "Net interest income was $23.5 billion. Book value per share "
                "was $113.64."
            ),
            "ticker": "JPM",
            "date": "2024-10-11",
            "source": "earnings_call",
            "outcome": {"direction": "up", "magnitude": 0.044, "timeframe": "1d"},
        },
    ]
    return samples


def generate_sample_news() -> list[dict[str, Any]]:
    """Generate sample financial news data for testing."""
    samples = [
        {
            "text": (
                "Federal Reserve holds interest rates steady at 5.25-5.50%, "
                "signals potential rate cut in September. Chair Powell says "
                "inflation has made further progress toward the 2% target."
            ),
            "ticker": "SPY",
            "date": "2024-07-31",
            "source": "news",
            "outcome": {"direction": "up", "magnitude": 0.015, "timeframe": "1d"},
        },
        {
            "text": (
                "Oil prices surge 4% as Middle East tensions escalate. "
                "Brent crude rises to $82 per barrel. Energy stocks rally "
                "broadly with XOM and CVX leading gains."
            ),
            "ticker": "XLE",
            "date": "2024-08-05",
            "source": "news",
            "outcome": {"direction": "up", "magnitude": 0.032, "timeframe": "1d"},
        },
        {
            "text": (
                "US unemployment rate rises to 4.3%, triggering Sahm Rule "
                "recession indicator. Nonfarm payrolls increase by only "
                "114,000, well below expectations of 175,000."
            ),
            "ticker": "SPY",
            "date": "2024-08-02",
            "source": "news",
            "outcome": {"direction": "down", "magnitude": 0.030, "timeframe": "1d"},
        },
        {
            "text": (
                "Semiconductor stocks plunge as new US export restrictions "
                "to China announced. ASML, NVDA, AMD all fall sharply. "
                "Biden administration expands chip export controls."
            ),
            "ticker": "SMH",
            "date": "2024-07-17",
            "source": "news",
            "outcome": {"direction": "down", "magnitude": 0.065, "timeframe": "1d"},
        },
        {
            "text": (
                "Warren Buffett's Berkshire Hathaway reveals massive Apple "
                "stake reduction, selling roughly half its AAPL position. "
                "Cash pile grows to record $277 billion."
            ),
            "ticker": "AAPL",
            "date": "2024-08-03",
            "source": "news",
            "outcome": {"direction": "down", "magnitude": 0.022, "timeframe": "1d"},
        },
        {
            "text": (
                "CPI report shows inflation cooling to 2.9% year over year, "
                "first time below 3% since March 2021. Core CPI rises 0.2% "
                "month over month. Markets rally on rate cut expectations."
            ),
            "ticker": "SPY",
            "date": "2024-08-14",
            "source": "news",
            "outcome": {"direction": "up", "magnitude": 0.019, "timeframe": "1d"},
        },
    ]
    return samples


def generate_sample_filing() -> list[dict[str, Any]]:
    """Generate sample SEC filing data for testing."""
    samples = [
        {
            "text": (
                "FORM 10-K ANNUAL REPORT - AAPL. Total net sales decreased 3% "
                "to $383.3 billion. iPhone revenue was $200.6 billion. "
                "Services revenue reached $85.2 billion, up 9%. Operating "
                "expenses were $54.8 billion. Long-term debt was $95.3 billion. "
                "Cash and equivalents totaled $30.7 billion."
            ),
            "ticker": "AAPL",
            "date": "2024-01-30",
            "source": "10-K",
            "outcome": {"direction": "up", "magnitude": 0.012, "timeframe": "5d"},
        },
        {
            "text": (
                "FORM 8-K CURRENT REPORT - TSLA. The Company announces a "
                "workforce reduction of approximately 10% globally. CEO "
                "states this is necessary to reduce costs and increase "
                "productivity for the next growth phase. Restructuring "
                "charges expected to be $350 million."
            ),
            "ticker": "TSLA",
            "date": "2024-04-15",
            "source": "8-K",
            "outcome": {"direction": "down", "magnitude": 0.055, "timeframe": "5d"},
        },
        {
            "text": (
                "FORM 10-Q QUARTERLY REPORT - MSFT. Revenue was $61.9 billion, "
                "up 17%. Intelligent Cloud revenue was $26.7 billion, up 21%. "
                "Azure and other cloud services grew 31%. Operating income "
                "was $27.6 billion. Unearned revenue was $53.2 billion."
            ),
            "ticker": "MSFT",
            "date": "2024-04-25",
            "source": "10-Q",
            "outcome": {"direction": "up", "magnitude": 0.018, "timeframe": "5d"},
        },
        {
            "text": (
                "FORM 8-K CURRENT REPORT - JPM. The Company reports a "
                "significant increase in provision for credit losses to "
                "$2.8 billion. Commercial real estate portfolio shows "
                "elevated stress with net charge-offs rising 45% quarter "
                "over quarter. Management cites office sector weakness."
            ),
            "ticker": "JPM",
            "date": "2024-03-15",
            "source": "8-K",
            "outcome": {"direction": "down", "magnitude": 0.021, "timeframe": "5d"},
        },
        {
            "text": (
                "FORM 10-K ANNUAL REPORT - AMZN. Net sales increased 12% to "
                "$574.8 billion. AWS segment had $90.8 billion in sales with "
                "operating income of $24.6 billion. Free cash flow improved "
                "to $36.8 billion. Headcount decreased 5% to 1,525,000."
            ),
            "ticker": "AMZN",
            "date": "2024-02-01",
            "source": "10-K",
            "outcome": {"direction": "up", "magnitude": 0.035, "timeframe": "5d"},
        },
    ]
    return samples


def generate_sample_macro() -> list[dict[str, Any]]:
    """Generate sample macroeconomic data for testing."""
    samples = [
        {
            "text": (
                "Federal Reserve Chair Powell's Jackson Hole speech: "
                "'The time has come for policy to adjust.' Signals confidence "
                "that inflation is on a sustainable path to 2%. Labor market "
                "has cooled considerably from overheated state. Emphasizes "
                "data-dependent approach to rate cuts."
            ),
            "ticker": "SPY",
            "date": "2024-08-23",
            "source": "fed_speech",
            "outcome": {"direction": "up", "magnitude": 0.018, "timeframe": "5d"},
        },
        {
            "text": (
                "US GDP grows at 2.8% annual rate in Q2 2024, well above "
                "the 2.0% consensus estimate. Consumer spending rises 2.3%. "
                "Business investment increases 5.2%. The economy shows "
                "resilience despite restrictive monetary policy."
            ),
            "ticker": "SPY",
            "date": "2024-07-25",
            "source": "economic_data",
            "outcome": {"direction": "up", "magnitude": 0.008, "timeframe": "1d"},
        },
        {
            "text": (
                "ISM Manufacturing PMI falls to 46.8 in July, signaling "
                "contraction for the fourth consecutive month. New orders "
                "index drops to 47.4. Employment index declines to 43.4, "
                "the lowest since June 2020. Prices paid index rises to 52.9."
            ),
            "ticker": "SPY",
            "date": "2024-08-01",
            "source": "economic_data",
            "outcome": {"direction": "down", "magnitude": 0.025, "timeframe": "1d"},
        },
        {
            "text": (
                "Bank of Japan raises interest rates to 0.25%, the highest "
                "since 2008. Governor Ueda signals further tightening ahead. "
                "Yen strengthens sharply. Global carry trade unwind accelerates. "
                "Japanese equities fall 5% in response."
            ),
            "ticker": "SPY",
            "date": "2024-07-31",
            "source": "central_bank",
            "outcome": {"direction": "down", "magnitude": 0.042, "timeframe": "5d"},
        },
        {
            "text": (
                "US Consumer Confidence Index rises to 103.3 in August, "
                "above expectations of 100.7. Present situation index "
                "increases to 134.4. Expectations index rises to 82.5. "
                "Consumers show improved outlook on labor market conditions "
                "and business environment."
            ),
            "ticker": "SPY",
            "date": "2024-08-27",
            "source": "economic_data",
            "outcome": {"direction": "up", "magnitude": 0.005, "timeframe": "1d"},
        },
    ]
    return samples


def write_sample_data(output_dir: str | Path | None = None) -> None:
    """Write all sample datasets to JSONL files."""
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / "data" / "sample"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generators = {
        "earnings": generate_sample_earnings,
        "news": generate_sample_news,
        "filing": generate_sample_filing,
        "macro": generate_sample_macro,
    }

    for name, gen_fn in generators.items():
        path = output_dir / f"{name}.jsonl"
        records = gen_fn()
        with open(path, "w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    write_sample_data()
    print("Sample data written to data/sample/")
