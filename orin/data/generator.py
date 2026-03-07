"""Procedural data generator for creating large synthetic financial text datasets.

Generates hundreds of varied records with realistic financial text patterns.
Text signals correlate with outcomes so RL agents can learn meaningful patterns,
but with enough noise and variation to require generalization.
"""

from __future__ import annotations

import random
from typing import Any

# -- Ticker pools by sector --

TECH_TICKERS = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "NVDA",
    "META",
    "TSLA",
    "CRM",
    "ORCL",
    "ADBE",
    "INTC",
    "AMD",
    "AVGO",
    "CSCO",
    "NOW",
    "SHOP",
    "SQ",
    "SNOW",
    "PLTR",
    "NET",
    "DDOG",
    "MDB",
    "ZS",
    "CRWD",
    "PANW",
]

FINANCE_TICKERS = [
    "JPM",
    "BAC",
    "WFC",
    "GS",
    "MS",
    "C",
    "BLK",
    "SCHW",
    "AXP",
    "V",
    "MA",
    "COF",
    "USB",
    "PNC",
    "TFC",
]

HEALTHCARE_TICKERS = [
    "UNH",
    "JNJ",
    "LLY",
    "PFE",
    "ABBV",
    "MRK",
    "TMO",
    "ABT",
    "DHR",
    "BMY",
    "AMGN",
    "GILD",
    "VRTX",
    "REGN",
    "ISRG",
]

CONSUMER_TICKERS = [
    "WMT",
    "COST",
    "HD",
    "MCD",
    "NKE",
    "SBUX",
    "TGT",
    "LOW",
    "TJX",
    "ROST",
    "DG",
    "DLTR",
    "YUM",
    "CMG",
    "DPZ",
]

INDUSTRIAL_TICKERS = [
    "CAT",
    "DE",
    "BA",
    "HON",
    "UPS",
    "RTX",
    "LMT",
    "GE",
    "MMM",
    "EMR",
]

ALL_TICKERS = (
    TECH_TICKERS + FINANCE_TICKERS + HEALTHCARE_TICKERS + CONSUMER_TICKERS + INDUSTRIAL_TICKERS
)

# -- Templates for earnings calls --

_EARNINGS_BEAT_TEMPLATES = [
    (
        "{ticker} {quarter} earnings call. Revenue was ${revenue}B, up {rev_growth}% "
        "year over year, beating estimates by ${beat_amount}M. {segment} revenue "
        "grew {seg_growth}%. Operating margin expanded {margin_bps} basis points "
        "to {margin}%. Management raised full-year guidance. {positive_comment}"
    ),
    (
        "Good afternoon. {ticker} reported strong {quarter} results. Revenue "
        "of ${revenue}B exceeded consensus by {beat_pct}%. {metric_name} surged "
        "{metric_growth}% to ${metric_value}B. EPS of ${eps} beat estimates by "
        "${eps_beat}. {positive_comment} Outlook remains very positive."
    ),
    (
        "{ticker} {quarter} earnings exceeded expectations across all metrics. "
        "Revenue was ${revenue}B, up {rev_growth}% YoY. Gross margin improved "
        "to {margin}%. Free cash flow was ${fcf}B, up {fcf_growth}%. "
        "{positive_comment} Announced ${buyback}B share buyback program."
    ),
    (
        "Welcome to {ticker} {quarter} earnings conference call. Record "
        "revenue of ${revenue}B, increasing {rev_growth}% year over year. "
        "{segment} delivered exceptional results with {seg_growth}% growth. "
        "Net income rose {ni_growth}% to ${ni}B. {positive_comment}"
    ),
]

_EARNINGS_MISS_TEMPLATES = [
    (
        "{ticker} {quarter} earnings call. Revenue was ${revenue}B, {rev_change} "
        "{rev_growth}% year over year, missing estimates by ${miss_amount}M. "
        "{segment} revenue {seg_direction} {seg_change}%. Operating margin "
        "contracted {margin_bps} basis points to {margin}%. Management "
        "lowered full-year guidance. {negative_comment}"
    ),
    (
        "{ticker} reported disappointing {quarter} results. Revenue of "
        "${revenue}B fell short of the ${expected}B consensus. {metric_name} "
        "{metric_direction} {metric_change}% to ${metric_value}B. EPS of "
        "${eps} missed estimates by ${eps_miss}. {negative_comment} "
        "Outlook was cautious."
    ),
    (
        "{ticker} {quarter} earnings missed expectations. Revenue was "
        "${revenue}B, {rev_change} {rev_growth}% YoY. Gross margin "
        "declined to {margin}%. Free cash flow was negative ${fcf}B. "
        "{negative_comment} Announced workforce reduction of {layoff_pct}%."
    ),
    (
        "{ticker} {quarter} results fell short across key metrics. Revenue "
        "of ${revenue}B declined {rev_growth}%. {segment} posted weakness "
        "with {seg_direction} {seg_change}%. Management cited {headwind}. "
        "Guidance cut for the {ordinal} consecutive quarter. {negative_comment}"
    ),
]

_POSITIVE_COMMENTS = [
    "Customer demand remains exceptionally strong.",
    "AI-driven growth is accelerating across the business.",
    "Record backlog gives us confidence in sustained growth.",
    "Seeing broad-based strength across all geographies.",
    "New product adoption is ahead of our expectations.",
    "Pipeline is the strongest in company history.",
    "Market share gains continued to accelerate.",
    "We are executing well on our strategic priorities.",
    "Strong momentum in digital transformation initiatives.",
    "Recurring revenue growth demonstrates business durability.",
]

_NEGATIVE_COMMENTS = [
    "Consumer spending has weakened significantly.",
    "Competitive pressures are intensifying in our core markets.",
    "Macro uncertainty continues to impact demand.",
    "We are taking decisive cost actions to protect margins.",
    "The demand environment remains challenging.",
    "Inventory destocking is taking longer than expected.",
    "Price erosion is accelerating in key segments.",
    "Customer deferrals increased in the quarter.",
    "We acknowledge execution shortfalls this quarter.",
    "Market conditions deteriorated more than anticipated.",
]

_NEWS_POSITIVE_TEMPLATES = [
    (
        "{catalyst} sends markets higher. {index} rallies {move}% on the news. "
        "{sector} stocks lead gains. Investor sentiment improves broadly. "
        "{follow_up}"
    ),
    (
        "Breaking: {catalyst}. Stocks surge as {driver}. {index} up {move}% "
        "in early trading. {sector} sector at {timeref} highs. {follow_up}"
    ),
    (
        "{catalyst}. Markets respond positively with {index} gaining {move}%. "
        "Broad-based rally across sectors. {follow_up} Risk appetite increases."
    ),
]

_NEWS_NEGATIVE_TEMPLATES = [
    (
        "{catalyst} rattles markets. {index} falls {move}% on the news. "
        "{sector} stocks lead losses. Fear gauge VIX spikes {vix_move}%. "
        "{follow_up}"
    ),
    (
        "Breaking: {catalyst}. Stocks plunge as {driver}. {index} down {move}% "
        "in heavy volume. {sector} sector sells off sharply. {follow_up}"
    ),
    (
        "{catalyst}. Markets react negatively with {index} dropping {move}%. "
        "Broad-based selling across sectors. {follow_up} Risk-off sentiment dominates."
    ),
]

_POSITIVE_CATALYSTS = [
    "Federal Reserve signals rate cuts ahead",
    "Inflation data comes in below expectations",
    "Jobs report shows strong but not overheating labor market",
    "GDP growth beats consensus estimates",
    "Trade deal breakthrough announced",
    "Consumer confidence hits multi-year high",
    "Corporate earnings season exceeds expectations",
    "Manufacturing PMI returns to expansion territory",
    "Retail sales surge past forecasts",
    "Housing starts jump on lower rate expectations",
    "Tech sector reports record AI-driven revenue",
    "Oil prices stabilize supporting energy sector",
    "US dollar weakens boosting multinational earnings outlook",
    "Credit conditions ease as banks loosen lending standards",
    "Productivity growth accelerates reducing inflation pressure",
]

_NEGATIVE_CATALYSTS = [
    "Federal Reserve signals rates will stay higher for longer",
    "Inflation data surprises to the upside",
    "Jobs report triggers recession fears",
    "GDP growth stalls amid consumer weakness",
    "Trade tensions escalate with new tariffs announced",
    "Consumer confidence plunges to multi-year low",
    "Earnings season reveals widespread margin compression",
    "Manufacturing PMI falls deeper into contraction",
    "Retail sales decline sharply missing all estimates",
    "Housing market freezes as mortgage rates spike",
    "Major bank reports surge in loan delinquencies",
    "Oil prices spike on supply disruption fears",
    "Geopolitical crisis sends shockwaves through markets",
    "Credit conditions tighten as defaults rise",
    "Yield curve inversion deepens signaling recession risk",
]

_QUARTERS = [
    "Q1 2023",
    "Q2 2023",
    "Q3 2023",
    "Q4 2023",
    "Q1 2024",
    "Q2 2024",
    "Q3 2024",
    "Q4 2024",
]

_DATES_2023 = [
    "2023-01-25",
    "2023-02-02",
    "2023-02-15",
    "2023-03-10",
    "2023-04-20",
    "2023-04-27",
    "2023-05-04",
    "2023-05-18",
    "2023-07-20",
    "2023-07-27",
    "2023-08-03",
    "2023-08-10",
    "2023-10-19",
    "2023-10-26",
    "2023-11-02",
    "2023-11-16",
]

_DATES_2024 = [
    "2024-01-25",
    "2024-02-01",
    "2024-02-15",
    "2024-03-07",
    "2024-04-18",
    "2024-04-25",
    "2024-05-01",
    "2024-05-22",
    "2024-07-18",
    "2024-07-25",
    "2024-08-01",
    "2024-08-08",
    "2024-10-17",
    "2024-10-24",
    "2024-10-31",
    "2024-11-07",
]

_ALL_DATES = _DATES_2023 + _DATES_2024

_SECTORS = ["Technology", "Financials", "Healthcare", "Consumer", "Industrials", "Energy"]
_INDICES = ["S&P 500", "Nasdaq", "Dow Jones", "Russell 2000"]
_SEGMENTS = [
    "Cloud services",
    "Digital advertising",
    "Subscription",
    "Enterprise software",
    "Hardware",
    "Services",
    "Data Center",
    "Consumer electronics",
    "Investment banking",
    "Wealth management",
    "Pharmaceutical",
    "Medical devices",
    "E-commerce",
    "Retail",
]
_HEADWINDS = [
    "macroeconomic uncertainty",
    "foreign exchange headwinds",
    "supply chain disruptions",
    "competitive pressures",
    "consumer weakness",
    "regulatory challenges",
    "input cost inflation",
    "inventory destocking",
]


def _rand_revenue(ticker: str) -> float:
    """Generate plausible revenue based on ticker."""
    large = {"AAPL", "AMZN", "GOOGL", "MSFT", "WMT", "UNH"}
    mid = {"META", "NVDA", "JPM", "BAC", "COST", "HD", "JNJ", "PFE"}
    if ticker in large:
        return round(random.uniform(50, 170), 1)
    elif ticker in mid:
        return round(random.uniform(10, 50), 1)
    else:
        return round(random.uniform(1, 15), 1)


def generate_earnings(
    n: int = 200,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Generate n synthetic earnings call records.

    Roughly 55% beats (up) and 45% misses (down) to reflect
    the real-world tendency for companies to beat estimates.

    Args:
        n: Number of records to generate.
        seed: Random seed for reproducibility.

    Returns:
        List of orin-format records.
    """
    rng = random.Random(seed)
    records = []

    for i in range(n):
        ticker = rng.choice(ALL_TICKERS)
        date = rng.choice(_ALL_DATES)
        quarter = rng.choice(_QUARTERS)
        revenue = _rand_revenue(ticker)
        segment = rng.choice(_SEGMENTS)

        is_beat = rng.random() < 0.55

        if is_beat:
            template = rng.choice(_EARNINGS_BEAT_TEMPLATES)
            rev_growth = rng.randint(5, 45)
            margin = round(rng.uniform(20, 55), 1)
            magnitude = round(rng.uniform(0.01, 0.15), 3)
            # Add some noise: 10% chance a "beat" actually goes down
            if rng.random() < 0.10:
                direction = "down"
                magnitude = round(rng.uniform(0.005, 0.03), 3)
            else:
                direction = "up"

            text = template.format(
                ticker=ticker,
                quarter=quarter,
                revenue=revenue,
                rev_growth=rev_growth,
                beat_amount=rng.randint(20, 500),
                beat_pct=round(rng.uniform(0.5, 5.0), 1),
                segment=segment,
                seg_growth=rng.randint(10, 60),
                margin_bps=rng.randint(50, 300),
                margin=margin,
                positive_comment=rng.choice(_POSITIVE_COMMENTS),
                metric_name=segment,
                metric_growth=rng.randint(15, 80),
                metric_value=round(revenue * rng.uniform(0.2, 0.6), 1),
                eps=round(rng.uniform(1.5, 8.0), 2),
                eps_beat=round(rng.uniform(0.05, 0.50), 2),
                fcf=round(revenue * rng.uniform(0.05, 0.25), 1),
                fcf_growth=rng.randint(10, 50),
                buyback=rng.randint(2, 25),
                ni=round(revenue * rng.uniform(0.1, 0.3), 1),
                ni_growth=rng.randint(8, 40),
            )
        else:
            template = rng.choice(_EARNINGS_MISS_TEMPLATES)
            rev_growth = rng.randint(1, 20)
            margin = round(rng.uniform(12, 35), 1)
            magnitude = round(rng.uniform(0.01, 0.20), 3)
            # 10% chance a "miss" actually goes up (sell-the-rumor)
            if rng.random() < 0.10:
                direction = "up"
                magnitude = round(rng.uniform(0.005, 0.03), 3)
            else:
                direction = "down"

            rev_change = rng.choice(["down", "declining", "falling"])
            seg_direction = rng.choice(["declined", "fell", "dropped"])
            text = template.format(
                ticker=ticker,
                quarter=quarter,
                revenue=revenue,
                rev_change=rev_change,
                rev_growth=rev_growth,
                miss_amount=rng.randint(30, 600),
                expected=round(revenue + rng.uniform(0.5, 3.0), 1),
                segment=segment,
                seg_direction=seg_direction,
                seg_change=rng.randint(3, 25),
                margin_bps=rng.randint(50, 250),
                margin=margin,
                negative_comment=rng.choice(_NEGATIVE_COMMENTS),
                metric_name=segment,
                metric_direction=rng.choice(["declined", "fell", "dropped"]),
                metric_change=rng.randint(5, 30),
                metric_value=round(revenue * rng.uniform(0.15, 0.5), 1),
                eps=round(rng.uniform(0.5, 4.0), 2),
                eps_miss=round(rng.uniform(0.05, 0.40), 2),
                fcf=round(revenue * rng.uniform(0.02, 0.10), 1),
                headwind=rng.choice(_HEADWINDS),
                ordinal=rng.choice(["second", "third"]),
                layoff_pct=rng.randint(5, 15),
            )

        records.append(
            {
                "text": text,
                "ticker": ticker,
                "date": date,
                "source": "earnings_call",
                "outcome": {
                    "direction": direction,
                    "magnitude": magnitude,
                    "timeframe": "1d",
                },
            }
        )

    return records


def generate_news(
    n: int = 150,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Generate n synthetic financial news records."""
    rng = random.Random(seed)
    records = []

    for i in range(n):
        is_positive = rng.random() < 0.50
        date = rng.choice(_ALL_DATES)
        sector = rng.choice(_SECTORS)
        index = rng.choice(_INDICES)

        if is_positive:
            template = rng.choice(_NEWS_POSITIVE_TEMPLATES)
            catalyst = rng.choice(_POSITIVE_CATALYSTS)
            move = round(rng.uniform(0.3, 3.0), 1)
            magnitude = round(rng.uniform(0.005, 0.04), 3)
            direction = "down" if rng.random() < 0.08 else "up"
            ticker = rng.choice(["SPY", "QQQ", "IWM", "DIA"])
            text = template.format(
                catalyst=catalyst,
                index=index,
                move=move,
                sector=sector,
                driver="optimism grows",
                timeref=rng.choice(["52-week", "multi-month", "record"]),
                follow_up=rng.choice(
                    [
                        "Bond yields decline.",
                        "Dollar weakens.",
                        "Breadth improves.",
                        "Volume surges on buying.",
                        "Options market signals bullish positioning.",
                    ]
                ),
            )
        else:
            template = rng.choice(_NEWS_NEGATIVE_TEMPLATES)
            catalyst = rng.choice(_NEGATIVE_CATALYSTS)
            move = round(rng.uniform(0.3, 4.0), 1)
            magnitude = round(rng.uniform(0.005, 0.06), 3)
            direction = "up" if rng.random() < 0.08 else "down"
            ticker = rng.choice(["SPY", "QQQ", "IWM", "DIA"])
            text = template.format(
                catalyst=catalyst,
                index=index,
                move=move,
                sector=sector,
                driver="panic selling accelerates",
                vix_move=rng.randint(10, 45),
                follow_up=rng.choice(
                    [
                        "Safe haven flows into treasuries.",
                        "Gold prices spike.",
                        "Credit spreads widen.",
                        "Put volume surges.",
                        "Defensive sectors outperform.",
                    ]
                ),
            )

        if rng.random() < 0.08:
            magnitude = round(rng.uniform(0.0, 0.004), 3)
            direction = "flat"

        records.append(
            {
                "text": text,
                "ticker": ticker,
                "date": date,
                "source": "news",
                "outcome": {
                    "direction": direction,
                    "magnitude": magnitude,
                    "timeframe": "1d",
                },
            }
        )

    return records


def generate_filings(
    n: int = 100,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Generate n synthetic SEC filing records."""
    rng = random.Random(seed)
    records = []
    form_types = ["10-K", "10-Q", "8-K"]

    _positive_8k = [
        "announces ${amount}B share buyback program",
        "declares special dividend of ${div} per share",
        "completes acquisition of {target} for ${amount}B",
        "reports record quarterly revenue and raises guidance",
        "announces strategic partnership with major technology provider",
    ]
    _negative_8k = [
        "announces workforce reduction of {pct}% globally",
        "reports material weakness in internal controls",
        "discloses SEC investigation into accounting practices",
        "announces CEO departure effective immediately",
        "reports significant increase in credit loss provisions",
        "suspends dividend and announces restructuring charges of ${amount}M",
    ]

    for i in range(n):
        ticker = rng.choice(ALL_TICKERS)
        date = rng.choice(_ALL_DATES)
        form = rng.choice(form_types)
        revenue = _rand_revenue(ticker)
        is_positive = rng.random() < 0.52

        if form == "8-K":
            if is_positive:
                event = rng.choice(_positive_8k).format(
                    amount=rng.randint(2, 30),
                    div=round(rng.uniform(0.5, 5.0), 2),
                    target="a leading industry player",
                )
                text = (
                    f"FORM 8-K CURRENT REPORT - {ticker}. The Company {event}. "
                    f"Management expressed confidence in the company's financial "
                    f"position and growth trajectory. {rng.choice(_POSITIVE_COMMENTS)}"
                )
                direction = "up"
                magnitude = round(rng.uniform(0.01, 0.10), 3)
            else:
                event = rng.choice(_negative_8k).format(
                    pct=rng.randint(5, 20),
                    amount=rng.randint(100, 800),
                )
                text = (
                    f"FORM 8-K CURRENT REPORT - {ticker}. The Company {event}. "
                    f"Management cited challenging market conditions and the need "
                    f"for operational efficiency. {rng.choice(_NEGATIVE_COMMENTS)}"
                )
                direction = "down"
                magnitude = round(rng.uniform(0.01, 0.08), 3)
        else:
            if is_positive:
                rev_growth = rng.randint(5, 35)
                margin = round(rng.uniform(22, 55), 1)
                text = (
                    f"FORM {form} {'ANNUAL' if form == '10-K' else 'QUARTERLY'} "
                    f"REPORT - {ticker}. Revenue {'increased' if rev_growth > 0 else 'was'} "
                    f"{rev_growth}% to ${revenue}B. Operating margin was {margin}%. "
                    f"Cash and equivalents totaled ${round(revenue * rng.uniform(0.1, 0.4), 1)}B. "
                    f"Free cash flow improved {rng.randint(10, 40)}%. "
                    f"{rng.choice(_POSITIVE_COMMENTS)}"
                )
                direction = "up"
                magnitude = round(rng.uniform(0.005, 0.06), 3)
            else:
                rev_decline = rng.randint(2, 20)
                margin = round(rng.uniform(8, 28), 1)
                text = (
                    f"FORM {form} {'ANNUAL' if form == '10-K' else 'QUARTERLY'} "
                    f"REPORT - {ticker}. Revenue declined {rev_decline}% to "
                    f"${revenue}B. Operating margin contracted to {margin}%. "
                    f"Long-term debt increased to "
                    f"${round(revenue * rng.uniform(0.5, 1.5), 1)}B. "
                    f"Goodwill impairment charges of "
                    f"${round(rng.uniform(0.5, 5.0), 1)}B recorded. "
                    f"{rng.choice(_NEGATIVE_COMMENTS)}"
                )
                direction = "down"
                magnitude = round(rng.uniform(0.005, 0.07), 3)

        # Noise: 8% chance direction flips
        if rng.random() < 0.08:
            direction = "up" if direction == "down" else "down"
            magnitude = round(rng.uniform(0.005, 0.02), 3)

        records.append(
            {
                "text": text,
                "ticker": ticker,
                "date": date,
                "source": form,
                "outcome": {
                    "direction": direction,
                    "magnitude": magnitude,
                    "timeframe": "5d",
                },
            }
        )

    return records


def generate_macro(
    n: int = 100,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Generate n synthetic macroeconomic data records."""
    rng = random.Random(seed)
    records = []

    _positive_macro_templates = [
        (
            "{indicator} {positive_verb} to {value}, {comparison} the "
            "{estimate} consensus estimate. {context} {follow_up}"
        ),
        (
            "Federal Reserve {dovish_action}. {detail} Markets rally on "
            "expectations of easier monetary policy. {follow_up}"
        ),
        (
            "{indicator} shows improvement, {positive_verb} to {value} in "
            "{month}. {context} Economic outlook brightens. {follow_up}"
        ),
    ]
    _negative_macro_templates = [
        (
            "{indicator} {negative_verb} to {value}, {comparison} the "
            "{estimate} consensus estimate. {context} {follow_up}"
        ),
        (
            "Federal Reserve {hawkish_action}. {detail} Markets sell off on "
            "expectations of tighter monetary policy. {follow_up}"
        ),
        (
            "{indicator} shows deterioration, {negative_verb} to {value} in "
            "{month}. {context} Recession concerns intensify. {follow_up}"
        ),
    ]

    _indicators = [
        "GDP growth",
        "CPI inflation",
        "Core PCE",
        "ISM Manufacturing PMI",
        "ISM Services PMI",
        "Nonfarm payrolls",
        "Unemployment rate",
        "Consumer Confidence Index",
        "Retail sales",
        "Housing starts",
        "Industrial production",
        "Durable goods orders",
        "University of Michigan sentiment",
        "JOLTS job openings",
    ]
    _months = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]

    for i in range(n):
        is_positive = rng.random() < 0.50
        date = rng.choice(_ALL_DATES)
        indicator = rng.choice(_indicators)
        month = rng.choice(_months)

        if is_positive:
            template = rng.choice(_positive_macro_templates)
            magnitude = round(rng.uniform(0.003, 0.03), 3)
            direction = "down" if rng.random() < 0.10 else "up"
            text = template.format(
                indicator=indicator,
                positive_verb=rng.choice(["rises", "increases", "improves", "surges"]),
                value=f"{rng.uniform(1.5, 5.5):.1f}%",
                comparison=rng.choice(["beating", "above", "exceeding"]),
                estimate=f"{rng.uniform(1.0, 4.0):.1f}%",
                context=rng.choice(
                    [
                        "Economic resilience continues.",
                        "Consumer spending remains robust.",
                        "Labor market shows strength.",
                        "Business investment accelerates.",
                    ]
                ),
                follow_up=rng.choice(
                    [
                        "Soft landing scenario gains credibility.",
                        "Bond yields fall on the data.",
                        "Equity markets respond positively.",
                        "Dollar strengthens modestly.",
                    ]
                ),
                dovish_action=rng.choice(
                    [
                        "signals rate cuts in coming meetings",
                        "announces end of quantitative tightening",
                        "adopts more dovish tone in statement",
                    ]
                ),
                detail=rng.choice(
                    [
                        "Dot plot shows 75bp of cuts by year end.",
                        "Powell emphasizes dual mandate balance.",
                        "Statement removes hawkish language.",
                    ]
                ),
                month=month,
            )
        else:
            template = rng.choice(_negative_macro_templates)
            magnitude = round(rng.uniform(0.005, 0.04), 3)
            direction = "up" if rng.random() < 0.10 else "down"
            text = template.format(
                indicator=indicator,
                negative_verb=rng.choice(["falls", "declines", "drops", "plunges"]),
                value=f"{rng.uniform(0.5, 4.5):.1f}%",
                comparison=rng.choice(["missing", "below", "worse than"]),
                estimate=f"{rng.uniform(1.5, 5.0):.1f}%",
                context=rng.choice(
                    [
                        "Recession fears mount.",
                        "Consumer weakness spreading.",
                        "Labor market softening rapidly.",
                        "Credit conditions tightening.",
                    ]
                ),
                follow_up=rng.choice(
                    [
                        "Flight to safety accelerates.",
                        "Bond yields spike on the news.",
                        "Equity markets sell off broadly.",
                        "VIX jumps on uncertainty.",
                    ]
                ),
                hawkish_action=rng.choice(
                    [
                        "signals rates will stay higher for longer",
                        "raises inflation forecast for the year",
                        "warns of persistent price pressures",
                    ]
                ),
                detail=rng.choice(
                    [
                        "Dot plot shows no cuts until next year.",
                        "Powell emphasizes inflation fight is not over.",
                        "Statement adds hawkish language on prices.",
                    ]
                ),
                month=month,
            )

        records.append(
            {
                "text": text,
                "ticker": rng.choice(["SPY", "QQQ", "IWM", "DIA", "TLT"]),
                "date": date,
                "source": rng.choice(["economic_data", "fed_speech", "central_bank"]),
                "outcome": {
                    "direction": direction,
                    "magnitude": magnitude,
                    "timeframe": rng.choice(["1d", "5d"]),
                },
            }
        )

    return records


def generate_all(
    n_earnings: int = 200,
    n_news: int = 150,
    n_filings: int = 100,
    n_macro: int = 100,
    seed: int = 42,
) -> dict[str, list[dict[str, Any]]]:
    """Generate all dataset types.

    Returns:
        Dict mapping env type to list of records.
    """
    return {
        "earnings": generate_earnings(n_earnings, seed),
        "news": generate_news(n_news, seed + 1),
        "filing": generate_filings(n_filings, seed + 2),
        "macro": generate_macro(n_macro, seed + 3),
    }
