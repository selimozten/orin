"""Data source connectors and sample data generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def generate_sample_earnings() -> list[dict[str, Any]]:
    """Generate sample earnings call data for testing.

    Records are designed with learnable text-outcome patterns:
    beats/records/growth/strong → tends to go up,
    misses/declines/weak/cuts → tends to go down.
    """
    samples = [
        # -- BEATS / POSITIVE --
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
                "JPM Q3 2024 earnings. Net revenue of $43.3 billion, up 6%. "
                "Net income was $12.9 billion. CIB revenue was $17.0 billion. "
                "Investment banking fees up 31%. Credit costs were $3.1 billion. "
                "Net interest income was $23.5 billion. Book value per share "
                "was $113.64. Results exceeded analyst expectations."
            ),
            "ticker": "JPM",
            "date": "2024-10-11",
            "source": "earnings_call",
            "outcome": {"direction": "up", "magnitude": 0.044, "timeframe": "1d"},
        },
        {
            "text": (
                "COST Q4 2024 earnings call. Net sales increased 9.1% to "
                "$78.9 billion. Comparable sales rose 6.9%. E-commerce sales "
                "surged 18.9%. Membership fee revenue was $1.51 billion, up "
                "7.4%. We renewed memberships at a record 93.1% rate in the "
                "US and Canada. Net income rose 12% to $2.35 billion."
            ),
            "ticker": "COST",
            "date": "2024-09-26",
            "source": "earnings_call",
            "outcome": {"direction": "up", "magnitude": 0.031, "timeframe": "1d"},
        },
        {
            "text": (
                "CRM fiscal Q2 2025 earnings. Revenue was $9.33 billion, up "
                "8% year over year, beating estimates by $70 million. "
                "Subscription revenue grew 9%. Operating margin expanded to "
                "33.7%, up 460 basis points. Raised full-year guidance. "
                "Customer AI adoption is accelerating significantly."
            ),
            "ticker": "CRM",
            "date": "2024-08-28",
            "source": "earnings_call",
            "outcome": {"direction": "up", "magnitude": 0.038, "timeframe": "1d"},
        },
        {
            "text": (
                "WMT Q2 FY2025 earnings. Revenue was $169.3 billion, up 4.8%. "
                "US comp sales grew 4.2%. E-commerce sales surged 22%. "
                "Operating income increased 8.5%. Company raised full-year "
                "guidance for the second time. Strong consumer spending "
                "trends continued across all channels."
            ),
            "ticker": "WMT",
            "date": "2024-08-15",
            "source": "earnings_call",
            "outcome": {"direction": "up", "magnitude": 0.065, "timeframe": "1d"},
        },
        {
            "text": (
                "LLY Q2 2024 earnings. Revenue surged 36% to $11.3 billion, "
                "driven by blockbuster Mounjaro and Zepbound sales. Mounjaro "
                "revenue was $3.1 billion, up 215%. Adjusted EPS of $3.92 "
                "beat estimates by $0.59. Company raised full-year revenue "
                "guidance by $3 billion. Pipeline remains very strong."
            ),
            "ticker": "LLY",
            "date": "2024-08-08",
            "source": "earnings_call",
            "outcome": {"direction": "up", "magnitude": 0.098, "timeframe": "1d"},
        },
        {
            "text": (
                "NFLX Q3 2024 earnings. Revenue increased 15% to $9.82 "
                "billion. Net subscriber additions of 5.07 million, well above "
                "expectations. Operating margin was 29.6%, up from 22.4% a "
                "year ago. Ad-supported tier membership grew 35% quarter over "
                "quarter. Raised Q4 revenue guidance above consensus."
            ),
            "ticker": "NFLX",
            "date": "2024-10-17",
            "source": "earnings_call",
            "outcome": {"direction": "up", "magnitude": 0.112, "timeframe": "1d"},
        },
        {
            "text": (
                "V fiscal Q3 2024 earnings. Net revenues increased 10% to "
                "$8.9 billion. Payment volumes grew 8%. Cross-border volumes "
                "surged 14%. Processed transactions up 10% to 58.7 billion. "
                "EPS of $2.42 beat consensus by $0.11. Strong consumer "
                "spending trends across all regions."
            ),
            "ticker": "V",
            "date": "2024-07-23",
            "source": "earnings_call",
            "outcome": {"direction": "up", "magnitude": 0.022, "timeframe": "1d"},
        },
        {
            "text": (
                "UNH Q3 2024 earnings. Revenue grew 9% to $100.8 billion. "
                "Optum health services revenue was $62.9 billion, up 13%. "
                "Medical care ratio remained favorable at 85.2%. Adjusted "
                "EPS of $7.15 exceeded estimates. Reaffirmed full-year "
                "guidance. Enrollment growth was robust across all segments."
            ),
            "ticker": "UNH",
            "date": "2024-10-15",
            "source": "earnings_call",
            "outcome": {"direction": "up", "magnitude": 0.041, "timeframe": "1d"},
        },
        {
            "text": (
                "MA Q2 2024 earnings. Net revenue rose 11% to $6.96 billion. "
                "Gross dollar volume increased 9%. Cross-border volume surged "
                "17%. Value-added services revenue grew 18%. Adjusted EPS of "
                "$3.59 beat estimates. Management highlighted strong global "
                "consumer spending and accelerating digital payment adoption."
            ),
            "ticker": "MA",
            "date": "2024-07-31",
            "source": "earnings_call",
            "outcome": {"direction": "up", "magnitude": 0.033, "timeframe": "1d"},
        },
        {
            "text": (
                "GS Q3 2024 earnings. Revenue was $12.7 billion, up 7%. "
                "Investment banking fees surged 20% to $1.87 billion. "
                "FICC revenue was $2.96 billion. Equities trading revenue "
                "hit a record $3.50 billion, up 18%. Asset management fees "
                "grew 9%. ROE improved to 10.4%. Beat expectations."
            ),
            "ticker": "GS",
            "date": "2024-10-15",
            "source": "earnings_call",
            "outcome": {"direction": "up", "magnitude": 0.029, "timeframe": "1d"},
        },
        {
            "text": (
                "ORCL fiscal Q1 2025 earnings. Total revenue was $13.31 "
                "billion, up 7%. Cloud services revenue grew 21% to $5.62 "
                "billion. Cloud infrastructure revenue surged 45%. "
                "Remaining performance obligations were $99 billion, up 52%. "
                "Operating margin expanded. Demand for AI infrastructure "
                "is driving record cloud bookings."
            ),
            "ticker": "ORCL",
            "date": "2024-09-09",
            "source": "earnings_call",
            "outcome": {"direction": "up", "magnitude": 0.087, "timeframe": "1d"},
        },
        {
            "text": (
                "AVGO fiscal Q3 2024 earnings. Revenue surged 47% to $13.07 "
                "billion. AI revenue was $3.1 billion, up 360% year over year. "
                "Infrastructure software revenue was $5.8 billion following "
                "VMware integration. Adjusted EBITDA margin was 63%. "
                "Raised full-year revenue guidance. AI networking demand "
                "continues to exceed expectations."
            ),
            "ticker": "AVGO",
            "date": "2024-09-05",
            "source": "earnings_call",
            "outcome": {"direction": "up", "magnitude": 0.042, "timeframe": "1d"},
        },
        {
            "text": (
                "HD Q2 FY2025 earnings. Revenue increased 0.6% to $43.2 "
                "billion. Comparable sales declined 1.3% but beat estimates "
                "of -2.5%. Pro customer segment showed strength with comp "
                "growth of 2.1%. Gross margin expanded 30 basis points to "
                "33.7%. Company raised full-year earnings guidance citing "
                "resilient demand from professional contractors."
            ),
            "ticker": "HD",
            "date": "2024-08-13",
            "source": "earnings_call",
            "outcome": {"direction": "up", "magnitude": 0.018, "timeframe": "1d"},
        },
        # -- MISSES / NEGATIVE --
        {
            "text": (
                "Welcome to MSFT fiscal Q4 2024 earnings conference call. "
                "Revenue was $64.7 billion, increasing 15% year over year. "
                "However, Azure growth of 29% missed expectations of 31%. "
                "Cloud guidance disappointed. Capital expenditure surged to "
                "$19 billion, raising concerns about AI spending returns."
            ),
            "ticker": "MSFT",
            "date": "2024-07-30",
            "source": "earnings_call",
            "outcome": {"direction": "down", "magnitude": 0.032, "timeframe": "1d"},
        },
        {
            "text": (
                "Good evening. AMZN Q2 2024 results. Net sales increased 10% "
                "to $148.0 billion but Q3 revenue guidance of $154-158.5 "
                "billion missed the $158.2 billion consensus. AWS growth was "
                "19%, below expectations. Operating margin pressure from "
                "increased AI infrastructure spending concerned investors."
            ),
            "ticker": "AMZN",
            "date": "2024-08-01",
            "source": "earnings_call",
            "outcome": {"direction": "down", "magnitude": 0.045, "timeframe": "1d"},
        },
        {
            "text": (
                "TSLA Q2 2024 earnings call. Total revenue was $25.5 billion, "
                "up only 2% year over year. Automotive revenue declined 7% to "
                "$19.9 billion. Automotive gross margin fell to 18.5%, well "
                "below expectations. Price cuts continued to pressure margins. "
                "Vehicle deliveries missed estimates significantly."
            ),
            "ticker": "TSLA",
            "date": "2024-07-23",
            "source": "earnings_call",
            "outcome": {"direction": "down", "magnitude": 0.082, "timeframe": "1d"},
        },
        {
            "text": (
                "GOOGL Q2 2024 earnings. Revenue of $84.7 billion was in "
                "line but YouTube ads revenue of $8.7 billion missed the "
                "$8.9 billion estimate. Cloud growth decelerated. Concerns "
                "about AI competition intensified. Capital expenditure of "
                "$13.2 billion raised questions about spending discipline."
            ),
            "ticker": "GOOGL",
            "date": "2024-07-23",
            "source": "earnings_call",
            "outcome": {"direction": "down", "magnitude": 0.051, "timeframe": "1d"},
        },
        {
            "text": (
                "META Q2 2024 earnings conference call. Revenue was $39.1 "
                "billion, up 22%, but Q3 guidance was below expectations. "
                "Reality Labs posted an operating loss of $4.5 billion. "
                "Capital expenditure guidance raised to $37-40 billion. "
                "Investors concerned about the pace of metaverse spending "
                "with unclear return on investment."
            ),
            "ticker": "META",
            "date": "2024-07-31",
            "source": "earnings_call",
            "outcome": {"direction": "down", "magnitude": 0.041, "timeframe": "1d"},
        },
        {
            "text": (
                "INTC Q2 2024 earnings. Revenue was $12.8 billion, down 1% "
                "year over year. Gross margin fell to 38.7%, well below "
                "guidance. Q3 revenue guidance of $12.5-13.5 billion was "
                "significantly below consensus of $14.4 billion. Announced "
                "15,000 job cuts and dividend suspension. Foundry business "
                "posted $2.8 billion operating loss."
            ),
            "ticker": "INTC",
            "date": "2024-08-01",
            "source": "earnings_call",
            "outcome": {"direction": "down", "magnitude": 0.261, "timeframe": "1d"},
        },
        {
            "text": (
                "DIS fiscal Q3 2024 earnings. Revenue was $23.2 billion, up "
                "4%, but parks and experiences segment missed expectations "
                "with 3% growth. Parks operating income fell 3%. Management "
                "warned of consumer softness in domestic parks. Streaming "
                "losses narrowed but subscriber growth disappointed."
            ),
            "ticker": "DIS",
            "date": "2024-08-07",
            "source": "earnings_call",
            "outcome": {"direction": "down", "magnitude": 0.043, "timeframe": "1d"},
        },
        {
            "text": (
                "SNAP Q2 2024 earnings. Revenue increased 16% to $1.24 "
                "billion but missed the $1.25 billion estimate. Daily active "
                "users grew 9% to 432 million. However, Q3 revenue guidance "
                "of $1.34-1.38 billion fell short of consensus. ARPU "
                "growth decelerated. Concerns about competitive pressure "
                "from TikTok and Instagram Reels."
            ),
            "ticker": "SNAP",
            "date": "2024-07-25",
            "source": "earnings_call",
            "outcome": {"direction": "down", "magnitude": 0.158, "timeframe": "1d"},
        },
        {
            "text": (
                "BA Q2 2024 earnings. Revenue was $16.9 billion, down 15% "
                "year over year. Commercial airplane deliveries fell 32%. "
                "Operating loss was $1.4 billion. Free cash flow was negative "
                "$4.3 billion. Quality issues continue to weigh on production. "
                "New CEO announced major restructuring plan."
            ),
            "ticker": "BA",
            "date": "2024-07-31",
            "source": "earnings_call",
            "outcome": {"direction": "down", "magnitude": 0.072, "timeframe": "1d"},
        },
        {
            "text": (
                "NKE fiscal Q4 2024 earnings. Revenue declined 2% to $12.6 "
                "billion, missing estimates. Direct-to-consumer sales fell 8%. "
                "Gross margin contracted 150 basis points. Full-year FY2025 "
                "guidance cut dramatically, expecting revenue to decline "
                "mid-single digits. Inventory levels remained elevated. "
                "CEO acknowledged execution missteps."
            ),
            "ticker": "NKE",
            "date": "2024-06-27",
            "source": "earnings_call",
            "outcome": {"direction": "down", "magnitude": 0.198, "timeframe": "1d"},
        },
        {
            "text": (
                "SBUX fiscal Q3 2024 earnings. Revenue was $9.1 billion, "
                "below the $9.2 billion estimate. Global comparable store "
                "sales declined 3%. US comp sales fell 2%. China comp sales "
                "plunged 14%. Traffic declined across all regions. Company "
                "lowered full-year guidance for the second consecutive "
                "quarter. Management cited consumer pullback."
            ),
            "ticker": "SBUX",
            "date": "2024-07-30",
            "source": "earnings_call",
            "outcome": {"direction": "down", "magnitude": 0.055, "timeframe": "1d"},
        },
        {
            "text": (
                "PYPL Q2 2024 earnings. Revenue was $7.89 billion, up 8%. "
                "However, transaction margin dollars grew only 3%, missing "
                "expectations. Total payment volume of $416.8 billion was "
                "below estimates. Active accounts declined to 429 million. "
                "Competition from Apple Pay and other digital wallets is "
                "intensifying. Lowered transaction margin guidance."
            ),
            "ticker": "PYPL",
            "date": "2024-07-30",
            "source": "earnings_call",
            "outcome": {"direction": "down", "magnitude": 0.068, "timeframe": "1d"},
        },
        {
            "text": (
                "PFE Q2 2024 earnings. Revenue was $13.3 billion, down 6% "
                "year over year excluding COVID products. COVID vaccine "
                "revenue was $195 million, down 87%. Paxlovid revenue "
                "dropped 67%. Adjusted EPS of $0.60 missed estimates. "
                "Cost savings program expanded by $1.5 billion. Pipeline "
                "setbacks with two trial failures announced."
            ),
            "ticker": "PFE",
            "date": "2024-07-30",
            "source": "earnings_call",
            "outcome": {"direction": "down", "magnitude": 0.035, "timeframe": "1d"},
        },
        # -- MORE BEATS --
        {
            "text": (
                "PANW fiscal Q4 2024 earnings. Revenue was $2.19 billion, "
                "up 12%. Annual recurring revenue surged to $4.22 billion. "
                "Next-generation security ARR grew 43%. Operating margin "
                "expanded to 27.3%. RPO grew 20% to $12.7 billion. "
                "Management reported record large deal activity and raised "
                "fiscal 2025 guidance above consensus."
            ),
            "ticker": "PANW",
            "date": "2024-08-19",
            "source": "earnings_call",
            "outcome": {"direction": "up", "magnitude": 0.072, "timeframe": "1d"},
        },
        {
            "text": (
                "AMAT fiscal Q3 2024 earnings. Revenue was $6.78 billion, "
                "up 5%, beating estimates by $210 million. Semiconductor "
                "systems revenue was $4.92 billion. ICAPS revenue grew 20%. "
                "Adjusted EPS of $2.12 exceeded consensus by $0.13. "
                "Guided Q4 revenue above expectations driven by AI chip "
                "manufacturing demand."
            ),
            "ticker": "AMAT",
            "date": "2024-08-15",
            "source": "earnings_call",
            "outcome": {"direction": "up", "magnitude": 0.034, "timeframe": "1d"},
        },
        {
            "text": (
                "NOW Q2 2024 earnings. Subscription revenues grew 23% to "
                "$2.54 billion, beating guidance. Remaining performance "
                "obligations of $18.6 billion, up 28%. Current RPO grew 22%. "
                "Operating margin was 29%. Net new ACV growth accelerated. "
                "AI-powered products seeing rapid customer adoption."
            ),
            "ticker": "NOW",
            "date": "2024-07-24",
            "source": "earnings_call",
            "outcome": {"direction": "up", "magnitude": 0.056, "timeframe": "1d"},
        },
        {
            "text": (
                "UBER Q2 2024 earnings. Gross bookings were $39.95 billion, "
                "up 19%. Revenue increased 16% to $10.7 billion. Mobility "
                "trips grew 21%. Delivery orders grew 17%. Free cash flow "
                "was $1.72 billion. Company announced a $7 billion share "
                "buyback program. GAAP profitability continued to improve."
            ),
            "ticker": "UBER",
            "date": "2024-08-06",
            "source": "earnings_call",
            "outcome": {"direction": "up", "magnitude": 0.025, "timeframe": "1d"},
        },
        # -- MORE MISSES --
        {
            "text": (
                "PARA Q2 2024 earnings. Revenue declined 11% to $6.44 "
                "billion, missing estimates. TV media revenue fell 17%. "
                "Direct-to-consumer losses widened despite subscriber growth. "
                "Advertising revenue declined 7%. The company announced "
                "merger discussions with Skydance Media, creating uncertainty. "
                "Free cash flow was negative $300 million."
            ),
            "ticker": "PARA",
            "date": "2024-08-07",
            "source": "earnings_call",
            "outcome": {"direction": "down", "magnitude": 0.062, "timeframe": "1d"},
        },
        {
            "text": (
                "COIN Q2 2024 earnings. Revenue was $1.45 billion, below the "
                "$1.63 billion estimate. Transaction revenue fell 27% quarter "
                "over quarter as crypto trading volumes declined. Consumer "
                "transaction revenue dropped 29%. Subscription revenue missed. "
                "Management warned of continued market headwinds."
            ),
            "ticker": "COIN",
            "date": "2024-08-01",
            "source": "earnings_call",
            "outcome": {"direction": "down", "magnitude": 0.083, "timeframe": "1d"},
        },
        {
            "text": (
                "MCD Q2 2024 earnings. Revenue was $6.49 billion, below "
                "estimates of $6.61 billion. Global comparable sales declined "
                "1%, the first drop since Q4 2020. US comp sales fell 0.7%. "
                "International comp sales declined 1.1%. Value perception "
                "issues persisted. Management announced $5 value meal to "
                "drive traffic recovery."
            ),
            "ticker": "MCD",
            "date": "2024-07-29",
            "source": "earnings_call",
            "outcome": {"direction": "down", "magnitude": 0.037, "timeframe": "1d"},
        },
    ]
    return samples


def generate_sample_news() -> list[dict[str, Any]]:
    """Generate sample financial news data for testing."""
    samples = [
        # -- POSITIVE NEWS --
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
                "CPI report shows inflation cooling to 2.9% year over year, "
                "first time below 3% since March 2021. Core CPI rises 0.2% "
                "month over month. Markets rally on rate cut expectations."
            ),
            "ticker": "SPY",
            "date": "2024-08-14",
            "source": "news",
            "outcome": {"direction": "up", "magnitude": 0.019, "timeframe": "1d"},
        },
        {
            "text": (
                "S&P 500 hits new all-time high as tech stocks surge. "
                "AI momentum continues to drive market gains. Breadth "
                "improves with financials and industrials joining the rally. "
                "Market sentiment is strongly bullish."
            ),
            "ticker": "SPY",
            "date": "2024-09-19",
            "source": "news",
            "outcome": {"direction": "up", "magnitude": 0.012, "timeframe": "1d"},
        },
        {
            "text": (
                "Federal Reserve cuts interest rates by 50 basis points, "
                "the first rate cut since 2020. Fed funds rate now at "
                "4.75-5.00%. Powell signals further cuts ahead. Markets "
                "cheer the aggressive move. Bond yields fall sharply."
            ),
            "ticker": "SPY",
            "date": "2024-09-18",
            "source": "news",
            "outcome": {"direction": "up", "magnitude": 0.025, "timeframe": "1d"},
        },
        {
            "text": (
                "Retail sales rise 1.0% in August, well above the 0.2% "
                "expected. Consumer spending remains strong despite rate "
                "hikes. Core retail sales ex-autos up 0.6%. Data suggests "
                "soft landing is achievable. Stocks rally broadly."
            ),
            "ticker": "SPY",
            "date": "2024-09-17",
            "source": "news",
            "outcome": {"direction": "up", "magnitude": 0.008, "timeframe": "1d"},
        },
        {
            "text": (
                "US-China trade talks resume with constructive tone. "
                "Both sides agree to reduce tariffs on key goods. Markets "
                "surge on de-escalation hopes. Semiconductor stocks lead "
                "gains. Risk appetite improves globally."
            ),
            "ticker": "SPY",
            "date": "2024-06-15",
            "source": "news",
            "outcome": {"direction": "up", "magnitude": 0.022, "timeframe": "1d"},
        },
        {
            "text": (
                "Initial jobless claims fall to 227,000, below expectations "
                "of 235,000. Continuing claims also decline. Labor market "
                "remains resilient. Goldilocks scenario supports equities. "
                "Markets see reduced recession probability."
            ),
            "ticker": "SPY",
            "date": "2024-08-22",
            "source": "news",
            "outcome": {"direction": "up", "magnitude": 0.011, "timeframe": "1d"},
        },
        {
            "text": (
                "Major pharmaceutical breakthrough: new Alzheimer's drug "
                "shows 35% slowing of cognitive decline in Phase 3 trial. "
                "FDA fast-track designation expected. Biotech sector rallies "
                "on the news. Multiple healthcare stocks at 52-week highs."
            ),
            "ticker": "XLV",
            "date": "2024-07-10",
            "source": "news",
            "outcome": {"direction": "up", "magnitude": 0.027, "timeframe": "1d"},
        },
        {
            "text": (
                "Housing starts surge 9.6% in August to 1.36 million "
                "annualized rate. Building permits up 4.9%. Homebuilder "
                "confidence improves. Lower mortgage rate expectations "
                "driving buyer demand. Construction stocks rally sharply."
            ),
            "ticker": "XHB",
            "date": "2024-09-18",
            "source": "news",
            "outcome": {"direction": "up", "magnitude": 0.018, "timeframe": "1d"},
        },
        # -- NEGATIVE NEWS --
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
                "Cash pile grows to record $277 billion. Signal of caution."
            ),
            "ticker": "AAPL",
            "date": "2024-08-03",
            "source": "news",
            "outcome": {"direction": "down", "magnitude": 0.022, "timeframe": "1d"},
        },
        {
            "text": (
                "Global stock markets crash as Japan's Nikkei plunges 12%, "
                "the worst day since 1987. Yen carry trade unwind "
                "accelerates. VIX spikes to 65. Panic selling across all "
                "asset classes. Circuit breakers triggered in multiple markets."
            ),
            "ticker": "SPY",
            "date": "2024-08-05",
            "source": "news",
            "outcome": {"direction": "down", "magnitude": 0.030, "timeframe": "1d"},
        },
        {
            "text": (
                "PPI rises 0.4% in July, above expectations of 0.2%. "
                "Core PPI up 0.3%. Inflation concerns resurface. Rate cut "
                "expectations pare back. Bond yields spike. Growth stocks "
                "sell off sharply on higher-for-longer fears."
            ),
            "ticker": "SPY",
            "date": "2024-08-13",
            "source": "news",
            "outcome": {"direction": "down", "magnitude": 0.015, "timeframe": "1d"},
        },
        {
            "text": (
                "Commercial real estate crisis deepens as major REIT "
                "announces $2.1 billion write-down on office portfolio. "
                "Office vacancy rates hit record 19.6%. Regional bank "
                "exposure to CRE raises systemic concerns. Financial "
                "stocks sell off broadly."
            ),
            "ticker": "XLF",
            "date": "2024-07-22",
            "source": "news",
            "outcome": {"direction": "down", "magnitude": 0.028, "timeframe": "1d"},
        },
        {
            "text": (
                "Geopolitical tensions spike as military conflict escalates "
                "in the Middle East. Oil prices jump 7%. Defense stocks "
                "rise but broader market sells off on uncertainty. Safe "
                "haven flows into treasuries and gold. VIX rises 25%."
            ),
            "ticker": "SPY",
            "date": "2024-10-01",
            "source": "news",
            "outcome": {"direction": "down", "magnitude": 0.019, "timeframe": "1d"},
        },
        {
            "text": (
                "Leading economic indicators decline for 7th consecutive "
                "month, falling 0.6% in August. Conference Board warns of "
                "elevated recession risk. Consumer expectations weaken. "
                "Credit conditions tightening. Stocks fall on growth fears."
            ),
            "ticker": "SPY",
            "date": "2024-09-20",
            "source": "news",
            "outcome": {"direction": "down", "magnitude": 0.014, "timeframe": "1d"},
        },
        {
            "text": (
                "Major US bank reports significant increase in loan "
                "delinquencies. Credit card delinquency rate hits 3.1%, "
                "highest since 2011. Auto loan defaults rising. Consumer "
                "credit stress spreading. Regional bank stocks plunge."
            ),
            "ticker": "KRE",
            "date": "2024-08-12",
            "source": "news",
            "outcome": {"direction": "down", "magnitude": 0.033, "timeframe": "1d"},
        },
        {
            "text": (
                "China's economic slowdown deepens. GDP growth falls to "
                "4.7%, missing the 5.1% target. Youth unemployment hits "
                "new record. Property sector continues to contract. "
                "Deflation concerns mount. Global commodity stocks fall."
            ),
            "ticker": "FXI",
            "date": "2024-07-15",
            "source": "news",
            "outcome": {"direction": "down", "magnitude": 0.041, "timeframe": "1d"},
        },
    ]
    return samples


def generate_sample_filing() -> list[dict[str, Any]]:
    """Generate sample SEC filing data for testing."""
    samples = [
        # -- POSITIVE FILINGS --
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
        {
            "text": (
                "FORM 10-Q QUARTERLY REPORT - NVDA. Revenue was $30.04 "
                "billion, up 122% year over year. Data Center revenue was "
                "$26.3 billion, up 154%. Gross margin was 75.1%. Operating "
                "income was $18.6 billion. Cash generation was $14.5 billion. "
                "Balance sheet remains very strong."
            ),
            "ticker": "NVDA",
            "date": "2024-08-28",
            "source": "10-Q",
            "outcome": {"direction": "up", "magnitude": 0.025, "timeframe": "5d"},
        },
        {
            "text": (
                "FORM 8-K CURRENT REPORT - GOOGL. The Company announces a "
                "$70 billion share buyback program, the largest in company "
                "history. Board also declares first-ever quarterly dividend "
                "of $0.20 per share. Reflects confidence in strong cash flow "
                "generation and commitment to shareholder returns."
            ),
            "ticker": "GOOGL",
            "date": "2024-04-25",
            "source": "8-K",
            "outcome": {"direction": "up", "magnitude": 0.102, "timeframe": "5d"},
        },
        {
            "text": (
                "FORM 10-K ANNUAL REPORT - V. Net revenues increased 11% "
                "to $35.9 billion. Payments volume grew 9% to $14.5 trillion. "
                "Cross-border volume up 16%. Operating margin was 67%. "
                "Returned $17.8 billion to shareholders through buybacks "
                "and dividends. Strong global payment trends continue."
            ),
            "ticker": "V",
            "date": "2024-02-15",
            "source": "10-K",
            "outcome": {"direction": "up", "magnitude": 0.015, "timeframe": "5d"},
        },
        {
            "text": (
                "FORM 8-K CURRENT REPORT - META. The Company announces "
                "first-ever quarterly dividend of $0.50 per share and a "
                "$50 billion share buyback authorization. Operating margin "
                "improved to 41%. Revenue grew 25%. Strong financial position "
                "supports increased capital returns to shareholders."
            ),
            "ticker": "META",
            "date": "2024-02-01",
            "source": "8-K",
            "outcome": {"direction": "up", "magnitude": 0.155, "timeframe": "5d"},
        },
        # -- NEGATIVE FILINGS --
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
                "FORM 10-Q QUARTERLY REPORT - INTC. Revenue declined 1% "
                "to $12.8 billion. Gross margin contracted to 38.7%. "
                "Foundry segment posted operating loss of $2.8 billion. "
                "Client computing revenue declined 8%. Management "
                "acknowledged significant execution challenges. Announced "
                "cost restructuring and workforce reduction of 15,000."
            ),
            "ticker": "INTC",
            "date": "2024-08-01",
            "source": "10-Q",
            "outcome": {"direction": "down", "magnitude": 0.089, "timeframe": "5d"},
        },
        {
            "text": (
                "FORM 8-K CURRENT REPORT - BA. The Company reports "
                "additional $4.7 billion charges related to the 737 MAX "
                "program. Production rate reduced to 25 per month from 38. "
                "Delivery delays expected to continue through 2025. "
                "Cash burn accelerated. Credit rating under review."
            ),
            "ticker": "BA",
            "date": "2024-04-22",
            "source": "8-K",
            "outcome": {"direction": "down", "magnitude": 0.065, "timeframe": "5d"},
        },
        {
            "text": (
                "FORM 10-K ANNUAL REPORT - PFE. Total revenue declined "
                "42% to $58.5 billion. COVID-19 product revenue collapsed "
                "70%. Non-COVID pipeline revenue was flat. Adjusted EPS "
                "declined 48%. Significant inventory write-downs of $5.6 "
                "billion. Long-term debt increased to $61.2 billion."
            ),
            "ticker": "PFE",
            "date": "2024-02-22",
            "source": "10-K",
            "outcome": {"direction": "down", "magnitude": 0.031, "timeframe": "5d"},
        },
        {
            "text": (
                "FORM 8-K CURRENT REPORT - COIN. The Company discloses "
                "SEC Wells notice for potential enforcement action regarding "
                "staking services and asset listings. Management states "
                "it will vigorously defend its position. Legal costs "
                "expected to be material. Regulatory uncertainty intensifies."
            ),
            "ticker": "COIN",
            "date": "2024-03-18",
            "source": "8-K",
            "outcome": {"direction": "down", "magnitude": 0.074, "timeframe": "5d"},
        },
    ]
    return samples


def generate_sample_macro() -> list[dict[str, Any]]:
    """Generate sample macroeconomic data for testing."""
    samples = [
        # -- POSITIVE MACRO --
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
        {
            "text": (
                "PCE price index rises 2.5% year over year in July, in line "
                "with expectations. Core PCE up 2.6%. Monthly increase of "
                "0.2% matches forecasts. Inflation continues to gradually "
                "approach the Fed's 2% target. Supports rate cut case."
            ),
            "ticker": "SPY",
            "date": "2024-08-30",
            "source": "economic_data",
            "outcome": {"direction": "up", "magnitude": 0.010, "timeframe": "1d"},
        },
        {
            "text": (
                "Federal Reserve cuts rates by 50 basis points to 4.75-5.00%. "
                "Dot plot shows 100 basis points of additional cuts expected "
                "in 2025. Powell emphasizes the economy is in a good place. "
                "Inflation progress gives confidence to begin easing cycle. "
                "Markets rally on dovish surprise."
            ),
            "ticker": "SPY",
            "date": "2024-09-18",
            "source": "fed_speech",
            "outcome": {"direction": "up", "magnitude": 0.022, "timeframe": "5d"},
        },
        {
            "text": (
                "Euro area GDP grows 0.3% in Q2, beating expectations of "
                "0.2%. German economy exits recession. Services PMI rises "
                "to 53.8. ECB expected to cut rates further in October. "
                "European stocks rally. Global growth outlook improves."
            ),
            "ticker": "SPY",
            "date": "2024-07-30",
            "source": "economic_data",
            "outcome": {"direction": "up", "magnitude": 0.006, "timeframe": "1d"},
        },
        {
            "text": (
                "US trade deficit narrows to $68.9 billion in July from "
                "$73.1 billion. Exports reach record $265.9 billion. "
                "Manufacturing exports increase 4.2%. Services surplus "
                "widens. Improving trade balance supports GDP growth "
                "and strengthens the economic outlook."
            ),
            "ticker": "SPY",
            "date": "2024-09-05",
            "source": "economic_data",
            "outcome": {"direction": "up", "magnitude": 0.009, "timeframe": "1d"},
        },
        {
            "text": (
                "US productivity grows 2.5% in Q2 2024, well above "
                "expectations of 1.8%. Unit labor costs rise only 0.9%. "
                "Strong productivity growth is disinflationary and supports "
                "corporate profit margins. Positive for equities and "
                "consistent with a soft landing scenario."
            ),
            "ticker": "SPY",
            "date": "2024-08-08",
            "source": "economic_data",
            "outcome": {"direction": "up", "magnitude": 0.012, "timeframe": "1d"},
        },
        # -- NEGATIVE MACRO --
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
                "US yield curve inverts further as 10-year drops below "
                "2-year by 35 basis points. Inversion has persisted for "
                "24 months. Historically reliable recession signal. Credit "
                "conditions continue to tighten. Bank lending standards "
                "reaching restrictive levels."
            ),
            "ticker": "SPY",
            "date": "2024-07-15",
            "source": "economic_data",
            "outcome": {"direction": "down", "magnitude": 0.011, "timeframe": "1d"},
        },
        {
            "text": (
                "China's Producer Price Index falls 0.8% year over year, "
                "extending deflationary streak to 21 months. Manufacturing "
                "PMI contracts to 49.4. Exports slow sharply. Property "
                "sector crisis deepens with major developer defaults. "
                "Global growth concerns intensify."
            ),
            "ticker": "SPY",
            "date": "2024-08-09",
            "source": "economic_data",
            "outcome": {"direction": "down", "magnitude": 0.018, "timeframe": "1d"},
        },
        {
            "text": (
                "JOLTS report shows job openings fall to 7.67 million, "
                "the lowest since January 2021. Quits rate declines to "
                "2.1%. Hiring rate drops to 3.3%. Labor market cooling "
                "faster than expected. Recession concerns rise. Markets "
                "sell off on weakening employment outlook."
            ),
            "ticker": "SPY",
            "date": "2024-09-04",
            "source": "economic_data",
            "outcome": {"direction": "down", "magnitude": 0.020, "timeframe": "1d"},
        },
        {
            "text": (
                "US existing home sales fall 5.4% month over month to "
                "3.86 million annualized rate, the lowest since 2010. "
                "Median price reaches record $426,900 despite weak volume. "
                "Affordability at worst level in decades. Mortgage rate "
                "remains above 7%. Housing market frozen."
            ),
            "ticker": "SPY",
            "date": "2024-08-22",
            "source": "economic_data",
            "outcome": {"direction": "down", "magnitude": 0.008, "timeframe": "1d"},
        },
        {
            "text": (
                "ECB raises rates unexpectedly by 25 basis points citing "
                "persistent inflation. European bond yields spike. Euro "
                "strengthens against dollar. Global liquidity tightening "
                "concerns. Equity markets sell off globally on hawkish "
                "surprise from central banks."
            ),
            "ticker": "SPY",
            "date": "2024-07-18",
            "source": "central_bank",
            "outcome": {"direction": "down", "magnitude": 0.015, "timeframe": "1d"},
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
