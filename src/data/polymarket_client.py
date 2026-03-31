"""
Polymarket CLOB data fetcher — read-only, no auth required.

Fetches open crypto markets expiring within 7 days with volume > $10K.
Never crashes on API failure — returns empty list with a warning log.
"""

import json
import logging
import urllib.request
from datetime import datetime, timezone
from typing import TypedDict

logger = logging.getLogger(__name__)

CLOB_BASE = "https://clob.polymarket.com"
CRYPTO_KEYWORDS = frozenset({"btc", "eth", "bitcoin", "ethereum", "crypto", "solana", "sol"})


class MarketSnapshot(TypedDict):
    market_id: str
    question: str
    yes_price: float
    no_price: float
    implied_prob_yes: float
    volume: float
    end_date: str


def _is_crypto_market(question: str) -> bool:
    q = question.lower()
    return any(kw in q for kw in CRYPTO_KEYWORDS)


def _parse_end_date(market: dict) -> datetime | None:
    """Try several field names Polymarket uses for end date."""
    for field in ("end_date_iso", "end_date", "endDateIso", "endDate"):
        raw = market.get(field)
        if raw:
            try:
                return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
            except (ValueError, TypeError):
                continue
    return None


def _extract_yes_no_prices(market: dict) -> tuple[float, float]:
    """Pull YES/NO prices from the tokens array, with fallback to bid/ask fields."""
    yes_price: float = 0.0
    no_price: float = 0.0

    for token in market.get("tokens", []):
        outcome = str(token.get("outcome", "")).upper()
        price = float(token.get("price") or 0)
        if outcome == "YES":
            yes_price = price
        elif outcome == "NO":
            no_price = price

    # Fallback: some markets use top-level bid/ask
    if yes_price == 0.0:
        yes_price = float(market.get("best_bid") or market.get("outcomePrices", [0])[0] or 0)
    if no_price == 0.0 and len(market.get("outcomePrices", [])) > 1:
        no_price = float(market.get("outcomePrices", [0, 0])[1] or 0)

    return yes_price, no_price


def fetch_markets(next_cursor: str | None = None) -> list[MarketSnapshot]:
    """
    Fetch Polymarket CLOB markets and filter to:
        - Questions mentioning BTC/ETH/bitcoin/crypto
        - End date within 7 days from now
        - Volume > $10,000

    Returns a list of MarketSnapshot dicts.
    Returns empty list on any API failure.
    """
    now = datetime.now(timezone.utc)
    results: list[MarketSnapshot] = []

    # Paginate through all markets (CLOB API returns cursor-based pages)
    cursor = next_cursor
    pages_fetched = 0
    MAX_PAGES = 20  # safety limit

    while pages_fetched < MAX_PAGES:
        url = f"{CLOB_BASE}/markets"
        if cursor:
            url += f"?next_cursor={cursor}"

        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            logger.warning(f"Polymarket API unavailable (URL: {url}): {e}")
            break

        # Response may be a list (legacy) or a paginated dict
        if isinstance(data, list):
            markets = data
            cursor = None  # no more pages
        elif isinstance(data, dict):
            markets = data.get("data", [])
            cursor = data.get("next_cursor")
        else:
            logger.warning(f"Unexpected Polymarket response type: {type(data)}")
            break

        pages_fetched += 1

        for market in markets:
            try:
                question = str(market.get("question", ""))
                if not _is_crypto_market(question):
                    continue

                end_date = _parse_end_date(market)
                if end_date is None:
                    continue

                days_remaining = (end_date - now).total_seconds() / 86400
                if days_remaining < 0 or days_remaining > 7:
                    continue

                volume = float(market.get("volume") or market.get("volume24hr") or 0)
                if volume < 10_000:
                    continue

                yes_price, no_price = _extract_yes_no_prices(market)

                results.append(
                    MarketSnapshot(
                        market_id=str(
                            market.get("condition_id")
                            or market.get("market_slug")
                            or market.get("id")
                            or ""
                        ),
                        question=question,
                        yes_price=yes_price,
                        no_price=no_price,
                        implied_prob_yes=yes_price,  # CLOB price == implied probability
                        volume=volume,
                        end_date=end_date.isoformat(),
                    )
                )
            except Exception as e:
                logger.debug(f"Skipping market parse error: {e}")
                continue

        if not cursor:
            break  # no more pages

    if not results:
        logger.info("Polymarket fetch returned 0 matching crypto markets")

    return results
