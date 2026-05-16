"""Price and currency from LLM rows or raw snippet text."""

from __future__ import annotations

from typing import Any

from app.llm.extract_listings.constants import PRICE_SNIFF_RE


def coerce_price_currency(item: dict[str, Any]) -> tuple[float, str]:
    raw_p = item.get("price")
    try:
        p = float(raw_p) if raw_p is not None else 0.0
    except (TypeError, ValueError):
        p = 0.0
    if p < 0.0:
        p = 0.0

    cur = item.get("currency")
    s = str(cur).strip().upper() if cur is not None else ""
    if len(s) != 3 or not s.isalpha():
        s = "EUR"
    return p, s


def sniff_price_currency(content: str) -> tuple[float, str]:
    text = (content or "")[:800].replace("\xa0", " ")
    m = PRICE_SNIFF_RE.search(text)
    if not m:
        return 0.0, "EUR"
    g = m.groups()
    raw = (g[0] or g[1] or g[2] or "").strip()
    if not raw:
        return 0.0, "EUR"
    raw = raw.replace(" ", "").replace(",", ".")
    try:
        p = float(raw)
    except ValueError:
        return 0.0, "EUR"
    cur = "GBP" if m.group(2) else "EUR"
    if m.group(3) and "USD" in text[max(0, m.start() - 2) : m.end() + 2].upper():
        cur = "USD"
    return p, cur
