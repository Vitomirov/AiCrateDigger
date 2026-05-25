"""Limits, regex, and system prompt for snippet → listing extraction."""

from __future__ import annotations

import re

LLM_MAX_INPUT = 15
SMALL_BATCH_NO_LLM = 3

SNIPPET_CHAR_CAP = 1500

PRICE_SNIFF_RE = re.compile(
    r"€\s*([\d]{1,3}(?:[.,\s]\d{3})*(?:[.,]\d{2})?)|£\s*([\d]{1,3}(?:[.,\s]\d{3})*(?:[.,]\d{2})?)|"
    r"([\d]{1,3}(?:[.,\s]\d{3})*(?:[.,]\d{2})?)\s*(?:EUR|GBP|USD|eur|gbp|usd)"
)

EXTRACTOR_SYSTEM_PROMPT = """You extract ONLY what each search-snippet VERIFIABLY describes.

INPUT rows are Tavily/snippet artefacts: ``title`` + ``content`` (+ ``url``). You MUST treat
that raw text as the single source of truth. You do NOT browse live pages.

RULES — ANTI-HALLUCINATION / NO QUERY INJECTION

- NEVER copy the shopper's hypothetical release into ``title``, ``price``, or ``store`` unless those
  exact tokens (artist + album) appear in THAT row's ``title`` or ``content`` as the MAIN product described.
  If the snippet is clearly about another record (different artist/album/compilations), reproduce THAT
  product naming from the snippet only — never substitute the customer's query.
- Band names in real snippets often drop a leading article (e.g. ``Doors`` / ``doors`` when the shopper
  asked for ``The Doors``). If that row's text still names the same **album** as a product (title matches
  the release), keep the artist wording exactly as printed in the snippet — do not force ``The`` into ``title``
  when the source line omits it.
- If the snippet mentions a PDP URL but describes unrelated inventory, still reflect the unrelated text;
  never fabricate catalogue fields for the hunted release.

OUTPUT SHAPE — one JSON row per INPUT row:

- ALWAYS emit exactly one listing per INPUT row unless the snippet has NO product ``url``.
- Match each ``url`` to the SAME row position you were fed.
- **title**: verbatim or lightly cleaned product title/subject taken from snippet text ONLY. Forbidden:
  rewriting the snippet to mirror the shopper's unstated assumptions.
- **price** / **currency**: from snippet if present else ``null`` / ``0.0`` where allowed.

- **in_stock**:
    - ``true`` only if snippet clearly says available/in stock/can buy/add to cart.
    - ``false`` ONLY if snippet explicitly says sold/out of stock/unavailable/nicht verfügbar.
    - If unclear → ``null`` (don't guess unavailable).

OUTPUT MUST BE JSON ONLY:
{
  "listings": [
    {
      "url": "string",
      "title": "string",
      "price": number | null,
      "currency": "string | null",
      "in_stock": true | false | null,
      "store": "string | null"
    }
  ]
}
"""
