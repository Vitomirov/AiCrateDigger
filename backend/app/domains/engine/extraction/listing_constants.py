"""Limits, regex, and system prompt for snippet → listing extraction."""

from __future__ import annotations

import re

LLM_MAX_INPUT = 15
SMALL_BATCH_NO_LLM = 3

SNIPPET_CHAR_CAP = 1500

#: Physical-format tokens. When present alongside a merch/digital token, the
#: row is treated as a legitimate bundle (e.g. "LP + Poster", "Vinyl +
#: Digital Download code") rather than pure merch/digital — avoids false
#: rejects on very common bundled-item wording.
_PHYSICAL_FORMAT_BLOB_TOKENS: tuple[str, ...] = (
    "vinyl", " lp", "lp ", "record", "schallplatte", "ploca", "vinil", "vinile",
    " cd", "cd ", "cassette", "kaseta", "picture disc", "boxset", "box set",
)

#: Band-merch / apparel / gift-shop snippet tokens — never a physical album.
#: Catches hosts that pass URL/host heuristics (e.g. a whitelisted local shop's
#: own merch subpage, or a "/shop/" collection URL) purely from snippet text.
MERCH_BLOB_TOKENS: tuple[str, ...] = (
    "merch", "merchandise", "gifts for sale", "gift shop", "t-shirt", "t shirt",
    "tshirt", "hoodie", "hoodies", "apparel", "clothing", "poster", "posters",
    "sticker", "stickers", "patch", "patches", "enamel pin", "pin badge",
    "tote bag", "phone case", "keychain", "beanie", "snapback", "baseball cap",
    "tank top", "long sleeve", "crewneck", "zip-up",
)

#: Digital-only download/streaming snippet tokens — never a buyable physical
#: copy. Guarded by ``_PHYSICAL_FORMAT_BLOB_TOKENS`` so a "vinyl + digital
#: download code" bundle listing (extremely common on indie PDPs) is not
#: mistaken for a pure digital release.
DIGITAL_FORMAT_BLOB_TOKENS: tuple[str, ...] = (
    "digital download", "mp3 download", "flac download", "wav download",
    "digital album", "instant download", "streaming only", "digital only release",
    "no physical copy", "digital single",
)


def blob_suggests_merch_or_digital_only(blob_lc: str) -> bool:
    """``True`` when snippet text signals band merch or a pure digital release.

    Defense-in-depth vs the host/path blacklists in
    :mod:`app.domains.engine.search.prefilter.constants`: catches merch/digital
    SKUs surfaced from an otherwise-trusted shop host (e.g. a whitelisted local
    shop's own ``/merch/`` category, which bypasses intent checks) purely from
    snippet wording. Never trips on bundled "vinyl + poster" / "LP + digital
    download code" listings — those keep a physical-format token in the blob.
    """
    b = (blob_lc or "").lower()
    if not b.strip():
        return False
    has_merch_or_digital = any(tok in b for tok in MERCH_BLOB_TOKENS) or any(
        tok in b for tok in DIGITAL_FORMAT_BLOB_TOKENS
    )
    if not has_merch_or_digital:
        return False
    return not any(tok in b for tok in _PHYSICAL_FORMAT_BLOB_TOKENS)

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

ARTIST-CATALOG MODE (when the shopper named artist + place only):
- Extract any vinyl/LP product by that artist described in each snippet.
- The album title must come from snippet text — never from unstated shopper assumptions.

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
