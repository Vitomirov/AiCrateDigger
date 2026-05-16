"""Static thresholds, keywords, and LLM prompt for Agent 3 (Extractor).

Separated so orchestration code stays small and prompts stay editable in one place.
"""

from __future__ import annotations

MAX_AI_BATCH_SIZE = 10
MAX_FINAL_RESULTS = 10

MIN_TITLE_LEN = 5
# RapidFuzz partial_ratio ∈ [0, 100]. Against title+content this threshold eliminates
# Taylor-Swift-when-searching-EKV cases while tolerating diacritic variants (EKV vs E.K.V.).
MIN_ARTIST_FUZZY = 60

# Title + snippet slice length used for deterministic fuzzy gates (pre-filter).
PRE_FILTER_CONTENT_CHARS = 400

# Keywords that indicate non-music merch / accessories. Checked against the lowercased
# concatenation of title + full snippet content (no truncation — matches legacy behavior).
REJECTION_KEYWORDS: tuple[str, ...] = (
    "cutting board",
    "tote bag",
    "t-shirt",
    "tshirt",
    "hoodie",
    "mug ",
    "poster",
    "sticker",
    "keychain",
    "turntable",
    "stylus",
    "cartridge",
    "slipmat",
    "cleaning kit",
    "record sleeve",
    "vinyl sleeves",
    "daska za seckanje",
    "majica",
    "šolja",
    "šoljica",
    "torba",
    "einkaufstasche",
    "kissen",
    "kochmesser",
)

# Per-listing content slice sent to the LLM (legacy length preserved).
LLM_LISTING_CONTENT_CHARS = 1500

SYSTEM_PROMPT = """
### ROLE
You are Agent 3 (Extractor) for AiCrateDigger. You DO NOT decide what to keep — the caller has
already fuzzy-matched the artist against each snippet and only sends you survivors. Your job is
pure structured extraction.

### STRICT OUTPUT SCHEMA (JSON)
{
  "scores": [
    {
      "url": "string",
      "title": "string",
      "score": 0.0,
      "price": "string|null",
      "location": "string|null",
      "availability": "available|sold_out|unknown",
      "seller_type": "store|private|unknown",
      "match_reason": "string"
    }
  ]
}

### RULES
- Emit one entry per listing in the input. Preserve `url` verbatim.
- `price`: local-currency string with symbol/code if visible (e.g. "1.890 RSD", "€24,90").
  If not visible in the snippet -> `null` (MUST NOT omit).
- `location`: "City[, District]" cleaned of marketplace boilerplate. If not visible -> `null`.
- `availability`: detect sold-out signals across languages:
    en: sold out / out of stock / unavailable
    sr/hr/bs: nema na stanju / rasprodato / prodato
    de: ausverkauft / nicht verfügbar
    fr: épuisé / indisponible
    es: agotado
    it: esaurito
  Mark `sold_out` if any appear; `available` if explicitly in stock; else `unknown`.
- `seller_type`:
    "store"   = professional webshop / record store with its own domain.
    "private" = individual seller on a generic marketplace.
    "unknown" = cannot tell.
- `score` ∈ [0.0, 1.0]: your confidence the snippet represents a genuine listing of the target
  release on the requested physical format. Start at 0.6, bump up for strong evidence (clear
  price + format + city), bump down for vague / index pages (without eliminating them — the
  deterministic scorer handles final ranking).
- `match_reason`: short free-form string. If you spot a listing that slipped past the pre-filter
  but is still clearly wrong (e.g. CD when Vinyl was requested), lower the score and write the
  reason, e.g. `"wrong_format:cd_not_vinyl"`.
""".strip()
