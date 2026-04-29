"""Runtime contract: AiCrateDigg system constraints (EU Physical Music Store Locator).

This module holds documentation and immutable constraint constants only.
No business logic belongs here — downstream code should import and enforce these values.
"""

from __future__ import annotations

from typing import Final, Literal

# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

SYSTEM_DESCRIPTION: Final[str] = (
    "Localized EU Physical Record Store Finder. No shipping. No digital. "
    "City-locked inventory resolution."
)

SYSTEM_SCOPE: Final[str] = "EU_ONLY"

# ---------------------------------------------------------------------------
# System-level safety flags
# ---------------------------------------------------------------------------

EU_ONLY: Final[bool] = True
PHYSICAL_ONLY_MODE: Final[bool] = True
CITY_LOCKED_SEARCH: Final[bool] = True

# ---------------------------------------------------------------------------
# 1. Allowed physical formats (only these may be surfaced as buyable physical media)
# ---------------------------------------------------------------------------

PHYSICAL_FORMAT_VINYL: Final[str] = "VINYL"
PHYSICAL_FORMAT_CD: Final[str] = "CD"
PHYSICAL_FORMAT_CASSETTE: Final[str] = "CASSETTE"

ALLOWED_PHYSICAL_FORMATS: Final[frozenset[str]] = frozenset(
    {
        PHYSICAL_FORMAT_VINYL,
        PHYSICAL_FORMAT_CD,
        PHYSICAL_FORMAT_CASSETTE,
    }
)

PhysicalFormatLiteral = Literal["VINYL", "CD", "CASSETTE"]

# ---------------------------------------------------------------------------
# 2. Forbidden output / result classes (must not be returned or promoted)
# ---------------------------------------------------------------------------

FORBIDDEN_OUTPUT_DIGITAL_STREAMING: Final[str] = "DIGITAL_STREAMING"
FORBIDDEN_OUTPUT_DIGITAL_DOWNLOAD: Final[str] = "DIGITAL_DOWNLOAD"
FORBIDDEN_OUTPUT_SHIPPING_RESULTS: Final[str] = "SHIPPING_RESULTS"
FORBIDDEN_OUTPUT_GLOBAL_ECOMMERCE_RESULTS: Final[str] = "GLOBAL_ECOMMERCE_RESULTS"

FORBIDDEN_OUTPUTS: Final[frozenset[str]] = frozenset(
    {
        FORBIDDEN_OUTPUT_DIGITAL_STREAMING,
        FORBIDDEN_OUTPUT_DIGITAL_DOWNLOAD,
        FORBIDDEN_OUTPUT_SHIPPING_RESULTS,
        FORBIDDEN_OUTPUT_GLOBAL_ECOMMERCE_RESULTS,
    }
)

# ---------------------------------------------------------------------------
# 3. Geography — priority ranking and controlled fallback (enforcement downstream)
# ---------------------------------------------------------------------------

GeographyFallbackMode = Literal[
    "city_only",
    "country_fallback",
    "regional",
]

# Fallback behavior is required for usability.
# Strict city-only mode causes false negatives in sparse inventory markets.
# Fallback order:
# 1. city
# 2. same country
# 3. EU regional expansion (lowest priority)

GEOGRAPHY_FALLBACK_MODE: GeographyFallbackMode = "country_fallback"

GEOGRAPHY_CITY_PRIORITY_STRONG: Final[bool] = True
GEOGRAPHY_COUNTRY_BACKOFF_ALLOWED: Final[bool] = True
GEOGRAPHY_CROSS_COUNTRY_ALLOWED_ONLY_AS_LAST_RESORT: Final[bool] = True

# ---------------------------------------------------------------------------
# 4. Allowed store / seller types (physical retail contexts)
# ---------------------------------------------------------------------------

# Only physical retail environments are valid sources.
# No marketplaces or ecommerce aggregators allowed.

STORE_TYPE_RECORD_STORE: Final[str] = "RECORD_STORE"
STORE_TYPE_MUSIC_CHAIN_STORE: Final[str] = "MUSIC_CHAIN_STORE"
STORE_TYPE_SECOND_HAND_RECORD_SHOP: Final[str] = "SECOND_HAND_RECORD_SHOP"

ALLOWED_STORE_TYPES: Final[frozenset[str]] = frozenset(
    {
        STORE_TYPE_RECORD_STORE,
        STORE_TYPE_MUSIC_CHAIN_STORE,
        STORE_TYPE_SECOND_HAND_RECORD_SHOP,
    }
)

AllowedStoreTypeLiteral = Literal[
    "RECORD_STORE",
    "MUSIC_CHAIN_STORE",
    "SECOND_HAND_RECORD_SHOP",
]

# ---------------------------------------------------------------------------
# 5. Forbidden store / platform types
# ---------------------------------------------------------------------------

STORE_TYPE_MARKETPLACES: Final[str] = "MARKETPLACES"
STORE_TYPE_GENERAL_ECOMMERCE: Final[str] = "GENERAL_ECOMMERCE"

FORBIDDEN_STORE_TYPES: Final[frozenset[str]] = frozenset(
    {
        STORE_TYPE_MARKETPLACES,
        STORE_TYPE_GENERAL_ECOMMERCE,
    }
)

ForbiddenStoreTypeLiteral = Literal[
    "MARKETPLACES",
    "GENERAL_ECOMMERCE",
]

__all__ = [
    "SYSTEM_DESCRIPTION",
    "SYSTEM_SCOPE",
    "EU_ONLY",
    "PHYSICAL_ONLY_MODE",
    "CITY_LOCKED_SEARCH",
    "ALLOWED_PHYSICAL_FORMATS",
    "PHYSICAL_FORMAT_VINYL",
    "PHYSICAL_FORMAT_CD",
    "PHYSICAL_FORMAT_CASSETTE",
    "PhysicalFormatLiteral",
    "FORBIDDEN_OUTPUTS",
    "FORBIDDEN_OUTPUT_DIGITAL_STREAMING",
    "FORBIDDEN_OUTPUT_DIGITAL_DOWNLOAD",
    "FORBIDDEN_OUTPUT_SHIPPING_RESULTS",
    "FORBIDDEN_OUTPUT_GLOBAL_ECOMMERCE_RESULTS",
    "GeographyFallbackMode",
    "GEOGRAPHY_FALLBACK_MODE",
    "GEOGRAPHY_CITY_PRIORITY_STRONG",
    "GEOGRAPHY_COUNTRY_BACKOFF_ALLOWED",
    "GEOGRAPHY_CROSS_COUNTRY_ALLOWED_ONLY_AS_LAST_RESORT",
    "ALLOWED_STORE_TYPES",
    "STORE_TYPE_RECORD_STORE",
    "STORE_TYPE_MUSIC_CHAIN_STORE",
    "STORE_TYPE_SECOND_HAND_RECORD_SHOP",
    "AllowedStoreTypeLiteral",
    "FORBIDDEN_STORE_TYPES",
    "STORE_TYPE_MARKETPLACES",
    "STORE_TYPE_GENERAL_ECOMMERCE",
    "ForbiddenStoreTypeLiteral",
]

_POLICY_OWNERSHIP_NOTE = """
NOTE:
This module defines policy only.

It does NOT enforce behavior at runtime.

Enforcement must happen in:
- query_generator agent
- search ranking layer
- result validation layer

These constants are not self-enforcing.
"""
