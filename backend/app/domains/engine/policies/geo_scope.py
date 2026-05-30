"""Commerce region mapping for store discovery."""

from __future__ import annotations

from typing import Literal

Region = Literal[
    "balkans",
    "central_europe",
    "western_europe",
    "eastern_europe",
    "southern_europe",
    "nordics",
    "uk",
    "baltics",
]

#: Commerce regions used for store filtering. Country-level only.
COUNTRY_TO_REGION: dict[str, Region] = {
    # BALKANS
    "RS": "balkans",
    "BA": "balkans",
    "ME": "balkans",
    "MK": "balkans",
    "AL": "balkans",
    "XK": "balkans",
    "HR": "balkans",
    "BG": "balkans",

    # CENTRAL EUROPE
    "DE": "central_europe",
    "AT": "central_europe",
    "CH": "central_europe",
    "CZ": "central_europe",
    "SK": "central_europe",
    "HU": "central_europe",
    "PL": "central_europe",
    "SI": "central_europe",
    "RO": "central_europe",

    # WESTERN EUROPE
    "FR": "western_europe",
    "NL": "western_europe",
    "BE": "western_europe",
    "LU": "western_europe",
    "IE": "western_europe",
    
    # UK
    "GB": "uk",

    # EASTERN EUROPE
    "MD": "eastern_europe",

    # SOUTHERN EUROPE
    "IT": "southern_europe",
    "ES": "southern_europe",
    "PT": "southern_europe",
    "MT": "southern_europe",
    "CY": "southern_europe",
    "GR": "southern_europe",

    # NORDICS
    "SE": "nordics",
    "NO": "nordics",
    "DK": "nordics",
    "FI": "nordics",
    "IS": "nordics",

    # BALTICS
    "EE": "baltics",
    "LV": "baltics",
    "LT": "baltics"
}


def country_to_region(country_code: str | None) -> Region | None:
    """Country (ISO-2) → commerce region (or ``None`` when unknown)."""
    if not country_code:
        return None
    cc = country_code.strip().upper()
    if cc == "UK":
        cc = "GB"
    return COUNTRY_TO_REGION.get(cc)
