"""Map ISO 3166-1 alpha-2 codes to Tavily ``country`` API tokens (English names)."""

from __future__ import annotations

# Tavily search API expects lower-case country names for ``country`` (topic=general).
# Coverage focuses on supported record-store geographies; unknown codes return None.
_ISO2_TO_TAVILY_COUNTRY: dict[str, str] = {
    "AD": "andorra",
    "AE": "united arab emirates",
    "AR": "argentina",
    "AT": "austria",
    "AU": "australia",
    "BA": "bosnia and herzegovina",
    "BE": "belgium",
    "BG": "bulgaria",
    "BR": "brazil",
    "BY": "belarus",
    "CA": "canada",
    "CH": "switzerland",
    "CL": "chile",
    "CO": "colombia",
    "CR": "costa rica",
    "CY": "cyprus",
    "CZ": "czech republic",
    "DE": "germany",
    "DK": "denmark",
    "EE": "estonia",
    "ES": "spain",
    "FI": "finland",
    "FR": "france",
    "GB": "united kingdom",
    "GR": "greece",
    "HR": "croatia",
    "HU": "hungary",
    "IE": "ireland",
    "IL": "israel",
    "IS": "iceland",
    "IT": "italy",
    "JP": "japan",
    "LT": "lithuania",
    "LU": "luxembourg",
    "LV": "latvia",
    "MD": "moldova",
    "ME": "montenegro",
    "MK": "north macedonia",
    "MT": "malta",
    "MX": "mexico",
    "NL": "netherlands",
    "NO": "norway",
    "NZ": "new zealand",
    "PL": "poland",
    "PT": "portugal",
    "RO": "romania",
    "RS": "serbia",
    "SE": "sweden",
    "SI": "slovenia",
    "SK": "slovakia",
    "TR": "turkey",
    "UA": "ukraine",
    "US": "united states",
    "UY": "uruguay",
}


def tavily_country_from_iso3166_alpha2(code: str | None) -> str | None:
    if not (code or "").strip():
        return None
    key = str(code).strip().upper()
    return _ISO2_TO_TAVILY_COUNTRY.get(key)
