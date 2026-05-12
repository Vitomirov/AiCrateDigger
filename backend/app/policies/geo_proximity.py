"""Geographic proximity scoring for store/list ranking (reusable, deterministic).

Scores are additive bonuses on top of other rank signals. They encode how well a
merchant aligns with the user's resolved location — **not** whether the store
ships EU-wide (that alone is a weak +5 signal).
"""

from __future__ import annotations

from rapidfuzz import fuzz

# Land borders between ISO-3166-1 alpha-2 codes (subset focused on EU coverage).
# Used only when HQ country differs from the user's country.
BORDER_NEIGHBORS: dict[str, frozenset[str]] = {
    "AT": frozenset({"CH", "DE", "IT", "LI", "CZ", "SK", "HU", "SI"}),
    "BE": frozenset({"FR", "DE", "LU", "NL"}),
    "BG": frozenset({"GR", "RO", "RS", "TR", "MK"}),
    "CH": frozenset({"AT", "DE", "FR", "IT", "LI"}),
    "CZ": frozenset({"AT", "DE", "PL", "SK"}),
    "DE": frozenset({"AT", "BE", "CH", "CZ", "DK", "FR", "LU", "NL", "PL"}),
    "DK": frozenset({"DE", "SE"}),
    "EE": frozenset({"LV", "RU"}),
    "ES": frozenset({"FR", "PT", "AD"}),
    "FI": frozenset({"NO", "SE", "RU"}),
    "FR": frozenset({"AD", "BE", "CH", "DE", "ES", "IT", "LU", "MC"}),
    "GR": frozenset({"AL", "BG", "TR", "MK"}),
    "HR": frozenset({"BA", "HU", "ME", "RS", "SI"}),
    "HU": frozenset({"AT", "HR", "RO", "RS", "SK", "SI", "UA"}),
    "IE": frozenset({"GB"}),
    "IT": frozenset({"AT", "CH", "FR", "SI", "SM", "VA"}),
    "LT": frozenset({"BY", "LV", "PL", "RU"}),
    "LU": frozenset({"BE", "DE", "FR"}),
    "LV": frozenset({"BY", "EE", "LT", "RU"}),
    "NL": frozenset({"BE", "DE"}),
    "NO": frozenset({"FI", "RU", "SE"}),
    "PL": frozenset({"BY", "CZ", "DE", "LT", "SK", "UA"}),
    "PT": frozenset({"ES"}),
    "RO": frozenset({"BG", "HU", "MD", "RS", "UA"}),
    "RS": frozenset({"AL", "BA", "BG", "HR", "HU", "ME", "MK", "RO"}),
    "SE": frozenset({"FI", "NO"}),
    "SI": frozenset({"AT", "HR", "IT", "HU"}),
    "SK": frozenset({"AT", "CZ", "HU", "PL", "UA"}),
    "GB": frozenset({"IE"}),
}


def cities_match(user_city: str | None, store_city: str | None, *, min_ratio: int = 88) -> bool:
    """Lenient city match for typos / casing (e.g. Barselona vs Barcelona)."""
    if not user_city or not store_city:
        return False
    a = user_city.strip().lower()
    b = store_city.strip().lower()
    if not a or not b:
        return False
    if a == b:
        return True
    return int(fuzz.ratio(a, b)) >= min_ratio


def geo_proximity_bonus(
    *,
    store_country: str | None,
    store_city: str | None,
    store_commerce_region: str | None,
    target_country: str | None,
    target_city: str | None,
    target_commerce_region: str | None,
    ships_expanded: frozenset[str],
) -> float:
    """Return a bonus in [0, 100] with the priority: city > country > border > region > EU ship.

    ``ships_expanded`` should be the output of :func:`app.policies.geo_scope.expand_ships_to`.
    """
    sc = (store_country or "").strip().upper()
    tc = (target_country or "").strip().upper()
    if sc == "UK":
        sc = "GB"
    if tc == "UK":
        tc = "GB"

    score = 0.0

    if tc and sc == tc:
        if target_city and store_city and cities_match(target_city, store_city):
            return 100.0
        score = max(score, 60.0)

    if tc and sc and sc != tc and sc in BORDER_NEIGHBORS.get(tc, frozenset()):
        score = max(score, 35.0)

    if (
        target_commerce_region
        and store_commerce_region
        and target_commerce_region == store_commerce_region
    ):
        score = max(score, 20.0)

    if tc and sc != tc and tc in ships_expanded:
        score = max(score, 5.0)

    return score
