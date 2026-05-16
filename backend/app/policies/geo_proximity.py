"""Geographic proximity scoring for store/list ranking (reusable, deterministic).

Scores are additive bonuses on top of other rank signals. They encode how well a
merchant aligns with the user's resolved location AND its commercial class:
a same-city coincidence is a true locality signal only for true local shops; for
regional ecommerce and marketplaces the same coordinates mostly mean
"ships-nationwide-from-a-warehouse", so the bonus is dampened.
"""

from __future__ import annotations

from rapidfuzz import fuzz

#: Locality strength per store class. Bonus components are multiplied by this
#: factor so a regional ecom HQ'd in the target city cannot tie with a true
#: local_shop in that city. Unknown / missing class falls back to
#: ``regional_ecommerce`` (mid-trust).
_STORE_TYPE_LOCALITY_FACTOR: dict[str, float] = {
    "local_shop": 1.0,
    "regional_ecommerce": 0.55,
    "marketplace": 0.30,
}


def _locality_factor(store_type: str | None) -> float:
    return _STORE_TYPE_LOCALITY_FACTOR.get(
        (store_type or "").strip().lower(),
        _STORE_TYPE_LOCALITY_FACTOR["regional_ecommerce"],
    )


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


# Common English↔local exonyms parsed users often mix with storefront DB spelling.
_SYNONYMOUS_CAPITAL_GROUPS: tuple[frozenset[str], ...] = (
    frozenset({"prague", "praha"}),
    frozenset({"bucharest", "bucurești", "bucuresti"}),
    frozenset({"vienna", "wien"}),
    frozenset({"copenhagen", "københavn", "kobenhavn"}),
    frozenset({"munich", "münchen", "muenchen"}),
    frozenset({"cologne", "köln", "koeln"}),
    frozenset({"florence", "firenze"}),
)


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
    for grp in _SYNONYMOUS_CAPITAL_GROUPS:
        if a in grp and b in grp:
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
    store_type: str | None = None,
) -> float:
    """Return a bonus in [0, 100], modulated by store class.

    Branch priority is unchanged (city > country > border > region > EU ship); each
    base value is multiplied by :data:`_STORE_TYPE_LOCALITY_FACTOR` so a
    same-city regional ecommerce (~55) cannot tie with a same-city local_shop (100).

    ``ships_expanded`` should be the output of :func:`app.policies.geo_scope.expand_ships_to`.
    """
    sc = (store_country or "").strip().upper()
    tc = (target_country or "").strip().upper()
    if sc == "UK":
        sc = "GB"
    if tc == "UK":
        tc = "GB"

    lf = _locality_factor(store_type)
    score = 0.0

    if tc and sc == tc:
        if target_city and store_city and cities_match(target_city, store_city):
            return 100.0 * lf
        score = max(score, 50.0 * lf)  # was 60 — widen the city→country gap

    if tc and sc and sc != tc and sc in BORDER_NEIGHBORS.get(tc, frozenset()):
        score = max(score, 30.0 * lf)  # was 35

    if (
        target_commerce_region
        and store_commerce_region
        and target_commerce_region == store_commerce_region
    ):
        score = max(score, 18.0 * lf)  # was 20

    if tc and sc != tc and tc in ships_expanded:
        score = max(score, 4.0 * lf)   # was 5

    return score
