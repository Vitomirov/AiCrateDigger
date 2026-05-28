"""City matching helpers for local-shop host selection."""

from __future__ import annotations

from rapidfuzz import fuzz

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
