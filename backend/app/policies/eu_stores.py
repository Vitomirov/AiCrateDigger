"""European-market vinyl store whitelist (code seed).

Hardcoded rows live in ``ALLOWED_STORES`` and seed PostgreSQL on first boot when
``DATABASE_URL`` is set (see :func:`app.db.store_loader.seed_whitelist_stores_if_empty`).

``listing_quality`` (1–10) biases domain selection toward strong ecommerce / vinyl
catalogues; it does not replace ``priority`` (merchant tiering).

Domain checks in :mod:`app.validators.listings` use the same domains the pipeline
loads for that request (DB-backed when configured).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

StoreType = Literal["local_shop", "regional_ecommerce", "marketplace"]


@dataclass(frozen=True, slots=True)
class StoreEntry:
    name: str
    domain: str
    country_code: str | None
    region: str | None
    ships_to: tuple[str, ...]
    priority: int
    is_active: bool = True
    #: Ecommerce signal for Tavily domain picking (higher = prefer for product search).
    listing_quality: int = 5
    city: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    store_type: StoreType = "regional_ecommerce"


ALLOWED_STORES: tuple[StoreEntry, ...] = (
    StoreEntry(
        "HHV",
        "hhv.de",
        "DE",
        "western_europe",
        ("EU",),
        10,
        listing_quality=9,
        city="Berlin",
    ),
    StoreEntry("JPC", "jpc.de", "DE", "western_europe", ("EU",), 10, listing_quality=9),
    StoreEntry("Rough Trade", "roughtrade.com", "GB", "uk", ("EU", "GB"), 10, listing_quality=9),
    StoreEntry("Juno Records", "juno.co.uk", "GB", "uk", ("EU", "GB"), 10, listing_quality=9),
    StoreEntry(
        "Fnac",
        "fnac.com",
        "FR",
        "western_europe",
        ("EU",),
        10,
        listing_quality=9,
        store_type="marketplace",
    ),
    StoreEntry("Decks", "decks.de", "DE", "western_europe", ("EU",), 9, listing_quality=9),
    StoreEntry("Concerto", "concerto.nl", "NL", "western_europe", ("EU",), 9, listing_quality=9),
    StoreEntry("Phonica Records", "phonica.co.uk", "GB", "uk", ("EU", "GB"), 9, listing_quality=9),
    StoreEntry("Rush Hour", "rushhour.nl", "NL", "western_europe", ("EU",), 9, listing_quality=9),
    StoreEntry("CDandLP", "cdandlp.com", "FR", "western_europe", ("EU",), 9, listing_quality=8),
    StoreEntry("Groove Attack", "grooveattack.de", "DE", "western_europe", ("EU",), 8, listing_quality=8),
    StoreEntry("Recordsale", "recordsale.de", "DE", "western_europe", ("EU",), 8, listing_quality=6),
    StoreEntry("Norman Records", "normanrecords.co.uk", "GB", "uk", ("EU", "GB"), 8, listing_quality=7),
    StoreEntry("Clone", "clone.nl", "NL", "western_europe", ("EU",), 8, listing_quality=9),
    StoreEntry("Piccadilly Records", "piccadillyrecords.com", "GB", "uk", ("EU", "GB"), 8, listing_quality=8),
    StoreEntry("Boomkat", "boomkat.com", "GB", "uk", ("EU", "GB"), 8, listing_quality=9),
    StoreEntry("Sound Ohm", "soundohm.com", "IT", "southern_europe", ("EU",), 7, listing_quality=8),
    StoreEntry(
        "Discos Revolver",
        "discosrevolver.com",
        "ES",
        "southern_europe",
        ("EU",),
        7,
        listing_quality=7,
        city="Barcelona",
        store_type="local_shop",
    ),
    StoreEntry("Cactus Music", "cactusmusic.be", "BE", "western_europe", ("EU",), 6, listing_quality=4),
    StoreEntry("Teuchtler Schallplatten", "teuchtler.at", "AT", "western_europe", ("EU",), 8, listing_quality=8),
    StoreEntry("Pet Sounds", "petsounds.se", "SE", "nordics", ("EU",), 8, listing_quality=8),
    StoreEntry("Beatshop", "beatshop.cz", "CZ", "central_europe", ("EU",), 6, listing_quality=4),
    StoreEntry("Dizonord", "dizonord.com", "FR", "western_europe", ("EU",), 7, listing_quality=8),
    StoreEntry("Born Bad Records", "bornbad.fr", "FR", "western_europe", ("EU",), 6, listing_quality=4),
    StoreEntry("Action Records", "actionrecords.co.uk", "GB", "uk", ("EU", "GB"), 7, listing_quality=6),
    StoreEntry(
        "Metropolis Music Company",
        "metropolismusic.rs",
        "RS",
        "balkans",
        ("RS", "BA", "HR", "ME", "MK", "SI"),
        7,
        listing_quality=9,
        city="Belgrade",
        store_type="local_shop",
    ),
    StoreEntry(
        "Mascom Store",
        "mascom.rs",
        "RS",
        "balkans",
        ("RS", "BA", "HR", "ME", "MK", "SI"),
        7,
        listing_quality=9,
        city="Belgrade",
        store_type="local_shop",
    ),
)


def get_active_stores() -> tuple[StoreEntry, ...]:
    return tuple(s for s in ALLOWED_STORES if s.is_active)
