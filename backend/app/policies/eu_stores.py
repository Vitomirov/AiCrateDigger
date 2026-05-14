"""European-market vinyl store whitelist (code seed).

Hardcoded rows live in ``ALLOWED_STORES`` and seed PostgreSQL on first boot when
``DATABASE_URL`` is set (see :func:`app.db.store_loader.seed_whitelist_stores_if_empty`).

PRIORITY POLICY (curated, locals-first):
    10  flagship local_shop in a major city (Rough Trade / Phonica / Concerto / Rush Hour)
     9  strong local_shop (London/Paris/Vienna/Stockholm/Barcelona/Milan locals)
     8  solid local_shop in a smaller city, OR strong country-wide local mailorder
     7  backup local / minor regional ecommerce
   5–6  non-curated regional ecommerce, or marketplaces (Fnac, CDandLP)

``listing_quality`` (1–10) biases domain selection toward strong product catalogues;
``priority`` is the merchant-trust dial that the country/region tier sorts on.

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
    # ---- London (GB) ----
    StoreEntry(
        "Rough Trade",
        "roughtrade.com",
        "GB",
        "uk",
        ("EU", "GB"),
        10,
        listing_quality=9,
        city="London",
        store_type="local_shop",
    ),
    StoreEntry(
        "Phonica Records",
        "phonica.co.uk",
        "GB",
        "uk",
        ("EU", "GB"),
        10,
        listing_quality=9,
        city="London",
        store_type="local_shop",
    ),
    StoreEntry(
        "Sister Ray",
        "sisterray.co.uk",
        "GB",
        "uk",
        ("EU", "GB"),
        9,
        listing_quality=8,
        city="London",
        store_type="local_shop",
    ),
    StoreEntry(
        "Sounds of the Universe",
        "soundsoftheuniverse.com",
        "GB",
        "uk",
        ("EU", "GB"),
        9,
        listing_quality=8,
        city="London",
        store_type="local_shop",
    ),
    StoreEntry(
        "Honest Jon's",
        "honestjons.com",
        "GB",
        "uk",
        ("EU", "GB"),
        9,
        listing_quality=7,
        city="London",
        store_type="local_shop",
    ),
    # London-based DJ mailorder — kept for catalogue depth, NOT a local walk-in shop.
    # MUST stay ``regional_ecommerce`` so it is excluded from the indie-only city tier.
    StoreEntry(
        "Juno Records",
        "juno.co.uk",
        "GB",
        "uk",
        ("EU", "GB"),
        8,
        listing_quality=9,
        city="London",
        store_type="regional_ecommerce",
    ),

    # ---- Manchester / Leeds / Preston (GB) ----
    StoreEntry(
        "Piccadilly Records",
        "piccadillyrecords.com",
        "GB",
        "uk",
        ("EU", "GB"),
        9,
        listing_quality=8,
        city="Manchester",
        store_type="local_shop",
    ),
    StoreEntry(
        "Boomkat",
        "boomkat.com",
        "GB",
        "uk",
        ("EU", "GB"),
        8,
        listing_quality=9,
        city="Manchester",
    ),
    StoreEntry(
        "Norman Records",
        "normanrecords.co.uk",
        "GB",
        "uk",
        ("EU", "GB"),
        8,
        listing_quality=7,
        city="Leeds",
        store_type="local_shop",
    ),
    StoreEntry(
        "Action Records",
        "actionrecords.co.uk",
        "GB",
        "uk",
        ("EU", "GB"),
        7,
        listing_quality=6,
        city="Preston",
        store_type="local_shop",
    ),

    # ---- Amsterdam / Rotterdam (NL) ----
    StoreEntry(
        "Concerto",
        "concerto.nl",
        "NL",
        "western_europe",
        ("EU",),
        10,
        listing_quality=9,
        city="Amsterdam",
        store_type="local_shop",
    ),
    StoreEntry(
        "Rush Hour",
        "rushhour.nl",
        "NL",
        "western_europe",
        ("EU",),
        10,
        listing_quality=9,
        city="Amsterdam",
        store_type="local_shop",
    ),
    StoreEntry(
        "Clone",
        "clone.nl",
        "NL",
        "western_europe",
        ("EU",),
        9,
        listing_quality=9,
        city="Rotterdam",
        store_type="local_shop",
    ),

    # ---- Berlin / Germany ----
    # HHV/JPC/Decks are large DE mailorders — useful for catalogue depth but NOT true Berlin locals.
    # MUST stay ``regional_ecommerce`` so HHV is excluded from the indie-only Berlin city tier.
    StoreEntry(
        "HHV",
        "hhv.de",
        "DE",
        "western_europe",
        ("EU",),
        9,
        listing_quality=9,
        city="Berlin",
        store_type="regional_ecommerce",
    ),
    StoreEntry("JPC", "jpc.de", "DE", "western_europe", ("EU",), 9, listing_quality=9),
    StoreEntry("Decks", "decks.de", "DE", "western_europe", ("EU",), 8, listing_quality=9),
    StoreEntry("Groove Attack", "grooveattack.de", "DE", "western_europe", ("EU",), 7, listing_quality=8),
    StoreEntry("Recordsale", "recordsale.de", "DE", "western_europe", ("EU",), 6, listing_quality=6),

    # ---- Paris (FR) ----
    StoreEntry(
        "Dizonord",
        "dizonord.com",
        "FR",
        "western_europe",
        ("EU",),
        9,
        listing_quality=8,
        city="Paris",
        store_type="local_shop",
    ),
    StoreEntry(
        "Born Bad Records",
        "bornbad.fr",
        "FR",
        "western_europe",
        ("EU",),
        8,
        listing_quality=5,
        city="Paris",
        store_type="local_shop",
    ),

    # ---- Vienna (AT) ----
    StoreEntry(
        "Teuchtler Schallplatten",
        "teuchtler.at",
        "AT",
        "western_europe",
        ("EU",),
        9,
        listing_quality=8,
        city="Vienna",
        store_type="local_shop",
    ),

    # ---- Stockholm (SE) ----
    StoreEntry(
        "Pet Sounds",
        "petsounds.se",
        "SE",
        "nordics",
        ("EU",),
        9,
        listing_quality=8,
        city="Stockholm",
        store_type="local_shop",
    ),

    # ---- Milan (IT) ----
    StoreEntry(
        "Sound Ohm",
        "soundohm.com",
        "IT",
        "southern_europe",
        ("EU",),
        9,
        listing_quality=8,
        city="Milan",
        store_type="local_shop",
    ),

    # ---- Barcelona (ES) ----
    StoreEntry(
        "Discos Revolver",
        "discosrevolver.com",
        "ES",
        "southern_europe",
        ("EU",),
        9,
        listing_quality=7,
        city="Barcelona",
        store_type="local_shop",
    ),

    # ---- Belgium / Czechia ----
    StoreEntry(
        "Cactus Music",
        "cactusmusic.be",
        "BE",
        "western_europe",
        ("EU",),
        7,
        listing_quality=5,
        store_type="local_shop",
    ),
    StoreEntry(
        "Beatshop",
        "beatshop.cz",
        "CZ",
        "central_europe",
        ("EU",),
        7,
        listing_quality=5,
        store_type="local_shop",
    ),

    # ---- Belgrade (RS) ----
    StoreEntry(
        "Metropolis Music Company",
        "metropolismusic.rs",
        "RS",
        "balkans",
        ("RS", "BA", "HR", "ME", "MK", "SI"),
        8,
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

    # ---- Marketplaces (kept for fallback only, deliberately low priority) ----
    StoreEntry(
        "Fnac",
        "fnac.com",
        "FR",
        "western_europe",
        ("EU",),
        5,
        listing_quality=8,
        store_type="marketplace",
    ),
    StoreEntry(
        "CDandLP",
        "cdandlp.com",
        "FR",
        "western_europe",
        ("EU",),
        5,
        listing_quality=8,
        store_type="marketplace",
    ),
)


def get_active_stores() -> tuple[StoreEntry, ...]:
    return tuple(s for s in ALLOWED_STORES if s.is_active)
