"""EU vinyl store whitelist (code seed).

Hardcoded rows live in ``ALLOWED_STORES`` and seed PostgreSQL on first boot when
``DATABASE_URL`` is set (see :func:`app.db.store_loader.seed_whitelist_stores_if_empty`).

After seeding, ops can change the live allowlist with SQL/ORM on ``whitelist_stores``
without redeploying — :func:`app.db.store_loader.load_active_stores` reads the DB.
If the table is empty or Postgres is off, the pipeline falls back to this tuple.

Domain checks in :mod:`app.validators.listings` use the same domains the pipeline
loads for that request (DB-backed when configured).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StoreEntry:
    name: str
    domain: str
    country: str
    ships_to: tuple[str, ...]
    priority: int
    is_active: bool


ALLOWED_STORES: tuple[StoreEntry, ...] = (
    StoreEntry("HHV", "hhv.de", "DE", ("EU",), 10, True),
    StoreEntry("JPC", "jpc.de", "DE", ("EU",), 10, True),
    StoreEntry("Rough Trade", "roughtrade.com", "UK", ("EU",), 10, True),
    StoreEntry("Juno Records", "juno.co.uk", "UK", ("EU",), 10, True),
    StoreEntry("Fnac", "fnac.com", "FR", ("EU",), 10, True),
    StoreEntry("Decks", "decks.de", "DE", ("EU",), 9, True),
    StoreEntry("Concerto", "concerto.nl", "NL", ("EU",), 9, True),
    StoreEntry("Phonica Records", "phonica.co.uk", "UK", ("EU",), 9, True),
    StoreEntry("Rush Hour", "rushhour.nl", "NL", ("EU",), 9, True),
    StoreEntry("CDandLP", "cdandlp.com", "FR", ("EU",), 9, True),
    StoreEntry("Groove Attack", "grooveattack.de", "DE", ("EU",), 8, True),
    StoreEntry("Recordsale", "recordsale.de", "DE", ("EU",), 8, True),
    StoreEntry("Norman Records", "normanrecords.co.uk", "UK", ("EU",), 8, True),
    StoreEntry("Clone", "clone.nl", "NL", ("EU",), 8, True),
    StoreEntry("Piccadilly Records", "piccadillyrecords.com", "UK", ("EU",), 8, True),
    StoreEntry("Boomkat", "boomkat.com", "UK", ("EU",), 8, True),
    StoreEntry("Sound Ohm", "soundohm.com", "IT", ("EU",), 7, True),
    StoreEntry("Discos Revolver", "discosrevolver.com", "ES", ("EU",), 7, True),
    StoreEntry("Cactus Music", "cactusmusic.be", "BE", ("EU",), 6, True),
    StoreEntry("Teuchtler Schallplatten", "teuchtler.at", "AT", ("EU",), 8, True),
    StoreEntry("Pet Sounds", "petsounds.se", "SE", ("EU",), 8, True),
    StoreEntry("Beatshop", "beatshop.cz", "CZ", ("EU",), 6, True),
    StoreEntry("Dizonord", "dizonord.com", "FR", ("EU",), 7, True),
    StoreEntry("Born Bad Records", "bornbad.fr", "FR", ("EU",), 6, True),
    StoreEntry("Action Records", "actionrecords.co.uk", "UK", ("EU",), 7, True),
    StoreEntry("Metropolis Music Company", "metropolismusic.rs", "RS", ("EU",), 7, True),
    StoreEntry("Mascom Store", "mascom.rs", "RS", ("EU",), 7, True),
)


def get_active_stores() -> tuple[StoreEntry, ...]:
    return tuple(s for s in ALLOWED_STORES if s.is_active)
