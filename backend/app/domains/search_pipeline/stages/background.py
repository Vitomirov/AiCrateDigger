"""Deferred store-discovery scheduling (BackgroundTasks or asyncio.create_task)."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

from fastapi import BackgroundTasks

logger = logging.getLogger(__name__)


async def run_background_store_discovery(
    work: Callable[[], Awaitable[object]],
    *,
    label: str,
) -> None:
    """Execute deferred discovery; log failures without affecting the HTTP response."""
    try:
        await work()
    except Exception:  # noqa: BLE001 — background work must never surface to the client.
        logger.exception(
            "background_store_discovery_failed",
            extra={"stage": "store_discovery", "discovery_kind": label},
        )


def schedule_background_store_discovery(
    background_tasks: BackgroundTasks | None,
    work: Callable[[], Awaitable[object]],
    *,
    label: str,
) -> None:
    """Run discovery after the response (BackgroundTasks) or concurrently (create_task)."""

    async def _runner() -> None:
        await run_background_store_discovery(work, label=label)

    if background_tasks is not None:
        background_tasks.add_task(_runner)
        return
    asyncio.create_task(_runner())
