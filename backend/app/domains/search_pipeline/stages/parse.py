"""Parse user query and resolve album search anchor."""

from __future__ import annotations

from app.domains.query_parser.parse_schema import ParsedQuery
from app.domains.query_parser.parse_user_query import parse_user_query
from app.domains.search_pipeline.pipeline_context import stage_timer
from app.domains.search_pipeline.search_intent import SearchIntent


async def stage_parse(query: str) -> ParsedQuery:
    """Run the parser inside the ``parse`` stage timer."""
    with stage_timer("parse", input={"query": query}) as rec:
        parsed = await parse_user_query(query)
        rec.output = parsed.model_dump()
    return parsed


async def stage_resolve_album_title(
    parsed: ParsedQuery,
    search_intent: SearchIntent,
) -> str | None:
    """Pick the album anchor for release searches; ``None`` is valid for artist catalog."""
    with stage_timer("album_resolve") as rec:
        if search_intent == "artist_catalog":
            title = None
        else:
            title = (parsed.effective_album or "").strip() or None
        rec.output = {"album_title": title, "search_intent": search_intent}
        rec.status = "success" if title or search_intent == "artist_catalog" else "empty"
    return title
