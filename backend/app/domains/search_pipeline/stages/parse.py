"""Parse user query and resolve album search anchor."""

from __future__ import annotations

from app.domains.query_parser.parse_schema import ParsedQuery
from app.domains.query_parser.parse_user_query import parse_user_query
from app.domains.search_pipeline.pipeline_context import stage_timer


async def stage_parse(query: str) -> ParsedQuery:
    """Run the parser inside the ``parse`` stage timer."""
    with stage_timer("parse", input={"query": query}) as rec:
        parsed = await parse_user_query(query)
        rec.output = parsed.model_dump()
    return parsed


async def stage_resolve_album_title(parsed: ParsedQuery) -> str | None:
    """Pick the album search anchor from parser output (fail closed when missing)."""
    with stage_timer("album_resolve") as rec:
        title = (parsed.effective_album or "").strip() or None
        rec.output = {"album_title": title}
        rec.status = "success" if title else "empty"
    return title
