"""Compatibility shim. Tavily logic moved to `app.services.tavily_service`.

Kept as a thin re-export so existing imports (`from app.agents.searcher import
run_tavily_search`) don't break during the Chunk A → Chunk B transition. The
router will be updated in Chunk B to import directly from the service module.
"""

from app.services.tavily_service import run_tavily_search

__all__ = ["run_tavily_search"]
