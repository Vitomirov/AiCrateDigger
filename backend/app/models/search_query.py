from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.models.result import ListingResult

ResolutionConfidence = Literal["high", "medium", "low", "unknown"]
IntentCompleteness = Literal["complete", "partial", "unknown"]


class ParsedQuery(BaseModel):
    """Permissive parser output (Deferred Resolution Architecture).

    Real user queries are partial, ambiguous, and intent-driven. The parser MUST
    NOT reject them — any missing field is tolerated, and `intent_completeness`
    signals how much the downstream pipeline is expected to fill in.
    """

    artist: str | None = Field(
        default=None,
        description="Extracted artist name (verbatim). Nullable — partial queries allowed.",
    )

    album: str | None = Field(
        default=None,
        description="Literal album title as provided by the user. Null for relative references.",
    )
    album_index: int | None = Field(
        default=None,
        description="1-based ordinal; -1 means 'latest'; null if no ordinal was used.",
    )
    resolved_album: str | None = Field(
        default=None,
        description="Deterministically resolved canonical album title (Discogs-backed).",
    )
    resolution_confidence: ResolutionConfidence = Field(
        default="unknown",
        description=(
            "`high`   = Discogs-backed resolution; "
            "`medium` = Discogs partial hit; "
            "`low`    = LLM-only fallback; "
            "`unknown` = user gave a specific title OR no resolution attempted."
        ),
    )

    format: Literal["Vinyl", "CD", "Cassette"] | None = Field(
        default=None, description="Requested physical format — may be null if unstated."
    )
    country: str | None = Field(
        default=None, description="Primary target country — may be null for truly global intents."
    )
    city: str | None = Field(default=None, description="Optional target city")
    language: str = Field(..., min_length=1, description="Detected query language")
    original_query: str = Field(..., description="Original user input")

    intent_completeness: IntentCompleteness = Field(
        default="unknown",
        description=(
            "`complete` = artist AND (album | resolved_album) present; "
            "`partial`  = at least artist known but album missing; "
            "`unknown`  = neither artist nor album known."
        ),
    )

    @property
    def effective_album(self) -> str:
        """Best-known album label. Empty string when truly unknown — callers MUST handle."""
        return (self.resolved_album or self.album or "").strip()

    @property
    def has_album(self) -> bool:
        return bool(self.effective_album)

    @property
    def has_artist(self) -> bool:
        return bool((self.artist or "").strip())

    def missing_fields(self) -> list[str]:
        """Fields that were expected but absent — fed into structured logs."""
        missing: list[str] = []
        if not self.has_artist:
            missing.append("artist")
        if not self.has_album:
            missing.append("album")
        if not self.format:
            missing.append("format")
        if not self.country:
            missing.append("country")
        return missing


class ParseRequest(BaseModel):
    # Example surfaces in /docs so manual testers don't post `{}` and get a 422.
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"query": "The Doors Strange Days vinyl in Belgrade"},
                {"query": "Disciplina Kičme - Sviđa mi se da ti ne bude prijatno na vinilu u Beogradu"},
            ]
        }
    )

    query: str = Field(
        ...,
        min_length=1,
        description="Raw natural-language user query (any language).",
    )


class SearchQueries(BaseModel):
    queries: list[str] = Field(
        ...,
        min_length=3,
        max_length=5,
        description="3-5 generated localized search query strings",
    )
    marketplaces: list[str] = Field(
        default_factory=list,
        description=(
            "Marketplaces / webshop domains each query targets, aligned by index with `queries`. "
            "Empty string at index `i` means the query is a bootstrap/discovery probe."
        ),
    )


class SearchResult(BaseModel):
    title: str = Field(..., description="Listing page title")
    url: str = Field(..., description="Listing URL")
    content: str = Field(..., description="Extracted snippet/content from search engine")
    score: float = Field(..., description="Relevance score from Tavily")
    price: str | None = Field(default=None, description="Extracted listing price with currency")
    extracted_location: str | None = Field(
        default=None,
        description="Most specific extracted location from snippet",
    )


class SearchResponse(BaseModel):
    results: list[ListingResult] = Field(
        default_factory=list,
        description="Aggregated deduplicated search results (strict ListingResult contract).",
    )
    debug: dict | None = Field(
        default=None,
        description=(
            "Populated only when `DEBUG=true` (env var). Contains the full PipelineContext "
            "trace (parser, query_gen, tavily, extractor, rag_*) for that request."
        ),
    )
