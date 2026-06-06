export type HealthResponse = {
  status: string;
  service: string;
  database_configured: boolean;
};

export type ListingResultDto = {
  url: string;
  title: string;
  score: number;
  price: string | null;
  location: string | null;
  availability: string;
  seller_type: string;
  domain: string | null;
};

/** Mirrors backend `domain.parse_schema.ParsedQuery` (subset for typing). */
export type ParsedQueryDto = {
  artist: string | null;
  album: string | null;
  album_index: number | null;
  resolved_album: string | null;
  resolution_confidence: "high" | "medium" | "low" | "unknown";
  location: string | null;
  country_code: string | null;
  search_scope: string;
  resolved_city: string | null;
  geo_confidence: number | null;
  geo_granularity: string | null;
  language: string;
  original_query: string;
};

/** Machine-readable empty-state code emitted by `/search`. Mirrors backend
 *  `models.search_query.SearchEmptyReason`. Keep in sync. */
export type SearchEmptyReason = "album_unresolved" | "intent_unresolved";

export type SearchResponseDto = {
  results: ListingResultDto[];
  parsed?: ParsedQueryDto | null;
  reason?: SearchEmptyReason | null;
  debug?: Record<string, unknown> | null;
};
