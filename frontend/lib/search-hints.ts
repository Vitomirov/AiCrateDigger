import type { ParsedQueryDto, SearchResponseDto } from "./api-types";

function hasGeoSignal(parsed: ParsedQueryDto): boolean {
  return Boolean(
    parsed.country_code ||
      parsed.resolved_city ||
      parsed.location ||
      parsed.search_scope === "regional",
  );
}

function isArtistCatalogQuery(parsed: ParsedQueryDto): boolean {
  const artist = (parsed.artist || "").trim();
  const album = (parsed.album || "").trim();
  const resolvedAlbum = (parsed.resolved_album || "").trim();
  return Boolean(artist && !album && !resolvedAlbum && hasGeoSignal(parsed));
}

/** Soft tip when the user searched artist + place without naming an album. */
export const ARTIST_CATALOG_TIP =
  "Tip: Add the album name for better results.";

export function getArtistCatalogTip(
  payload: SearchResponseDto | null,
  loading: boolean,
  error: string | null,
): string | null {
  if (!payload || loading || error) {
    return null;
  }
  if (payload.reason === "intent_unresolved" || payload.reason === "album_unresolved") {
    return null;
  }
  const parsed = payload.parsed;
  if (!parsed || !isArtistCatalogQuery(parsed)) {
    return null;
  }
  return ARTIST_CATALOG_TIP;
}
