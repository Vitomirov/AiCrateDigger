import type { SearchResponseDto } from "./api-types";

/** Human copy for structured empty-state codes returned by `/search`. */
export const EMPTY_REASON_COPY: Record<NonNullable<SearchResponseDto["reason"]>, string> = {
  album_unresolved:
    "Couldn’t resolve which album to hunt — name the release (or spell the artist) so we can search shops.",
};

export function getEmptySearchMessage(
  payload: SearchResponseDto | null,
  loading: boolean,
  error: string | null,
  resultsCount: number,
): string | null {
  if (!payload || loading || error || resultsCount > 0) {
    return null;
  }
  if (payload.reason && EMPTY_REASON_COPY[payload.reason]) {
    return EMPTY_REASON_COPY[payload.reason];
  }
  return "Nothing this pass — tweak the title or add a city hint.";
}
