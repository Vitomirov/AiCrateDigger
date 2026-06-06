import type { SearchResponseDto } from "./api-types";

/** Search finished with zero listings — generic fallback when no structured `reason`. */
const NO_LISTINGS_MESSAGE =
  "Nothing turned up in shops — try the full album name, or add a city you're near.";

/** Human copy for structured empty-state codes returned by `/search`. */
export const EMPTY_REASON_COPY: Record<NonNullable<SearchResponseDto["reason"]>, string> = {
  album_unresolved:
    "We couldn't tell which album you meant — add the record name, like Tool · Aenima.",
  intent_unresolved:
    "Add an artist and a city or country — e.g. Mgła in Poland, or Tool · Aenima in Berlin.",
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
  if (payload.reason) {
    return EMPTY_REASON_COPY[payload.reason] ?? NO_LISTINGS_MESSAGE;
  }
  return NO_LISTINGS_MESSAGE;
}
