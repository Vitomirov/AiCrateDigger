import type { ListingResultDto } from "./api-types";

export function prettyDomain(domain: string | null, url: string): string {
  if (domain && domain.trim()) {
    return domain.replace(/^www\./, "").toUpperCase();
  }
  try {
    const u = new URL(url).hostname.replace(/^www\./, "");
    return u.toUpperCase();
  } catch {
    return "WEB SHOP";
  }
}

export function deriveArtistAlbum(title: string): { artist: string; albumLine: string } {
  const t = title.trim();
  const separators = [" — ", " – ", " - ", " / ", " • "];

  for (const sep of separators) {
    const i = t.indexOf(sep);
    if (i >= 2 && i < t.length - 2) {
      return {
        artist: t.slice(0, i).trim(),
        albumLine: t.slice(i + sep.length).trim(),
      };
    }
  }

  return { artist: "", albumLine: t };
}

export type ListingCardLabels = {
  artistLabel: string;
  albumLabel: string;
  storeLabel: string;
};

export function formatListingCardLabels(listing: ListingResultDto): ListingCardLabels {
  const { artist, albumLine } = deriveArtistAlbum(listing.title);
  const artistLabel = artist || "Various / Unknown";
  const albumLabel = (artist && albumLine ? albumLine : albumLine || listing.title).trim() || listing.title;
  const storeLabel = prettyDomain(listing.domain, listing.url);

  return { artistLabel, albumLabel, storeLabel };
}
