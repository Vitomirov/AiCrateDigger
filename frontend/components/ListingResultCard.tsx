import type { ListingResultDto } from "../lib/api";

function prettyDomain(domain: string | null, url: string): string {
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

function deriveArtistAlbum(title: string): { artist: string; albumLine: string } {
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

type ListingResultCardProps = {
  listing: ListingResultDto;
  compact?: boolean;
};

export default function ListingResultCard({ listing, compact }: ListingResultCardProps) {
  const { artist, albumLine } = deriveArtistAlbum(listing.title);
  const artistLabel = artist || "Various / Unknown";
  const albumLabel = (artist && albumLine ? albumLine : albumLine || listing.title).trim() || listing.title;

  const storeLabel = prettyDomain(listing.domain, listing.url);

  if (compact) {
    return (
      <article className="rounded-md border-2 border-crate-rust bg-crate-night/90 p-3 shadow-lg backdrop-blur-sm">
        <div className="flex flex-wrap items-start justify-between gap-2">
          <div className="min-w-0">
            <p className="text-[9px] font-bold uppercase tracking-[0.35em] text-crate-gold/90">Artist</p>
            <p className="font-slab text-lg uppercase leading-tight text-crate-cream">{artistLabel}</p>
            <p className="mt-2 text-[9px] font-bold uppercase tracking-[0.35em] text-crate-gold/90">Album</p>
            <p className="text-sm font-bold leading-snug text-crate-cream/95 line-clamp-2">{albumLabel}</p>
          </div>
          <div className="text-right text-[10px] font-mono font-semibold uppercase text-crate-gold">
            {listing.price ?? "—"}
          </div>
        </div>
        <div className="mt-3 border-t border-dashed border-crate-rust/80 pt-3">
          <p className="text-[9px] font-black uppercase tracking-[0.4em] text-crate-gold">Store</p>
          <div className="mt-1.5 flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
            <span className="font-slab text-sm uppercase tracking-wide text-crate-gold">{storeLabel}</span>
            <a
              href={listing.url}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex w-full items-center justify-center rounded border-2 border-crate-amber bg-crate-gold/90 px-3 py-2 font-slab text-[10px] uppercase tracking-[0.25em] text-crate-night hover:bg-crate-amber sm:w-auto"
            >
              Store →
            </a>
          </div>
        </div>
      </article>
    );
  }

  return (
    <article className="rounded-lg border-2 border-crate-rust bg-gradient-to-br from-[#2a231c]/95 via-crate-night/90 to-crate-night/98 p-5 shadow-xl">
      <div className="flex flex-wrap items-start justify-between gap-y-5">
        <div className="min-w-0 max-w-xl space-y-1 pr-6">
          <p className="text-[11px] font-bold uppercase tracking-[0.4em] text-crate-gold/90">Artist</p>
          <p className="font-slab text-2xl uppercase leading-tight tracking-wide text-crate-cream">{artistLabel}</p>
          <div className="pt-6">
            <p className="text-[11px] font-bold uppercase tracking-[0.4em] text-crate-gold/90">Album</p>
            <p className="text-lg font-bold leading-snug text-crate-cream/95">{albumLabel}</p>
          </div>
        </div>
        <div className="text-right font-mono text-xs font-semibold uppercase tracking-[0.3em] text-crate-gold">
          BIN • {listing.price ?? "CALL"}
        </div>
      </div>

      <div className="mt-7 border-t border-dashed border-crate-rust pt-7">
        <p className="text-[11px] font-black uppercase tracking-[0.52em] text-crate-gold">Store shelf</p>
        <div className="mt-4 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <span className="font-slab text-2xl uppercase tracking-[0.2em] text-crate-gold">{storeLabel}</span>
          <a
            href={listing.url}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex w-full shrink-0 items-center justify-center rounded-md border-[3px] border-crate-amber bg-crate-gold/90 px-6 py-3 text-center font-slab text-base uppercase tracking-[0.38em] text-crate-night transition hover:bg-crate-amber sm:w-auto sm:min-w-[12rem]"
          >
            Dig In Store →
          </a>
        </div>
      </div>
    </article>
  );
}
