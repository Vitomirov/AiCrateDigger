"use client";

import SearchDevInspector from "@/components/dev/SearchDevInspector";
import SearchResultsList from "@/components/listing/SearchResultsList";
import DigSearchForm from "@/components/search/DigSearchForm";
import SearchHero from "@/components/search/SearchHero";
import SearchStatusBanner from "@/components/search/SearchStatusBanner";
import HugeVinylRecordBg from "@/components/ui/HugeVinylRecord";
import RateLimitModal from "@/components/ui/RateLimitModal";
import { useDigSearch } from "@/hooks/useDigSearch";
import { isDevInspectorEnabled } from "@/lib/config";
import { getEmptySearchMessage } from "@/lib/search-empty";

export default function SearchExperience() {
  const {
    query,
    setQuery,
    loading,
    error,
    payload,
    rateLimitOpen,
    setRateLimitOpen,
    hasRunInspect,
    dig,
    progressPct,
  } = useDigSearch();

  const results = payload?.results ?? [];
  const showDevInspector = isDevInspectorEnabled();
  const emptyHint = getEmptySearchMessage(payload, loading, error, results.length);

  return (
    <div className="relative flex min-h-[100dvh] w-full flex-col">
      <div className="pointer-events-none fixed inset-0 z-0 flex items-center justify-center opacity-[0.52] sm:opacity-[0.58] md:opacity-[0.6]">
        <HugeVinylRecordBg />
      </div>

      <div className="relative z-[2] flex w-full flex-col">
        <div className="flex shrink-0 flex-col items-center px-3 pt-3 pb-2 max-sm:min-h-[calc(50dvh+43.2vmin+0.75rem)] sm:px-5 sm:pt-4 sm:pb-3">
          <SearchHero />
          {/* Mobile: label top aligns to record hole; scrolls away with page (not fixed). sm+: normal flow. */}
          <div className="z-[3] flex w-full justify-center max-sm:absolute max-sm:left-1/2 max-sm:top-[calc(50dvh-43.2vmin)] max-sm:-translate-x-1/2 max-sm:px-3 sm:relative sm:left-auto sm:top-auto sm:translate-x-0">
            <DigSearchForm
              query={query}
              loading={loading}
              progressPct={progressPct}
              onQueryChange={setQuery}
              onDig={() => void dig()}
            />
          </div>
        </div>

        {error ? <SearchStatusBanner message={error} variant="error" /> : null}
        {!error && emptyHint ? <SearchStatusBanner message={emptyHint} variant="info" /> : null}

        {showDevInspector && hasRunInspect ? (
          <SearchDevInspector loading={loading} error={error} payload={payload} />
        ) : null}

        {results.length > 0 ? <SearchResultsList listings={results} compact /> : null}
      </div>

      <footer className="relative z-[2] mt-auto shrink-0 px-4 pb-[max(0.75rem,env(safe-area-inset-bottom,0px))] pt-3 text-center text-[0.72rem] font-medium uppercase tracking-[0.18em] text-crate-cream/55 sm:text-[0.75rem] sm:pb-[max(0.875rem,env(safe-area-inset-bottom,0px))]">
        <span className="block sm:inline">Designed and developed by</span>{" "}
        <a
          href="https://dejanvitomirov.com"
          target="_blank"
          rel="noopener noreferrer"
          className="block text-crate-gold underline decoration-crate-amber/60 underline-offset-2 transition hover:text-crate-amber sm:inline"
        >
          Dejan Vitomirov
        </a>
      </footer>

      <RateLimitModal open={rateLimitOpen} onClose={() => setRateLimitOpen(false)} />
    </div>
  );
}
