"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { ChangeEvent, KeyboardEvent } from "react";

import {
  postSearch,
  type ListingResultDto,
  type SearchResponseDto,
} from "../lib/api";
import { buildPipelineInspectPayload, DevJsonPanel } from "./DevJsonInspector";
import HugeVinylRecordBg from "./HugeVinylRecord";
import ListingResultCard from "./ListingResultCard";

/** Human copy for structured empty-state codes returned by `/search`. */
const EMPTY_REASON_COPY: Record<NonNullable<SearchResponseDto["reason"]>, string> = {
  album_unresolved:
    "Couldn’t resolve which album to hunt — name the release (or spell the artist) so we can search shops.",
};

export default function SearchExperience() {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [payload, setPayload] = useState<SearchResponseDto | null>(null);
  /** After the first Dig, keep JSON panels mounted for debugging. */
  const [hasRunInspect, setHasRunInspect] = useState(false);
  const progressTimer = useRef<ReturnType<typeof setInterval> | null>(null);
  const t0 = useRef(0);

  const cancelProgress = useCallback(() => {
    if (progressTimer.current) {
      clearInterval(progressTimer.current);
      progressTimer.current = null;
    }
  }, []);

  useEffect(() => () => cancelProgress(), [cancelProgress]);

  const rampProgress = useCallback(() => {
    cancelProgress();
    t0.current = Date.now();
    setProgress(5);
    progressTimer.current = setInterval(() => {
      const elapsed = Date.now() - t0.current;
      const next = Math.min(90, 6 + (elapsed / 10000) * 84);
      setProgress((p: number) => (p < next ? Math.floor(next) : p));
    }, 140);
  }, [cancelProgress]);

  async function onDig() {
    const q = query.trim();
    if (!q || loading) {
      return;
    }
    setError(null);
    setHasRunInspect(true);
    setLoading(true);
    rampProgress();
    try {
      // Single round-trip: `/search` now returns `parsed` alongside `results`,
      // so the dev inspector and the result list both render from one response.
      // This halves parser latency and OpenAI token cost per Dig.
      const search = await postSearch(q);
      setPayload(search);
      setProgress(100);
    } catch (e) {
      setProgress(0);
      setPayload(null);
      setError(e instanceof Error ? e.message : "Search failed.");
    } finally {
      cancelProgress();
      setLoading(false);
      setTimeout(() => setProgress(0), 500);
    }
  }

  const results: ListingResultDto[] = payload?.results ?? [];
  const hasHits = results.length > 0;
  const pct = Math.min(100, Math.max(0, progress));

  const emptyHint = (() => {
    if (!payload || loading || error || results.length > 0) {
      return null;
    }
    if (payload.reason && EMPTY_REASON_COPY[payload.reason]) {
      return EMPTY_REASON_COPY[payload.reason];
    }
    return "Nothing this pass — tweak the title or add a city hint.";
  })();

  return (
    <div className="relative flex min-h-[100dvh] w-full flex-col">
      {/* LP backdrop — fixed so it fills the viewport while the page scrolls with results */}
      <div className="pointer-events-none fixed inset-0 z-0 flex items-center justify-center opacity-[0.52] sm:opacity-[0.58] md:opacity-[0.6]">
        <HugeVinylRecordBg />
      </div>

      <div className="relative z-[2] flex w-full flex-col">
        {/* Hero — compact so JSON inspector + cards get vertical room */}
        <div className="flex shrink-0 flex-col items-center px-3 pt-3 pb-2 sm:px-5 sm:pt-4 sm:pb-3">
          <header className="max-w-lg shrink-0 text-center md:max-w-xl">
            <p className="font-slab text-[0.74rem] font-normal uppercase tracking-[0.48em] text-crate-amber sm:text-[1.5rem]">
              find favourite records
            </p>
            <h1 className="mt-0.5 font-slab text-[2.05rem] leading-[0.95] uppercase text-crate-cream sm:text-[2.65rem] md:text-[2.85rem]">
              Ai Crate Digger
            </h1>
            <p className="mx-auto mt-2 max-w-md text-[0.82rem] font-semibold leading-snug tracking-wide text-crate-cream/95 sm:mt-3 sm:max-w-lg sm:text-[0.95rem] md:text-[1.02rem]">
            Tell us what record you’re after — we’ll find shops that actually have it.
            </p>
          </header>

          {/* Circular label — +20% scale; input + CTA share one height, centered on disc */}
          <div className="mt-3 shrink-0 sm:mt-4">
            <div
              className="relative flex aspect-square w-[min(86.4vmin,calc(19rem*1.2))] flex-col overflow-hidden rounded-full border-[5px] border-crate-gold bg-[rgba(253,243,220,0.97)] shadow-2xl ring-2 ring-black/25 sm:w-[min(91.2vmin,calc(24rem*1.2))] sm:border-[6px] md:w-[min(96vmin,calc(28rem*1.2))]"
            >
              <div className="flex h-full min-h-0 flex-col px-[10%] py-[26%] sm:py-[22%]">
                <p className="shrink-0 pb-2 text-center font-slab text-[0.62rem] uppercase tracking-[0.42em] text-crate-rust sm:pb-2 sm:text-[1.2rem]">
                  drop your ask
                </p>
                <label htmlFor="dig-query" className="sr-only">
                  Search for a record
                </label>
                {/* Vertically centered block: eyebrow above + hint below stay light so field + CTA sit on the optical center */}
                <div className="flex min-h-0 flex-1 flex-col justify-center gap-2.5 sm:gap-3">
                  <textarea
                    id="dig-query"
                    rows={2}
                    disabled={loading}
                    value={query}
                    onChange={(e: ChangeEvent<HTMLTextAreaElement>) => setQuery(e.target.value)}
                    onKeyDown={(e: KeyboardEvent<HTMLTextAreaElement>) => {
                      if (e.key === "Enter" && !e.shiftKey) {
                        e.preventDefault();
                        void onDig();
                      }
                    }}
                    placeholder={'Tool - Aenima in Belgrade…'}
                    className="box-border h-12 min-h-[3rem] w-full shrink-0 resize-none overflow-y-auto rounded-lg border-[3px] border-crate-rust bg-[#0b0907]/95 px-3 py-2 text-[13px] font-semibold leading-tight text-crate-cream shadow-inner outline-none placeholder:text-crate-cream/45 placeholder:italic focus:border-crate-amber disabled:opacity-55 sm:h-[3.25rem] sm:min-h-[3.25rem] sm:text-[14px] md:h-14 md:min-h-[3.5rem] md:text-[15px]"
                  />
                  <button
                    type="button"
                    disabled={loading || !query.trim()}
                    onClick={() => void onDig()}
                    className="relative box-border flex h-12 min-h-[3rem] w-full shrink-0 items-center justify-center overflow-hidden rounded-lg border-[3px] border-crate-cream bg-crate-rust py-0 font-slab text-[0.95rem] uppercase tracking-[0.2em] text-crate-cream shadow-lg transition-colors enabled:active:translate-y-px disabled:opacity-35 sm:h-[3.25rem] sm:min-h-[3.25rem] sm:text-[1.05rem] sm:tracking-[0.26em] md:h-14 md:min-h-[3.5rem] md:text-lg md:tracking-[0.28em]"
                  >
                    {loading ? (
                      <>
                        <span
                          aria-hidden
                          className="absolute inset-y-0 left-0 bg-crate-amber/85"
                          style={{ width: `${pct}%`, transition: "width 140ms linear" }}
                        />
                        <span className="relative z-[2] block text-sm md:text-base">{pct}% DIGGING…</span>
                      </>
                    ) : (
                      <span className="relative z-[2] block">Dig That LP</span>
                    )}
                  </button>
                </div>
                <div className="shrink-0">
                  {loading ? (
                    <div className="h-2 w-full overflow-hidden rounded border-2 border-crate-rust/90 bg-black/55">
                      <div
                        className="h-full bg-crate-amber transition-[width] duration-150"
                        style={{ width: `${pct}%` }}
                      />
                    </div>
                  ) : (
                    <p className="text-center text-[0.58rem] font-bold uppercase tracking-[0.28em] text-crate-rust/80 sm:text-[0.6rem]">
                      enter • dig
                    </p>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Status strips — never their own scroll (short copy) */}
        {error ? (
          <div className="shrink-0 px-4 pb-2" role="alert">
            <p className="rounded-md border border-red-700/55 bg-black/75 px-3 py-2 text-center text-[0.78rem] font-semibold leading-snug text-red-200">
              {error}
            </p>
          </div>
        ) : null}
        {!error && emptyHint ? (
          <div className="shrink-0 px-4 pb-2">
            <p className="rounded-md border border-crate-gold/35 bg-black/65 px-3 py-2 text-center text-[0.78rem] leading-snug text-crate-cream/80">
              {emptyHint}
            </p>
          </div>
        ) : null}

        {/* Dev / debug: parse, pipeline (tavily & stages), listings as JSON */}
        {hasRunInspect ? (
          <div className="relative z-[2] w-full px-3 pb-3 sm:px-5">
            <p className="mb-2 text-center font-slab text-[0.62rem] font-semibold uppercase tracking-[0.28em] text-crate-cream/50">
              Pipeline inspector
            </p>
            <div className="mx-auto grid max-w-6xl grid-cols-1 gap-3 lg:grid-cols-3">
              <DevJsonPanel
                title="Parse"
                subtitle={
                  error
                    ? "POST /search → parsed (single round-trip)"
                    : payload?.reason
                      ? `reason: ${payload.reason} · POST /search → parsed`
                      : "POST /search → parsed (single round-trip)"
                }
                loading={loading}
                error={error}
                data={error ? null : (payload?.parsed ?? null)}
              />
              <DevJsonPanel
                title="Query & pipeline"
                subtitle="debug.stages: geo_norm / geo_tier / tavily / extract / validate / ranking / geo_widening_summary"
                loading={loading}
                error={error}
                data={error ? null : buildPipelineInspectPayload(payload?.debug ?? null)}
              />
              <DevJsonPanel
                title="Listings"
                subtitle="POST /search — `results` array"
                loading={loading}
                error={error}
                data={error ? null : (payload?.results ?? [])}
              />
            </div>
          </div>
        ) : null}

        {/* Results — document scroll (no nested scroll pane) */}
        {hasHits ? (
          <div className="relative z-[2] w-full px-3 pb-6 pt-2 sm:px-5 md:pb-10">
            <div className="mx-auto flex max-w-xl flex-col gap-4 md:max-w-2xl">
              {results.map((row) => (
                <ListingResultCard key={row.url} listing={row} compact />
              ))}
            </div>
          </div>
        ) : null}
      </div>

      <footer className="relative z-[2] mt-auto shrink-0 px-4 pb-[max(0.75rem,env(safe-area-inset-bottom,0px))] pt-3 text-center text-[0.72rem] font-medium uppercase tracking-[0.18em] text-crate-cream/55 sm:text-[0.75rem] sm:pb-[max(0.875rem,env(safe-area-inset-bottom,0px))]">
        Designed and developed by{" "}
        <a
          href="https://dejanvitomirov.com"
          target="_blank"
          rel="noopener noreferrer"
          className="text-crate-gold underline decoration-crate-amber/60 underline-offset-2 transition hover:text-crate-amber"
        >
          Dejan Vitomirov
        </a>
      </footer>
    </div>
  );
}
