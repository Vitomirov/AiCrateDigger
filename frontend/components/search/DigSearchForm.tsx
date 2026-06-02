"use client";

import type { ChangeEvent, KeyboardEvent, Ref } from "react";

type DigSearchFormProps = {
  query: string;
  loading: boolean;
  progressPct: number;
  textareaRef: Ref<HTMLTextAreaElement>;
  onQueryChange: (value: string) => void;
  onDig: () => void;
};

export default function DigSearchForm({
  query,
  loading,
  progressPct,
  textareaRef,
  onQueryChange,
  onDig,
}: DigSearchFormProps) {
  const pct = progressPct;

  return (
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
          <div className="flex min-h-0 flex-1 flex-col justify-center gap-2.5 sm:gap-3">
            <textarea
              ref={textareaRef}
              id="dig-query"
              rows={2}
              disabled={loading}
              value={query}
              onChange={(e: ChangeEvent<HTMLTextAreaElement>) => onQueryChange(e.target.value)}
              onKeyDown={(e: KeyboardEvent<HTMLTextAreaElement>) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  onDig();
                }
              }}
              placeholder={'Tool - Aenima in Belgrade…'}
              className="box-border h-12 min-h-[3rem] w-full shrink-0 resize-none overflow-y-auto rounded-lg border-[3px] border-crate-rust bg-[#0b0907]/95 px-3 py-2 text-[13px] font-semibold leading-tight text-crate-cream shadow-inner outline-none placeholder:text-crate-cream/45 placeholder:italic focus:border-crate-amber disabled:opacity-55 sm:h-[3.25rem] sm:min-h-[3.25rem] sm:text-[14px] md:h-14 md:min-h-[3.5rem] md:text-[15px]"
            />
            <button
              type="button"
              disabled={loading || !query.trim()}
              onClick={onDig}
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
  );
}
