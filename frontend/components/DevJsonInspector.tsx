"use client";

import { useCallback, useState } from "react";

function stableStringify(value: unknown): string {
  if (value === undefined) {
    return "";
  }
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

type DevJsonPanelProps = {
  title: string;
  subtitle?: string;
  /** When set, shown instead of JSON (e.g. request error). */
  error: string | null;
  data: unknown;
  loading?: boolean;
};

export function DevJsonPanel({ title, subtitle, error, data, loading }: DevJsonPanelProps) {
  const [copied, setCopied] = useState(false);
  const bodyText = error
    ? error
    : loading
      ? "…"
      : data === null || data === undefined
        ? "Run a search to populate this panel."
        : stableStringify(data);

  const onCopy = useCallback(async () => {
    if (!bodyText || loading) {
      return;
    }
    try {
      await navigator.clipboard.writeText(bodyText);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1600);
    } catch {
      setCopied(false);
    }
  }, [bodyText, loading]);

  return (
    <section className="flex min-h-0 flex-1 flex-col rounded-lg border-2 border-crate-gold/40 bg-black/70 shadow-inner">
      <div className="flex shrink-0 items-start justify-between gap-2 border-b border-crate-gold/25 px-3 py-2">
        <div className="min-w-0">
          <h2 className="font-slab text-[0.68rem] font-semibold uppercase tracking-[0.22em] text-crate-amber">
            {title}
          </h2>
          {subtitle ? (
            <p className="mt-0.5 text-[0.65rem] font-medium leading-snug text-crate-cream/55">{subtitle}</p>
          ) : null}
        </div>
        <button
          type="button"
          disabled={loading || !bodyText || bodyText === "…"}
          onClick={() => void onCopy()}
          className="shrink-0 rounded border border-crate-cream/25 bg-crate-rust/80 px-2 py-1 text-[0.6rem] font-bold uppercase tracking-wider text-crate-cream transition enabled:hover:border-crate-amber enabled:hover:text-crate-amber disabled:opacity-35"
        >
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
      <pre
        className={`min-h-[12rem] flex-1 overflow-auto p-3 font-mono text-[0.68rem] leading-relaxed sm:min-h-[14rem] sm:text-[0.72rem] ${
          error ? "text-red-200/95" : "text-crate-cream/90"
        }`}
      >
        {bodyText}
      </pre>
    </section>
  );
}

/** Middle column: emphasize tavily / parse stages when `debug` is present. */
export function buildPipelineInspectPayload(debug: Record<string, unknown> | null | undefined): unknown {
  if (!debug || typeof debug !== "object") {
    return {
      _note:
        "No `debug` on the search response. Set DEBUG=true on the backend to record tavily, extract, and validate stages.",
    };
  }
  const stages = debug.stages;
  const trace = debug.trace;
  const requestId = debug.request_id;
  return {
    request_id: requestId,
    stages,
    trace,
  };
}
