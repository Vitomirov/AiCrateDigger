"use client";

import { useCallback, useState } from "react";

import { postSearch, RateLimitError, type SearchResponseDto } from "@/lib/api";
import { useRampProgress } from "@/hooks/useRampProgress";

export function useDigSearch() {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [rateLimitOpen, setRateLimitOpen] = useState(false);
  const [payload, setPayload] = useState<SearchResponseDto | null>(null);
  /** After the first Dig, keep JSON panels mounted for debugging. */
  const [hasRunInspect, setHasRunInspect] = useState(false);

  const { pct, start: startProgress, cancel: cancelProgress, complete: completeProgress, reset: resetProgress } =
    useRampProgress();

  const dig = useCallback(async () => {
    const q = query.trim();
    if (!q || loading) {
      return;
    }
    setError(null);
    setHasRunInspect(true);
    setLoading(true);
    startProgress();
    try {
      const search = await postSearch(q);
      setPayload(search);
      completeProgress();
    } catch (e) {
      resetProgress();
      setPayload(null);
      if (e instanceof RateLimitError) {
        setError(null);
        setRateLimitOpen(true);
      } else {
        setError(e instanceof Error ? e.message : "Search failed.");
      }
    } finally {
      cancelProgress();
      setLoading(false);
      setTimeout(() => resetProgress(), 500);
    }
  }, [query, loading, startProgress, completeProgress, resetProgress, cancelProgress]);

  return {
    query,
    setQuery,
    loading,
    error,
    payload,
    rateLimitOpen,
    setRateLimitOpen,
    hasRunInspect,
    dig,
    progressPct: pct,
  };
}
