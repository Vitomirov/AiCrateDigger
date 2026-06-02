"use client";

import { useCallback, useEffect, useRef, useState } from "react";

export function useRampProgress() {
  const [progress, setProgress] = useState(0);
  const progressTimer = useRef<ReturnType<typeof setInterval> | null>(null);
  const t0 = useRef(0);

  const cancel = useCallback(() => {
    if (progressTimer.current) {
      clearInterval(progressTimer.current);
      progressTimer.current = null;
    }
  }, []);

  useEffect(() => () => cancel(), [cancel]);

  const start = useCallback(() => {
    cancel();
    t0.current = Date.now();
    setProgress(5);
    progressTimer.current = setInterval(() => {
      const elapsed = Date.now() - t0.current;
      const next = Math.min(90, 6 + (elapsed / 10000) * 84);
      setProgress((p: number) => (p < next ? Math.floor(next) : p));
    }, 140);
  }, [cancel]);

  const complete = useCallback(() => {
    setProgress(100);
  }, []);

  const reset = useCallback(() => {
    setProgress(0);
  }, []);

  const pct = Math.min(100, Math.max(0, progress));

  return { progress, pct, start, cancel, complete, reset };
}
