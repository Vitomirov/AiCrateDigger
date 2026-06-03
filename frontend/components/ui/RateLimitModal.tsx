"use client";

import { useCallback, useEffect } from "react";

type RateLimitModalProps = {
  open: boolean;
  onClose: () => void;
};

export default function RateLimitModal({ open, onClose }: RateLimitModalProps) {
  const handleKeyDown = useCallback(
    (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        onClose();
      }
    },
    [onClose],
  );

  useEffect(() => {
    if (!open) {
      return;
    }
    document.addEventListener("keydown", handleKeyDown);
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", handleKeyDown);
      document.body.style.overflow = previousOverflow;
    };
  }, [open, handleKeyDown]);

  if (!open) {
    return null;
  }

  return (
    <div
      className="fixed inset-0 z-[100] flex items-center justify-center px-4 py-8"
      role="dialog"
      aria-modal="true"
      aria-labelledby="rate-limit-title"
    >
      <button
        type="button"
        aria-label="Close rate limit dialog"
        className="absolute inset-0 bg-black/72 backdrop-blur-[2px]"
        onClick={onClose}
      />

      <div className="relative z-[1] w-full max-w-md overflow-hidden rounded-2xl border-[3px] border-crate-gold bg-[rgba(253,243,220,0.98)] shadow-platter ring-2 ring-black/30">
        <div className="border-b border-crate-rust/20 bg-crate-rust px-6 py-4">
          <p className="font-slab text-[0.65rem] uppercase tracking-[0.42em] text-crate-amber">
            daily dig limit
          </p>
          <h2
            id="rate-limit-title"
            className="mt-1 font-slab text-[1.65rem] uppercase leading-none text-crate-cream"
          >
            Five digs per day
          </h2>
        </div>

        <div className="px-6 py-5">
          <p className="text-[0.92rem] font-medium leading-relaxed text-crate-rust">
            Hey there! Glad you are digging for crates with us! To keep AiCrateDigger free and
            protect our API budgets, we limit searches to{" "}
            <span className="font-bold text-crate-rust">5 requests per 24 hours</span>
            .Grab a coffee, spin a record, and come back tomorrow!
          </p>

          <button
            type="button"
            onClick={onClose}
            className="mt-6 w-full rounded-lg border-[3px] border-crate-cream bg-crate-rust py-3 font-slab text-[0.95rem] uppercase tracking-[0.22em] text-crate-cream shadow-lg transition-colors hover:bg-[#4a2c20] active:translate-y-px"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
