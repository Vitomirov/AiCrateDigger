"use client";

import { useEffect, useState } from "react";

const TYPE_MS = 48;
const DELETE_MS = 28;
const PAUSE_MS = 2600;

type Phase = "type" | "pause" | "delete";

export function useTypewriterLoop(examples: readonly string[], enabled: boolean): string {
  const [text, setText] = useState("");

  useEffect(() => {
    if (!enabled || examples.length === 0) {
      setText("");
      return;
    }

    let exampleIndex = 0;
    let charIndex = 0;
    let phase: Phase = "type";
    let timer: ReturnType<typeof setTimeout>;

    const tick = () => {
      const full = examples[exampleIndex] ?? "";

      if (phase === "type") {
        charIndex += 1;
        setText(full.slice(0, charIndex));
        if (charIndex >= full.length) {
          phase = "pause";
          timer = setTimeout(tick, PAUSE_MS);
          return;
        }
        timer = setTimeout(tick, TYPE_MS);
        return;
      }

      if (phase === "pause") {
        phase = "delete";
        timer = setTimeout(tick, DELETE_MS);
        return;
      }

      charIndex -= 1;
      setText(full.slice(0, charIndex));
      if (charIndex <= 0) {
        exampleIndex = (exampleIndex + 1) % examples.length;
        phase = "type";
        timer = setTimeout(tick, TYPE_MS);
        return;
      }
      timer = setTimeout(tick, DELETE_MS);
    };

    timer = setTimeout(tick, TYPE_MS);
    return () => clearTimeout(timer);
  }, [enabled, examples]);

  return text;
}
