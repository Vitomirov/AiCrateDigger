"use client";

type SearchExampleHintsProps = {
  examples: readonly string[];
  recipe: readonly string[];
  onPickExample: (example: string) => void;
};

export default function SearchExampleHints({
  examples,
  recipe,
  onPickExample,
}: SearchExampleHintsProps) {
  return (
    <div
      className="mt-3 flex w-full max-w-md flex-col items-center gap-2.5 px-2 sm:mt-4 sm:max-w-lg"
      aria-label="Search tips"
    >
      <p className="flex flex-wrap items-center justify-center gap-x-1.5 gap-y-0.5 text-[0.56rem] font-bold uppercase tracking-[0.34em] text-crate-cream/65 sm:text-[0.6rem]">
        {recipe.map((part, index) => (
          <span key={part} className="inline-flex items-center gap-1.5">
            {index > 0 ? (
              <span aria-hidden className="text-crate-amber/70">
                ·
              </span>
            ) : null}
            <span
              className="animate-pulse text-crate-cream/80"
              style={{ animationDelay: `${index * 450}ms`, animationDuration: "2.4s" }}
            >
              {part}
            </span>
          </span>
        ))}
      </p>
      <div className="flex w-full flex-wrap justify-center gap-2">
        {examples.map((example) => (
          <button
            key={example}
            type="button"
            onClick={() => onPickExample(example)}
            className="rounded-full border border-crate-cream/20 bg-black/40 px-3 py-1.5 text-[0.7rem] font-semibold leading-snug text-crate-cream/90 shadow-sm backdrop-blur-[2px] transition hover:border-crate-amber/55 hover:bg-black/55 hover:text-crate-cream active:scale-[0.98] sm:px-3.5 sm:py-2 sm:text-[0.76rem]"
          >
            {example}
          </button>
        ))}
      </div>
    </div>
  );
}
