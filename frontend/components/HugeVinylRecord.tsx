/** Full-bleed LP background — center cut out so the label UI sits in the groove. */

const LABEL_HOLE_R = 94;

/** Concentric groove rings; all radii > hole so every stroke is a full circle in the SVG (no false “oval” arcs from masking sub-rings). */
/** Three extra rings (194–202) extend past the label / title band so the grooves read like a full side, not a short runout. */
const GROOVE_RADII = [202, 198, 194, 190, 178, 166, 154, 142, 130, 118, 106];

export default function HugeVinylRecordBg({ className = "" }: { className?: string }) {
  return (
    <div
      className={`pointer-events-none flex items-center justify-center ${className}`}
      aria-hidden
    >
      {/* Strict square box + uniform SVG scale → circles stay circular on screen */}
      <div
        className="aspect-square w-[min(168vmin,78rem)] shrink-0 sm:w-[min(178vmin,90rem)] md:w-[min(192vmin,102rem)] lg:w-[min(208vmin,118rem)]"
      >
        <svg
          viewBox="0 0 420 420"
          width="420"
          height="420"
          preserveAspectRatio="xMidYMid meet"
          shapeRendering="geometricPrecision"
          className="block h-full w-full max-w-none drop-shadow-platter"
        >
          <defs>
            <linearGradient id="grooveRadialBg" x1="50%" y1="50%" x2="76%" y2="76%">
              <stop offset="0%" stopColor="#141210" />
              <stop offset="100%" stopColor="#070605" />
            </linearGradient>
            <mask id="crateLabelHole">
              <rect width="420" height="420" fill="white" />
              <circle cx="210" cy="210" r={LABEL_HOLE_R} fill="black" />
            </mask>
          </defs>
          <g mask="url(#crateLabelHole)">
            {/* Vinyl body */}
            <circle cx="210" cy="210" r="204" fill="url(#grooveRadialBg)" stroke="#050403" strokeWidth="8" />
            {/* Outer bands carry past the headline; last three rings are a touch softer */}
            {GROOVE_RADII.map((r, i) => (
              <circle
                key={r}
                cx="210"
                cy="210"
                r={r}
                fill="none"
                stroke={
                  i % 3 === 0 ? "#231f1c" : i % 3 === 1 ? "#171410" : "#2c2620"
                }
                strokeWidth={r >= 194 ? "0.9" : "1"}
                strokeOpacity={r >= 194 ? 0.82 : 0.94}
              />
            ))}
          </g>
        </svg>
      </div>
    </div>
  );
}
