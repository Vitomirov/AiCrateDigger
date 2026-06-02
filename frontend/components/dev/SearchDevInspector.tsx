import type { SearchResponseDto } from "@/lib/api-types";
import { buildPipelineInspectPayload } from "@/lib/search-inspector";

import DevJsonPanel from "./DevJsonPanel";

type SearchDevInspectorProps = {
  loading: boolean;
  error: string | null;
  payload: SearchResponseDto | null;
};

export default function SearchDevInspector({ loading, error, payload }: SearchDevInspectorProps) {
  const parseSubtitle = error
    ? "POST /search → parsed (single round-trip)"
    : payload?.reason
      ? `reason: ${payload.reason} · POST /search → parsed`
      : "POST /search → parsed (single round-trip)";

  return (
    <div className="relative z-[2] w-full px-3 pb-3 sm:px-5">
      <p className="mb-2 text-center font-slab text-[0.62rem] font-semibold uppercase tracking-[0.28em] text-crate-cream/50">
        Pipeline inspector
      </p>
      <div className="mx-auto grid max-w-6xl grid-cols-1 gap-3 lg:grid-cols-3">
        <DevJsonPanel
          title="Parse"
          subtitle={parseSubtitle}
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
  );
}
