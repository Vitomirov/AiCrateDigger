/** Shape the middle dev-inspector column from `SearchResponseDto.debug`. */

export function buildPipelineInspectPayload(
  debug: Record<string, unknown> | null | undefined,
): unknown {
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
