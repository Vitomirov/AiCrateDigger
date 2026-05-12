"""Pure helpers for splitting store hostnames for Tavily ``include_domains`` limits."""


def chunk_include_domains(domains: list[str], max_per_chunk: int) -> list[list[str]]:
    """Split hostname list so each batch has at most ``max_per_chunk`` entries."""
    if not domains:
        return []
    cap = max(1, int(max_per_chunk))
    if len(domains) <= cap:
        return [domains]
    return [domains[i : i + cap] for i in range(0, len(domains), cap)]
