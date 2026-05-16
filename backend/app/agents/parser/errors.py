"""Structured parser failures."""


class ParserError(RuntimeError):
    """Structured parser failure (e.g. LLM dead, invalid JSON). Not raised for partial intents."""
