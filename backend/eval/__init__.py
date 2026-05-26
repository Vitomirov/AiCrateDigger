"""Pipeline evaluation harness for AiCrateDigger.

Run locally::

    cd backend && poetry run python -m eval.cli

Run via Docker Compose::

    docker compose --profile eval run --rm eval
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
