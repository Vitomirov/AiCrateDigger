"""Fail fast when production env is misconfigured."""

from __future__ import annotations

import logging

from app.core.config import Settings

logger = logging.getLogger(__name__)


def validate_production_settings(settings: Settings) -> None:
    """Raise on startup when ``APP_ENV=production`` but security knobs are off."""
    if settings.app_env != "production":
        return

    errors: list[str] = []

    if settings.debug:
        errors.append("DEBUG must be false when APP_ENV=production")

    if not (settings.internal_api_secret or "").strip():
        errors.append("INTERNAL_API_SECRET must be set when APP_ENV=production")

    if not settings.search_rate_limit_enabled:
        errors.append("SEARCH_RATE_LIMIT_ENABLED must be true when APP_ENV=production")

    if not settings.search_rate_limit_fail_closed:
        errors.append("SEARCH_RATE_LIMIT_FAIL_CLOSED must be true when APP_ENV=production")

    if not settings.global_daily_quota_enabled:
        errors.append("GLOBAL_DAILY_QUOTA_ENABLED must be true when APP_ENV=production")

    if not settings.global_daily_quota_fail_closed:
        errors.append("GLOBAL_DAILY_QUOTA_FAIL_CLOSED must be true when APP_ENV=production")

    if not settings.database_enabled:
        errors.append("DATABASE_URL (or POSTGRES_*) must be set when APP_ENV=production")

    if not (settings.redis_url or "").strip():
        errors.append("REDIS_URL must be set when APP_ENV=production")

    if errors:
        message = "Production configuration invalid:\n- " + "\n- ".join(errors)
        logger.critical("production_config_invalid", extra={"stage": "startup", "errors": errors})
        raise RuntimeError(message)

    if settings.log_format != "json":
        logger.warning(
            "production_log_format_recommendation",
            extra={
                "stage": "startup",
                "status": "warn",
                "reason": "Set LOG_FORMAT=json for log aggregation in production",
            },
        )
