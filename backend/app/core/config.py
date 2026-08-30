"""Application configuration.

All environment-specific values are read from environment variables (or an
optional local ``.env`` file) via ``pydantic-settings``. No secrets or
production endpoints are hardcoded here: the defaults below are safe,
non-sensitive values intended for local development only, and every value is
overridable through the environment.

Requirements: 13.1 (env-var config), 13.4 (no secrets in repo),
18.6 (credentials loaded from the environment only).
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strongly-typed application settings sourced from the environment.

    Field values are resolved in this order: explicit environment variable,
    then a matching entry in a local ``.env`` file (if present), then the
    safe local-development default declared below.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # PostgreSQL connection URL (SQLAlchemy format). Overridden in every
    # deployed environment; this default only serves local development and
    # contains no real credentials.
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/pricetruth_db"

    # Redis connection URL used for caching.
    REDIS_URL: str = "redis://localhost:6379/0"

    # Open Food Facts public API base URL and version. v2 is the documented
    # default and can be upgraded (e.g. to v3) purely through configuration,
    # without code changes.
    OFF_BASE_URL: str = "https://world.openfoodfacts.org"
    OFF_VERSION: str = "v2"

    # The single frontend origin permitted to make cross-origin requests.
    CORS_ALLOWED_ORIGIN: str = "http://localhost:5173"


@lru_cache
def get_settings() -> Settings:
    """Return a process-wide cached :class:`Settings` instance.

    The result is cached so configuration is parsed once per process and the
    same immutable settings object is reused across all requests.
    """

    return Settings()
